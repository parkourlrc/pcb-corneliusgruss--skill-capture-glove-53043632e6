"""Validate that sessions are training-ready.

Checks every session end-to-end: file completeness, data alignment,
trajectory quality, encoder variance, ToF sanity, image loadability,
and normalization stats. Run this before queueing a training job.

Usage:
    python -m glove.scripts.validate_dataset --data-dir data/raw
    python -m glove.scripts.validate_dataset --dataset datasets/mug_coaster_v1.yaml
    python -m glove.scripts.validate_dataset --data-dir data/raw --fix  # auto-skip bad sessions
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from glove.data.session import SessionLoader
from glove.data.transforms import compute_relative_transform, relative_transform_to_13d
from glove.data.training import (
    TrainingDataset,
    NormalizationStats,
    compute_normalization_stats,
)


# Thresholds
MAX_TRAJECTORY_M = 50.0        # reject if total path > 50m (real divergence is 100m+)
MIN_TRAJECTORY_M = 0.05        # reject if hand barely moved
MAX_JUMP_M = 0.5               # single-frame jump threshold
MAX_JUMP_FRACTION = 0.02       # reject if >2% of frames have jumps
MIN_ENCODER_STD = 0.5          # degrees — reject if encoders are stuck
MIN_FRAMES = 50                # need at least 50 frames
MAX_MISSING_FRAMES = 0.05      # reject if >5% of top/bottom frames missing
MIN_TOF_NONZERO_FRACTION = 0.5 # at least 50% of ToF readings should be nonzero


def validate_session(session_dir: Path, verbose: bool = True) -> dict:
    """Validate a single session for training readiness.

    Returns dict with 'valid', 'warnings', 'errors', and per-check details.
    """
    result = {
        "session": session_dir.name,
        "valid": True,
        "warnings": [],
        "errors": [],
        "checks": {},
    }

    def error(name: str, msg: str):
        result["checks"][name] = {"pass": False, "detail": msg}
        result["errors"].append(f"{name}: {msg}")
        result["valid"] = False
        if verbose:
            print(f"    [FAIL] {name}: {msg}")

    def warn(name: str, msg: str):
        result["checks"][name] = {"pass": True, "detail": msg, "warning": True}
        result["warnings"].append(f"{name}: {msg}")
        if verbose:
            print(f"    [WARN] {name}: {msg}")

    def ok(name: str, msg: str):
        result["checks"][name] = {"pass": True, "detail": msg}
        if verbose:
            print(f"    [ OK ] {name}: {msg}")

    if verbose:
        print(f"\n  {session_dir.name}")

    # --- 1. Load session ---
    try:
        session = SessionLoader(session_dir, require_poses=True)
    except Exception as e:
        error("load", str(e))
        return result

    # --- 2. File completeness ---
    top_dir = session_dir / "top"
    bottom_dir = session_dir / "bottom"
    poses_file = session_dir / "poses.json"
    encoders_file = session_dir / "encoders.json"

    if not poses_file.exists():
        error("poses_exist", "No poses.json — run OKVIS first")
        return result

    if session.num_poses < MIN_FRAMES:
        error("min_frames", f"{session.num_poses} poses < {MIN_FRAMES} minimum")
        return result
    ok("poses_exist", f"{session.num_poses} poses")

    if not top_dir.is_dir():
        error("top_exist", "No top/ directory — run process_sessions.sh")
        return result
    ok("top_exist", f"{session.num_rgb_frames} top camera frames")

    if not bottom_dir.is_dir():
        error("bottom_exist", "No bottom/ directory — run process_sessions.sh")
        return result
    ok("bottom_exist", f"{session.num_rgb_frames_right} bottom camera frames")

    if not encoders_file.exists():
        error("encoders_exist", "No encoders.json")
        return result
    ok("encoders_exist", f"{session.num_encoder_readings} readings")

    # --- 3. Frame count alignment ---
    n_poses = session.num_poses
    n_rgb = session.num_rgb_frames
    n_rgb_r = session.num_rgb_frames_right

    if n_rgb < n_poses:
        missing_pct = (n_poses - n_rgb) / n_poses
        if missing_pct > MAX_MISSING_FRAMES:
            error("top_alignment", f"top has {n_rgb} frames but {n_poses} poses ({missing_pct:.0%} missing)")
        else:
            warn("top_alignment", f"top has {n_rgb} frames for {n_poses} poses ({missing_pct:.1%} missing)")
    else:
        ok("top_alignment", f"{n_rgb} top camera frames for {n_poses} poses")

    if n_rgb_r < n_poses:
        missing_pct = (n_poses - n_rgb_r) / n_poses
        if missing_pct > MAX_MISSING_FRAMES:
            error("bottom_alignment", f"bottom has {n_rgb_r} frames but {n_poses} poses ({missing_pct:.0%} missing)")
        else:
            warn("bottom_alignment", f"bottom has {n_rgb_r} for {n_poses} poses ({missing_pct:.1%} missing)")
    else:
        ok("bottom_alignment", f"{n_rgb_r} bottom camera frames for {n_poses} poses")

    # --- 4. Image loadability (spot check) ---
    check_indices = [0, n_poses // 2, min(n_poses - 1, n_rgb - 1)]
    load_failures = 0
    for idx in check_indices:
        try:
            img = session.get_rgb_frame(idx)
            if img.ndim != 3 or img.shape[2] != 3:
                load_failures += 1
        except Exception:
            load_failures += 1
        try:
            img_r = session.get_rgb_frame_right(idx)
            if img_r.ndim != 3 or img_r.shape[2] != 3:
                load_failures += 1
        except Exception:
            load_failures += 1

    if load_failures > 0:
        error("image_load", f"{load_failures}/{len(check_indices)*2} spot-check images failed to load")
    else:
        ok("image_load", f"{len(check_indices)*2}/{len(check_indices)*2} spot-check images OK")

    # --- 5. Trajectory quality ---
    transforms = session.get_all_transforms()  # (T, 4, 4)
    positions = transforms[:, :3, 3]  # (T, 3)

    # Total path length
    deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    path_length = float(deltas.sum())
    result["checks"]["path_length"] = {"value": path_length}

    if path_length > MAX_TRAJECTORY_M:
        error("trajectory_range", f"Path length {path_length:.2f}m > {MAX_TRAJECTORY_M}m — likely OKVIS divergence")
    elif path_length < MIN_TRAJECTORY_M:
        error("trajectory_range", f"Path length {path_length:.3f}m < {MIN_TRAJECTORY_M}m — hand barely moved")
    else:
        ok("trajectory_range", f"Path length {path_length:.2f}m")

    # Frame-to-frame jumps
    jumps = deltas > MAX_JUMP_M
    jump_count = int(jumps.sum())
    jump_frac = jump_count / max(len(deltas), 1)

    if jump_frac > MAX_JUMP_FRACTION:
        error("trajectory_jumps", f"{jump_count} jumps > {MAX_JUMP_M}m ({jump_frac:.1%} of frames)")
    elif jump_count > 0:
        warn("trajectory_jumps", f"{jump_count} jumps > {MAX_JUMP_M}m ({jump_frac:.1%})")
    else:
        ok("trajectory_jumps", "No large jumps")

    # NaN check
    nan_count = int(np.isnan(transforms).any(axis=(1, 2)).sum())
    if nan_count > 0:
        error("trajectory_nan", f"{nan_count} poses contain NaN")
    else:
        ok("trajectory_nan", "No NaN in poses")

    # --- 6. Encoder quality ---
    encoders = session.get_aligned_encoders()  # (T, 4)
    encoder_std = encoders.std(axis=0)
    encoder_names = ["index_mcp", "index_pip", "three_mcp", "three_pip"]

    stuck_channels = []
    for i, (name, std) in enumerate(zip(encoder_names, encoder_std)):
        if std < MIN_ENCODER_STD:
            stuck_channels.append(name)

    if len(stuck_channels) == 4:
        error("encoder_variance", f"All encoders stuck (std < {MIN_ENCODER_STD}°)")
    elif stuck_channels:
        warn("encoder_variance", f"Low variance: {', '.join(stuck_channels)} (std < {MIN_ENCODER_STD}°)")
    else:
        ok("encoder_variance", f"std = [{', '.join(f'{s:.1f}°' for s in encoder_std)}]")

    # --- 7. ToF quality ---
    tof = session.get_aligned_tof()  # (T, 2)
    tof_nonzero = (tof > 0.001).mean(axis=0)  # fraction of nonzero readings per sensor

    if tof_nonzero.mean() < MIN_TOF_NONZERO_FRACTION:
        warn("tof_quality", f"Low ToF coverage: top={tof_nonzero[0]:.0%}, bot={tof_nonzero[1]:.0%}")
    else:
        tof_range_top = f"{tof[tof[:, 0] > 0, 0].min():.3f}-{tof[:, 0].max():.3f}m" if tof_nonzero[0] > 0 else "none"
        tof_range_bot = f"{tof[tof[:, 1] > 0, 1].min():.3f}-{tof[:, 1].max():.3f}m" if tof_nonzero[1] > 0 else "none"
        ok("tof_quality", f"top={tof_range_top}, bot={tof_range_bot}")

    # --- 8. Metadata ---
    meta = session.metadata
    if not meta.get("task"):
        warn("metadata_task", "No task tag in metadata.json")
    else:
        ok("metadata_task", f"task={meta['task']}")

    if meta.get("slamRejected"):
        error("slam_rejected", "Session was rejected by OKVIS quality gate")

    return result


def validate_normalization(stats: NormalizationStats, verbose: bool = True) -> list[str]:
    """Check normalization stats for degenerate values."""
    issues = []

    # Action range should be nonzero for all dimensions
    action_range = stats.action_max - stats.action_min
    zero_dims = np.where(action_range < 1e-6)[0]
    if len(zero_dims) > 0:
        issues.append(f"Zero action range on dims {zero_dims.tolist()} — data has no variance")

    # Proprio std should be nonzero
    low_std = np.where(stats.proprio_std < 1e-6)[0]
    if len(low_std) > 0:
        issues.append(f"Zero proprio std on dims {low_std.tolist()}")

    # Position range sanity (first 3 dims should be < 5m for tabletop)
    pos_range = action_range[:3]
    if (pos_range > 5.0).any():
        issues.append(f"Position range suspiciously large: {pos_range} — check for OKVIS divergence")

    # Rotation dims (3:9) should have range roughly in [0, 2] for 6D rotation
    rot_range = action_range[3:9]
    if (rot_range > 4.0).any():
        issues.append(f"Rotation range large: {rot_range}")

    if verbose:
        print("\n  Normalization stats:")
        print(f"    Proprio mean: {stats.proprio_mean}")
        print(f"    Proprio std:  {stats.proprio_std}")
        print(f"    Action min:   {stats.action_min}")
        print(f"    Action max:   {stats.action_max}")
        print(f"    Action range: {action_range}")
        if issues:
            for issue in issues:
                print(f"    [WARN] {issue}")
        else:
            print("    [ OK ] Stats look reasonable")

    return issues


def validate_dataset_sampling(
    session_dirs: list[Path],
    obs_horizon: int = 2,
    action_horizon: int = 16,
    verbose: bool = True,
) -> dict:
    """Try to construct a dataset and sample from it."""
    result = {"valid": True, "errors": [], "num_samples": 0}

    try:
        dataset = TrainingDataset(
            session_dirs,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
        )
        result["num_samples"] = len(dataset)

        if len(dataset) == 0:
            result["valid"] = False
            result["errors"].append("Dataset has 0 samples")
            return result

        # Sample a few items
        for i in [0, len(dataset) // 2, len(dataset) - 1]:
            images, proprio, actions = dataset[i]
            assert images.shape == (obs_horizon, 2, 3, 224, 224), f"Image shape: {images.shape}"
            assert proprio.shape == (obs_horizon, 15), f"Proprio shape: {proprio.shape}"
            assert actions.shape == (action_horizon, 13), f"Action shape: {actions.shape}"
            assert not torch.isnan(images).any(), "NaN in images"
            assert not torch.isnan(proprio).any(), "NaN in proprio"
            assert not torch.isnan(actions).any(), "NaN in actions"

        # Validate normalization stats
        norm_issues = validate_normalization(dataset.stats, verbose=verbose)
        if norm_issues:
            result["warnings"] = norm_issues

        if verbose:
            print(f"\n  Dataset: {len(dataset)} samples from {len(dataset.sessions)} sessions")
            print(f"    Sample shapes: images={images.shape}, proprio={proprio.shape}, actions={actions.shape}")
            print(f"    [ OK ] 3 random samples loaded and validated")

    except Exception as e:
        result["valid"] = False
        result["errors"].append(str(e))
        if verbose:
            print(f"\n    [FAIL] Dataset construction failed: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate dataset for training readiness")
    parser.add_argument("--data-dir", default="data/raw", help="Session data directory")
    parser.add_argument("--dataset", default=None, help="Dataset manifest YAML")
    parser.add_argument("--skip-sampling", action="store_true", help="Skip dataset sampling test")
    parser.add_argument("--quiet", action="store_true", help="Only print summary")
    args = parser.parse_args()

    verbose = not args.quiet

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    # Discover sessions
    if args.dataset:
        from glove.data.dataset import DatasetManifest
        manifest = DatasetManifest.load(args.dataset)
        session_dirs = manifest.discover_sessions()
        print(f"Manifest: {manifest.name} (task={manifest.task})")
    else:
        data_dir = Path(args.data_dir)
        session_dirs = sorted(
            d for d in data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and (d / "poses.json").exists()
        )
        print(f"Data dir: {data_dir}")

    print(f"Sessions found: {len(session_dirs)}")

    if not session_dirs:
        print("\nNo sessions found. Nothing to validate.")
        sys.exit(1)

    # Validate each session
    results = []
    for session_dir in session_dirs:
        r = validate_session(session_dir, verbose=verbose)
        results.append(r)

    valid = [r for r in results if r["valid"]]
    invalid = [r for r in results if not r["valid"]]
    warned = [r for r in valid if r["warnings"]]

    print(f"\n{'=' * 60}")
    print("SESSION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Valid:    {len(valid)}/{len(results)}")
    print(f"  Invalid:  {len(invalid)}/{len(results)}")
    print(f"  Warnings: {len(warned)}/{len(results)}")

    if invalid:
        print(f"\n  Rejected sessions:")
        for r in invalid:
            print(f"    {r['session']}: {r['errors'][0]}")

    if warned:
        print(f"\n  Sessions with warnings:")
        for r in warned:
            for w in r["warnings"]:
                print(f"    {r['session']}: {w}")

    # Dataset sampling test
    valid_dirs = [Path(args.data_dir) / r["session"] if not args.dataset else
                  next(d for d in session_dirs if d.name == r["session"])
                  for r in valid]

    if not args.skip_sampling and valid_dirs:
        print(f"\n{'=' * 60}")
        print("DATASET SAMPLING TEST")
        print(f"{'=' * 60}")

        import torch  # late import — only needed for sampling
        ds_result = validate_dataset_sampling(valid_dirs, verbose=verbose)

        if not ds_result["valid"]:
            print(f"\n  [FAIL] {ds_result['errors']}")
            sys.exit(1)

    # Final verdict
    print(f"\n{'=' * 60}")
    if len(valid) == 0:
        print("RESULT: NO VALID SESSIONS — cannot train")
        sys.exit(1)
    elif len(invalid) > 0:
        print(f"RESULT: {len(valid)} sessions ready, {len(invalid)} rejected")
        print(f"Training-ready sessions:")
        for r in valid:
            print(f"  {r['session']}")
        sys.exit(0)
    else:
        print(f"RESULT: ALL {len(valid)} SESSIONS READY FOR TRAINING")
        sys.exit(0)


if __name__ == "__main__":
    main()
