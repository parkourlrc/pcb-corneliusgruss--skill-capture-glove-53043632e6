"""
End-to-end pipeline smoke test: generate → train → evaluate.

Validates the full pipeline works with synthetic data using a tiny model.
This is the kind of regression gate that labs like DeepMind and Sunday run
on every PR — if this breaks, the pipeline has a bug.

Usage:
    python -m glove.scripts.smoke_test                    # full pipeline
    python -m glove.scripts.smoke_test --keep-data        # don't delete temp data
    python -m glove.scripts.smoke_test --output /tmp/smoke # custom output dir

Expected runtime: ~1-2 min on CPU (M-series Mac), ~30s on GPU.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from glove.data.synthetic import generate_synthetic_dataset
from glove.data.session import SessionLoader
from glove.data.training import TrainingDataset, compute_normalization_stats
from glove.data.transforms import rot6d_angular_error
from glove.policy.diffusion import DiffusionPolicy, PolicyConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def step_generate(output_dir: Path, num_sessions: int, num_frames: int) -> list[Path]:
    """Step 1: Generate synthetic sessions."""
    data_dir = output_dir / "sessions"
    sessions = generate_synthetic_dataset(
        data_dir,
        num_sessions=num_sessions,
        frames_per_session=num_frames,
        seed=42,
    )
    # Verify structure
    for s in sessions:
        assert (s / "data" / "episode.parquet").exists(), f"Missing episode.parquet in {s}"
        assert (s / "data" / "poses.parquet").exists(), f"Missing poses.parquet in {s}"
        assert (s / "videos" / "cam_top.mp4").exists(), f"Missing cam_top.mp4 in {s}"
        assert (s / "videos" / "cam_bottom.mp4").exists(), f"Missing cam_bottom.mp4 in {s}"
    return sessions


def step_load_and_validate(session_dirs: list[Path]) -> list[SessionLoader]:
    """Step 2: Load sessions via SessionLoader, validate data shapes."""
    sessions = []
    for d in session_dirs:
        s = SessionLoader(d)
        transforms = s.get_all_transforms()
        joints = s.get_aligned_encoders()
        tof = s.get_aligned_tof()
        T = len(transforms)

        assert transforms.shape == (T, 4, 4), f"Bad transform shape: {transforms.shape}"
        assert joints.shape == (T, 4), f"Bad joints shape: {joints.shape}"
        assert tof.shape == (T, 2), f"Bad tof shape: {tof.shape}"

        # Check video frames loadable
        frame = s.get_rgb_frame(0)
        assert frame.ndim == 3 and frame.shape[2] == 3, f"Bad frame shape: {frame.shape}"

        frame_r = s.get_rgb_frame_right(0)
        assert frame_r.ndim == 3 and frame_r.shape[2] == 3

        sessions.append(s)
    return sessions


def step_train(
    train_dirs: list[Path],
    val_dirs: list[Path],
    config: PolicyConfig,
    device: torch.device,
    output_dir: Path,
    num_epochs: int,
) -> tuple[Path, list[float], list[float]]:
    """Step 3: Train for a few epochs, return checkpoint path and loss curves."""
    from torch.utils.data import DataLoader

    train_dataset = TrainingDataset(
        train_dirs,
        obs_horizon=config.obs_horizon,
        action_horizon=config.action_horizon,
        image_size=config.image_size,
        action_dim=config.action_dim,
        augment=False,
    )
    val_dataset = TrainingDataset(
        val_dirs,
        obs_horizon=config.obs_horizon,
        action_horizon=config.action_horizon,
        image_size=config.image_size,
        action_dim=config.action_dim,
        stats=train_dataset.stats,
    )

    assert len(train_dataset) > 0, "No training samples"
    assert len(val_dataset) > 0, "No validation samples"

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    policy = DiffusionPolicy(config).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)

    param_count = sum(p.numel() for p in policy.parameters())
    trainable_count = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,} total, {trainable_count:,} trainable")
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Train
        policy.train()
        epoch_loss = 0.0
        count = 0
        for images, proprio, actions in train_loader:
            images = images.to(device)
            proprio = proprio.to(device)
            actions = actions.to(device)

            loss_dict = policy(images, proprio, actions)
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * images.shape[0]
            count += images.shape[0]

        train_loss = epoch_loss / max(count, 1)
        train_losses.append(train_loss)

        # Validate
        policy.eval()
        val_loss_total = 0.0
        val_count = 0
        with torch.no_grad():
            for images, proprio, actions in val_loader:
                images = images.to(device)
                proprio = proprio.to(device)
                actions = actions.to(device)
                loss_dict = policy(images, proprio, actions)
                val_loss_total += loss_dict["loss"].item() * images.shape[0]
                val_count += images.shape[0]

        val_loss = val_loss_total / max(val_count, 1)
        val_losses.append(val_loss)

        print(f"  Epoch {epoch+1}/{num_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    # Save checkpoint
    checkpoint_path = output_dir / "checkpoint_smoke.pt"
    torch.save({
        "epoch": num_epochs - 1,
        "model": policy.state_dict(),
        "ema_model": policy.state_dict(),  # no EMA in smoke test
        "optimizer": optimizer.state_dict(),
        "config": vars(config),
        "policy_type": "diffusion",
        "normalizer": train_dataset.stats.state_dict(),
        "val_loss": val_losses[-1],
    }, checkpoint_path)

    return checkpoint_path, train_losses, val_losses


def step_evaluate(
    checkpoint_path: Path,
    val_dirs: list[Path],
    config: PolicyConfig,
    device: torch.device,
) -> dict:
    """Step 4: Load checkpoint and evaluate on val sessions."""
    from glove.data.training import NormalizationStats
    from glove.data.transforms import compute_relative_transform, relative_transform_to_13d

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy = DiffusionPolicy(config).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

    stats = NormalizationStats.from_state_dict(checkpoint["normalizer"])

    from glove.data.training import IMAGENET_MEAN, IMAGENET_STD
    from PIL import Image as PILImage

    all_pos_errors = []
    all_rot_errors = []
    all_joint_errors = []

    for session_dir in val_dirs:
        session = SessionLoader(session_dir)
        transforms = session.get_all_transforms()
        joints = session.get_aligned_encoders()
        tof = session.get_aligned_tof()
        T = len(transforms)

        if T < config.obs_horizon + config.action_horizon:
            continue

        # Evaluate a few windows (not all, for speed)
        eval_indices = list(range(
            config.obs_horizon - 1,
            min(T - config.action_horizon, config.obs_horizon - 1 + 10),
        ))

        for t in eval_indices:
            obs_start = t - config.obs_horizon + 1

            # Build obs images
            obs_images = np.zeros(
                (1, config.obs_horizon, 2, 3, config.image_size, config.image_size),
                dtype=np.float32,
            )
            for i in range(config.obs_horizon):
                fi = obs_start + i
                for cam_idx, get_fn in enumerate([session.get_rgb_frame, session.get_rgb_frame_right]):
                    img = get_fn(fi)
                    pil = PILImage.fromarray(img)
                    pil = pil.resize((config.image_size, config.image_size), PILImage.BILINEAR)
                    arr = np.array(pil, dtype=np.float32) / 255.0
                    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
                    obs_images[0, i, cam_idx] = arr.transpose(2, 0, 1)

            # Build obs proprio
            T_ref = transforms[t]
            obs_proprio = np.zeros((config.obs_horizon, config.obs_dim), dtype=np.float32)
            for i in range(config.obs_horizon):
                fi = obs_start + i
                T_rel = compute_relative_transform(T_ref, transforms[fi])
                vec13 = relative_transform_to_13d(T_rel, joints[fi])
                obs_proprio[i, :13] = vec13
                obs_proprio[i, 13:] = tof[fi]
            obs_proprio = stats.normalize_proprio(obs_proprio)[np.newaxis]

            # Predict
            with torch.no_grad():
                pred_norm = policy.predict(
                    torch.from_numpy(obs_images).to(device),
                    torch.from_numpy(obs_proprio).to(device),
                ).cpu().numpy()[0]

            pred = stats.unnormalize_action(pred_norm)

            # Ground truth
            gt = np.zeros((config.action_horizon, config.action_dim), dtype=np.float32)
            for i in range(config.action_horizon):
                fi = t + 1 + i
                T_rel = compute_relative_transform(T_ref, transforms[fi])
                gt[i] = relative_transform_to_13d(T_rel, joints[fi])

            all_pos_errors.append(np.linalg.norm(pred[:, :3] - gt[:, :3], axis=-1).mean())
            all_rot_errors.append(rot6d_angular_error(pred[:, 3:9], gt[:, 3:9]).mean())
            all_joint_errors.append(np.abs(pred[:, 9:] - gt[:, 9:]).mean())

    return {
        "position_error": float(np.mean(all_pos_errors)),
        "rotation_error_deg": float(np.mean(all_rot_errors)),
        "joint_error_deg": float(np.mean(all_joint_errors)),
        "num_windows": len(all_pos_errors),
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline smoke test")
    parser.add_argument("--output", default=None, help="Output directory (default: temp dir)")
    parser.add_argument("--keep-data", action="store_true", help="Keep generated data after test")
    parser.add_argument("--num-sessions", type=int, default=6, help="Number of synthetic sessions")
    parser.add_argument("--num-frames", type=int, default=50, help="Frames per session")
    parser.add_argument("--num-epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--device", default=None, help="Device override")
    args = parser.parse_args()

    if args.output:
        output_dir = Path(args.output)
    else:
        import tempfile
        output_dir = Path(tempfile.mkdtemp(prefix="glove_smoke_"))

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else get_device()

    # Tiny config for fast iteration
    config = PolicyConfig(
        obs_dim=15,
        action_dim=13,
        image_size=64,
        action_horizon=4,
        obs_horizon=2,
        num_diffusion_steps=10,
        hidden_dim=32,
        num_layers=2,
        batch_size=4,
        learning_rate=1e-3,
        num_epochs=args.num_epochs,
        freeze_backbone=True,
    )

    print("=" * 60)
    print("GLOVE PIPELINE SMOKE TEST")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"  Sessions: {args.num_sessions} x {args.num_frames} frames")
    print(f"  Config: hidden={config.hidden_dim}, layers={config.num_layers}, "
          f"img={config.image_size}, horizon={config.action_horizon}")
    print("=" * 60)

    t0 = time.time()
    passed = True

    # --- Step 1: Generate ---
    print("\n[1/4] Generating synthetic data...")
    t1 = time.time()
    session_dirs = step_generate(output_dir, args.num_sessions, args.num_frames)
    print(f"  Generated {len(session_dirs)} sessions ({time.time()-t1:.1f}s)")

    # --- Step 2: Load & Validate ---
    print("\n[2/4] Loading sessions & validating data shapes...")
    t2 = time.time()
    sessions = step_load_and_validate(session_dirs)
    print(f"  All {len(sessions)} sessions valid ({time.time()-t2:.1f}s)")

    # --- Step 3: Train ---
    print("\n[3/4] Training diffusion policy...")
    t3 = time.time()

    # Split: first 80% train, last 20% val
    n_val = max(1, len(session_dirs) // 5)
    train_dirs = session_dirs[:-n_val]
    val_dirs = session_dirs[-n_val:]
    print(f"  Split: {len(train_dirs)} train, {len(val_dirs)} val")

    checkpoint_path, train_losses, val_losses = step_train(
        train_dirs, val_dirs, config, device, output_dir, args.num_epochs,
    )
    print(f"  Training complete ({time.time()-t3:.1f}s)")
    print(f"  Checkpoint: {checkpoint_path}")

    # Check training loss decreased
    if len(train_losses) >= 2 and train_losses[-1] >= train_losses[0]:
        print(f"  WARNING: training loss did not decrease ({train_losses[0]:.4f} → {train_losses[-1]:.4f})")
        print(f"  This is OK for 3 epochs on random data — it would decrease with more epochs.")

    # --- Step 4: Evaluate ---
    print("\n[4/4] Evaluating on validation set...")
    t4 = time.time()
    metrics = step_evaluate(checkpoint_path, val_dirs, config, device)
    print(f"  Evaluated {metrics['num_windows']} windows ({time.time()-t4:.1f}s)")
    print(f"  Position error:  {metrics['position_error']:.4f}")
    print(f"  Rotation error:  {metrics['rotation_error_deg']:.1f} deg")
    print(f"  Joint error:     {metrics['joint_error_deg']:.1f} deg")

    # Sanity checks — we don't expect good performance, just finite values
    for key in ["position_error", "rotation_error_deg", "joint_error_deg"]:
        if not np.isfinite(metrics[key]):
            print(f"  FAIL: {key} is not finite!")
            passed = False

    if metrics["num_windows"] == 0:
        print("  FAIL: no evaluation windows produced!")
        passed = False

    # --- Summary ---
    total_time = time.time() - t0
    print("\n" + "=" * 60)
    if passed:
        print(f"SMOKE TEST PASSED ({total_time:.1f}s)")
    else:
        print(f"SMOKE TEST FAILED ({total_time:.1f}s)")
    print("=" * 60)

    if not args.keep_data and not args.output:
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"Cleaned up {output_dir}")
    else:
        print(f"Data preserved at {output_dir}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
