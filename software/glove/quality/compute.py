"""Compute quality metrics from `aligned/episode.parquet`.

Reference implementation for the schema in `glove.quality.schema`. Pure:
read episode.parquet + meta.yaml → QualityBlock. The caller decides whether
to persist (see `glove.quality.backfill`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import yaml

from glove.quality.schema import (
    QualityBlock,
    QualityMetrics,
    THRESHOLDS,
    compute_verdict,
)


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angle between two quaternion arrays (N,4) in degrees. Unit-quat assumed."""
    dot = np.abs(np.sum(q1 * q2, axis=1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.degrees(np.arccos(dot))


def _longest_run(mask: np.ndarray) -> int:
    """Longest consecutive True run in a boolean array."""
    if mask.size == 0:
        return 0
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _local_straightness(xyz: np.ndarray, valid: np.ndarray, fps: float, window_s: float = 0.25) -> float:
    """Median chord/path-length over rolling `window_s` windows.

    1.0 = perfectly straight motion inside the window.
    Values < ~0.8 indicate back-and-forth wobble (path doubles back on itself
    within the window, so cumulative path length grows faster than the chord).

    Threshold values were tuned against a known-clean reference demo:
    median straightness ≈0.85 over 0.25 s windows for a clean trajectory;
    pose-graph re-anchoring artifacts typically sit in the 0.68–0.76 range.

    Returns 1.0 when there are not enough valid frames to score.
    """
    n = len(xyz)
    if n < 4:
        return 1.0
    win = max(2, int(round(fps * window_s)))
    if win >= n:
        return 1.0

    # Per-frame step lengths (n-1,) and cumulative path length (n,).
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    # Mask out steps that touch a lost frame so they don't poison the path length.
    step_valid = valid[:-1] & valid[1:]
    steps = np.where(step_valid, steps, 0.0)
    cum = np.concatenate([[0.0], np.cumsum(steps)])

    # Slide the window: chord = ||x[i+win] - x[i]||, path = cum[i+win] - cum[i].
    starts = xyz[:-win]
    ends = xyz[win:]
    chord = np.linalg.norm(ends - starts, axis=1)
    path = cum[win:] - cum[:-win]

    # Window is valid iff every endpoint inside it is valid.
    win_valid = np.array([valid[i:i + win + 1].all() for i in range(n - win)])
    # Drop windows where the hand barely moved (chord and path both tiny):
    # a 0.1mm-amplitude wobble that happens to stay still gives ratio≈0 noise.
    moving = path > 1e-3  # > 1mm of path inside the window
    keep = win_valid & moving
    if not keep.any():
        return 1.0

    ratio = chord[keep] / np.maximum(path[keep], 1e-9)
    return float(np.median(np.clip(ratio, 0.0, 1.0)))


def _per_axis_hifreq_rms_mm(xyz: np.ndarray, valid: np.ndarray, fps: float, cutoff_s: float = 0.05) -> float:
    """Max-axis RMS of high-passed position trajectory (mm).

    High-pass = signal − Gaussian-smoothed signal with sigma = `cutoff_s` * fps.
    With cutoff_s=0.05 (≈10 Hz cutoff at 60 fps) we isolate the tremor band
    above human-motion bandwidth, which is where SLAM noise lives.

    Threshold values were tuned against a known-clean reference demo:
    per-axis hifreq max ≈1 mm for a clean trajectory; pose-graph re-anchoring
    artifacts pile up to 2–3 mm even when each individual jump is small.

    Returns 0.0 if too few valid frames.
    """
    if len(xyz) < 4 or valid.sum() < 4:
        return 0.0
    sigma = max(1.0, cutoff_s * fps)
    radius = int(round(3.0 * sigma))
    if radius < 1:
        return 0.0
    k = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()

    # NaN-out invalid frames, then convolve with edge-pad ('reflect'-equivalent
    # via np.pad) using nan-aware arithmetic.
    masked = xyz.astype(np.float64).copy()
    masked[~valid] = np.nan

    rms_per_axis = np.zeros(3)
    for ax in range(3):
        s = masked[:, ax]
        # Pad with reflection
        pad = np.pad(s, radius, mode="edge")
        # NaN-aware convolution: replace NaNs with local mean of valid samples
        if np.isnan(pad).any():
            mean = np.nanmean(pad)
            pad = np.where(np.isnan(pad), mean, pad)
        smooth = np.convolve(pad, kernel, mode="valid")
        hp = s - smooth
        hp_valid = hp[valid]
        if hp_valid.size == 0:
            rms_per_axis[ax] = 0.0
        else:
            rms_per_axis[ax] = np.sqrt(np.mean(hp_valid ** 2))
    return float(rms_per_axis.max() * 1000.0)


def _coherence(d_pose: np.ndarray, d_hall: np.ndarray, dt: float = 1.0, fps: float = 60.0) -> float:
    """Pearson correlation of rolling 1s-window energy between pose and hall motion.

    Returns 0.0 when either signal is constant or too short.
    """
    win = max(1, int(round(fps * dt)))
    if len(d_pose) < win * 2:
        return 0.0
    kernel = np.ones(win, dtype=np.float64) / win
    ep = np.convolve(d_pose, kernel, mode="valid")
    eh = np.convolve(d_hall, kernel, mode="valid")
    if ep.std() < 1e-9 or eh.std() < 1e-9:
        return 0.0
    r = float(np.corrcoef(ep, eh)[0, 1])
    if np.isnan(r):
        return 0.0
    return r


def compute_from_episode(
    episode_parquet: Path,
    duration_s: Optional[float] = None,
) -> QualityBlock:
    """Compute a full QualityBlock from a session's episode.parquet.

    Parameters
    ----------
    episode_parquet : Path
        Path to `{session}/aligned/episode.parquet`.
    duration_s : float, optional
        Video duration from meta.yaml. If None, derived from `t_rel`.
    """
    table = pq.read_table(episode_parquet)
    df = table.to_pandas()

    n_frames = len(df)
    n_keyframes = int(df["is_keyframe"].sum()) if "is_keyframe" in df.columns else 0
    is_lost = df["is_lost"].to_numpy() if "is_lost" in df.columns else np.zeros(n_frames, dtype=bool)

    # Pose arrays — prefer `pose_*` (body-frame MTCP) for the metric surface the policy sees.
    xyz = df[["pose_x", "pose_y", "pose_z"]].to_numpy(dtype=np.float64)
    quat = df[["pose_qx", "pose_qy", "pose_qz", "pose_qw"]].to_numpy(dtype=np.float64)
    t_rel = df["t_rel"].to_numpy(dtype=np.float64)

    # Δpose between consecutive frames, masked where either endpoint is lost
    d_xyz = np.diff(xyz, axis=0)
    d_norm = np.linalg.norm(d_xyz, axis=1)  # meters
    valid_pair = ~(is_lost[:-1] | is_lost[1:])
    d_norm_valid = d_norm[valid_pair]
    d_norm_mm = d_norm_valid * 1000.0

    pose_jump_max_mm = float(d_norm_mm.max()) if d_norm_mm.size else 0.0
    pose_jump_count_20mm = int((d_norm_mm > THRESHOLDS.pose_jump_max_mm_warn).sum())

    # Indices (in original frame space) where jump > warn threshold
    flagged_mask = np.zeros(n_frames, dtype=bool)
    jump_over = d_norm_mm > THRESHOLDS.pose_jump_max_mm_warn
    if jump_over.any():
        # d_norm_valid came from valid_pair; map back to original indices of the *second* frame
        valid_idx_second = np.where(valid_pair)[0] + 1  # shape (n_valid_pairs,)
        flagged_idx = valid_idx_second[jump_over]
        flagged_mask[flagged_idx] = True
    flagged_frames = np.where(flagged_mask)[0].tolist()

    # Velocity (m/s) — use Δt between frames
    dt = np.diff(t_rel)
    dt_safe = np.where(dt > 1e-6, dt, np.nan)
    vel = d_norm / dt_safe
    vel_valid = vel[valid_pair & ~np.isnan(vel)]
    pose_vel_p95_mps = float(np.percentile(vel_valid, 95)) if vel_valid.size else 0.0

    # Acceleration p95 (m/s²) — finite-difference of valid velocities over mean dt
    if vel_valid.size >= 2:
        mean_dt = float(np.nanmean(dt_safe)) if np.isfinite(np.nanmean(dt_safe)) else 1.0 / 60.0
        accel = np.abs(np.diff(vel_valid)) / max(mean_dt, 1e-6)
        pose_accel_p95 = float(np.percentile(accel, 95)) if accel.size else 0.0
    else:
        pose_accel_p95 = 0.0

    # Quaternion angular delta
    q_angles = _quat_angle_deg(quat[:-1], quat[1:])
    q_angles_valid = q_angles[valid_pair]
    quat_angle_jump_max_deg = float(q_angles_valid.max()) if q_angles_valid.size else 0.0

    # Relocalization suspects: count "burst events" — 0.5s windows containing
    # ≥3 warn-level pose jumps. This catches both ORB-SLAM3 silent relocalizations
    # (single large jump → one burst) and rail/flip patterns where the estimator
    # oscillates during tracking (is_lost stays false; multiple clustered jumps).
    # The prior is_lost-based definition missed rails entirely, which was the
    # dominant failure mode on the mug_coaster batch.
    if n_frames >= 2 and flagged_mask.any():
        mean_dt = float(np.nanmean(dt_safe)) if np.isfinite(np.nanmean(dt_safe)) else 1.0 / 60.0
        win = max(3, int(round(0.5 / max(mean_dt, 1e-6))))
        kernel = np.ones(win, dtype=np.int32)
        flagged_count = np.convolve(flagged_mask.astype(np.int32), kernel, mode="valid")
        burst_window = flagged_count >= 3
        # Count non-overlapping bursts (collapse runs of consecutive burst windows)
        transitions = np.diff(burst_window.astype(np.int8), prepend=0)
        reloc = int((transitions == 1).sum())
    else:
        reloc = 0

    # Lost ratio + longest lost run
    lost_ratio = float(is_lost.mean()) if n_frames else 0.0
    lost_run_max = _longest_run(is_lost)

    # Proprio-pose coherence: correlate rolling-window Δpose energy vs. Δhall energy
    hall_cols = [f"hall_{i}_{ax}" for i in range(4) for ax in ("bx", "by", "bz")]
    hall_present = [c for c in hall_cols if c in df.columns]
    if hall_present:
        hall = df[hall_present].to_numpy(dtype=np.float64)
        d_hall = np.linalg.norm(np.diff(hall, axis=0), axis=1)
        # Pad d_norm to same length as d_hall (both are diffs, same length N-1)
        coherence = _coherence(d_norm, d_hall)
    else:
        coherence = 0.0

    # Duration
    if duration_s is None:
        duration_s = float(t_rel[-1] - t_rel[0]) if n_frames > 1 else 0.0

    # Wobble gates (v2). FPS estimated from median dt; falls back to 60Hz
    # when t_rel is degenerate.
    if dt.size and np.isfinite(np.nanmedian(dt_safe)):
        fps_est = 1.0 / max(float(np.nanmedian(dt_safe)), 1e-6)
    else:
        fps_est = 60.0
    valid_frame = ~is_lost
    local_straightness = _local_straightness(xyz, valid_frame, fps=fps_est, window_s=0.25)
    per_axis_hifreq = _per_axis_hifreq_rms_mm(xyz, valid_frame, fps=fps_est, cutoff_s=0.05)

    metrics = QualityMetrics(
        pose_jump_max_mm=round(pose_jump_max_mm, 2),
        pose_jump_count_20mm=pose_jump_count_20mm,
        pose_vel_p95_mps=round(pose_vel_p95_mps, 3),
        pose_accel_p95=round(pose_accel_p95, 3),
        quat_angle_jump_max_deg=round(quat_angle_jump_max_deg, 2),
        relocalization_suspects=reloc,
        lost_ratio=round(lost_ratio, 4),
        lost_run_max=lost_run_max,
        proprio_pose_coherence=round(coherence, 3),
        duration_s=round(float(duration_s), 2),
        n_frames=n_frames,
        n_keyframes=n_keyframes,
        local_straightness_0_25s=round(local_straightness, 3),
        per_axis_hifreq_max_mm=round(per_axis_hifreq, 2),
    )

    return compute_verdict(metrics, flagged_frames=flagged_frames)


def score_session(session_dir: Path, persist: bool = True) -> Optional[QualityBlock]:
    """Score a session and (optionally) write the result into `meta.yaml`.

    Returns None if the session has no `aligned/episode.parquet` yet.
    """
    session_dir = Path(session_dir)
    meta_path = session_dir / "meta.yaml"
    episode_path = session_dir / "aligned" / "episode.parquet"
    if not meta_path.exists() or not episode_path.exists():
        return None

    meta = yaml.safe_load(meta_path.read_text()) or {}
    duration_s = None
    try:
        duration_s = float(meta.get("video", {}).get("duration_s"))
    except (TypeError, ValueError):
        duration_s = None

    block = compute_from_episode(episode_path, duration_s=duration_s)
    if persist:
        meta["quality"] = block.to_dict()
        meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    return block
