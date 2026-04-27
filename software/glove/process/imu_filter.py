"""Reject SLAM pose frames whose motion is inconsistent with the DAQ IMU.

ORB-SLAM3 monocular-inertial mode occasionally produces single-frame or
short-burst pose jumps during nominal tracking — bundle-adjustment flips that
are not real motion. The DAQ IMU (~680Hz QMI8658) is a fully independent sensor
from the GoPro's GPMF IMU that drove those flips, so we use it as the oracle.

Detection: for each frame k, compute its deviation from the midpoint of its
neighbors: dev = ||p[k] - 0.5*(p[k-1] + p[k+1])||. A real hand motion at
observed IMU acceleration excess `a = max(|a_body| - g, 0)` over the 2-frame
window can produce at most 0.5 * a * (2*dt)² of deviation. Anything beyond
that (plus a 25mm floor for noise) is an estimator flip.

Filter: linearly interpolate position across rejected frames; slerp quaternion.
Iterate 2-5 times — a single pass catches isolated flips but contiguous bad
segments need the midpoints to get re-evaluated after their neighbors heal.

The filter NEVER modifies cam_trajectory.csv on disk; align.py applies it
in-memory just before writing episode.parquet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

G = 9.81


@dataclass
class FilterResult:
    xyz: np.ndarray             # (N,3) filtered world positions
    quat_xyzw: np.ndarray       # (N,4) filtered orientation quats (scipy order)
    reject: np.ndarray          # (N,) bool — True where replaced (lost OR flip)
    n_lost_in: int              # frames already lost before filter
    n_flips: int                # frames rejected as estimator flips
    max_jump_before_mm: float
    max_jump_after_mm: float
    iterations: int


def _imu_excess_around_frame(
    frame_wall: np.ndarray,
    imu_wall: np.ndarray,
    imu_mag: np.ndarray,
    half_window_s: float = 0.020,
) -> np.ndarray:
    """max(|a_body|-g, 0) over IMU samples within ±half_window_s of each frame."""
    out = np.full(len(frame_wall), np.nan, dtype=np.float64)
    if len(imu_wall) == 0:
        return out
    lo = np.searchsorted(imu_wall, frame_wall - half_window_s, side="left")
    hi = np.searchsorted(imu_wall, frame_wall + half_window_s, side="right")
    for i in range(len(frame_wall)):
        a, b = lo[i], hi[i]
        if b > a:
            out[i] = max(0.0, imu_mag[a:b].max() - G)
    return out


def _detect_rejections(
    xyz: np.ndarray,
    t_rel: np.ndarray,
    lost: np.ndarray,
    imu_excess_frame: np.ndarray,
    base_deviation_mm: float,
    hard_ceiling_mm: float,
) -> np.ndarray:
    N = len(xyz)
    reject = lost.copy()
    if N < 3:
        return reject
    dt = np.diff(t_rel)
    window_2dt = dt[:-1] + dt[1:]
    mid = 0.5 * (xyz[:-2] + xyz[2:])
    dev_mm = np.linalg.norm(xyz[1:-1] - mid, axis=1) * 1000.0

    excess = np.where(np.isnan(imu_excess_frame[1:-1]), 0.0, imu_excess_frame[1:-1])
    allowed = base_deviation_mm + 0.5 * excess * (window_2dt ** 2) * 1000.0
    allowed = np.minimum(allowed, hard_ceiling_mm)

    flag = (dev_mm > allowed) & ~lost[1:-1]
    reject[1:-1] |= flag
    return reject


def _fill_rejected(
    xyz: np.ndarray,
    quat: np.ndarray,
    t_rel: np.ndarray,
    reject: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(xyz)
    valid = ~reject
    if not valid.any():
        return xyz.copy(), quat.copy()

    xyz_out = xyz.copy()
    for axis in range(3):
        xyz_out[reject, axis] = np.interp(t_rel[reject], t_rel[valid], xyz[valid, axis])

    quat_out = quat.copy()
    if reject.any() and valid.sum() >= 2:
        v_idx = np.where(valid)[0]
        rots = Rotation.from_quat(quat[valid])
        slerp = Slerp(t_rel[valid], rots)
        t_q = np.clip(t_rel[reject], t_rel[v_idx[0]], t_rel[v_idx[-1]])
        quat_out[reject] = slerp(t_q).as_quat()
    return xyz_out, quat_out


def filter_poses(
    xyz: np.ndarray,           # (N, 3) world-frame camera positions
    quat_xyzw: np.ndarray,     # (N, 4) scipy xyzw
    t_rel: np.ndarray,         # (N,) seconds since video start
    lost: np.ndarray,          # (N,) bool
    frame_wall: np.ndarray,    # (N,) epoch seconds
    imu_wall: np.ndarray,      # (M,) epoch seconds
    imu_accel_body: np.ndarray,  # (M, 3) or (M,) magnitude
    base_deviation_mm: float = 25.0,
    hard_ceiling_mm: float = 100.0,
    max_iterations: int = 5,
) -> FilterResult:
    """Apply IMU-consistency filter. See module docstring for algorithm."""
    if imu_accel_body.ndim == 2:
        imu_mag = np.linalg.norm(imu_accel_body, axis=1)
    else:
        imu_mag = imu_accel_body

    excess_frame = _imu_excess_around_frame(frame_wall, imu_wall, imu_mag)

    dxyz_before = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000
    max_jump_before = float(dxyz_before.max()) if len(dxyz_before) else 0.0

    reject = lost.copy()
    xyz_f, quat_f = xyz.copy(), quat_xyzw.copy()
    iteration = 0
    for iteration in range(max_iterations):
        prev_count = reject.sum()
        new_reject = _detect_rejections(
            xyz_f, t_rel, lost, excess_frame, base_deviation_mm, hard_ceiling_mm
        )
        reject = reject | new_reject
        if reject.sum() == prev_count:
            break
        xyz_f, quat_f = _fill_rejected(xyz, quat_xyzw, t_rel, reject)

    dxyz_after = np.linalg.norm(np.diff(xyz_f, axis=0), axis=1) * 1000
    max_jump_after = float(dxyz_after.max()) if len(dxyz_after) else 0.0

    return FilterResult(
        xyz=xyz_f,
        quat_xyzw=quat_f,
        reject=reject,
        n_lost_in=int(lost.sum()),
        n_flips=int((reject & ~lost).sum()),
        max_jump_before_mm=max_jump_before,
        max_jump_after_mm=max_jump_after,
        iterations=iteration + 1,
    )


def load_daq_imu(session_dir) -> tuple[np.ndarray, np.ndarray]:
    """Load DAQ IMU from <session>/raw/imu_daq.parquet.

    Returns (wall_time, accel_xyz). Empty arrays if the file is missing or empty.
    """
    from pathlib import Path
    p = Path(session_dir) / "raw" / "imu_daq.parquet"
    if not p.exists():
        return np.array([]), np.array([]).reshape(0, 3)
    df = pd.read_parquet(p)
    if len(df) == 0:
        return np.array([]), np.array([]).reshape(0, 3)
    wall = df["wall_time"].to_numpy()
    accel = df[["accel_x", "accel_y", "accel_z"]].to_numpy()
    return wall, accel
