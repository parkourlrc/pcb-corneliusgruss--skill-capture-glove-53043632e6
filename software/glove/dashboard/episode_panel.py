"""Loader: `aligned/episode.parquet` → Plotly-ready arrays for the review panel.

Kept separate from the scanner so the grid-summary code path doesn't pay the
parquet-read cost (~15ms each). Only the open-session detail view hits this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow.parquet as pq


def _downsample(arr: np.ndarray, max_points: int = 1500) -> np.ndarray:
    if arr.size <= max_points:
        return arr
    idx = np.linspace(0, arr.size - 1, max_points).astype(int)
    return arr[idx]


def load_episode_panel(episode_parquet: Path, max_points: int = 1500) -> Optional[dict[str, Any]]:
    """Return a JSON-ready dict of arrays for the detail panel.

    Keys:
      t_rel[]                    time since video start (s), downsampled
      pose_xy[2][]               xy-plane trajectory (m)
      pose_xz[2][]               xz-plane trajectory (m)
      pose_jump_mm[]             per-frame |Δpose| in mm (length n_frames-1, aligned to t_rel[1:])
      pose_jump_threshold_mm     the warn threshold (for dashed line in plot)
      is_lost_mask[]             boolean, downsampled
      hall_angles[4][]           atan2(by, bx) in radians, downsampled
      hall_bz[4][]               Bz per sensor, downsampled
      tof_top[], tof_bottom[]    mm, downsampled (NaN for sentinel)
      flagged_frames[]           frame indices (pre-downsample) where jumps triggered
      n_frames                   int, original
    """
    if not episode_parquet.exists():
        return None

    df = pq.read_table(episode_parquet).to_pandas()
    n = len(df)
    if n == 0:
        return None

    t_rel = df["t_rel"].to_numpy(dtype=np.float64)
    if n > 1 and t_rel[-1] > t_rel[0]:
        fps = float((n - 1) / (t_rel[-1] - t_rel[0]))
    else:
        fps = 60.0

    xyz = df[["pose_x", "pose_y", "pose_z"]].to_numpy(dtype=np.float64)
    d_xyz = np.diff(xyz, axis=0)
    jump_mm = np.linalg.norm(d_xyz, axis=1) * 1000.0  # length n-1
    # prepend 0 so it's aligned to frame-index axis
    jump_mm = np.concatenate([[0.0], jump_mm])

    is_lost = df["is_lost"].to_numpy() if "is_lost" in df.columns else np.zeros(n, dtype=bool)

    # Hall angle per sensor = atan2(by, bx); Bz is the force-ish channel
    hall_angles: list[list[float]] = []
    hall_bz: list[list[float]] = []
    for i in range(4):
        bx_col, by_col, bz_col = f"hall_{i}_bx", f"hall_{i}_by", f"hall_{i}_bz"
        if bx_col in df.columns and by_col in df.columns:
            bx = df[bx_col].to_numpy(dtype=np.float32)
            by = df[by_col].to_numpy(dtype=np.float32)
            angle = np.arctan2(by, bx)
            hall_angles.append(_downsample(angle, max_points).tolist())
        else:
            hall_angles.append([])
        if bz_col in df.columns:
            hall_bz.append(_downsample(df[bz_col].to_numpy(dtype=np.float32), max_points).tolist())
        else:
            hall_bz.append([])

    def _tof(col: str) -> list[float]:
        if col not in df.columns:
            return []
        v = df[col].to_numpy(dtype=np.float32)
        v = np.where(v < 0, np.nan, v)
        return _downsample(v, max_points).tolist()

    # Flagged frames: pose_jump > 20mm and neither endpoint lost
    valid_pair = ~(is_lost[:-1] | is_lost[1:])
    flagged_idx = np.where((jump_mm[1:] > 20.0) & valid_pair)[0] + 1  # second-frame index
    flagged_times = t_rel[flagged_idx].tolist() if flagged_idx.size else []
    flagged_mag = jump_mm[flagged_idx].tolist() if flagged_idx.size else []
    flagged = flagged_idx.tolist()

    return {
        "t_rel": _downsample(t_rel, max_points).tolist(),
        "pose_xy": [
            _downsample(xyz[:, 0], max_points).tolist(),
            _downsample(xyz[:, 1], max_points).tolist(),
            _downsample(xyz[:, 2], max_points).tolist(),
        ],
        "pose_jump_mm": _downsample(jump_mm, max_points).tolist(),
        "pose_jump_threshold_mm": 20.0,
        "is_lost_mask": _downsample(is_lost.astype(np.int8), max_points).tolist(),
        "hall_angles": hall_angles,
        "hall_bz": hall_bz,
        "tof_top": _tof("tof_top_mm"),
        "tof_bottom": _tof("tof_bottom_mm"),
        "flagged_frames": flagged,
        "flagged_times": flagged_times,
        "flagged_mag": flagged_mag,
        "n_frames": int(n),
        "fps": round(fps, 3),
    }
