"""Compute map-quality metrics from `slam/camera_trajectory.csv`.

Maps don't have an `aligned/episode.parquet` (they're not training data), so
their quality signal comes from SLAM's own output during map-build: tracking
success, keyframe count, spatial extent of keyframe positions.

Derivative metrics (how well demos localize against this map) are computed
separately from the demo scan — see `dashboard/routes/maps.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


@dataclass
class MapQuality:
    n_frames_total: int
    n_tracked: int
    mapping_tracking_ratio: float
    n_keyframes: int
    duration_s: float
    map_extent_m: list[float]      # [x, y, z] bounding box size
    keyframe_positions: list[list[float]] = field(default_factory=list)  # downsampled for minimap
    scored_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MapQuality":
        return cls(
            n_frames_total=d.get("n_frames_total", 0),
            n_tracked=d.get("n_tracked", 0),
            mapping_tracking_ratio=d.get("mapping_tracking_ratio", 0.0),
            n_keyframes=d.get("n_keyframes", 0),
            duration_s=d.get("duration_s", 0.0),
            map_extent_m=list(d.get("map_extent_m", [0.0, 0.0, 0.0])),
            keyframe_positions=list(d.get("keyframe_positions", [])),
            scored_at=d.get("scored_at", ""),
        )


def compute_from_trajectory(traj_csv: Path, max_minimap_points: int = 400) -> MapQuality:
    df = pd.read_csv(traj_csv)
    n_total = len(df)

    # Normalize bool columns (emitted as strings by ORB-SLAM3 wrapper)
    is_lost = df["is_lost"].astype(str).str.lower().eq("true").to_numpy() if "is_lost" in df.columns else np.zeros(n_total, dtype=bool)
    is_keyframe = df["is_keyframe"].astype(str).str.lower().eq("true").to_numpy() if "is_keyframe" in df.columns else np.zeros(n_total, dtype=bool)
    n_tracked = int((~is_lost).sum())
    n_keyframes = int(is_keyframe.sum())

    # Duration from timestamps (t_rel seconds since video start)
    if "timestamp" in df.columns and n_total > 0:
        t = df["timestamp"].to_numpy(dtype=np.float64)
        duration_s = float(t[-1] - t[0])
    else:
        duration_s = 0.0

    # Spatial extent — use KEYFRAME positions when available (more stable than
    # per-frame poses which include briefly-drifting estimates).
    kf_mask = is_keyframe & ~is_lost
    ref_mask = kf_mask if kf_mask.any() else ~is_lost
    if ref_mask.any() and all(c in df.columns for c in ("x", "y", "z")):
        xyz = df.loc[ref_mask, ["x", "y", "z"]].to_numpy(dtype=np.float64)
        extent = (xyz.max(axis=0) - xyz.min(axis=0)).tolist()
        if len(xyz) > max_minimap_points:
            idx = np.linspace(0, len(xyz) - 1, max_minimap_points).astype(int)
            minimap = xyz[idx].tolist()
        else:
            minimap = xyz.tolist()
    else:
        extent = [0.0, 0.0, 0.0]
        minimap = []

    return MapQuality(
        n_frames_total=n_total,
        n_tracked=n_tracked,
        mapping_tracking_ratio=round(n_tracked / max(n_total, 1), 4),
        n_keyframes=n_keyframes,
        duration_s=round(duration_s, 2),
        map_extent_m=[round(float(v), 3) for v in extent],
        keyframe_positions=[[round(float(v), 3) for v in p] for p in minimap],
        scored_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def score_map(session_dir: Path, persist: bool = True) -> Optional[MapQuality]:
    session_dir = Path(session_dir)
    meta_path = session_dir / "meta.yaml"
    traj_path = session_dir / "slam" / "camera_trajectory.csv"
    if not (meta_path.exists() and traj_path.exists()):
        return None

    mq = compute_from_trajectory(traj_path)
    if persist:
        meta = yaml.safe_load(meta_path.read_text()) or {}
        meta["map_quality"] = mq.to_dict()
        meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    return mq
