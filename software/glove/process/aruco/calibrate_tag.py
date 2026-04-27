"""Compute `tx_slam_tag` for a mapping session.

Ports UMI's `scripts/calibrate_slam_tag.py` — for each frame where the table
ArUco tag is detected, compute

    T_slam_tag_i = T_slam_cam_i @ T_cam_tag_i

robustly average the samples to produce a single 4x4 transform, and write it to

    <map-session>/slam/tx_slam_tag.json

This transform lets downstream code express demo trajectories in the tag frame,
which is **gravity-aligned by construction** (tag lies flat on the table; its
normal is the vertical axis). That corrects ORB-SLAM3's residual IMU-bootstrap
tilt (~5–6° in our measurements) without needing a hand-eye calibration pass.

Pass the mapping session directory; we look for `aruco/tag_detection.pkl` +
`slam/camera_trajectory.csv` inside it.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from glove.process.aruco._util import pose_to_mat


def _geometric_median(pts: np.ndarray, n_iter: int = 100, eps: float = 1e-8) -> np.ndarray:
    """Weiszfeld iteration. Standalone to avoid a skfda dependency."""
    y = np.mean(pts, axis=0)
    for _ in range(n_iter):
        d = np.linalg.norm(pts - y, axis=1)
        nz = d > eps
        if not nz.any():
            return y
        w = 1.0 / d[nz]
        y_new = (w[:, None] * pts[nz]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            return y_new
        y = y_new
    return y


def calibrate_tag(
    map_session_dir: Path,
    tag_id: int = 13,
    keyframe_only: bool = False,
    dist_min_m: float = 0.3,
    dist_max_m: float = 2.0,
    image_center_tol: float = 0.6,
) -> Path:
    map_session_dir = Path(map_session_dir).resolve()
    tag_pkl = map_session_dir / "aruco" / "tag_detection.pkl"
    traj_csv = map_session_dir / "slam" / "camera_trajectory.csv"
    if not tag_pkl.exists():
        raise FileNotFoundError(
            f"No tag_detection.pkl in {map_session_dir/'aruco'} — run `glove.process detect-aruco` first."
        )
    if not traj_csv.exists():
        raise FileNotFoundError(
            f"No camera_trajectory.csv in {map_session_dir/'slam'} — run `glove.process build-map` first."
        )

    df = pd.read_csv(traj_csv)
    detections = pickle.loads(tag_pkl.read_bytes())

    # Valid SLAM poses (drop lost; optional keyframe-only)
    is_valid = ~df["is_lost"]
    if keyframe_only:
        is_valid &= df["is_keyframe"]
    if not is_valid.any():
        raise RuntimeError("No valid SLAM poses in camera_trajectory.csv")

    cam_pose_timestamps = df["timestamp"].loc[is_valid].to_numpy()
    cam_pos = df[["x", "y", "z"]].loc[is_valid].to_numpy()
    cam_rot_quat_xyzw = df[["q_x", "q_y", "q_z", "q_w"]].loc[is_valid].to_numpy()

    from scipy.spatial.transform import Rotation
    cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
    cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float64)
    cam_pose[:, 3, 3] = 1
    cam_pose[:, :3, 3] = cam_pos
    cam_pose[:, :3, :3] = cam_rot.as_matrix()

    # Nearest-timestamp match between trajectory rows and video-frame detections
    video_timestamps = np.array([r["time"] for r in detections])
    matches = [int(np.argmin(np.abs(video_timestamps - t))) for t in cam_pose_timestamps]

    all_tx_slam_tag = []
    n_tag_hits = 0
    n_filtered_dist = 0
    n_filtered_edge = 0
    img_center = None
    for tum_idx, vid_idx in enumerate(matches):
        td = detections[vid_idx]
        tag_dict = td["tag_dict"]
        if tag_id not in tag_dict:
            continue
        n_tag_hits += 1
        tag = tag_dict[tag_id]

        # T_cam_tag from PnP
        pose = np.concatenate([tag["tvec"], tag["rvec"]])
        tx_cam_tag = pose_to_mat(pose)
        tx_slam_cam = cam_pose[tum_idx]

        # Filter: only use detections where the tag is close enough for a
        # well-conditioned PnP, and not near the fisheye edge where distortion
        # residuals are worst. Thresholds taken from UMI.
        dist_to_cam = float(np.linalg.norm(tx_cam_tag[:3, 3]))
        if dist_to_cam < dist_min_m or dist_to_cam > dist_max_m:
            n_filtered_dist += 1
            continue
        corners = tag["corners"]
        tag_center_pix = corners.mean(axis=0)
        # Image center comes from the first record with the detected tag so we
        # don't hardcode a 2704×2028 assumption.
        if img_center is None:
            # Corners are in original (unscaled) image coords; we infer width/height
            # from the max corner coord across the whole pkl. Practical and safe
            # because every UMI-like pipeline uses the native resolution.
            max_xy = np.zeros(2)
            for r in detections:
                for v in r["tag_dict"].values():
                    max_xy = np.maximum(max_xy, v["corners"].max(axis=0))
            img_center = max_xy / 2.0
        dist_to_center = float(np.linalg.norm(tag_center_pix - img_center) / img_center[1])
        if dist_to_center > image_center_tol:
            n_filtered_edge += 1
            continue

        all_tx_slam_tag.append(tx_slam_cam @ tx_cam_tag)

    if not all_tx_slam_tag:
        raise RuntimeError(
            f"No usable tag detections for id={tag_id} after filtering "
            f"(hits={n_tag_hits} filtered_dist={n_filtered_dist} filtered_edge={n_filtered_edge})"
        )
    all_tx_slam_tag = np.array(all_tx_slam_tag)

    # Robust central estimate: geometric median on position, then pick the
    # nearest sample. Matches UMI's approach (tiny edit: Weiszfeld instead of skfda).
    positions = all_tx_slam_tag[:, :3, 3]
    median = _geometric_median(positions)
    dists = np.linalg.norm(positions - median, axis=-1)
    thresh = np.quantile(dists, 0.9)
    keep = dists < thresh
    std = positions[keep].std(axis=0)
    mean = positions[keep].mean(axis=0)
    inner_dists = np.linalg.norm(positions[keep] - mean, axis=-1)
    nn_idx = int(np.argmin(inner_dists))
    tx_slam_tag = all_tx_slam_tag[keep][nn_idx]

    out_path = map_session_dir / "slam" / "tx_slam_tag.json"
    out_path.write_text(
        json.dumps(
            {
                "tx_slam_tag": tx_slam_tag.tolist(),
                "tag_id": tag_id,
                "n_samples_used": int(keep.sum()),
                "n_samples_total": int(len(all_tx_slam_tag)),
                "position_std_cm_90pct": [round(float(s * 100), 3) for s in std],
            },
            indent=2,
        )
    )
    print(f"Wrote {out_path}")
    print(
        f"  tag_id={tag_id}  hits={n_tag_hits}  used={int(keep.sum())}/{len(all_tx_slam_tag)}  "
        f"std_cm_90pct={[round(float(s*100), 2) for s in std]}"
    )
    return out_path


def build_parser(sub) -> None:
    p = sub.add_parser(
        "calibrate-tag",
        help="Compute tx_slam_tag.json for a mapping session (gravity-aligns the world frame).",
    )
    p.add_argument("map_session_dir", type=Path)
    p.add_argument("--tag-id", type=int, default=13)
    p.add_argument("--keyframe-only", action="store_true",
                   help="Use only SLAM keyframes (slower rows, often lower-noise).")
    p.set_defaults(func=cli)


def cli(args) -> int:
    calibrate_tag(
        args.map_session_dir,
        tag_id=args.tag_id,
        keyframe_only=args.keyframe_only,
    )
    return 0
