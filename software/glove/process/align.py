"""Align proprio stream to per-frame SLAM poses → aligned/episode.parquet.

For each row in slam/camera_trajectory.csv (one per GoPro video frame), this
interpolates proprio samples at the frame's wall-clock time, applies the
camera→body-frame transform (when calibration is available), and writes a
fused table that is the training-ready artifact for the policy stack.

Columns in episode.parquet:
    frame_idx, t_rel (s since video start), wall_time (epoch s),
    pose_x, pose_y, pose_z, pose_qw, pose_qx, pose_qy, pose_qz,   # MTCP in world frame (or cam if no calib)
    cam_x, cam_y, cam_z, cam_qw, cam_qx, cam_qy, cam_qz,          # always GoPro camera pose in world
    is_lost, is_keyframe,
    hall_{0..3}_{bx,by,bz} (int16),
    tof_top_mm, tof_bottom_mm (int16).

`pose_*` is the primary body-frame pose expected by downstream training code.
When `T_mtcp_gopro` is calibrated in mtcp_sensors.json, `pose_*` = MTCP body
pose; otherwise `pose_*` falls back to camera pose and a WARNING is printed.

World-frame handling: by default, poses are expressed in the **tag frame** (the
ArUco anchor at map-build time). This requires `<parent_map>/slam/tx_slam_tag.json`
to exist — produced by `glove.process calibrate-tag`. The tag frame is gravity-
aligned by construction (tag lies flat on the table), so `pose_z` is literally
height above the table and two demos localized against the same map share a
common gravity axis. If `tx_slam_tag.json` is missing, we fall back to the raw
ORB-SLAM3 world frame and log a WARNING — `pose_z` then carries ORB's residual
IMU-bootstrap tilt (empirically 5–6°), which makes vertical-height comparisons
unreliable.

Proprio fields are linearly interpolated from ~30 Hz to ~60 Hz video frames.
ToF sentinels (-1) are treated as NaN during interp.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy.spatial.transform import Rotation

from glove.process.imu_filter import filter_poses, load_daq_imu
from glove.quality.compute import score_session
from glove.quality.schema import SCHEMA_VERSION as QUALITY_SCHEMA_VERSION

CALIB_PATH = Path("software/configs/calibration/mtcp_sensors.json")
BODY_FRAME_KEY = "T_mtcp_gopro"


def _load_T_mtcp_gopro(calib_path: Path = CALIB_PATH) -> Optional[np.ndarray]:
    if not calib_path.exists():
        return None
    data = json.loads(calib_path.read_text())
    arr = data.get(BODY_FRAME_KEY)
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float64)


def _apply_body_transform(
    cam_xyz: np.ndarray,           # (N, 3)
    cam_quat_xyzw: np.ndarray,     # (N, 4) — scipy order
    T_mtcp_cam: np.ndarray,        # (4, 4) — "cam expressed in mtcp"
) -> tuple[np.ndarray, np.ndarray]:
    """T_world_mtcp = T_world_cam @ inv(T_mtcp_cam).

    Returns (mtcp_xyz, mtcp_quat_xyzw) both shaped per-frame.
    """
    T_cam_mtcp = np.linalg.inv(T_mtcp_cam)
    N = len(cam_xyz)

    T_world_cam = np.tile(np.eye(4), (N, 1, 1))
    T_world_cam[:, :3, :3] = Rotation.from_quat(cam_quat_xyzw).as_matrix()
    T_world_cam[:, :3, 3] = cam_xyz

    T_world_mtcp = T_world_cam @ T_cam_mtcp

    mtcp_xyz = T_world_mtcp[:, :3, 3]
    mtcp_quat = Rotation.from_matrix(T_world_mtcp[:, :3, :3]).as_quat()
    return mtcp_xyz, mtcp_quat


def _apply_world_transform(
    xyz: np.ndarray,               # (N, 3)
    quat_xyzw: np.ndarray,         # (N, 4)
    T_new_world: np.ndarray,       # (4, 4) — left-multiplied onto T_world_obj
) -> tuple[np.ndarray, np.ndarray]:
    """Re-express a batch of `T_world_obj` poses in a new world frame.

    `T_new_obj = T_new_world @ T_world_obj`. Used to move cam poses from the
    raw SLAM world into the gravity-aligned tag frame.
    """
    N = len(xyz)
    T_world_obj = np.tile(np.eye(4), (N, 1, 1))
    T_world_obj[:, :3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T_world_obj[:, :3, 3] = xyz
    T_new_obj = T_new_world @ T_world_obj
    return T_new_obj[:, :3, 3], Rotation.from_matrix(T_new_obj[:, :3, :3]).as_quat()


def _load_tx_slam_tag(
    session_dir: Path,
    meta: dict,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Resolve the tag transform for this demo session.

    Two stacks produce the same shape transform but store it in different places:
      - ORB-SLAM3 stack: <map>/slam/tx_slam_tag.json   (key: "tx_slam_tag")
      - HLoc stack:      <map>/hloc/tx_hloc_tag.json   (key: "tx_hloc_tag")

    The choice is dictated by `meta.slam_provenance.stack` on the session (the demo
    for demo sessions; the session itself for mapping sessions). Falls through to
    looking at both locations if the provenance is missing.

    Returns (T_world_tag as 4×4, source string) or (None, None) when unavailable.
    """
    kind = meta.get("kind", "demo")
    sessions_root = session_dir.parent
    stack = (meta.get("slam_provenance") or {}).get("stack")

    def _candidates(map_dir: Path) -> list[tuple[Path, str]]:
        # Order matches the resolved `stack`; if unspecified, try HLoc first
        # (newer stack) then ORB.
        h = (map_dir / "hloc" / "tx_hloc_tag.json", "tx_hloc_tag")
        s = (map_dir / "slam" / "tx_slam_tag.json", "tx_slam_tag")
        if stack == "orbslam3":
            return [s, h]
        elif stack == "hloc":
            return [h, s]
        else:
            return [h, s]

    if kind == "mapping":
        cands = _candidates(session_dir)
    else:
        parent = meta.get("parent_map")
        if not parent:
            return None, None
        cands = _candidates(sessions_root / parent)

    for tag_path, key in cands:
        if tag_path.exists():
            break
    else:
        return None, None

    source = str(tag_path.relative_to(sessions_root)) if tag_path.is_relative_to(sessions_root) else str(tag_path)
    data = json.loads(tag_path.read_text())
    # Both stacks store a 4×4 SE(3) under their own key — same numerical role
    # (T_world_tag, mapping the SLAM/HLoc world into the gravity-aligned tag frame).
    arr = data.get(key) or data.get("tx_slam_tag") or data.get("tx_hloc_tag")
    if arr is None:
        raise KeyError(f"{tag_path}: no tx_slam_tag/tx_hloc_tag key in JSON")
    return np.array(arr, dtype=np.float64), source


def align(session_dir: Path, imu_filter: bool = True) -> Path:
    session_dir = Path(session_dir).resolve()
    meta = yaml.safe_load((session_dir / "meta.yaml").read_text())
    wall_start = float(meta["video"]["wall_start"])

    traj_path = session_dir / "slam" / "camera_trajectory.csv"
    proprio_path = session_dir / "raw" / "proprio.parquet"
    if not traj_path.exists():
        raise FileNotFoundError(f"No camera_trajectory.csv in {session_dir/'slam'}")
    if not proprio_path.exists():
        raise FileNotFoundError(f"No proprio.parquet in {session_dir/'raw'}")

    # Trajectory (one row per video frame)
    traj = pd.read_csv(traj_path)
    traj["is_lost"] = traj["is_lost"].astype(str).str.lower().eq("true")
    traj["is_keyframe"] = traj["is_keyframe"].astype(str).str.lower().eq("true")
    frame_t_rel = traj["timestamp"].to_numpy(dtype=np.float64)
    frame_wall = wall_start + frame_t_rel

    cam_xyz = traj[["x", "y", "z"]].to_numpy(dtype=np.float64)
    cam_quat = traj[["q_x", "q_y", "q_z", "q_w"]].to_numpy(dtype=np.float64)
    # Replace any lost-frame zero quats with identity to avoid scipy complaints
    norms = np.linalg.norm(cam_quat, axis=1)
    lost = traj["is_lost"].to_numpy() | (norms < 1e-6)
    cam_quat[lost] = np.array([0.0, 0.0, 0.0, 1.0])

    # Tag-frame alignment. The mapping session's tx_slam_tag.json defines a
    # gravity-aligned world (tag plane = XY, normal = +Z). Applying inv(T_slam_tag)
    # rotates & translates every cam pose from ORB's tilted slam frame into that
    # gravity-aligned tag frame. Fall back silently if the file is missing but
    # log loud — downstream readers assume tag-frame by default now.
    tx_slam_tag, tag_frame_source = _load_tx_slam_tag(session_dir, meta)
    if tx_slam_tag is not None:
        tx_tag_slam = np.linalg.inv(tx_slam_tag)
        cam_xyz, cam_quat = _apply_world_transform(cam_xyz, cam_quat, tx_tag_slam)
        print(f"  [align] world-frame = tag frame (source: {tag_frame_source})")
    else:
        print("  [align] WARNING: no tx_slam_tag.json — poses in raw ORB-SLAM3 world "
              "frame (may have 5–6° residual tilt). Run `glove.process detect-aruco` "
              "+ `calibrate-tag` on the parent map to enable gravity alignment.")

    # IMU-consistency filter (rejects estimator flips, fills by interp+slerp).
    filter_info: dict | None = None
    if imu_filter:
        imu_wall, imu_accel = load_daq_imu(session_dir)
        if len(imu_wall) == 0:
            print("  [imu-filter] DAQ IMU unavailable — skipping (raw SLAM poses used)")
        else:
            fr = filter_poses(
                cam_xyz, cam_quat, frame_t_rel, lost,
                frame_wall, imu_wall, imu_accel,
            )
            cam_xyz = fr.xyz
            cam_quat = fr.quat_xyzw
            print(f"  [imu-filter] rejected {fr.n_flips} flips + {fr.n_lost_in} lost → "
                  f"max jump {fr.max_jump_before_mm:.0f}mm → {fr.max_jump_after_mm:.0f}mm "
                  f"({fr.iterations} passes)")
            filter_info = {
                "applied": True,
                "n_flips": fr.n_flips,
                "n_lost": fr.n_lost_in,
                "max_jump_before_mm": fr.max_jump_before_mm,
                "max_jump_after_mm": fr.max_jump_after_mm,
                "iterations": fr.iterations,
            }

    # Body-frame transform (if calibration available)
    T_mtcp_cam = _load_T_mtcp_gopro()
    if T_mtcp_cam is not None:
        mtcp_xyz, mtcp_quat = _apply_body_transform(cam_xyz, cam_quat, T_mtcp_cam)
        pose_xyz, pose_quat = mtcp_xyz, mtcp_quat
        body_frame_applied = True
    else:
        print(f"WARNING: no {BODY_FRAME_KEY} in {CALIB_PATH}; pose_* will be the CAMERA pose, not MTCP.")
        pose_xyz, pose_quat = cam_xyz, cam_quat
        body_frame_applied = False

    # Proprio. Empty proprio (master tape wasn't running during capture) is tolerated
    # so the pose-only trajectory still becomes viewable in the review dashboard —
    # hall/tof columns are filled with sentinels and quality metrics on those channels
    # will be NaN/degenerate. Not a green-light state for training, but legible for
    # investigation.
    pro = pq.read_table(proprio_path).to_pandas()
    proprio_missing = len(pro) == 0
    if proprio_missing:
        print(f"[align] WARNING: proprio.parquet is empty in {session_dir} — "
              f"filling hall/tof with sentinels; episode.parquet will be pose-only.")
        t_src = np.array([], dtype=np.float64)
    else:
        pro = pro.sort_values("wall_time").reset_index(drop=True)
        t_src = pro["wall_time"].to_numpy(dtype=np.float64)

    out = {
        "frame_idx": traj["frame_idx"].to_numpy(dtype=np.int32),
        "t_rel": frame_t_rel.astype(np.float32),
        "wall_time": frame_wall,
        "pose_x": pose_xyz[:, 0].astype(np.float32),
        "pose_y": pose_xyz[:, 1].astype(np.float32),
        "pose_z": pose_xyz[:, 2].astype(np.float32),
        # scipy returns [x,y,z,w], we store as (qw, qx, qy, qz) for readability
        "pose_qw": pose_quat[:, 3].astype(np.float32),
        "pose_qx": pose_quat[:, 0].astype(np.float32),
        "pose_qy": pose_quat[:, 1].astype(np.float32),
        "pose_qz": pose_quat[:, 2].astype(np.float32),
        "cam_x": cam_xyz[:, 0].astype(np.float32),
        "cam_y": cam_xyz[:, 1].astype(np.float32),
        "cam_z": cam_xyz[:, 2].astype(np.float32),
        "cam_qw": cam_quat[:, 3].astype(np.float32),
        "cam_qx": cam_quat[:, 0].astype(np.float32),
        "cam_qy": cam_quat[:, 1].astype(np.float32),
        "cam_qz": cam_quat[:, 2].astype(np.float32),
        "is_lost": traj["is_lost"].to_numpy(dtype=bool),
        "is_keyframe": traj["is_keyframe"].to_numpy(dtype=bool),
    }

    for i in range(4):
        for ax in ("bx", "by", "bz"):
            col = f"hall_{i}_{ax}"
            if proprio_missing:
                out[col] = np.zeros(len(frame_wall), dtype=np.int16)
            else:
                y = pro[col].to_numpy(dtype=np.float32)
                interp = np.interp(frame_wall, t_src, y)
                out[col] = np.rint(interp).astype(np.int16)

    for col in ("tof_top_mm", "tof_bottom_mm"):
        if proprio_missing:
            out[col] = np.full(len(frame_wall), -1, dtype=np.int16)
            continue
        y = pro[col].to_numpy(dtype=np.float32)
        y[y < 0] = np.nan
        valid = ~np.isnan(y)
        if valid.any():
            interp = np.interp(frame_wall, t_src[valid], y[valid])
        else:
            interp = np.full_like(frame_wall, -1.0, dtype=np.float64)
        out[col] = np.rint(interp).astype(np.int16)

    aligned_dir = session_dir / "aligned"
    aligned_dir.mkdir(exist_ok=True)
    out_path = aligned_dir / "episode.parquet"

    fields = [
        pa.field("frame_idx", pa.int32()),
        pa.field("t_rel", pa.float32()),
        pa.field("wall_time", pa.float64()),
        pa.field("pose_x", pa.float32()), pa.field("pose_y", pa.float32()), pa.field("pose_z", pa.float32()),
        pa.field("pose_qw", pa.float32()), pa.field("pose_qx", pa.float32()),
        pa.field("pose_qy", pa.float32()), pa.field("pose_qz", pa.float32()),
        pa.field("cam_x", pa.float32()), pa.field("cam_y", pa.float32()), pa.field("cam_z", pa.float32()),
        pa.field("cam_qw", pa.float32()), pa.field("cam_qx", pa.float32()),
        pa.field("cam_qy", pa.float32()), pa.field("cam_qz", pa.float32()),
        pa.field("is_lost", pa.bool_()),
        pa.field("is_keyframe", pa.bool_()),
    ]
    for i in range(4):
        for ax in ("bx", "by", "bz"):
            fields.append(pa.field(f"hall_{i}_{ax}", pa.int16()))
    fields.append(pa.field("tof_top_mm", pa.int16()))
    fields.append(pa.field("tof_bottom_mm", pa.int16()))

    table = pa.table(out, schema=pa.schema(fields))
    pq.write_table(table, out_path, compression="zstd")

    # Stamp meta.yaml with the provenance of the body-frame transform
    meta.setdefault("align", {})
    meta["align"]["body_frame_applied"] = body_frame_applied
    meta["align"]["body_frame_source"] = BODY_FRAME_KEY if body_frame_applied else None
    meta["align"]["world_frame_source"] = tag_frame_source if tx_slam_tag is not None else "slam_raw"
    meta["align"]["imu_filter"] = filter_info or {"applied": False}
    meta["align"]["quality_schema_version"] = QUALITY_SCHEMA_VERSION
    (session_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    # Auto-score: writes a `quality` block into meta.yaml for the review gate.
    score_session(session_dir, persist=True)

    return out_path


def cli(args) -> int:
    path = align(args.session_dir, imu_filter=not args.no_imu_filter)
    tbl = pq.read_table(path)
    n = len(tbl)
    n_tracked = int(np.sum(~tbl["is_lost"].to_numpy()))
    print(f"Wrote {path}")
    print(f"  {n:,} frames, {n_tracked:,} tracked ({100*n_tracked/max(n,1):.1f}%)")
    return 0
