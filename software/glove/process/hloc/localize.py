"""HLoc localization for a demo session against a parent mapping session's model.

Replaces `gopro_slam --load_map` from the ORB-SLAM3 path. Same output schema:
writes `<demo>/slam/camera_trajectory.csv` + `parent_map` in meta.yaml so the
existing `align.py` consumes it unchanged.

Pipeline per demo:
    1. extract demo frames (stride=2 by default → 30Hz from 60fps)
    2. SuperPoint + NetVLAD on demo frames
    3. retrieval against the mapping clip's NetVLAD descriptors (top-K=20)
    4. filter retrieval pairs to images registered in the SfM model
    5. SuperPoint+LightGlue match
    6. PnP localize against the SfM model (hloc.localize_sfm)
    7. pose-graph batch smooth (manifold-aware rotation, robust position LM)
    8. write camera_trajectory.csv (same schema as ORB-SLAM3) — in HLoc-world frame
       so downstream `align.py` can apply tx_hloc_tag uniformly with ORB's path

Outputs:
    <demo>/hloc/feats.h5, netvlad.h5, pairs_filtered.txt, matches.h5
    <demo>/hloc/hloc_results.txt    raw per-frame PnP results
    <demo>/hloc/meta.yaml           per-demo provenance
    <demo>/slam/camera_trajectory.csv  smoothed trajectory (HLoc world frame)
                                       — backs up any prior ORB-SLAM3 file under slam/orbslam3_backup/

`align.py` recognizes `slam_provenance.stack == "hloc"` and reads
`<parent_map>/hloc/tx_hloc_tag.json` to apply the tag-frame transform.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from glove.process.hloc.smooth import HLocPoseObservation, smooth_trajectory


DEFAULT_INTRINSICS_JSON = Path("software/configs/calibration/gopro_intrinsics_2_7k.json")
DEFAULT_DEMO_FRAME_STRIDE = 2
DEFAULT_NUM_RETRIEVAL_PAIRS = 20


@dataclass
class HLocLocalizeStats:
    n_video_frames: int           # video frames in the original MP4
    n_query_frames: int           # extracted frames (after stride subsample)
    n_localized: int              # PnP-successful frames
    n_smoothed: int               # contiguous frames after gap-fill + smooth
    pose_jump_max_mm: float
    pose_jump_p99_mm: float
    n_jumps_over_20mm: int
    n_jumps_over_50mm: int
    verdict: str                  # CLEAN / NOISY / RAILS / auto_pass-style label
    extent_cm: list[float]
    path_length_m: float
    wall_clock_s: float


def _load_intrinsics_string(json_path: Path) -> str:
    raw = json.loads(json_path.read_text())
    intr = raw["intrinsics"]
    f = intr["focal_length"]
    return (
        f"OPENCV_FISHEYE {raw['image_width']} {raw['image_height']} "
        f"{f} {f} {intr['principal_pt_x']} {intr['principal_pt_y']} "
        f"{intr['radial_distortion_1']} {intr['radial_distortion_2']} "
        f"{intr['radial_distortion_3']} {intr['radial_distortion_4']}"
    )


def _extract_frames(mp4: Path, out_dir: Path, stride: int, prefix: str,
                     static_mask: Optional[np.ndarray] = None) -> list[tuple[int, str]]:
    """Decode every `stride`-th frame to JPEG. If `static_mask` is provided,
    white pixels (>127) are blacked out before writing — same convention as
    `glove.process.hloc.build_model._extract_frames`."""
    import av
    import cv2
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    with av.open(str(mp4)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(s)):
            if i % stride != 0:
                continue
            img = frame.to_ndarray(format="bgr24")
            if static_mask is not None:
                img[static_mask > 127] = 0
            name = f"{prefix}_{i:06d}.jpg"
            cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            names.append((i, name))
    return names


def _load_static_mask_for_frame(path: Path, frame_shape) -> np.ndarray:
    import cv2
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    h, w = frame_shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def _verdict(d_mm: np.ndarray) -> str:
    """Same thresholds as glove.quality.compute / SlamStats.verdict."""
    if (d_mm > 50).any():
        return "RAILS"
    if (d_mm > 20).sum() > 5:
        return "NOISY"
    if (d_mm > 20).sum() == 0:
        return "auto_pass"
    return f"NOISY({(d_mm>20).sum()})"


def _write_camera_trajectory_csv(traj_csv: Path, frames: np.ndarray,
                                  positions: np.ndarray, quats_xyzw: np.ndarray,
                                  fps: float = 60.0) -> None:
    """Same schema as ORB-SLAM3's camera_trajectory.csv so align.py is unchanged."""
    traj_csv.parent.mkdir(exist_ok=True)
    with traj_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "timestamp", "state", "is_lost", "is_keyframe",
                    "x", "y", "z", "q_x", "q_y", "q_z", "q_w"])
        for k, fi in enumerate(frames):
            t_rel = fi / fps
            w.writerow([
                int(fi), f"{t_rel:.6f}", "OK", "False", str(k % 10 == 0),
                f"{positions[k, 0]:.6f}", f"{positions[k, 1]:.6f}", f"{positions[k, 2]:.6f}",
                f"{quats_xyzw[k, 0]:.6f}", f"{quats_xyzw[k, 1]:.6f}",
                f"{quats_xyzw[k, 2]:.6f}", f"{quats_xyzw[k, 3]:.6f}",
            ])


def localize(
    demo_session_dir: Path,
    map_session_dir: Path,
    intrinsics_json: Path = DEFAULT_INTRINSICS_JSON,
    demo_frame_stride: int = DEFAULT_DEMO_FRAME_STRIDE,
    num_retrieval_pairs: int = DEFAULT_NUM_RETRIEVAL_PAIRS,
    mask_img: Optional[Path] = None,
) -> HLocLocalizeStats:
    """Localize a demo session against a parent mapping session's HLoc model."""
    from scipy.spatial.transform import Rotation
    from hloc import (
        extract_features,
        match_features,
        pairs_from_retrieval,
        localize_sfm,
    )
    import pycolmap

    demo = Path(demo_session_dir).resolve()
    mp4 = demo / "raw" / "raw_video.mp4"
    if not mp4.exists():
        raise FileNotFoundError(f"No raw_video.mp4 in {demo/'raw'}")
    map_ = Path(map_session_dir).resolve()
    map_hloc = map_ / "hloc"
    if not (map_hloc / "tx_hloc_tag.json").exists():
        raise FileNotFoundError(
            f"No HLoc model at {map_hloc} — run "
            f"`glove.process build-model {map_}` first."
        )

    out = demo / "hloc"
    out.mkdir(exist_ok=True)
    img_dir = out / "images"

    # Resolve SfM model path (build_model wrote a hint file)
    sfm_path_hint = map_hloc / "sfm_model_path.txt"
    if sfm_path_hint.exists():
        rel = sfm_path_hint.read_text().strip()
        sfm_model_path = (map_ / rel).resolve()
    else:
        sfm_model_path = map_hloc / "sfm" / "models" / "0"
        if not (sfm_model_path / "cameras.bin").exists():
            sfm_model_path = map_hloc / "sfm"
    m = pycolmap.Reconstruction(str(sfm_model_path))
    registered = {img.name for img in m.images.values()}
    map_feats = map_hloc / "feats.h5"
    map_gd = map_hloc / "netvlad.h5"
    if not (map_feats.exists() and map_gd.exists()):
        raise FileNotFoundError(
            f"Missing feats.h5/netvlad.h5 in {map_hloc} — re-run `build-model` to regenerate."
        )

    intr_str = _load_intrinsics_string(intrinsics_json)
    name = demo.name
    t0 = time.time()
    print(f"[hloc.localize] {name} → {map_.name}")

    # If the parent map was built with a mask, use the same mask here unless the
    # caller overrode. Inconsistent masking (mapped without mask + localize with
    # mask, or vice versa) makes the SuperPoint feature distributions
    # non-comparable and tanks PnP.
    map_meta = yaml.safe_load((map_hloc / "meta.yaml").read_text())
    map_mask_path = map_meta.get("mask_img")
    if mask_img is None and map_mask_path:
        mask_img = Path(map_mask_path)
    elif mask_img is not None and map_mask_path and str(mask_img) != map_mask_path:
        print(f"  WARNING: localize mask ({mask_img}) differs from map's "
              f"mask ({map_mask_path}). PnP may underperform.")

    # 1. Frames
    frames_json = out / "frames.json"
    if not img_dir.exists() or not any(img_dir.iterdir()):
        mask = None
        if mask_img is not None:
            import av as _av
            with _av.open(str(mp4)) as _c:
                _frame = next(_c.decode(_c.streams.video[0]))
                mask = _load_static_mask_for_frame(
                    mask_img, _frame.to_ndarray(format="bgr24").shape
                )
            print(f"  applying static mask {mask_img.name}")
        print(f"  extracting demo frames (stride={demo_frame_stride})...")
        frames = _extract_frames(mp4, img_dir, demo_frame_stride, name, static_mask=mask)
        frames_json.write_text(json.dumps(frames))
    else:
        frames = json.loads(frames_json.read_text())
    image_names = [n for _, n in frames]
    print(f"  query frames: {len(image_names)}")

    # 2-3. Features + globals
    # Match build_model.py: superpoint_aachen gives tighter per-frame PnP than
    # superpoint_max on our handheld GoPro footage. See note in build_model.py.
    feature_conf = extract_features.confs["superpoint_aachen"]
    retrieval_conf = extract_features.confs["netvlad"]
    feats = extract_features.main(feature_conf, img_dir, feature_path=out / "feats.h5",
                                   image_list=image_names, overwrite=False)
    gd = extract_features.main(retrieval_conf, img_dir, feature_path=out / "netvlad.h5",
                                image_list=image_names, overwrite=False)

    # 4. Retrieval
    pairs_raw = out / "pairs.txt"
    pairs_from_retrieval.main(
        gd, pairs_raw, num_matched=num_retrieval_pairs,
        db_prefix="map_", query_prefix=f"{name}_", db_descriptors=map_gd,
    )

    # 4b. Filter retrieval to only images registered in the SfM (avoid the
    #     localize_sfm IndexError when retrieval returns map images that didn't
    #     register into the final reconstruction).
    pairs_filtered = out / "pairs_filtered.txt"
    qhas: dict[str, int] = {}
    with pairs_raw.open() as f, pairs_filtered.open("w") as g:
        for line in f:
            q, db = line.strip().split()
            if db in registered:
                g.write(line)
                qhas[q] = qhas.get(q, 0) + 1
    valid = [n for n in image_names if qhas.get(n, 0) > 0]
    print(f"  valid queries: {len(valid)}/{len(image_names)}")
    if not valid:
        raise RuntimeError(
            f"Zero queries had retrieval candidates registered in the SfM model — "
            f"either the demo viewpoint doesn't overlap the map, or the SfM has only a "
            f"small fragment registered."
        )

    # 5. Match
    matcher_conf = match_features.confs["superpoint+lightglue"]
    matches = match_features.main(matcher_conf, pairs_filtered, features=feats,
                                   matches=out / "matches.h5",
                                   features_ref=map_feats, overwrite=False)

    # 6. PnP localize
    ql = out / "query_list.txt"
    with ql.open("w") as f:
        for n in valid:
            f.write(f"{n} {intr_str}\n")
    loc_results = out / "hloc_results.txt"
    localize_sfm.main(
        reference_sfm=sfm_model_path, queries=ql, retrieval=pairs_filtered,
        features=feats, matches=matches, results=loc_results, ransac_thresh=12,
    )

    # 7. Parse results + smooth
    pkl = out / "hloc_results.txt_logs.pkl"
    inliers_by_frame: dict[int, int] = {}
    if pkl.exists():
        logs = pickle.loads(pkl.read_bytes())
        for nm, entry in logs.get("loc", {}).items():
            try:
                fi = int(nm.rsplit("_", 1)[1].split(".")[0])
            except Exception:
                continue
            n = None
            if isinstance(entry.get("PnP_ret"), dict):
                n = entry["PnP_ret"].get("num_inliers")
            if n is None:
                n = entry.get("num_inliers", 50)
            inliers_by_frame[fi] = int(n)

    scale_data = json.loads((map_hloc / "tx_hloc_tag.json").read_text())
    scale = scale_data["scale_m_per_colmap_unit"]

    obs = []
    for line in loc_results.read_text().splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        nm = parts[0]
        try:
            fi = int(nm.rsplit("_", 1)[1].split(".")[0])
        except Exception:
            continue
        q_w_x_y_z = np.array([float(x) for x in parts[1:5]])  # qw qx qy qz (world→cam)
        tv = np.array([float(x) for x in parts[5:8]])
        # world→cam quat is qw,qx,qy,qz; convert to scipy xyzw
        R_wc = Rotation.from_quat([q_w_x_y_z[1], q_w_x_y_z[2], q_w_x_y_z[3], q_w_x_y_z[0]]).as_matrix()
        cam_pos_hloc = -R_wc.T @ tv * scale
        # Camera orientation (R_world_cam) — invert
        quat_world_cam = Rotation.from_quat(
            [q_w_x_y_z[1], q_w_x_y_z[2], q_w_x_y_z[3], q_w_x_y_z[0]]
        ).inv().as_quat()
        obs.append(HLocPoseObservation(
            frame_idx=fi, pos_world=cam_pos_hloc, quat_xyzw=quat_world_cam,
            n_inliers=inliers_by_frame.get(fi, 50),
        ))

    if len(obs) < 3:
        raise RuntimeError(f"Only {len(obs)} valid PnP poses — need ≥3 for smoothing.")

    pos_smooth, quat_smooth, smooth_info = smooth_trajectory(obs)

    # 8. Write camera_trajectory.csv (HLoc world frame — align will apply tx_hloc_tag)
    fr_out = np.arange(smooth_info["frame_idx_start"], smooth_info["frame_idx_end"] + 1)

    traj_csv = demo / "slam" / "camera_trajectory.csv"
    if traj_csv.exists():
        backup = demo / "slam" / "orbslam3_backup"
        backup.mkdir(exist_ok=True)
        if not (backup / "camera_trajectory.csv").exists():
            shutil.copy2(traj_csv, backup / "camera_trajectory.csv")
    _write_camera_trajectory_csv(traj_csv, fr_out, pos_smooth, quat_smooth)

    # Verdict on smoothed trajectory in HLoc-world frame (rigid transform —
    # equivalent jumps to tag frame). Position-only metric.
    d_mm = np.linalg.norm(np.diff(pos_smooth, axis=0), axis=1) * 1000
    extent_cm = ((pos_smooth.max(0) - pos_smooth.min(0)) * 100).tolist()
    stats = HLocLocalizeStats(
        n_video_frames=int(fr_out.max() + 1),
        n_query_frames=len(image_names),
        n_localized=len(obs),
        n_smoothed=len(fr_out),
        pose_jump_max_mm=float(d_mm.max()) if len(d_mm) else 0,
        pose_jump_p99_mm=float(np.percentile(d_mm, 99)) if len(d_mm) else 0,
        n_jumps_over_20mm=int((d_mm > 20).sum()),
        n_jumps_over_50mm=int((d_mm > 50).sum()),
        verdict=_verdict(d_mm),
        extent_cm=[round(x, 1) for x in extent_cm],
        path_length_m=round(float(d_mm.sum() / 1000), 3),
        wall_clock_s=time.time() - t0,
    )

    (out / "meta.yaml").write_text(yaml.safe_dump({
        "stack": "hloc",
        "parent_map": map_.name,
        "demo_frame_stride": demo_frame_stride,
        "num_retrieval_pairs": num_retrieval_pairs,
        "stats": stats.__dict__,
    }, sort_keys=False))

    # Stamp parent_map + slam_provenance into the demo's main meta.yaml so align
    # routes through the HLoc tx_hloc_tag.json under parent_map/hloc/.
    main_meta_path = demo / "meta.yaml"
    if main_meta_path.exists():
        meta = yaml.safe_load(main_meta_path.read_text()) or {}
        meta["parent_map"] = map_.name
        meta.setdefault("slam_provenance", {})
        meta["slam_provenance"]["stack"] = "hloc"
        meta["slam_provenance"]["localized_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        main_meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))

    print(f"[hloc.localize] {stats.verdict}  max_jump={stats.pose_jump_max_mm:.1f}mm  "
          f">20={stats.n_jumps_over_20mm}  >50={stats.n_jumps_over_50mm}  "
          f"path={stats.path_length_m}m  t={stats.wall_clock_s:.1f}s")
    return stats


def cli(args) -> int:
    stats = localize(
        args.demo_session_dir, args.map_session_dir,
        intrinsics_json=args.intrinsics_json,
        demo_frame_stride=args.demo_frame_stride,
        num_retrieval_pairs=args.num_retrieval_pairs,
        mask_img=args.mask_img,
    )
    print(f"\nLocalize summary for {args.demo_session_dir.name}:")
    print(f"  query frames:   {stats.n_query_frames}")
    print(f"  localized:      {stats.n_localized}")
    print(f"  smoothed:       {stats.n_smoothed} frames")
    print(f"  verdict:        {stats.verdict}")
    print(f"  pose_jump_max:  {stats.pose_jump_max_mm:.1f}mm")
    print(f"  >20mm / >50mm:  {stats.n_jumps_over_20mm} / {stats.n_jumps_over_50mm}")
    print(f"  extent (cm):    {stats.extent_cm}")
    print(f"  path:           {stats.path_length_m}m")
    print(f"  wall clock:     {stats.wall_clock_s:.1f}s")
    return 0


def build_parser(sub) -> None:
    p = sub.add_parser(
        "localize-hloc",
        help="HLoc PnP+smoothing for a demo (replaces ORB-SLAM3 localize on the HLoc stack).",
    )
    p.add_argument("demo_session_dir", type=Path)
    p.add_argument("--map", dest="map_session_dir", type=Path, required=True,
                   help="Parent mapping session (must have run `build-model` first).")
    p.add_argument("--intrinsics-json", type=Path, default=DEFAULT_INTRINSICS_JSON)
    p.add_argument("--demo-frame-stride", type=int, default=DEFAULT_DEMO_FRAME_STRIDE)
    p.add_argument("--num-retrieval-pairs", type=int, default=DEFAULT_NUM_RETRIEVAL_PAIRS)
    p.add_argument("--mask-img", type=Path, default=None,
                   help="Optional override of the parent map's static mask. By default "
                        "we inherit whatever mask the map was built with (or none).")
    p.set_defaults(func=cli)
