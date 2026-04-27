"""Build a COLMAP scene model from a mapping session's MP4.

Replaces the role that `gopro_slam --save_map` plays in the ORB-SLAM3 path.
Outputs go under `<mapping_session>/hloc/` so the existing `<mapping_session>/slam/`
tree (ORB-SLAM3 atlas) is untouched and the two stacks can coexist.

Outputs:
    <session>/hloc/sfm/                COLMAP reconstruction binaries
    <session>/hloc/feats.h5            SuperPoint features (re-used at localize)
    <session>/hloc/netvlad.h5          NetVLAD globals (re-used at localize)
    <session>/hloc/matches.h5          intra-mapping LightGlue matches
    <session>/hloc/pairs.txt           retrieval-guided pair list
    <session>/hloc/tx_hloc_tag.json    HLoc-world → tag-frame transform (4×4)
    <session>/hloc/meta.yaml           provenance: frame stride, KP config, scale, KF count

If a usable SfM model already exists at `hloc/sfm/cameras.bin`, the COLMAP build
is skipped and the existing model is loaded — makes reruns cheap.

Tag-frame calibration uses the existing `<session>/aruco/tag_detection.pkl`
(produced by `glove.process detect-aruco`). Run that step first.
"""

from __future__ import annotations

import json
import os
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# Suppress hloc/torch import-time noise on first call
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# Default config — matches what we proved on the smoke test
DEFAULT_INTRINSICS_JSON = Path("software/configs/calibration/gopro_intrinsics_2_7k.json")
DEFAULT_TAG_ID = 13
DEFAULT_MAP_FRAME_STRIDE = 30
DEFAULT_NUM_RETRIEVAL_PAIRS = 20


@dataclass
class HLocBuildStats:
    n_mapping_frames: int
    n_sfm_registered: int
    n_sfm_points: int
    scale_m_per_colmap_unit: float
    n_tag_samples_used: int
    wall_clock_s: float


def _load_intrinsics(json_path: Path) -> dict:
    raw = json.loads(json_path.read_text())
    intr = raw["intrinsics"]
    f = intr["focal_length"]
    px, py = intr["principal_pt_x"], intr["principal_pt_y"]
    k1, k2, k3, k4 = (intr[f"radial_distortion_{i}"] for i in range(1, 5))
    return {
        "image_width": raw["image_width"],
        "image_height": raw["image_height"],
        "K": np.array([[f, 0, px], [0, f, py], [0, 0, 1]], dtype=np.float64),
        "D": np.array([[k1], [k2], [k3], [k4]], dtype=np.float64),
        "intr_str": (
            f"OPENCV_FISHEYE {raw['image_width']} {raw['image_height']} "
            f"{f} {f} {px} {py} {k1} {k2} {k3} {k4}"
        ),
        "image_options": {
            "camera_model": "OPENCV_FISHEYE",
            "camera_params": ",".join(str(x) for x in [f, f, px, py, k1, k2, k3, k4]),
        },
    }


def _extract_frames(mp4: Path, out_dir: Path, stride: int, prefix: str,
                     static_mask: Optional[np.ndarray] = None) -> list[tuple[int, str]]:
    """Decode every `stride`-th frame from `mp4` to JPEGs in `out_dir`.

    If `static_mask` is provided (uint8 array, shape matching frame, white=masked),
    masked pixels are blacked out before writing the JPEG. SuperPoint and NetVLAD
    later read these JPEGs directly and never see the original glove pixels.

    Returns `[(orig_frame_idx, filename), …]` so we can map back to video frames later.
    """
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
            name = f"map_{i:06d}.jpg"
            cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            names.append((i, name))
    return names


def _load_static_mask(path: Path, frame_shape: tuple[int, int, int]) -> np.ndarray:
    """Load a static-mask PNG; resize to match the frame if needed.

    The mask convention is white pixels (>127) = masked region (will be blacked out).
    """
    import cv2
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    h_target, w_target = frame_shape[:2]
    if mask.shape != (h_target, w_target):
        mask = cv2.resize(mask, (w_target, h_target), interpolation=cv2.INTER_NEAREST)
    return mask


def _recover_scale_and_tag(sfm_model, tag_results: list[dict], tag_id: int
                            ) -> tuple[float, np.ndarray, dict]:
    """From SfM camera poses + ArUco tag detections, recover (scale, T_hloc_tag).

    Scale: median of cam-to-cam distance ratios in tag-frame meters / hloc-world units.
    T_hloc_tag: geometric-median + 90%-quantile filter on the per-frame tx_hloc_tag samples.
    """
    from scipy.spatial.transform import Rotation
    sfm_image_by_name = {img.name: img for img in sfm_model.images.values()}
    common = []
    for r in tag_results:
        if tag_id not in r["tag_dict"]:
            continue
        td = r["tag_dict"][tag_id]
        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3] = Rotation.from_rotvec(td["rvec"]).as_matrix()
        T_cam_tag[:3, 3] = td["tvec"]
        img = sfm_image_by_name.get(f"map_{r['frame_idx']:06d}.jpg")
        if img is None:
            continue
        cfw = img.cam_from_world()
        R_cw = Rotation.from_quat(cfw.rotation.quat).as_matrix()
        t_cw = np.asarray(cfw.translation)
        T_hloc_cam = np.eye(4)
        T_hloc_cam[:3, :3] = R_cw.T
        T_hloc_cam[:3, 3] = -R_cw.T @ t_cw
        cam_in_tag = -T_cam_tag[:3, :3].T @ T_cam_tag[:3, 3]
        common.append((r["frame_idx"], T_hloc_cam, T_cam_tag, cam_in_tag))

    if len(common) < 3:
        raise RuntimeError(
            f"Only {len(common)} mapping frames have both SfM pose AND tag {tag_id} "
            f"detection. Need ≥3 for scale recovery."
        )

    # Scale via cam-pair distance ratios (sub-sampled to O(N) for large N)
    ratios = []
    n = len(common)
    for i in range(0, n, max(1, n // 200)):
        for j in range(i + 1, min(i + 30, n)):
            d_m = np.linalg.norm(common[i][3] - common[j][3])
            d_c = np.linalg.norm(common[i][1][:3, 3] - common[j][1][:3, 3])
            if d_m > 0.05 and d_c > 0.01:
                ratios.append(d_m / d_c)
    scale = float(np.median(ratios))

    # T_hloc_tag — geometric median on positions, then 90%-quantile filter, then nearest sample
    samples = []
    for _, T_hloc_cam, T_cam_tag, _ in common:
        T_hcs = T_hloc_cam.copy()
        T_hcs[:3, 3] *= scale
        d = float(np.linalg.norm(T_cam_tag[:3, 3]))
        if 0.3 <= d <= 2.0:  # Same UMI calibrate_slam_tag filter
            samples.append(T_hcs @ T_cam_tag)
    samples = np.array(samples)
    if len(samples) == 0:
        raise RuntimeError(
            f"No tag detections passed distance filter (0.3 < d_cam_tag < 2.0 m)."
        )

    pos = samples[:, :3, 3]
    y = pos.mean(axis=0)
    for _ in range(100):  # Weiszfeld iteration
        d = np.linalg.norm(pos - y, axis=1)
        nz = d > 1e-9
        w = 1.0 / d[nz]
        y_new = (w[:, None] * pos[nz]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < 1e-9:
            break
        y = y_new
    dists = np.linalg.norm(pos - y, axis=1)
    keep = dists < np.quantile(dists, 0.9)
    nn_idx = int(np.argmin(np.linalg.norm(pos[keep] - pos[keep].mean(axis=0), axis=1)))
    T_hloc_tag = samples[keep][nn_idx]

    info = {
        "n_ratios": len(ratios),
        "scale_p25": float(np.percentile(ratios, 25)),
        "scale_p75": float(np.percentile(ratios, 75)),
        "n_tag_samples_total": int(len(samples)),
        "n_tag_samples_used": int(keep.sum()),
        "tag_position_std_cm_90pct": [
            round(float(s * 100), 3) for s in pos[keep].std(axis=0)
        ],
    }
    return scale, T_hloc_tag, info


def build_model(
    session_dir: Path,
    intrinsics_json: Path = DEFAULT_INTRINSICS_JSON,
    tag_id: int = DEFAULT_TAG_ID,
    map_frame_stride: int = DEFAULT_MAP_FRAME_STRIDE,
    num_retrieval_pairs: int = DEFAULT_NUM_RETRIEVAL_PAIRS,
    mask_img: Optional[Path] = None,
) -> HLocBuildStats:
    """Build the HLoc COLMAP model + tag-frame calibration for a mapping session."""
    session_dir = Path(session_dir).resolve()
    mp4 = session_dir / "raw" / "raw_video.mp4"
    if not mp4.exists():
        raise FileNotFoundError(f"No raw_video.mp4 in {session_dir/'raw'}")
    tag_pkl = session_dir / "aruco" / "tag_detection.pkl"
    if not tag_pkl.exists():
        raise FileNotFoundError(
            f"No aruco/tag_detection.pkl in {session_dir} — run "
            f"`glove.process detect-aruco {session_dir}` first."
        )

    out = session_dir / "hloc"
    out.mkdir(exist_ok=True)
    img_dir = out / "map_images"
    sfm_dir = out / "sfm"
    sfm_dir.mkdir(exist_ok=True)

    # Lazy hloc imports — heavy (downloads model weights on first call)
    from hloc import (
        extract_features,
        match_features,
        pairs_from_retrieval,
        reconstruction,
    )
    import pycolmap

    intr = _load_intrinsics(intrinsics_json)
    t0 = time.time()
    print(f"[hloc.build_model] {session_dir.name}")

    # 1. Frames (with optional static mask applied during extraction)
    frames_json = out / "map_frames.json"
    if not img_dir.exists() or not any(img_dir.iterdir()):
        # Probe one frame to know the mask resize target
        mask = None
        if mask_img is not None:
            import av as _av
            with _av.open(str(mp4)) as _c:
                _frame = next(_c.decode(_c.streams.video[0]))
                mask = _load_static_mask(mask_img, _frame.to_ndarray(format="bgr24").shape)
            print(f"  applying static mask {mask_img.name}: "
                  f"{(mask>127).sum():,}/{mask.size:,} pixels masked "
                  f"({100*(mask>127).mean():.1f}%)")
        print(f"  extracting mapping frames (stride={map_frame_stride})...")
        frames = _extract_frames(mp4, img_dir, map_frame_stride, "map",
                                  static_mask=mask)
        frames_json.write_text(json.dumps(frames))
    else:
        frames = json.loads(frames_json.read_text())
    image_names = [n for _, n in frames]
    print(f"  mapping images: {len(image_names)}")

    # 2-3. SuperPoint + NetVLAD
    # Empirically, superpoint_aachen gives tighter PnP than superpoint_max:
    # larger keypoint counts let more low-quality correspondences survive
    # RANSAC, and raw frame-to-frame jumps run ~3× higher with _max.
    feature_conf = extract_features.confs["superpoint_aachen"]
    retrieval_conf = extract_features.confs["netvlad"]
    feats = extract_features.main(feature_conf, img_dir, feature_path=out / "feats.h5",
                                   image_list=image_names, overwrite=False)
    gd = extract_features.main(retrieval_conf, img_dir, feature_path=out / "netvlad.h5",
                                image_list=image_names, overwrite=False)

    # 4. Pairs + matches
    pairs_path = out / "pairs.txt"
    pairs_from_retrieval.main(gd, pairs_path, num_matched=num_retrieval_pairs)
    matcher_conf = match_features.confs["superpoint+lightglue"]
    matches = match_features.main(matcher_conf, pairs_path, features=feats,
                                   matches=out / "matches.h5", overwrite=False)

    # 5. SfM (skip if cached). When COLMAP can't merge everything into one model,
    # it produces multiple sub-models in `models/0`, `models/1`, … and *also*
    # consolidates the largest one to the top level of `sfm_dir`. We probe every
    # candidate and pick the one with the most registered images so a leftover
    # fragment from a previous run doesn't outvote the real model.
    candidates = []
    if (sfm_dir / "cameras.bin").exists():
        candidates.append(sfm_dir)
    if (sfm_dir / "models").is_dir():
        for sub in sorted((sfm_dir / "models").iterdir()):
            if sub.is_dir() and (sub / "cameras.bin").exists():
                candidates.append(sub)

    if candidates:
        ranked = []
        for c in candidates:
            try:
                rc = pycolmap.Reconstruction(str(c))
                ranked.append((rc.num_reg_images(), c, rc))
            except Exception:
                continue
        if not ranked:
            raise RuntimeError(
                f"Cached SfM dirs exist at {sfm_dir} but none could be loaded by pycolmap."
            )
        ranked.sort(reverse=True)
        n_imgs, sfm_model_path, m = ranked[0]
        print(f"  [skip-sfm] cached SfM, picked largest: "
              f"{sfm_model_path.relative_to(session_dir)} ({n_imgs} images)")
        if len(ranked) > 1:
            others = ", ".join(
                f"{p.relative_to(session_dir)}={n}" for n, p, _ in ranked[1:]
            )
            print(f"  [skip-sfm] (other candidates: {others})")
    else:
        print(f"  running COLMAP incremental SfM (CPU-bound, can take 30-90 min)...")
        model = reconstruction.main(sfm_dir, img_dir, pairs_path, feats, matches,
                                     image_list=image_names,
                                     image_options=intr["image_options"])
        if model is None:
            raise RuntimeError("COLMAP reconstruction returned None")
        sfm_model_path = sfm_dir / "models" / "0"
        if not (sfm_model_path / "cameras.bin").exists():
            sfm_model_path = sfm_dir
        m = pycolmap.Reconstruction(str(sfm_model_path))
    print(f"  SfM: {m.num_reg_images()} images, {m.num_points3D()} points")

    # 6. Scale + T_hloc_tag from existing tag detections
    tag_results = pickle.loads(tag_pkl.read_bytes())
    scale, T_hloc_tag, scale_info = _recover_scale_and_tag(m, tag_results, tag_id)
    print(f"  scale: {scale:.4f} m/colmap-unit  ({scale_info['n_ratios']} pairs, "
          f"std cm {scale_info['tag_position_std_cm_90pct']})")

    # 7. Persist
    np.save(out / "tx_hloc_tag.npy", T_hloc_tag)
    (out / "tx_hloc_tag.json").write_text(json.dumps({
        "tx_hloc_tag": T_hloc_tag.tolist(),
        "tag_id": tag_id,
        "scale_m_per_colmap_unit": scale,
        **scale_info,
    }, indent=2))
    # Persist the path under sfm_dir that downstream localize should load from
    (out / "sfm_model_path.txt").write_text(
        str(sfm_model_path.relative_to(session_dir)) + "\n"
    )

    stats = HLocBuildStats(
        n_mapping_frames=len(image_names),
        n_sfm_registered=m.num_reg_images(),
        n_sfm_points=m.num_points3D(),
        scale_m_per_colmap_unit=scale,
        n_tag_samples_used=scale_info["n_tag_samples_used"],
        wall_clock_s=time.time() - t0,
    )
    (out / "meta.yaml").write_text(yaml.safe_dump({
        "stack": "hloc",
        "feature_extractor": "superpoint_aachen",
        "matcher": "superpoint+lightglue",
        "mask_img": str(mask_img) if mask_img is not None else None,
        "retrieval": "netvlad",
        "map_frame_stride": map_frame_stride,
        "num_retrieval_pairs": num_retrieval_pairs,
        "tag_id": tag_id,
        "intrinsics_json": str(intrinsics_json),
        "stats": stats.__dict__,
    }, sort_keys=False))

    # Stamp slam_provenance into the session's main meta.yaml so localize / align
    # can find this map's HLoc artifacts via `parent_map`.
    main_meta_path = session_dir / "meta.yaml"
    if main_meta_path.exists():
        meta = yaml.safe_load(main_meta_path.read_text()) or {}
        meta.setdefault("slam_provenance", {})
        meta["slam_provenance"]["stack"] = "hloc"
        meta["slam_provenance"]["build_model_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        main_meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))

    print(f"[hloc.build_model] done in {stats.wall_clock_s:.1f}s")
    return stats


def cli(args) -> int:
    stats = build_model(
        args.session_dir,
        intrinsics_json=args.intrinsics_json,
        tag_id=args.tag_id,
        map_frame_stride=args.map_frame_stride,
        num_retrieval_pairs=args.num_retrieval_pairs,
        mask_img=args.mask_img,
    )
    print(f"\nBuild model summary for {args.session_dir.name}:")
    print(f"  mapping frames: {stats.n_mapping_frames}")
    print(f"  SfM registered: {stats.n_sfm_registered}")
    print(f"  3D points:      {stats.n_sfm_points}")
    print(f"  scale:          {stats.scale_m_per_colmap_unit:.4f} m/unit")
    print(f"  wall clock:     {stats.wall_clock_s:.1f}s")
    return 0


def build_parser(sub) -> None:
    p = sub.add_parser(
        "build-model",
        help="HLoc COLMAP build for a mapping session (replaces ORB-SLAM3 build-map).",
    )
    p.add_argument("session_dir", type=Path)
    p.add_argument("--intrinsics-json", type=Path, default=DEFAULT_INTRINSICS_JSON)
    p.add_argument("--tag-id", type=int, default=DEFAULT_TAG_ID)
    p.add_argument("--map-frame-stride", type=int, default=DEFAULT_MAP_FRAME_STRIDE)
    p.add_argument("--num-retrieval-pairs", type=int, default=DEFAULT_NUM_RETRIEVAL_PAIRS)
    p.add_argument("--mask-img", type=Path, default=None,
                   help="Optional static-mask PNG. White pixels (>127) are blacked "
                        "out before SuperPoint extraction. Mirrors UMI's static mask. "
                        "The decision to mask is per-experiment; not the default.")
    p.set_defaults(func=cli)
