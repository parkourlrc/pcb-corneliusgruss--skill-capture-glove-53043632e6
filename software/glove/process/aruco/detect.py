"""Detect ArUco tags frame-by-frame in a session's MP4.

Mirrors UMI's `scripts/detect_aruco.py`. Output:

    <session>/aruco/tag_detection.pkl

Each record is `{"frame_idx": int, "time": float, "tag_dict": {id: {rvec, tvec, corners}}}`
where `time` is the frame timestamp in seconds (video's own timeline — matches
the `timestamp` column in `slam/camera_trajectory.csv`).

Must be run after `ingest` / `extract-gpmf` (needs `raw/raw_video.mp4`) and
before `calibrate-tag`.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import av
import cv2
import numpy as np
import yaml

from glove.process.aruco._util import (
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)


CALIB_ROOT = Path(__file__).resolve().parents[3] / "configs" / "calibration"
DEFAULT_INTRINSICS = CALIB_ROOT / "gopro_intrinsics_2_7k.json"
DEFAULT_ARUCO_CFG = CALIB_ROOT / "aruco_config.yaml"


def detect_video(
    session_dir: Path,
    intrinsics_json: Path = DEFAULT_INTRINSICS,
    aruco_yaml: Path = DEFAULT_ARUCO_CFG,
    num_workers: int = 4,
    overwrite: bool = False,
) -> Path:
    session_dir = Path(session_dir).resolve()
    mp4 = session_dir / "raw" / "raw_video.mp4"
    if not mp4.exists():
        raise FileNotFoundError(f"No raw_video.mp4 in {session_dir/'raw'}")
    out_dir = session_dir / "aruco"
    out_dir.mkdir(exist_ok=True)
    out_pkl = out_dir / "tag_detection.pkl"
    if out_pkl.exists() and not overwrite:
        print(f"tag_detection.pkl already exists — skipping ({out_pkl}); pass --overwrite to redo")
        return out_pkl

    import json
    aruco_cfg = parse_aruco_config(yaml.safe_load(aruco_yaml.read_text()))
    aruco_dict = aruco_cfg["aruco_dict"]
    marker_size_map = aruco_cfg["marker_size_map"]

    raw_intr = parse_fisheye_intrinsics(json.loads(intrinsics_json.read_text()))

    cv2.setNumThreads(num_workers)
    results: list[dict] = []
    with av.open(str(mp4)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = num_workers
        in_res = np.array([stream.height, stream.width])[::-1]  # (w, h)
        fisheye_intr = convert_fisheye_intrinsics_resolution(
            opencv_intr_dict=raw_intr, target_resolution=in_res
        )
        n_frames = stream.frames or 0
        print(f"Scanning {n_frames} frames at {in_res[0]}×{in_res[1]}…")
        for i, frame in enumerate(c.decode(stream)):
            img = frame.to_ndarray(format="rgb24")
            frame_cts_sec = frame.pts * stream.time_base
            tag_dict = detect_localize_aruco_tags(
                img=img,
                aruco_dict=aruco_dict,
                marker_size_map=marker_size_map,
                fisheye_intr_dict=fisheye_intr,
                refine_subpix=True,
            )
            results.append(
                {"frame_idx": i, "time": float(frame_cts_sec), "tag_dict": tag_dict}
            )
            if (i + 1) % 500 == 0:
                n_hits = sum(1 for r in results if r["tag_dict"])
                print(f"  frame {i+1}/{n_frames or '?'} — tags detected on {n_hits} so far")

    out_pkl.write_bytes(pickle.dumps(results))
    n_hits = sum(1 for r in results if r["tag_dict"])
    by_tag: dict[int, int] = {}
    for r in results:
        for tid in r["tag_dict"].keys():
            by_tag[tid] = by_tag.get(tid, 0) + 1
    print(f"Wrote {out_pkl}")
    print(f"  frames scanned: {len(results):,}   frames with any tag: {n_hits:,}")
    if by_tag:
        per_tag = "  ".join(f"id={k}:{v}" for k, v in sorted(by_tag.items()))
        print(f"  per-tag detections: {per_tag}")
    else:
        print("  no tags detected — check that the ArUco is visible and matches the dict")
    return out_pkl


def build_parser(sub) -> None:
    p = sub.add_parser(
        "detect-aruco",
        help="Detect ArUco tags in a session's MP4 → aruco/tag_detection.pkl.",
    )
    p.add_argument("session_dir", type=Path)
    p.add_argument("--intrinsics-json", type=Path, default=DEFAULT_INTRINSICS)
    p.add_argument("--aruco-yaml", type=Path, default=DEFAULT_ARUCO_CFG)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cli)


def cli(args) -> int:
    detect_video(
        args.session_dir,
        intrinsics_json=args.intrinsics_json,
        aruco_yaml=args.aruco_yaml,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
    )
    return 0
