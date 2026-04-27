"""CLI entry point for glove.process.

ORB-SLAM3 stack:
    ingest         Create a demo session from an MP4 + master-tape raw stream.
    extract-gpmf   Extract GoPro GPMF IMU into raw/imu_data.json.
    build-map      Run gopro_slam on a mapping session → slam/map_atlas.
    detect-aruco   Detect ArUco tags in a session MP4 → aruco/tag_detection.pkl.
    calibrate-tag  Fit tx_slam_tag from a mapping session's tags + trajectory.
    localize       Localize a demo session against a parent mapping session's map.

HLoc stack (SuperPoint + LightGlue + COLMAP — replaces ORB-SLAM3 build-map / localize):
    build-model    Build a COLMAP scene model from a mapping clip → hloc/sfm/.
    localize-hloc  PnP + pose-graph smoothing per demo → slam/camera_trajectory.csv.

Common downstream:
    align          Fuse proprio + per-frame poses into aligned/episode.parquet.
                   Reads tx_slam_tag.json (ORB) or tx_hloc_tag.json (HLoc) per
                   the demo's slam_provenance.stack.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from glove.process import align as align_mod
from glove.process import ingest as ingest_mod
from glove.process import preprocess as preprocess_mod
from glove.process.aruco import calibrate_tag as calibrate_tag_mod
from glove.process.aruco import detect as detect_aruco_mod
from glove.process.hloc import build_model as hloc_build_mod
from glove.process.hloc import localize as hloc_localize_mod
from glove.process.slam import build_map as build_map_mod
from glove.process.slam import extract_gpmf as extract_gpmf_mod
from glove.process.slam import localize as localize_mod


def main() -> int:
    parser = argparse.ArgumentParser(prog="glove.process")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ingest
    ing = sub.add_parser("ingest",
                         help="Create a demo session from one MP4 + master-tape raw stream.")
    ing.add_argument("--mp4", type=Path, required=True)
    ing.add_argument("--stream-dir", type=Path, default=None,
                     help="Master-tape stream dir (required for demo sessions, "
                          "optional for --kind mapping since mapping clips don't use proprio).")
    ing.add_argument("--sessions-root", type=Path, default=Path("data/sessions"))
    ing.add_argument("--task", type=str, default="demo")
    ing.add_argument("--ordinal", type=int, default=None)
    ing.add_argument("--kind", type=str, default="demo", choices=["demo", "mapping"])
    ing.add_argument("--mp4-tz-offset-hours", type=float, default=0.0,
                     help="Hours to add to MP4 timestamps to reach laptop UTC epoch. "
                          "Default 0.0 assumes the GoPro was Labs-synced to UTC. For cameras "
                          "writing local-time-as-UTC (non-Labs default), pass your local tz "
                          "offset (e.g. EDT → 4.0).")
    ing.add_argument("--mp4-wall-offset-seconds", type=float, default=0.0,
                     help="Manual fine offset for residual clock drift. Default 0.0.")
    ing.set_defaults(func=ingest_mod.cli)

    # extract-gpmf
    egp = sub.add_parser("extract-gpmf", help="Extract GoPro GPMF IMU for a session.")
    egp.add_argument("session_dir", type=Path)
    egp.add_argument("--overwrite", action="store_true")
    egp.set_defaults(func=extract_gpmf_mod.cli)

    # build-map
    bm = sub.add_parser("build-map", help="Build a map atlas from a mapping session.")
    bm.add_argument("session_dir", type=Path)
    bm.add_argument("--aruco-dict-id", type=int, default=0,
                    help="OpenCV ArUco dict id (default 0 = DICT_4X4_50).")
    bm.add_argument("--init-tag-id", type=int, default=13)
    bm.add_argument("--init-tag-size-m", type=float, default=0.16)
    bm.add_argument("--mask", action="store_true",
                    help="Pass a static-mask PNG to gopro_slam via --mask_img "
                         "(matches UMI's approach — no video re-encode). "
                         "Defaults to software/configs/calibration/hero10_static_mask.png "
                         "unless --mask-img overrides.")
    bm.add_argument("--mask-img", type=Path, default=None,
                    help="Explicit path to a static-mask PNG. Implies --mask.")
    bm.set_defaults(func=build_map_mod.cli)

    # localize
    loc = sub.add_parser("localize",
                         help="Localize a demo session against a parent mapping session's atlas.")
    loc.add_argument("demo_session_dir", type=Path)
    loc.add_argument("--map", dest="map_session_dir", type=Path, required=True,
                     help="Mapping session directory (contains slam/map_atlas).")
    loc.add_argument("--max-lost-frames", type=int, default=60,
                     help="Lost-frame budget (default 60 = ~1s at 60fps). Default is conservative "
                          "ON PURPOSE — high values let SLAM latch onto spurious frames during "
                          "extended init, producing silently-wrong trajectories. Only bump if "
                          "you know the clip is clean and just has a slow start.")
    loc.add_argument("--mask", action="store_true",
                     help="Pass a static-mask PNG to gopro_slam via --mask_img "
                          "(matches UMI's approach — no video re-encode). "
                          "Defaults to software/configs/calibration/hero10_static_mask.png "
                          "unless --mask-img overrides.")
    loc.add_argument("--mask-img", type=Path, default=None,
                     help="Explicit path to a static-mask PNG. Implies --mask.")
    loc.set_defaults(func=localize_mod.cli)

    # align
    al = sub.add_parser("align", help="Fuse proprio + trajectory → aligned/episode.parquet.")
    al.add_argument("session_dir", type=Path)
    al.add_argument("--no-imu-filter", action="store_true",
                    help="Disable DAQ-IMU consistency filter. By default, align rejects "
                         "estimator-flip frames (pose jumps the DAQ IMU does not corroborate) "
                         "and fills them by linear interp + slerp.")
    al.set_defaults(func=align_mod.cli)

    # detect-aruco
    detect_aruco_mod.build_parser(sub)

    # calibrate-tag
    calibrate_tag_mod.build_parser(sub)

    # build-model (HLoc COLMAP build — replaces build-map on the HLoc stack)
    hloc_build_mod.build_parser(sub)

    # localize-hloc (HLoc PnP + smoothing — replaces localize on the HLoc stack)
    hloc_localize_mod.build_parser(sub)

    # preprocess
    pp = sub.add_parser("preprocess",
                        help="Downscale session MP4s to 224×224 for training (raw_video_224.mp4). "
                             "Drastically reduces upload size to a remote GPU cluster.")
    pp.add_argument("session_dirs", type=Path, nargs="+",
                    help="One or more session dirs, OR a parent dir of session dirs.")
    pp.add_argument("--overwrite", action="store_true")
    pp.set_defaults(func=preprocess_mod.cli)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
