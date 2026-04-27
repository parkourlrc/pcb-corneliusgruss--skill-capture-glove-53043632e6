"""Localize a demo session against an existing map atlas.

Reads:
    <demo>/raw/raw_video.mp4
    <demo>/raw/imu_data.json
    <demo>/slam/slam_settings.yaml  (frozen on first call)
    <map>/slam/map_atlas            (from build_map on the parent mapping session)

Writes:
    <demo>/slam/camera_trajectory.csv
    <demo>/logs/slam.log

Demo trajectories produced this way share the world frame of the map atlas, so
poses from multiple demos are directly comparable and align to the ArUco origin.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from glove.process.slam import DEFAULT_STATIC_MASK, DOCKER_SLAM, freeze_slam_settings
from glove.process.slam.build_map import SlamStats, _stats, print_slam_stats


def localize(demo_session_dir: Path, map_session_dir: Path,
             max_lost_frames: int = 60,
             mask_img: Path | None = None) -> SlamStats:
    demo = Path(demo_session_dir).resolve()
    map_s = Path(map_session_dir).resolve()

    raw = demo / "raw"
    slam = demo / "slam"
    logs = demo / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    video_name = "raw_video.mp4"
    for needed in [video_name, "imu_data.json"]:
        if not (raw / needed).exists():
            raise FileNotFoundError(f"Missing {needed} in {raw} (did you run extract_gpmf?)")
    atlas = map_s / "slam" / "map_atlas"
    if not atlas.exists():
        raise FileNotFoundError(f"No map_atlas in {map_s / 'slam'} (did you run build_map?)")

    freeze_slam_settings(demo)

    extra_volumes: list[str] = []
    extra_args: list[str] = []
    if mask_img is not None:
        mask_img = Path(mask_img).resolve()
        if not mask_img.exists():
            raise FileNotFoundError(f"mask_img not found: {mask_img}")
        extra_volumes += ["--volume", f"{mask_img.parent}:/mask:ro"]
        extra_args += ["--mask_img", f"/mask/{mask_img.name}"]

    log = logs / "slam.log"
    cmd = [
        "docker", "run", "--rm", "--platform", "linux/amd64",
        "--volume", f"{raw}:/raw",
        "--volume", f"{slam}:/slam",
        "--volume", f"{map_s / 'slam'}:/map",
        *extra_volumes,
        DOCKER_SLAM,
        "/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam",
        "--vocabulary", "/ORB_SLAM3/Vocabulary/ORBvoc.txt",
        "--setting", "/slam/slam_settings.yaml",
        "--input_video", f"/raw/{video_name}",
        "--input_imu_json", "/raw/imu_data.json",
        "--output_trajectory_csv", "/slam/camera_trajectory.csv",
        "--load_map", "/map/map_atlas",
        *extra_args,
    ]
    if max_lost_frames > 0:
        cmd += ["--max_lost_frames", str(max_lost_frames)]

    with log.open("w") as logf:
        # Timeout guards against the relocalize→fail→relocalize loop: each successful
        # Relocalized!! resets ORB-SLAM3's internal lost counter, so --max_lost_frames
        # never accumulates to 60 consecutive lost frames. Without a wall-clock limit the
        # process runs all frames while IMU velocity diverges (→100+ m/s), making each
        # BA step progressively slower and turning a 10s video into an 11-minute hang.
        try:
            result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=300)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"gopro_slam timed out after 300s; see {log}")
    if result.returncode != 0:
        raise RuntimeError(f"gopro_slam exited {result.returncode}; see {log}")

    traj = slam / "camera_trajectory.csv"
    if not traj.exists():
        raise RuntimeError(f"SLAM did not produce {traj}")

    # Record the parent map so downstream tools (dashboard /maps usage panel,
    # dataset gates) can follow demo → map links without path inspection.
    meta_path = demo / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text()) or {}
        meta["parent_map"] = map_s.name
        meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))

    return _stats(traj)


def cli(args) -> int:
    mask_img = None
    if getattr(args, "mask_img", None):
        mask_img = args.mask_img
    elif getattr(args, "mask", False):
        mask_img = DEFAULT_STATIC_MASK
    stats = localize(args.demo_session_dir, args.map_session_dir,
                     max_lost_frames=args.max_lost_frames,
                     mask_img=mask_img)
    print_slam_stats(
        f"Localized {args.demo_session_dir} against {args.map_session_dir}", stats)
    return 0
