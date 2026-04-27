"""Build a SLAM map from a mapping session.

Runs UMI's `gopro_slam` via docker in `--save_map` mode. Reads:
    <session>/raw/raw_video.mp4
    <session>/raw/imu_data.json  (from extract_gpmf)
    <session>/slam/slam_settings.yaml  (frozen by freeze_slam_settings)

Writes:
    <session>/slam/map_atlas
    <session>/slam/camera_trajectory.csv
    <session>/logs/slam.log
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from glove.process.slam import DEFAULT_STATIC_MASK, DOCKER_SLAM, freeze_slam_settings


@dataclass
class SlamStats:
    n_frames: int
    n_tracked: int
    n_keyframes: int
    # Pose-quality metrics computed from camera_trajectory.csv. These exist because
    # `tracking_fraction` alone lied badly — a "rail pattern" (relocalize → brief
    # track → lose → relocalize at a different atlas pose) keeps `is_lost=false`
    # while the trajectory teleports meters between frames. Always look at
    # `pose_jump_max_mm` and `verdict` before calling a run "good".
    pose_jump_max_mm: float
    pose_jump_p99_mm: float
    n_jumps_over_20mm: int
    n_jumps_over_50mm: int
    path_length_m: float

    @property
    def tracking_fraction(self) -> float:
        return self.n_tracked / self.n_frames if self.n_frames else 0.0

    @property
    def verdict(self) -> str:
        # Conservative thresholds matching `glove.quality.schema.THRESHOLDS`:
        # warn = 20mm per-frame jump, fail = 50mm. Rails produce many >50mm jumps.
        if self.n_jumps_over_50mm > 0 or self.pose_jump_max_mm > 100.0:
            return "RAILS"
        if self.n_jumps_over_20mm > 5:
            return "NOISY"
        return "CLEAN"


def _stats(traj_csv: Path) -> SlamStats:
    with traj_csv.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return SlamStats(0, 0, 0, 0.0, 0.0, 0, 0, 0.0)
    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    d_mm = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000.0
    return SlamStats(
        n_frames=len(rows),
        n_tracked=sum(r["is_lost"].lower() == "false" for r in rows),
        n_keyframes=sum(r["is_keyframe"].lower() == "true" for r in rows),
        pose_jump_max_mm=float(d_mm.max()) if d_mm.size else 0.0,
        pose_jump_p99_mm=float(np.percentile(d_mm, 99)) if d_mm.size else 0.0,
        n_jumps_over_20mm=int((d_mm > 20.0).sum()),
        n_jumps_over_50mm=int((d_mm > 50.0).sum()),
        path_length_m=float(d_mm.sum() / 1000.0),
    )


def build_map(
    session_dir: Path,
    aruco_dict_id: int = 0,       # DICT_4X4_50
    init_tag_id: int = 13,
    init_tag_size_m: float = 0.16,
    mask_img: Path | None = None,
) -> SlamStats:
    session_dir = Path(session_dir).resolve()
    raw = session_dir / "raw"
    slam = session_dir / "slam"
    logs = session_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    video_name = "raw_video.mp4"
    for needed in [video_name, "imu_data.json"]:
        if not (raw / needed).exists():
            raise FileNotFoundError(f"Missing {needed} in {raw} (did you run extract_gpmf?)")

    freeze_slam_settings(session_dir)

    # Optional static mask passed through to gopro_slam's --mask_img,
    # matching UMI's approach. The PNG lives outside the session tree,
    # so mount its parent dir read-only.
    extra_volumes: list[str] = []
    extra_args: list[str] = []
    if mask_img is not None:
        mask_img = Path(mask_img).resolve()
        if not mask_img.exists():
            raise FileNotFoundError(f"mask_img not found: {mask_img}")
        extra_volumes += ["--volume", f"{mask_img.parent}:/mask:ro"]
        extra_args += ["--mask_img", f"/mask/{mask_img.name}"]

    log = logs / "slam.log"
    with log.open("w") as logf:
        result = subprocess.run([
            "docker", "run", "--rm", "--platform", "linux/amd64",
            "--volume", f"{raw}:/raw",
            "--volume", f"{slam}:/slam",
            *extra_volumes,
            DOCKER_SLAM,
            "/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam",
            "--vocabulary", "/ORB_SLAM3/Vocabulary/ORBvoc.txt",
            "--setting", "/slam/slam_settings.yaml",
            "--input_video", f"/raw/{video_name}",
            "--input_imu_json", "/raw/imu_data.json",
            "--output_trajectory_csv", "/slam/camera_trajectory.csv",
            "--save_map", "/slam/map_atlas",
            "--aruco_dict_id", str(aruco_dict_id),
            "--init_tag_id", str(init_tag_id),
            "--init_tag_size", f"{init_tag_size_m}",
            *extra_args,
        ], stdout=logf, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"gopro_slam exited {result.returncode}; see {log}")

    traj = slam / "camera_trajectory.csv"
    atlas = slam / "map_atlas"
    if not traj.exists() or not atlas.exists():
        raise RuntimeError(f"SLAM did not produce expected outputs in {slam}")

    # Auto-score: writes a `map_quality` block into meta.yaml for the /maps panel.
    from glove.quality.map_compute import score_map
    score_map(session_dir, persist=True)

    return _stats(traj)


def cli(args) -> int:
    mask_img = None
    if getattr(args, "mask_img", None):
        mask_img = args.mask_img
    elif getattr(args, "mask", False):
        mask_img = DEFAULT_STATIC_MASK
    stats = build_map(args.session_dir,
                      aruco_dict_id=args.aruco_dict_id,
                      init_tag_id=args.init_tag_id,
                      init_tag_size_m=args.init_tag_size_m,
                      mask_img=mask_img)
    print_slam_stats(f"Map built for {args.session_dir}", stats)
    return 0


def print_slam_stats(header: str, stats: SlamStats) -> None:
    """Print SLAM result with pose-quality verdict.

    We deliberately pair `tracking %` with pose-jump stats and a verdict because
    100% tracked can still mean a railed trajectory that teleports meters between
    frames. Never read the tracking % without also reading the verdict.
    """
    flag = {"CLEAN": "", "NOISY": "  ⚠ NOISY", "RAILS": "  ✗ RAILS"}[stats.verdict]
    print(header)
    print(f"  frames: {stats.n_frames:,}  tracked: {stats.n_tracked:,} "
          f"({100*stats.tracking_fraction:.1f}%)  keyframes: {stats.n_keyframes:,}")
    print(f"  pose: max_jump={stats.pose_jump_max_mm:.1f}mm  "
          f"p99={stats.pose_jump_p99_mm:.1f}mm  "
          f">20mm={stats.n_jumps_over_20mm}  >50mm={stats.n_jumps_over_50mm}  "
          f"path={stats.path_length_m:.2f}m  → {stats.verdict}{flag}")
