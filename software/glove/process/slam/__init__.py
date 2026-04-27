"""SLAM wrappers around UMI's dockerized ORB-SLAM3 + openicc toolchain.

Submodules:
- extract_gpmf   — GoPro MP4 → raw/imu_data.json (IMU used by gopro_slam)
- build_map      — mapping session → slam/map_atlas (+ trajectory)
- localize       — demo session + parent map → slam/camera_trajectory.csv

All three expect and produce files inside the new session schema:

    session/
    ├── raw/raw_video.mp4 + imu_data.json
    ├── slam/map_atlas, camera_trajectory.csv, slam_settings.yaml
    └── logs/slam.log
"""

from __future__ import annotations

import subprocess
from pathlib import Path


DOCKER_SLAM = "chicheng/orb_slam3:latest"
DOCKER_OPENICC = "chicheng/openicc:latest"
SLAM_SETTINGS_IN_IMAGE = "/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml"

# Default static-mask PNG. gopro_slam accepts `--mask_img`; UMI uses this
# to block the fisheye mirror + static gripper. For the glove we use it
# to block the palm + thumb + mount. Same approach, different silhouette.
DEFAULT_STATIC_MASK = (Path(__file__).resolve().parents[3]
                       / "configs" / "calibration" / "hero10_static_mask.png")


def freeze_slam_settings(session_dir: Path) -> Path:
    """Copy the Hero 10 settings YAML out of the SLAM docker image into
    `session/slam/slam_settings.yaml`. Captured once per session for reproducibility:
    the file inside a future docker tag might differ.
    """
    dst = session_dir / "slam" / "slam_settings.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    with dst.open("wb") as f:
        subprocess.run(
            ["docker", "run", "--rm", "--platform", "linux/amd64", DOCKER_SLAM,
             "cat", SLAM_SETTINGS_IN_IMAGE],
            stdout=f, check=True,
        )
    return dst
