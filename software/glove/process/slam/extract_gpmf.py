"""Extract GoPro GPMF IMU from an MP4 into the format gopro_slam expects.

Runs `chicheng/openicc` via docker (it's a node.js tool from OpenImuCameraCalibrator).
Writes `<session>/raw/imu_data.json` ready for `build_map` / `localize`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from glove.process.slam import DOCKER_OPENICC


def extract_gpmf(session_dir: Path, overwrite: bool = False) -> Path:
    """Extract GPMF IMU for the session's raw_video.mp4. Returns the imu_data.json path."""
    session_dir = Path(session_dir).resolve()
    raw = session_dir / "raw"
    video = raw / "raw_video.mp4"
    out = raw / "imu_data.json"
    if not video.exists():
        raise FileNotFoundError(f"No raw_video.mp4 in {raw}")
    if out.exists() and not overwrite:
        return out

    subprocess.run(
        ["docker", "run", "--rm",
         "--platform", "linux/amd64",
         "--volume", f"{raw}:/data",
         "-w", "/OpenImuCameraCalibrator/javascript",
         DOCKER_OPENICC,
         "node", "extract_metadata_single.js",
         "/data/raw_video.mp4", "/data/imu_data.json"],
        check=True,
    )
    return out


def cli(args) -> int:
    out = extract_gpmf(args.session_dir, overwrite=args.overwrite)
    size_kb = out.stat().st_size / 1024
    print(f"Extracted {out}  ({size_kb:,.1f} KB)")
    return 0
