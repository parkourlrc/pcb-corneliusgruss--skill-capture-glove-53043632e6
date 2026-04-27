"""Sweep trigger-mode ISP settings for OV9782 cameras.

This is a focused follow-up to `isp_test.py`. It targets the controls that are
most likely to hurt ORB-SLAM3 on the OV9782 cameras:

- backlight-compensation
- exposure-time-abs
- gamma
- sharpness

Each preset captures a short triggered burst and reports:
- cam0 strong ORB count (FAST15)
- cam0 local contrast / sharpness
- stereo raw matches / inliers

Typical usage:

    python -m glove.scripts.isp_sweep \
        --orb-config <path-to-orb-config>/configs/orb_slam3_stereo_inertial.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from glove.scripts.isp_test import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    apply_controls,
    capture_phase,
    find_cameras,
    load_mask,
    load_orb_config,
)

DEFAULT_OUT_DIR = "/tmp/isp_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep trigger-mode ISP settings for VIO.")
    parser.add_argument(
        "--orb-config",
        type=Path,
        required=True,
        help="ORB-SLAM3 YAML to load ORB thresholds from",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--settle-frames", type=int, default=30)
    parser.add_argument("--sample-frames", type=int, default=12)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--mask-left", type=Path, default=None)
    parser.add_argument("--mask-right", type=Path, default=None)
    return parser.parse_args()


def build_targeted_presets() -> list[dict]:
    base_manual = {
        "auto-white-balance-temp": "0",
        "white-balance-temp": "4600",
        "auto-exposure-mode": "1",
        "auto-exposure-priority": "1",
        "gain": "0",
        "contrast": "32",
        "saturation": "90",
        "brightness": "0",
        "hue": "0",
    }

    presets = [
        {
            "name": "trigger_auto_baseline",
            "controls": {
                "auto-white-balance-temp": "1",
                "auto-exposure-mode": "8",
                "auto-exposure-priority": "1",
                "backlight-compensation": "1",
            },
        },
        {
            "name": "manual_bl1_e157_g100_s3",
            "controls": {
                **base_manual,
                "backlight-compensation": "1",
                "exposure-time-abs": "157",
                "gamma": "100",
                "sharpness": "3",
            },
        },
        {
            "name": "manual_bl0_e157_g100_s3",
            "controls": {
                **base_manual,
                "backlight-compensation": "0",
                "exposure-time-abs": "157",
                "gamma": "100",
                "sharpness": "3",
            },
        },
        {
            "name": "manual_bl0_e120_g100_s3",
            "controls": {
                **base_manual,
                "backlight-compensation": "0",
                "exposure-time-abs": "120",
                "gamma": "100",
                "sharpness": "3",
            },
        },
        {
            "name": "manual_bl0_e100_g100_s3",
            "controls": {
                **base_manual,
                "backlight-compensation": "0",
                "exposure-time-abs": "100",
                "gamma": "100",
                "sharpness": "3",
            },
        },
        {
            "name": "manual_bl0_e080_g100_s3",
            "controls": {
                **base_manual,
                "backlight-compensation": "0",
                "exposure-time-abs": "80",
                "gamma": "100",
                "sharpness": "3",
            },
        },
        {
            "name": "manual_bl0_e120_g072_s3",
            "controls": {
                **base_manual,
                "backlight-compensation": "0",
                "exposure-time-abs": "120",
                "gamma": "72",
                "sharpness": "3",
            },
        },
        {
            "name": "manual_bl0_e100_g072_s3",
            "controls": {
                **base_manual,
                "backlight-compensation": "0",
                "exposure-time-abs": "100",
                "gamma": "72",
                "sharpness": "3",
            },
        },
        {
            "name": "manual_bl0_e120_g072_s6",
            "controls": {
                **base_manual,
                "backlight-compensation": "0",
                "exposure-time-abs": "120",
                "gamma": "72",
                "sharpness": "6",
            },
        },
    ]
    return presets


def sort_key(result: dict) -> tuple[float, float, float, float]:
    cam0 = result["phase"].frame.get(0, {})
    stereo = result["phase"].stereo or {}
    return (
        float(cam0.get("orb_primary_kp", 0.0)),
        float(cam0.get("laplacian_var", 0.0)),
        float(cam0.get("gray_std", 0.0)),
        float(stereo.get("stereo_inliers", 0.0)),
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orb_cfg = load_orb_config(args.orb_config)
    cameras = find_cameras()
    if not cameras:
        print("ERROR: No OV9782 cameras found")
        return

    masks: dict[int, object | None] = {}
    if len(cameras) >= 1:
        masks[cameras[0]] = load_mask(args.mask_left, (args.height, args.width)) if args.mask_left else None
    if len(cameras) >= 2:
        masks[cameras[1]] = load_mask(args.mask_right, (args.height, args.width)) if args.mask_right else None

    print(f"Found cameras: {cameras}")
    print(
        f"ORB config: nfeatures={orb_cfg.nfeatures} "
        f"ini_fast={orb_cfg.ini_fast} min_fast={orb_cfg.min_fast}"
    )

    results = []
    presets = build_targeted_presets()
    try:
        for idx, preset in enumerate(presets, start=1):
            print(f"\n[{idx}/{len(presets)}] {preset['name']}")
            controls = list(preset["controls"].items())
            apply_controls(cameras, controls)
            time.sleep(1.0)
            phase = capture_phase(
                preset["name"],
                cameras,
                args.settle_frames,
                args.sample_frames,
                args.width,
                args.height,
                args.fps,
                orb_cfg,
                masks,
                out_dir,
            )
            cam0 = phase.frame.get(cameras[0], {})
            cam1 = phase.frame.get(cameras[1], {}) if len(cameras) > 1 else {}
            stereo = phase.stereo or {}
            print(
                "  cam0:"
                f" orb15={cam0.get('orb_primary_kp', 0):.0f}"
                f" gray_std={cam0.get('gray_std', 0):.1f}"
                f" lap={cam0.get('laplacian_var', 0):.1f}"
                f" edges={cam0.get('edge_density_pct', 0):.2f}%"
                f" dup={cam0.get('duplicate_frame_pct', 0):.1f}%"
            )
            if cam1:
                print(
                    "  cam1:"
                    f" orb15={cam1.get('orb_primary_kp', 0):.0f}"
                    f" gray_std={cam1.get('gray_std', 0):.1f}"
                    f" lap={cam1.get('laplacian_var', 0):.1f}"
                    f" dup={cam1.get('duplicate_frame_pct', 0):.1f}%"
                )
            if stereo:
                print(
                    "  stereo:"
                    f" raw={stereo.get('stereo_raw_matches', 0):.1f}"
                    f" inliers={stereo.get('stereo_inliers', 0):.1f}"
                    f" ratio={100.0 * stereo.get('stereo_inlier_ratio', 0):.1f}%"
                )
            results.append(
                {
                    "name": preset["name"],
                    "controls": preset["controls"],
                    "phase": phase,
                }
            )
    finally:
        apply_controls(
            cameras,
            [
                ("auto-exposure-mode", "8"),
                ("auto-exposure-priority", "0"),
                ("auto-white-balance-temp", "1"),
                ("backlight-compensation", "1"),
            ],
        )
        print("\nCameras reset to free-running defaults.")

    ranked = sorted(results, key=sort_key, reverse=True)
    print("\n=== Ranked by cam0 strong-feature health ===")
    for rank, item in enumerate(ranked, start=1):
        phase = item["phase"]
        cam0 = phase.frame.get(cameras[0], {})
        stereo = phase.stereo or {}
        print(
            f"{rank:>2}. {item['name']:<24}"
            f" orb15={cam0.get('orb_primary_kp', 0):>6.0f}"
            f" lap={cam0.get('laplacian_var', 0):>7.1f}"
            f" gray={cam0.get('gray_std', 0):>5.1f}"
            f" stereo_inliers={stereo.get('stereo_inliers', 0):>5.1f}"
        )

    json_ready = {
        "camera_indices": cameras,
        "orb": {
            "nfeatures": orb_cfg.nfeatures,
            "ini_fast": orb_cfg.ini_fast,
            "min_fast": orb_cfg.min_fast,
        },
        "results": [
            {
                "name": item["name"],
                "controls": item["controls"],
                "frame": item["phase"].frame,
                "stereo": item["phase"].stereo,
                "isp": item["phase"].isp,
            }
            for item in ranked
        ],
    }
    (out_dir / "ranked_results.json").write_text(json.dumps(json_ready, indent=2) + "\n")
    print(f"\nSaved ranked results to {out_dir / 'ranked_results.json'}")


if __name__ == "__main__":
    main()
