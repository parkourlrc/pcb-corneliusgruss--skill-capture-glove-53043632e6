"""Camera ISP/VIO diagnostic for OV9782 stereo cameras.

This script compares three capture modes:

1. Free-running auto ISP
2. Triggered with auto exposure still enabled
3. Triggered with per-camera ISP values locked from the free-running settle

The old version only reported generic image-quality stats from a single frame.
This version captures short bursts, then scores them using metrics that are
closer to what ORB-SLAM3 actually cares about:

- contrast / clipping
- blur proxies (Laplacian variance, edge density)
- ORB feature counts with the project's FAST thresholds
- stereo descriptor matches + RANSAC inliers
- duplicate-frame detection in triggered mode

Typical usage:

    python -m glove.scripts.isp_test
    python -m glove.scripts.isp_test \
        --orb-config <path-to-orb-config>/configs/orb_slam3_stereo_inertial.yaml
    python -m glove.scripts.isp_test \
        --mask-left /path/to/cam_top_mask.png

Outputs:
    /tmp/isp_test/analysis.txt
    /tmp/isp_test/report.json
    /tmp/isp_test/*_last.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

UVC_UTIL = os.path.expanduser("~/.local/bin/uvc-util")
DEFAULT_OUT_DIR = "/tmp/isp_test"
DEFAULT_SETTLE_FRAMES = 90
DEFAULT_SAMPLE_FRAMES = 24
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 800
DEFAULT_FPS = 100
BACKEND = cv2.CAP_AVFOUNDATION

UVC_ISP_KEYS = [
    "white-balance-temp",
    "exposure-time-abs",
    "gain",
    "gamma",
    "sharpness",
    "contrast",
    "saturation",
    "brightness",
    "hue",
    "auto-white-balance-temp",
    "auto-exposure-mode",
    "auto-exposure-priority",
]
LOCKABLE_ISP_KEYS = [
    "white-balance-temp",
    "exposure-time-abs",
    "gain",
    "gamma",
    "sharpness",
    "contrast",
    "saturation",
    "brightness",
    "hue",
]
MANUAL_EXPOSURE_MODE = "1"


@dataclass
class ORBConfig:
    nfeatures: int = 1200
    ini_fast: int = 15
    min_fast: int = 5


@dataclass
class PhaseResult:
    name: str
    frame: dict[int, dict]
    stereo: dict | None
    isp: dict[int, dict]


class Reporter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def write(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OV9782 ISP/trigger modes for VIO.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--settle-frames", type=int, default=DEFAULT_SETTLE_FRAMES)
    parser.add_argument("--sample-frames", type=int, default=DEFAULT_SAMPLE_FRAMES)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--orb-config",
        type=Path,
        default=None,
        help="Optional ORB-SLAM3 YAML to load ORBextractor thresholds from",
    )
    parser.add_argument("--mask-left", type=Path, default=None)
    parser.add_argument("--mask-right", type=Path, default=None)
    parser.add_argument(
        "--skip-trigger-locked",
        action="store_true",
        help="Skip the trigger + manual-lock phase",
    )
    return parser.parse_args()


def load_orb_config(path: Path | None) -> ORBConfig:
    cfg = ORBConfig()
    if path is None:
        return cfg
    text = path.read_text()

    def extract_scalar(key: str, default: int) -> int:
        pattern = rf"(?m)^{re.escape(key)}\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$"
        match = re.search(pattern, text)
        if not match:
            return default
        value = match.group(1)
        return int(float(value))

    return ORBConfig(
        nfeatures=extract_scalar("ORBextractor.nFeatures", cfg.nfeatures),
        ini_fast=extract_scalar("ORBextractor.iniThFAST", cfg.ini_fast),
        min_fast=extract_scalar("ORBextractor.minThFAST", cfg.min_fast),
    )


def uvc_set(dev: int, controls: list[tuple[str, str]]) -> None:
    import subprocess

    cmd = [UVC_UTIL, "-I", str(dev)]
    for name, value in controls:
        cmd.extend(["-s", f"{name}={value}"])
    subprocess.run(cmd, capture_output=True, timeout=5)


def uvc_get(dev: int, name: str) -> str:
    import subprocess

    result = subprocess.run(
        [UVC_UTIL, "-I", str(dev), "-S", name],
        capture_output=True,
        text=True,
        timeout=5,
    )
    for line in result.stdout.splitlines():
        if "current-value:" in line:
            return line.split("current-value:")[-1].strip()
    return "?"


def read_all_isp(dev: int) -> dict[str, str]:
    return {key: uvc_get(dev, key) for key in UVC_ISP_KEYS}


def apply_controls(camera_indices: list[int], controls: list[tuple[str, str]]) -> None:
    for cam_idx in camera_indices:
        uvc_set(cam_idx, controls)


def find_cameras(max_cameras: int = 2) -> list[int]:
    cams: list[int] = []
    for idx in range(8):
        cap = cv2.VideoCapture(idx, BACKEND)
        if cap.isOpened():
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if height == DEFAULT_HEIGHT:
                cams.append(idx)
                if len(cams) == max_cameras:
                    break
        else:
            cap.release()
    return cams


def load_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def capture_burst(
    cam_idx: int,
    settle_frames: int,
    sample_frames: int,
    width: int,
    height: int,
    fps: int,
) -> list[np.ndarray]:
    cap = cv2.VideoCapture(cam_idx, BACKEND)
    if not cap.isOpened():
        return []
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    frames: list[np.ndarray] = []
    total_reads = settle_frames + sample_frames
    for read_idx in range(total_reads):
        ret, frame = cap.read()
        if not ret:
            continue
        if read_idx >= settle_frames:
            frames.append(frame.copy())
    cap.release()
    return frames


def prepare_gray(frame: np.ndarray, exclude_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if exclude_mask is None:
        keep_mask = np.ones(gray.shape, dtype=bool)
        return gray, keep_mask
    keep_mask = ~exclude_mask
    gray = gray.copy()
    gray[exclude_mask] = 0
    return gray, keep_mask


def _masked_values(arr: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    return arr[keep_mask] if keep_mask is not None else arr.reshape(-1)


def analyze_orb(gray: np.ndarray, orb_cfg: ORBConfig) -> dict[str, float]:
    def detect_count(fast_threshold: int) -> tuple[int, int]:
        orb = cv2.ORB_create(nfeatures=orb_cfg.nfeatures, fastThreshold=fast_threshold)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        return len(keypoints), 0 if descriptors is None else len(descriptors)

    primary_kp, primary_desc = detect_count(orb_cfg.ini_fast)
    rescue_kp, rescue_desc = detect_count(orb_cfg.min_fast)
    return {
        "orb_primary_kp": float(primary_kp),
        "orb_primary_desc": float(primary_desc),
        "orb_rescue_kp": float(rescue_kp),
        "orb_rescue_desc": float(rescue_desc),
    }


def analyze_frame(frame: np.ndarray, label: str, orb_cfg: ORBConfig, exclude_mask: np.ndarray | None = None) -> dict:
    gray, keep_mask = prepare_gray(frame, exclude_mask)
    b, g, r = cv2.split(frame)

    gray_vals = _masked_values(gray, keep_mask).astype(np.float32)
    r_vals = _masked_values(r, keep_mask).astype(np.float32)
    g_vals = _masked_values(g, keep_mask).astype(np.float32)
    b_vals = _masked_values(b, keep_mask).astype(np.float32)

    stats = {
        "label": label,
        "R_mean": float(r_vals.mean()),
        "G_mean": float(g_vals.mean()),
        "B_mean": float(b_vals.mean()),
        "R_std": float(r_vals.std()),
        "G_std": float(g_vals.std()),
        "B_std": float(b_vals.std()),
        "gray_mean": float(gray_vals.mean()),
        "gray_std": float(gray_vals.std()),
    }
    stats["WB_R/G"] = stats["R_mean"] / max(stats["G_mean"], 1.0)
    stats["WB_B/G"] = stats["B_mean"] / max(stats["G_mean"], 1.0)
    stats["WB_error"] = abs(stats["WB_R/G"] - 1.0) + abs(stats["WB_B/G"] - 1.0)

    hist = np.bincount(gray_vals.astype(np.uint8), minlength=256).astype(np.float64)
    total = max(hist.sum(), 1.0)
    stats["pct_underexposed"] = float(hist[:16].sum() / total * 100.0)
    stats["pct_overexposed"] = float(hist[240:].sum() / total * 100.0)
    stats["pct_midtones"] = float(hist[64:192].sum() / total * 100.0)
    stats["dynamic_range_pct"] = float(np.percentile(gray_vals, 99) - np.percentile(gray_vals, 1))

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_vals = _masked_values(lap, keep_mask)
    stats["laplacian_var"] = float(lap_vals.var())
    stats["laplacian_mean"] = float(np.abs(lap_vals).mean())

    edges = cv2.Canny(gray, 50, 150)
    edge_vals = _masked_values(edges, keep_mask)
    stats["edge_density_pct"] = float(edge_vals.sum() / 255.0 / max(edge_vals.size, 1) * 100.0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    neutral_mask = (hsv[:, :, 1] < 30) & keep_mask
    if neutral_mask.sum() > 100:
        neutral_r = float(r[neutral_mask].mean())
        neutral_g = float(g[neutral_mask].mean())
        neutral_b = float(b[neutral_mask].mean())
        stats["neutral_R"] = neutral_r
        stats["neutral_G"] = neutral_g
        stats["neutral_B"] = neutral_b
        stats["neutral_cast"] = "none"
        if neutral_r > neutral_g * 1.1 and neutral_r > neutral_b * 1.1:
            stats["neutral_cast"] = "RED"
        elif neutral_g > neutral_r * 1.1 and neutral_g > neutral_b * 1.1:
            stats["neutral_cast"] = "GREEN"
        elif neutral_b > neutral_r * 1.1 and neutral_b > neutral_g * 1.1:
            stats["neutral_cast"] = "BLUE"
        elif neutral_r > neutral_b * 1.1 and neutral_g > neutral_b * 1.1:
            stats["neutral_cast"] = "YELLOW (R+G high, B low)"

    stats.update(analyze_orb(gray, orb_cfg))
    return stats


def analyze_stereo_pair(
    frame_left: np.ndarray,
    frame_right: np.ndarray,
    orb_cfg: ORBConfig,
    mask_left: np.ndarray | None = None,
    mask_right: np.ndarray | None = None,
) -> dict[str, float]:
    gray_left, _ = prepare_gray(frame_left, mask_left)
    gray_right, _ = prepare_gray(frame_right, mask_right)
    orb = cv2.ORB_create(nfeatures=orb_cfg.nfeatures, fastThreshold=orb_cfg.min_fast)
    kp_left, desc_left = orb.detectAndCompute(gray_left, None)
    kp_right, desc_right = orb.detectAndCompute(gray_right, None)

    stats = {
        "stereo_left_kp": float(len(kp_left)),
        "stereo_right_kp": float(len(kp_right)),
        "stereo_raw_matches": 0.0,
        "stereo_inliers": 0.0,
        "stereo_inlier_ratio": 0.0,
        "stereo_match_ratio": 0.0,
        "stereo_hamming_med": 0.0,
    }
    if desc_left is None or desc_right is None or len(kp_left) < 8 or len(kp_right) < 8:
        return stats

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(desc_left, desc_right, k=2)
    good_matches = []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, alt = pair
        if best.distance < 0.8 * alt.distance:
            good_matches.append(best)

    stats["stereo_raw_matches"] = float(len(good_matches))
    stats["stereo_match_ratio"] = float(
        len(good_matches) / max(min(len(kp_left), len(kp_right)), 1)
    )
    if good_matches:
        stats["stereo_hamming_med"] = float(np.median([m.distance for m in good_matches]))

    if len(good_matches) < 8:
        return stats

    pts_left = np.float32([kp_left[m.queryIdx].pt for m in good_matches])
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in good_matches])
    _, inlier_mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC, 2.0, 0.99)
    if inlier_mask is not None:
        inliers = int(inlier_mask.astype(bool).sum())
        stats["stereo_inliers"] = float(inliers)
        stats["stereo_inlier_ratio"] = float(inliers / max(len(good_matches), 1))
    return stats


def summarize_numeric(stats_list: list[dict]) -> dict:
    if not stats_list:
        return {}
    summary: dict[str, float] = {}
    numeric_keys = set()
    for item in stats_list:
        numeric_keys.update(k for k, v in item.items() if isinstance(v, (int, float)))
    for key in sorted(numeric_keys):
        values = [float(item[key]) for item in stats_list if key in item]
        if values:
            summary[key] = float(np.mean(values))
    return summary


def analyze_burst_motion(frames: list[np.ndarray], exclude_mask: np.ndarray | None = None) -> dict[str, float]:
    if len(frames) < 2:
        return {"interframe_absdiff": 0.0, "duplicate_frame_pct": 0.0}
    diffs = []
    for prev, curr in zip(frames, frames[1:]):
        prev_gray, keep_mask = prepare_gray(prev, exclude_mask)
        curr_gray, _ = prepare_gray(curr, exclude_mask)
        diff = cv2.absdiff(prev_gray, curr_gray)
        diffs.append(float(_masked_values(diff, keep_mask).mean()))
    diff_arr = np.array(diffs, dtype=np.float64)
    return {
        "interframe_absdiff": float(diff_arr.mean()),
        "duplicate_frame_pct": float(np.mean(diff_arr < 1.0) * 100.0),
    }


def score_vio(frame_stats: dict, stereo_stats: dict | None) -> float:
    stereo = stereo_stats or {}
    return float(
        0.03 * frame_stats.get("gray_std", 0.0)
        + 1.0 * np.log1p(frame_stats.get("orb_primary_kp", 0.0))
        + 0.8 * np.log1p(frame_stats.get("orb_rescue_kp", 0.0))
        + 1.5 * np.log1p(stereo.get("stereo_inliers", 0.0))
        + 0.8 * np.log1p(stereo.get("stereo_raw_matches", 0.0))
        - 0.04 * frame_stats.get("pct_underexposed", 0.0)
        - 0.04 * frame_stats.get("pct_overexposed", 0.0)
        - 0.15 * frame_stats.get("duplicate_frame_pct", 0.0)
    )


def print_phase_summary(reporter: Reporter, phase: PhaseResult, orb_cfg: ORBConfig) -> None:
    reporter.line(f"\n{'=' * 68}")
    reporter.line(f"  {phase.name}")
    reporter.line(f"{'=' * 68}")
    for cam_idx, stats in sorted(phase.frame.items()):
        reporter.line(f"  cam{cam_idx}")
        reporter.line(
            "    Channels    : "
            f"R={stats['R_mean']:.0f}+-{stats['R_std']:.0f}  "
            f"G={stats['G_mean']:.0f}+-{stats['G_std']:.0f}  "
            f"B={stats['B_mean']:.0f}+-{stats['B_std']:.0f}"
        )
        reporter.line(
            f"    Gray        : mean={stats['gray_mean']:.0f}  std={stats['gray_std']:.0f}"
        )
        reporter.line(
            "    WB ratios   : "
            f"R/G={stats['WB_R/G']:.3f}  B/G={stats['WB_B/G']:.3f}  "
            f"error={stats['WB_error']:.3f}"
        )
        if "neutral_cast" in stats:
            reporter.line(
                "    Neutral cast: "
                f"{stats['neutral_cast']}  "
                f"(R={stats.get('neutral_R', 0):.0f} "
                f"G={stats.get('neutral_G', 0):.0f} "
                f"B={stats.get('neutral_B', 0):.0f})"
            )
        reporter.line(
            "    Exposure    : "
            f"{stats['pct_underexposed']:.1f}% under  "
            f"{stats['pct_midtones']:.1f}% mid  "
            f"{stats['pct_overexposed']:.1f}% over  "
            f"DR={stats['dynamic_range_pct']:.0f}"
        )
        reporter.line(
            "    Sharpness   : "
            f"laplacian_var={stats['laplacian_var']:.0f}  "
            f"edges={stats['edge_density_pct']:.1f}%"
        )
        reporter.line(
            "    ORB         : "
            f"FAST{orb_cfg.ini_fast}={stats['orb_primary_kp']:.0f}  "
            f"FAST{orb_cfg.min_fast}={stats['orb_rescue_kp']:.0f}"
        )
        reporter.line(
            "    New frames  : "
            f"absdiff={stats['interframe_absdiff']:.2f}  "
            f"duplicates={stats['duplicate_frame_pct']:.1f}%"
        )
    if phase.stereo:
        reporter.line("  stereo")
        reporter.line(
            "    Matches     : "
            f"raw={phase.stereo['stereo_raw_matches']:.0f}  "
            f"inliers={phase.stereo['stereo_inliers']:.0f}  "
            f"inlier_ratio={phase.stereo['stereo_inlier_ratio'] * 100:.1f}%"
        )
        reporter.line(
            "    Coverage    : "
            f"left={phase.stereo['stereo_left_kp']:.0f}  "
            f"right={phase.stereo['stereo_right_kp']:.0f}  "
            f"match_ratio={phase.stereo['stereo_match_ratio'] * 100:.1f}%"
        )
        reporter.line(
            f"    Hamming med : {phase.stereo['stereo_hamming_med']:.1f}"
        )
    if phase.frame:
        score = float(np.mean([stats["vio_score"] for stats in phase.frame.values()]))
        reporter.line(f"  Heuristic VIO score: {score:.2f}")


def print_comparison(reporter: Reporter, base: PhaseResult, other: PhaseResult) -> None:
    reporter.line(f"\n{'=' * 68}")
    reporter.line(f"  COMPARISON: {base.name} -> {other.name}")
    reporter.line(f"{'=' * 68}")
    for cam_idx in sorted(set(base.frame) & set(other.frame)):
        a = base.frame[cam_idx]
        b = other.frame[cam_idx]
        reporter.line(f"  cam{cam_idx}")
        reporter.line(
            f"    Gray std    : {a['gray_std']:.1f} -> {b['gray_std']:.1f}  ({b['gray_std'] - a['gray_std']:+.1f})"
        )
        reporter.line(
            f"    Laplacian   : {a['laplacian_var']:.0f} -> {b['laplacian_var']:.0f}  ({b['laplacian_var'] - a['laplacian_var']:+.0f})"
        )
        reporter.line(
            f"    ORB primary : {a['orb_primary_kp']:.0f} -> {b['orb_primary_kp']:.0f}  ({b['orb_primary_kp'] - a['orb_primary_kp']:+.0f})"
        )
        reporter.line(
            f"    ORB rescue  : {a['orb_rescue_kp']:.0f} -> {b['orb_rescue_kp']:.0f}  ({b['orb_rescue_kp'] - a['orb_rescue_kp']:+.0f})"
        )
        reporter.line(
            f"    Duplicates  : {a['duplicate_frame_pct']:.1f}% -> {b['duplicate_frame_pct']:.1f}%  ({b['duplicate_frame_pct'] - a['duplicate_frame_pct']:+.1f}%)"
        )
        reporter.line(
            f"    VIO score   : {a['vio_score']:.2f} -> {b['vio_score']:.2f}  ({b['vio_score'] - a['vio_score']:+.2f})"
        )
    if base.stereo and other.stereo:
        reporter.line("  stereo")
        reporter.line(
            f"    Raw matches : {base.stereo['stereo_raw_matches']:.0f} -> {other.stereo['stereo_raw_matches']:.0f}  ({other.stereo['stereo_raw_matches'] - base.stereo['stereo_raw_matches']:+.0f})"
        )
        reporter.line(
            f"    Inliers     : {base.stereo['stereo_inliers']:.0f} -> {other.stereo['stereo_inliers']:.0f}  ({other.stereo['stereo_inliers'] - base.stereo['stereo_inliers']:+.0f})"
        )


def capture_phase(
    phase_name: str,
    camera_indices: list[int],
    settle_frames: int,
    sample_frames: int,
    width: int,
    height: int,
    fps: int,
    orb_cfg: ORBConfig,
    masks: dict[int, np.ndarray | None],
    out_dir: Path,
) -> PhaseResult:
    frame_results: dict[int, dict] = {}
    isp_results: dict[int, dict] = {}
    bursts: dict[int, list[np.ndarray]] = {}

    for cam_idx in camera_indices:
        frames = capture_burst(cam_idx, settle_frames, sample_frames, width, height, fps)
        bursts[cam_idx] = frames
        isp_results[cam_idx] = read_all_isp(cam_idx)
        if not frames:
            continue
        per_frame = [
            analyze_frame(frame, f"{phase_name} cam{cam_idx}", orb_cfg, masks.get(cam_idx))
            for frame in frames
        ]
        summary = summarize_numeric(per_frame)
        summary.update(analyze_burst_motion(frames, masks.get(cam_idx)))
        summary["vio_score"] = score_vio(summary, None)
        frame_results[cam_idx] = summary
        cv2.imwrite(str(out_dir / f"cam{cam_idx}_{phase_name.lower().replace(' ', '_')}_last.png"), frames[-1])

    stereo_summary = None
    if len(camera_indices) >= 2:
        left_frames = bursts.get(camera_indices[0], [])
        right_frames = bursts.get(camera_indices[1], [])
        n_pairs = min(len(left_frames), len(right_frames))
        if n_pairs > 0:
            stereo_per_pair = [
                analyze_stereo_pair(
                    left_frames[i],
                    right_frames[i],
                    orb_cfg,
                    masks.get(camera_indices[0]),
                    masks.get(camera_indices[1]),
                )
                for i in range(n_pairs)
            ]
            stereo_summary = summarize_numeric(stereo_per_pair)
            for cam_idx, stats in frame_results.items():
                stats["vio_score"] = score_vio(stats, stereo_summary)

    return PhaseResult(name=phase_name, frame=frame_results, stereo=stereo_summary, isp=isp_results)


def lock_from_freerun(camera_indices: list[int]) -> dict[int, dict[str, str]]:
    locked: dict[int, dict[str, str]] = {}
    for cam_idx in camera_indices:
        locked[cam_idx] = {
            key: uvc_get(cam_idx, key)
            for key in LOCKABLE_ISP_KEYS
        }
    return locked


def apply_trigger_locked(camera_indices: list[int], locked_values: dict[int, dict[str, str]]) -> None:
    for cam_idx in camera_indices:
        controls = [
            ("auto-white-balance-temp", "0"),
            ("auto-exposure-mode", MANUAL_EXPOSURE_MODE),
            ("auto-exposure-priority", "1"),
        ]
        for key in LOCKABLE_ISP_KEYS:
            value = locked_values.get(cam_idx, {}).get(key)
            if value is not None and value != "?":
                controls.append((key, value))
        uvc_set(cam_idx, controls)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orb_cfg = load_orb_config(args.orb_config)
    reporter = Reporter()

    camera_indices = find_cameras()
    if not camera_indices:
        reporter.line("ERROR: No OV9782 cameras found")
        return

    reporter.line(f"Found {len(camera_indices)} camera(s): indices {camera_indices}")
    reporter.line(
        f"ORB config: nfeatures={orb_cfg.nfeatures} ini_fast={orb_cfg.ini_fast} min_fast={orb_cfg.min_fast}"
    )

    masks: dict[int, np.ndarray | None] = {}
    left_mask = None
    right_mask = None
    if len(camera_indices) >= 1:
        left_mask = load_mask(args.mask_left, (args.height, args.width)) if args.mask_left else None
        masks[camera_indices[0]] = left_mask
    if len(camera_indices) >= 2:
        right_mask = load_mask(args.mask_right, (args.height, args.width)) if args.mask_right else None
        masks[camera_indices[1]] = right_mask

    reporter.line("\n--- Phase 1: Free-running auto ISP ---")
    apply_controls(
        camera_indices,
        [
            ("auto-exposure-mode", "8"),
            ("auto-exposure-priority", "0"),
            ("auto-white-balance-temp", "1"),
        ],
    )
    freerun = capture_phase(
        "FREE-RUN AUTO",
        camera_indices,
        args.settle_frames,
        args.sample_frames,
        args.width,
        args.height,
        args.fps,
        orb_cfg,
        masks,
        out_dir,
    )
    print_phase_summary(reporter, freerun, orb_cfg)

    locked_values = lock_from_freerun(camera_indices)
    reporter.line("\nPer-camera lock candidates from free-run settle:")
    for cam_idx in camera_indices:
        reporter.line(f"  cam{cam_idx}: {locked_values[cam_idx]}")

    reporter.line("\n--- Phase 2: Triggered auto exposure ---")
    apply_controls(
        camera_indices,
        [
            ("auto-exposure-mode", "8"),
            ("auto-exposure-priority", "1"),
            ("auto-white-balance-temp", "1"),
        ],
    )
    time.sleep(2.0)
    triggered_auto = capture_phase(
        "TRIGGERED AUTO",
        camera_indices,
        args.settle_frames,
        args.sample_frames,
        args.width,
        args.height,
        args.fps,
        orb_cfg,
        masks,
        out_dir,
    )
    print_phase_summary(reporter, triggered_auto, orb_cfg)

    phases = [freerun, triggered_auto]
    if not args.skip_trigger_locked:
        reporter.line("\n--- Phase 3: Triggered + manual lock from free-run ---")
        apply_trigger_locked(camera_indices, locked_values)
        time.sleep(1.0)
        triggered_locked = capture_phase(
            "TRIGGERED LOCKED",
            camera_indices,
            args.settle_frames,
            args.sample_frames,
            args.width,
            args.height,
            args.fps,
            orb_cfg,
            masks,
            out_dir,
        )
        print_phase_summary(reporter, triggered_locked, orb_cfg)
        phases.append(triggered_locked)

    for phase in phases[1:]:
        print_comparison(reporter, freerun, phase)

    apply_controls(
        camera_indices,
        [
            ("auto-exposure-mode", "8"),
            ("auto-exposure-priority", "0"),
            ("auto-white-balance-temp", "1"),
        ],
    )
    reporter.line(f"\nCameras reset to free-running. Outputs saved to {out_dir}")

    reporter.write(out_dir / "analysis.txt")
    report_payload = {
        "camera_indices": camera_indices,
        "orb": asdict(orb_cfg),
        "phases": [
            {
                "name": phase.name,
                "frame": phase.frame,
                "stereo": phase.stereo,
                "isp": phase.isp,
            }
            for phase in phases
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report_payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
