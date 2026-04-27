"""Estimate GoPro pose from the 6×6 Kalibr AprilGrid validation board.

A measurement harness: the printed Kalibr target serves as external truth
to bound the SLAM stack's absolute accuracy. Detects tag36h11, solves a
multi-tag PnP per frame, and writes per-frame poses plus a drift/smoothness
summary. Used to produce the validation result reported in the paper.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import av
import cv2
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.spatial.transform as st
import yaml
from pupil_apriltags import Detector

from glove.process.aruco._util import parse_fisheye_intrinsics


DEFAULT_INTRINSICS = Path("software/configs/calibration/gopro_intrinsics_2_7k.json")
DEFAULT_TARGET = Path("software/configs/calibration/kalibr/target.yaml")


@dataclass(frozen=True)
class AprilGrid:
    rows: int
    cols: int
    tag_size_m: float
    tag_spacing: float

    @property
    def pitch_m(self) -> float:
        return self.tag_size_m * (1.0 + self.tag_spacing)

    def has_tag(self, tag_id: int) -> bool:
        return 0 <= tag_id < self.rows * self.cols

    def tag_origin_xy(self, tag_id: int) -> tuple[float, float]:
        """Lower-left corner of a tag in board coordinates.

        The Kalibr-generated target IDs increase left-to-right along the bottom
        row, then row-by-row upward:

            top:    30 31 32 33 34 35
            bottom:  0  1  2  3  4  5

        Board frame here is X right, Y up, Z out of the board toward camera.
        """
        row_from_bottom = tag_id // self.cols
        col = tag_id % self.cols
        return col * self.pitch_m, row_from_bottom * self.pitch_m

    def object_corners(self, tag_id: int) -> np.ndarray:
        """3D tag corners in pupil_apriltags order: BL, BR, TR, TL."""
        x0, y0 = self.tag_origin_xy(tag_id)
        s = self.tag_size_m
        return np.array(
            [
                [x0, y0, 0.0],
                [x0 + s, y0, 0.0],
                [x0 + s, y0 + s, 0.0],
                [x0, y0 + s, 0.0],
            ],
            dtype=np.float64,
        )


def load_grid(path: Path) -> AprilGrid:
    data = yaml.safe_load(path.read_text())
    if data.get("target_type") != "aprilgrid":
        raise ValueError(f"Expected aprilgrid target config, got {data.get('target_type')}")
    return AprilGrid(
        rows=int(data["tagRows"]),
        cols=int(data["tagCols"]),
        tag_size_m=float(data["tagSize"]),
        tag_spacing=float(data["tagSpacing"]),
    )


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    intr = parse_fisheye_intrinsics(json.loads(path.read_text()))
    return intr["K"], intr["D"]


def iter_frames(mp4: Path, stride: int) -> Iterable[tuple[int, np.ndarray]]:
    with av.open(str(mp4)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx % stride != 0:
                continue
            yield frame_idx, frame.to_ndarray(format="bgr24")


def preprocess(
    gray: np.ndarray,
    crop: tuple[int, int, int, int],
    upscale: float,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = crop
    cropped = gray[y0:y1, x0:x1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(cropped)
    if upscale != 1.0:
        clahe = cv2.resize(
            clahe,
            None,
            fx=upscale,
            fy=upscale,
            interpolation=cv2.INTER_CUBIC,
        )
    return clahe, np.array([x0, y0], dtype=np.float64)


def detect_tags(
    detector: Detector,
    image: np.ndarray,
    crop_offset: np.ndarray,
    upscale: float,
    grid: AprilGrid,
    min_margin: float,
) -> dict[int, object]:
    detections = detector.detect(image, estimate_tag_pose=False)
    best: dict[int, object] = {}
    for det in detections:
        tag_id = int(det.tag_id)
        if not grid.has_tag(tag_id):
            continue
        if float(det.decision_margin) < min_margin:
            continue
        if tag_id not in best or float(det.decision_margin) > float(best[tag_id].decision_margin):
            best[tag_id] = det

    # Convert corners from preprocessed crop pixels back to original frame pixels.
    for det in best.values():
        det.corners = np.asarray(det.corners, dtype=np.float64) / upscale + crop_offset
        det.center = np.asarray(det.center, dtype=np.float64) / upscale + crop_offset
    return best


def solve_board_pose(
    detections: dict[int, object],
    grid: AprilGrid,
    K: np.ndarray,
    D: np.ndarray,
    min_tags: int,
    ransac_reproj_px: float,
) -> dict[str, object] | None:
    if len(detections) < min_tags:
        return None

    obj_points = []
    img_points = []
    margins = []
    tag_ids = []
    for tag_id, det in sorted(detections.items()):
        obj_points.append(grid.object_corners(tag_id))
        img_points.append(np.asarray(det.corners, dtype=np.float64))
        margins.append(float(det.decision_margin))
        tag_ids.append(tag_id)

    obj = np.concatenate(obj_points, axis=0).astype(np.float64)
    img = np.concatenate(img_points, axis=0).astype(np.float64)
    img_und = cv2.fisheye.undistortPoints(img.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj,
        img_und,
        K,
        np.zeros((5, 1), dtype=np.float64),
        iterationsCount=100,
        reprojectionError=float(ransac_reproj_px),
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            obj,
            img_und,
            K,
            np.zeros((5, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        inliers = None
        if not ok:
            return None

    if inliers is not None and len(inliers) >= max(4, min_tags) * 4:
        idx = inliers.reshape(-1)
        obj_refine = obj[idx]
        img_refine = img_und[idx]
        ok, rvec, tvec = cv2.solvePnP(
            obj_refine,
            img_refine,
            K,
            np.zeros((5, 1), dtype=np.float64),
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, np.zeros((5, 1), dtype=np.float64))
    err = np.linalg.norm(proj.reshape(-1, 2) - img_und, axis=1)

    R_cam_board = st.Rotation.from_rotvec(rvec.reshape(3)).as_matrix()
    t_cam_board = tvec.reshape(3)
    R_board_cam = R_cam_board.T
    t_board_cam = -R_board_cam @ t_cam_board
    q_board_cam = st.Rotation.from_matrix(R_board_cam).as_quat()  # x, y, z, w

    return {
        "n_tags": len(tag_ids),
        "n_inlier_corners": int(len(inliers)) if inliers is not None else len(obj),
        "tag_ids": ",".join(str(x) for x in tag_ids),
        "margin_min": float(np.min(margins)),
        "margin_median": float(np.median(margins)),
        "reproj_rmse_px": float(np.sqrt(np.mean(err**2))),
        "board_in_cam_x_m": float(t_cam_board[0]),
        "board_in_cam_y_m": float(t_cam_board[1]),
        "board_in_cam_z_m": float(t_cam_board[2]),
        "cam_in_board_x_m": float(t_board_cam[0]),
        "cam_in_board_y_m": float(t_board_cam[1]),
        "cam_in_board_z_m": float(t_board_cam[2]),
        "cam_in_board_qx": float(q_board_cam[0]),
        "cam_in_board_qy": float(q_board_cam[1]),
        "cam_in_board_qz": float(q_board_cam[2]),
        "cam_in_board_qw": float(q_board_cam[3]),
    }


def summarize_trajectory(df: pd.DataFrame, fps: float, smooth_sigma_frames: float) -> dict[str, float]:
    solved = df[df["valid"]].copy()
    out: dict[str, float] = {
        "n_frames": float(len(df)),
        "n_solved": float(df["solved"].sum()),
        "n_valid": float(len(solved)),
        "valid_rate": float(len(solved) / max(len(df), 1)),
    }
    if solved.empty:
        return out

    pos = solved[["cam_in_board_x_m", "cam_in_board_y_m", "cam_in_board_z_m"]].to_numpy(float)
    center = np.nanmedian(pos, axis=0)
    dev_mm = np.linalg.norm(pos - center[None, :], axis=1) * 1000.0

    sigma = max(float(smooth_sigma_frames), 0.0)
    if sigma > 0 and len(pos) > int(4 * sigma):
        smooth = scipy.ndimage.gaussian_filter1d(pos, sigma=sigma, axis=0, mode="nearest")
        hf_mm = np.linalg.norm(pos - smooth, axis=1) * 1000.0
    else:
        hf_mm = np.zeros(len(pos), dtype=np.float64)

    out.update(
        {
            "fps_assumed": float(fps),
            "n_tags_median": float(np.nanmedian(solved["n_tags"])),
            "n_tags_min": float(np.nanmin(solved["n_tags"])),
            "reproj_rmse_median_px": float(np.nanmedian(solved["reproj_rmse_px"])),
            "reproj_rmse_p95_px": float(np.nanpercentile(solved["reproj_rmse_px"], 95)),
            "static_dev_median_mm": float(np.nanmedian(dev_mm)),
            "static_dev_p95_mm": float(np.nanpercentile(dev_mm, 95)),
            "hifreq_median_mm": float(np.nanmedian(hf_mm)),
            "hifreq_p95_mm": float(np.nanpercentile(hf_mm, 95)),
        }
    )
    return out


def analyze_session(
    session: Path,
    out_dir: Path,
    grid: AprilGrid,
    K: np.ndarray,
    D: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    mp4 = session / "raw" / "raw_video.mp4"
    if not mp4.exists():
        raise FileNotFoundError(mp4)

    detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        refine_edges=True,
        decode_sharpening=0.25,
    )
    rows = []
    crop = tuple(int(x) for x in args.crop)
    for frame_idx, bgr in iter_frames(mp4, args.stride):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        processed, offset = preprocess(gray, crop, args.upscale)
        detections = detect_tags(detector, processed, offset, args.upscale, grid, args.min_margin)
        pose = solve_board_pose(detections, grid, K, D, args.min_tags, args.ransac_reproj_px)
        row = {
            "frame_idx": frame_idx,
            "t_rel": frame_idx / args.fps,
            "solved": pose is not None,
            "valid": False,
            "n_tags_raw": len(detections),
        }
        if pose is not None:
            row.update(pose)
            row["valid"] = pose["reproj_rmse_px"] <= args.max_rmse_px
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{session.name}_aprilgrid_pose.csv"
    df.to_csv(csv_path, index=False)
    summary = summarize_trajectory(df, fps=args.fps, smooth_sigma_frames=args.smooth_sigma_frames)
    summary.update({"session": session.name, "csv": str(csv_path)})
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("data/experiments/pose_validation/aprilgrid"))
    parser.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=float, default=59.94005994005994)
    parser.add_argument("--crop", nargs=4, type=int, default=(0, 520, 2100, 2028), metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--upscale", type=float, default=2.0)
    parser.add_argument("--min-margin", type=float, default=30.0)
    parser.add_argument("--min-tags", type=int, default=4)
    parser.add_argument("--ransac-reproj-px", type=float, default=8.0)
    parser.add_argument("--max-rmse-px", type=float, default=12.0)
    parser.add_argument("--smooth-sigma-frames", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = load_grid(args.target)
    K, D = load_intrinsics(args.intrinsics)

    summaries = []
    for session in args.sessions:
        summaries.append(analyze_session(session, args.out_dir, grid, K, D, args))

    summary_df = pd.DataFrame(summaries)
    summary_csv = args.out_dir / "summary.csv"
    summary_yaml = args.out_dir / "summary.yaml"
    summary_df.to_csv(summary_csv, index=False)
    summary_yaml.write_text(yaml.safe_dump(summaries, sort_keys=False))

    cols = [
        "session",
        "n_solved",
        "n_valid",
        "valid_rate",
        "n_tags_median",
        "reproj_rmse_median_px",
        "static_dev_median_mm",
        "static_dev_p95_mm",
        "hifreq_median_mm",
        "hifreq_p95_mm",
    ]
    print(summary_df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {summary_csv}")


if __name__ == "__main__":
    main()
