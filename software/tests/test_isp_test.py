import cv2
import numpy as np

from glove.scripts.isp_test import (
    ORBConfig,
    analyze_burst_motion,
    analyze_frame,
    analyze_stereo_pair,
)


def _make_checkerboard(width: int = 320, height: int = 240, block: int = 16) -> np.ndarray:
    yy, xx = np.indices((height, width))
    pattern = (((xx // block) + (yy // block)) % 2) * 255
    img = np.stack([pattern, pattern, pattern], axis=-1).astype(np.uint8)
    return img


def _make_feature_scene(width: int = 320, height: int = 240) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(80):
        center = (int(rng.integers(20, width - 20)), int(rng.integers(20, height - 20)))
        radius = int(rng.integers(4, 10))
        color = int(rng.integers(120, 255))
        cv2.circle(img, center, radius, (color, color, color), thickness=-1)
    return img


def test_analyze_frame_reports_orb_features_on_textured_image():
    frame = _make_checkerboard()
    stats = analyze_frame(frame, "checker", ORBConfig(nfeatures=800, ini_fast=15, min_fast=5))

    assert stats["gray_std"] > 50
    assert stats["orb_primary_kp"] > 100
    assert stats["orb_rescue_kp"] >= stats["orb_primary_kp"]


def test_analyze_burst_motion_flags_duplicate_frames():
    frame = _make_checkerboard()
    motion = analyze_burst_motion([frame, frame.copy(), frame.copy()])

    assert motion["interframe_absdiff"] == 0.0
    assert motion["duplicate_frame_pct"] == 100.0


def test_analyze_stereo_pair_reports_matches_for_shifted_pattern():
    left = _make_feature_scene()
    right = np.roll(left, shift=8, axis=1)

    stats = analyze_stereo_pair(left, right, ORBConfig(nfeatures=1000, ini_fast=15, min_fast=5))

    assert stats["stereo_left_kp"] > 50
    assert stats["stereo_raw_matches"] > 20
    assert stats["stereo_inliers"] > 10
