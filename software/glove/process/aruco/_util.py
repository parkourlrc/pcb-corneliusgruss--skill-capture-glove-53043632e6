"""Minimal ports of UMI's `umi.common.cv_util` helpers.

Only the functions we need for detect + calibrate-tag: fisheye intrinsics
parsing/scaling, ArUco config parsing, per-frame tag detection + PnP, and a
`pose_to_mat` equivalent.

Kept in one small file (rather than vendoring all of UMI) so the provenance is
obvious and future ORB-SLAM3 / OpenCV version bumps have one place to diff.
"""

from __future__ import annotations

import copy
from typing import Dict

import cv2
import numpy as np
import scipy.spatial.transform as st


# ---------- fisheye intrinsics (Kannala-Brandt) ----------

def parse_fisheye_intrinsics(json_data: dict) -> Dict[str, np.ndarray]:
    """Convert OpenCameraImuCalibration FISHEYE JSON to an OpenCV dict."""
    assert json_data["intrinsic_type"] == "FISHEYE"
    intr = json_data["intrinsics"]
    h = json_data["image_height"]
    w = json_data["image_width"]
    f = intr["focal_length"]
    px = intr["principal_pt_x"]
    py = intr["principal_pt_y"]
    kb8 = [
        intr["radial_distortion_1"],
        intr["radial_distortion_2"],
        intr["radial_distortion_3"],
        intr["radial_distortion_4"],
    ]
    return {
        "DIM": np.array([w, h], dtype=np.int64),
        "K": np.array([[f, 0, px], [0, f, py], [0, 0, 1]], dtype=np.float64),
        "D": np.array([kb8]).T,
    }


def convert_fisheye_intrinsics_resolution(
    opencv_intr_dict: Dict[str, np.ndarray],
    target_resolution,
) -> Dict[str, np.ndarray]:
    """Rescale intrinsics to a new (w, h), assuming symmetric horizontal crop/pad only."""
    iw, ih = opencv_intr_dict["DIM"]
    iK = opencv_intr_dict["K"]
    ifx, ify = iK[0, 0], iK[1, 1]
    ipx, ipy = iK[0, 2], iK[1, 2]
    ow, oh = target_resolution
    ofx = ifx / ih * oh
    ofy = ify / ih * oh
    opx = (ipx - (iw / 2)) / ih * oh + (ow / 2)
    opy = ipy / ih * oh
    out = copy.deepcopy(opencv_intr_dict)
    out["DIM"] = np.array([ow, oh], dtype=np.int64)
    out["K"] = np.array([[ofx, 0, opx], [0, ofy, opy], [0, 0, 1]], dtype=np.float64)
    return out


# ---------- ArUco config ----------

def _get_aruco_dict(predefined: str) -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, predefined))


def parse_aruco_config(aruco_config_dict: dict):
    aruco_dict = _get_aruco_dict(**aruco_config_dict["aruco_dict"])
    n_markers = len(aruco_dict.bytesList)
    marker_size_map = aruco_config_dict["marker_size_map"]
    default_size = marker_size_map.get("default", None)
    out = dict()
    for mid in range(n_markers):
        size = default_size
        if mid in marker_size_map:
            size = marker_size_map[mid]
        out[mid] = size
    return {"aruco_dict": aruco_dict, "marker_size_map": out}


# ---------- ArUco detect + PnP ----------

def detect_localize_aruco_tags(
    img: np.ndarray,
    aruco_dict: cv2.aruco.Dictionary,
    marker_size_map: Dict[int, float],
    fisheye_intr_dict: Dict[str, np.ndarray],
    refine_subpix: bool = True,
):
    """Per-frame detect + fisheye-undistort + PnP. Returns `{tag_id: {rvec, tvec, corners}}`.

    Cross-version OpenCV: uses the `ArucoDetector` class (cv2 ≥4.7) and `solvePnP`
    directly. The previous module-level `cv2.aruco.detectMarkers` and
    `cv2.aruco.estimatePoseSingleMarkers` were removed in cv2 4.11.
    """
    K = fisheye_intr_dict["K"]
    D = fisheye_intr_dict["D"]
    param = cv2.aruco.DetectorParameters()
    if refine_subpix:
        param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, param)
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None or len(corners) == 0:
        return dict()
    tag_dict: Dict[int, dict] = dict()
    for this_id, this_corners in zip(ids, corners):
        this_id = int(this_id[0])
        size = marker_size_map.get(this_id)
        if size is None:
            continue
        # Undistort corners through the fisheye model, then PnP in the
        # undistorted pinhole space — this is what UMI does.
        undistorted = cv2.fisheye.undistortPoints(this_corners, K, D, P=K)
        # Replacement for the removed `estimatePoseSingleMarkers`. The marker is
        # a flat square of side `size` centered at origin in marker frame, with
        # corners in the same order OpenCV's ArucoDetector returns them
        # (TL, TR, BR, BL when looking at the marker face).
        h = size / 2.0
        obj_pts = np.array([
            [-h,  h, 0.0],
            [ h,  h, 0.0],
            [ h, -h, 0.0],
            [-h, -h, 0.0],
        ], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, undistorted.reshape(-1, 2).astype(np.float64),
            K, np.zeros((1, 5)), flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            continue
        tag_dict[this_id] = {
            "rvec": rvec.squeeze(),
            "tvec": tvec.squeeze(),
            "corners": this_corners.squeeze(),
        }
    return tag_dict


# ---------- pose helpers ----------

def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """`[tx, ty, tz, rx, ry, rz]` (rotvec) → 4x4 homogeneous matrix."""
    pos = pose[..., :3]
    rot = st.Rotation.from_rotvec(pose[..., 3:])
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=np.float64)
    mat[..., :3, 3] = pos
    mat[..., :3, :3] = rot.as_matrix()
    mat[..., 3, 3] = 1
    return mat
