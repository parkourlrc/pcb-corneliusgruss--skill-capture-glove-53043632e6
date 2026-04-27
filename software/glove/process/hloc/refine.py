"""Pose-only global bundle adjustment for HLoc per-frame trajectories.

# Why this exists

HLoc (SuperPoint + LightGlue + NetVLAD + COLMAP-based PnP) localizes every
demo frame **independently** against the map. Adjacent frames at 60 fps see
nearly the same scene, but their PnP solutions are computed in isolation:

  - frame i   matches features against 3D points {P1, P2, P3, P5, P8, ...}
  - frame i+1 matches features against 3D points {P1, P2, P4, P5, P8, ...}
  - different inlier sets → different RANSAC realizations → mm-scale jitter
    even on stationary frames.

Our pose-graph smoother (`smooth.py`) post-hoc smooths these noisy estimates
with a constant-velocity prior, but it can't undo the per-frame noise — it
just blurs it.

This module replaces the per-frame independence with **global pose-only
bundle adjustment**: jointly optimize all N demo poses by minimizing the
total reprojection error of every inlier match across the whole trajectory.
3D map points stay fixed (they came from a globally-BA-optimized SfM model;
we trust them). Adjacent frames share 80%+ of their inlier 3D points, so the
BA optimum cannot have inconsistent poses without paying a reprojection cost.

This is what ORB-SLAM3's Local Bundle Adjustment does in its tracking thread.
The architecture is textbook (Hartley & Zisserman ch. 18); we just do it on
top of HLoc's PnP outputs instead of a custom feature tracker.

# Inputs

For each demo frame, HLoc's `hloc_results.txt_logs.pkl` provides:
  - `keypoints_query`: (M, 2) pixel coords of detected SuperPoint keypoints
  - `points3D_ids`: list of M map-point IDs (one per keypoint match)
  - `PnP_ret.inlier_mask`: (M,) bool — which matches survived RANSAC
  - `PnP_ret.cam_from_world`: pycolmap.Rigid3d — initial pose estimate
  - `PnP_ret.camera`: pycolmap.Camera (OPENCV_FISHEYE intrinsics)

The SfM `pycolmap.Reconstruction` provides 3D point coordinates by ID.

# Output

Refined per-frame poses (cam_from_world) in the same COLMAP coords as input.
Apply downstream scale + tag-frame transform separately (same as `smooth.py`
output).

# Validation target

A clean ORB-SLAM3 reference demo reaches roughly:
    local_straightness_0_25s ≈ 0.96
    per_axis_hifreq_max_mm   ≈ 0.9

The HLoc + pose-graph smoother stack lands around 0.73–0.84 straightness
and 2–4 mm hifreq on a typical demo batch. Goal for BA-refined HLoc is
≥ 0.90 straightness and ≤ 1.5 mm hifreq.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


@dataclass
class FrameInliers:
    """Per-frame data for BA: inlier 2D-3D matches + initial pose."""
    frame_idx: int                 # for ordering
    name: str                      # image name (for diagnostics)
    keypoints: np.ndarray          # (Ni, 2) inlier pixel coords
    points3D: np.ndarray           # (Ni, 3) inlier 3D points (in COLMAP coords)
    pose_init_R: np.ndarray        # (3, 3) cam_from_world rotation
    pose_init_t: np.ndarray        # (3,)   cam_from_world translation


def load_inliers_from_hloc_log(
    logs_pkl_path: Path,
    sfm_recon,                     # pycolmap.Reconstruction
    name_to_frame_idx,             # callable: image name → int (e.g. "demo_..._000060.jpg" → 60)
) -> tuple[list[FrameInliers], dict]:
    """Parse HLoc log + SfM model into per-frame inlier data for BA.

    Returns (frames, camera_dict). camera_dict has fx/fy/cx/cy/k1..k4 from the
    first frame's PnP_ret.camera (assumed shared across the demo).
    """
    with open(logs_pkl_path, "rb") as f:
        d = pickle.load(f)
    loc = d["loc"]

    frames: list[FrameInliers] = []
    cam = None
    for name, entry in loc.items():
        pnp = entry["PnP_ret"]
        if pnp is None or pnp.get("num_inliers", 0) == 0:
            continue
        mask = pnp["inlier_mask"]
        if not mask.any():
            continue

        kpts_all = np.asarray(entry["keypoints_query"], dtype=np.float64)  # (M, 2)
        ids_all = entry["points3D_ids"]
        kpts = kpts_all[mask]
        ids = [pid for pid, m in zip(ids_all, mask) if m]

        # Look up 3D positions (skip any IDs not in the reconstruction — rare,
        # but possible if the SfM model was filtered after HLoc ran).
        # pycolmap.Reconstruction.points3D.__contains__ requires a Python int —
        # np.int64 from the HLoc pickle silently returns False. Cast first.
        xyz_list = []
        kpt_keep = []
        for i, pid in enumerate(ids):
            pid_int = int(pid)
            if pid_int in sfm_recon.points3D:
                xyz_list.append(sfm_recon.points3D[pid_int].xyz)
                kpt_keep.append(i)
        if not xyz_list:
            continue
        xyz = np.asarray(xyz_list, dtype=np.float64)
        kpts = kpts[kpt_keep]

        # Initial pose
        rigid = pnp["cam_from_world"]
        # pycolmap.Rigid3d.rotation is a Rotation3d (quat xyzw); .translation is (3,)
        try:
            qxyzw = np.array(rigid.rotation.quat, dtype=np.float64)  # xyzw
        except AttributeError:
            # newer pycolmap may expose .rotation as Quat / matrix; try .matrix
            qxyzw = Rotation.from_matrix(np.asarray(rigid.rotation.matrix())).as_quat()
        R = Rotation.from_quat(qxyzw).as_matrix()
        t = np.asarray(rigid.translation, dtype=np.float64)

        if cam is None:
            c = pnp["camera"]
            params = list(c.params)
            cam = dict(
                model=str(c.model),
                width=int(c.width),
                height=int(c.height),
                fx=params[0], fy=params[1], cx=params[2], cy=params[3],
                k1=params[4], k2=params[5], k3=params[6], k4=params[7],
            )

        frames.append(FrameInliers(
            frame_idx=name_to_frame_idx(name),
            name=name,
            keypoints=kpts,
            points3D=xyz,
            pose_init_R=R,
            pose_init_t=t,
        ))

    frames.sort(key=lambda f: f.frame_idx)
    return frames, cam


# ===== Fisheye projection (OPENCV_FISHEYE / Kannala-Brandt 4-coeff) =====

def _project_opencv_fisheye(
    Xc: np.ndarray,                # (N, 3) points in camera frame
    fx: float, fy: float, cx: float, cy: float,
    k1: float, k2: float, k3: float, k4: float,
) -> np.ndarray:
    """Project 3D points in camera frame to pixels using OPENCV_FISHEYE model.

    Mirrors cv2.fisheye.projectPoints / COLMAP's OPENCV_FISHEYE (Kannala-Brandt).
    Vectorized + differentiable enough for scipy's numerical Jacobian.

    Points behind the camera (Z <= 0) are projected with their |Z| to avoid
    division blow-ups during BA iterations; the optimizer should never want
    a converged pose that puts inlier points behind.
    """
    Z = Xc[:, 2]
    Z = np.where(np.abs(Z) < 1e-12, 1e-12, Z)
    Xn = Xc[:, 0] / Z
    Yn = Xc[:, 1] / Z
    r2 = Xn * Xn + Yn * Yn
    r = np.sqrt(r2)
    # atan(r) is well-defined; r→0 limit is 1, handle below
    theta = np.arctan(r)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta_d = theta * (1 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)
    # scale = theta_d / r (with r→0 limit = 1, since theta_d ≈ theta ≈ r)
    scale = np.where(r < 1e-9, 1.0, theta_d / np.where(r < 1e-9, 1.0, r))
    x = scale * Xn
    y = scale * Yn
    u = fx * x + cx
    v = fy * y + cy
    return np.stack([u, v], axis=1)


# ===== Pose parameterization: rotation as axis-angle + translation =====

def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """(3,) axis-angle → (3,3) rotation matrix. Uses scipy.Rotation."""
    return Rotation.from_rotvec(rotvec).as_matrix()


def _matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


# ===== Bundle adjustment =====

def refine_trajectory_ba(
    frames: list[FrameInliers],
    camera: dict,
    huber_pixel: float = 2.0,
    max_nfev: int = 200,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run global pose-only BA over all frames.

    Returns (R_refined, t_refined, info):
      R_refined: (N, 3, 3) cam_from_world rotation matrices
      t_refined: (N, 3)    cam_from_world translations (same units as input)
      info: dict with iteration count, initial/final cost, residual stats
    """
    N = len(frames)
    if N == 0:
        raise ValueError("refine_trajectory_ba: no frames provided")

    # Each pose: 3 rotvec + 3 translation = 6 params
    # Initial parameter vector: [rotvec_0, t_0, rotvec_1, t_1, ..., rotvec_{N-1}, t_{N-1}]
    x0 = np.zeros(6 * N)
    for i, fr in enumerate(frames):
        x0[6*i:6*i+3] = _matrix_to_rotvec(fr.pose_init_R)
        x0[6*i+3:6*i+6] = fr.pose_init_t

    # Pre-collect inliers + offsets for efficient batched residual computation
    inlier_offsets = [0]
    for fr in frames:
        inlier_offsets.append(inlier_offsets[-1] + len(fr.keypoints))
    total_inliers = inlier_offsets[-1]
    if verbose:
        print(f"  BA: {N} frames, {total_inliers} total inlier 2D-3D matches")
        print(f"      params: {6*N}, residuals: {2*total_inliers}")

    fx = camera["fx"]; fy = camera["fy"]; cx = camera["cx"]; cy = camera["cy"]
    k1 = camera["k1"]; k2 = camera["k2"]; k3 = camera["k3"]; k4 = camera["k4"]

    def residuals(x: np.ndarray) -> np.ndarray:
        out = np.empty(2 * total_inliers)
        for i, fr in enumerate(frames):
            R = _rotvec_to_matrix(x[6*i:6*i+3])
            t = x[6*i+3:6*i+6]
            Xc = (R @ fr.points3D.T).T + t
            uv = _project_opencv_fisheye(Xc, fx, fy, cx, cy, k1, k2, k3, k4)
            r = (uv - fr.keypoints).reshape(-1)
            # Huber down-weighting: r_i_eff = sign(r) * sqrt(2δ|r| - δ²) for |r| > δ
            ar = np.abs(r)
            mask = ar > huber_pixel
            r_eff = r.copy()
            r_eff[mask] = np.sign(r[mask]) * np.sqrt(
                np.maximum(2 * huber_pixel * ar[mask] - huber_pixel**2, 0.0)
            )
            out[2*inlier_offsets[i]:2*inlier_offsets[i+1]] = r_eff
        return out

    # Sparsity pattern: residual j (frame f_j) only depends on params of frame f_j.
    # Each frame contributes 2*Ni residuals × 6 params, all 1s.
    rows = 2 * total_inliers
    cols = 6 * N
    sp = lil_matrix((rows, cols), dtype=np.int8)
    for i, fr in enumerate(frames):
        r0 = 2 * inlier_offsets[i]
        r1 = 2 * inlier_offsets[i + 1]
        sp[r0:r1, 6*i:6*i+6] = 1
    sp = sp.tocsr()

    if verbose:
        r0 = residuals(x0)
        print(f"      initial RMS pixel error: {np.sqrt((r0**2).mean()):.3f} px, "
              f"max: {np.abs(r0).max():.1f} px")

    result = least_squares(
        residuals, x0,
        jac_sparsity=sp,
        method="trf",
        loss="linear",   # Huber already applied inside residuals
        x_scale="jac",
        max_nfev=max_nfev,
        verbose=2 if verbose else 0,
    )

    rf = residuals(result.x)
    if verbose:
        print(f"      final   RMS pixel error: {np.sqrt((rf**2).mean()):.3f} px, "
              f"max: {np.abs(rf).max():.1f} px")
        print(f"      LM iters: {result.nfev}, success: {result.success}")

    R_out = np.empty((N, 3, 3))
    t_out = np.empty((N, 3))
    for i in range(N):
        R_out[i] = _rotvec_to_matrix(result.x[6*i:6*i+3])
        t_out[i] = result.x[6*i+3:6*i+6]

    info = {
        "n_frames": N,
        "n_inliers_total": total_inliers,
        "rms_initial_px": float(np.sqrt((residuals(x0)**2).mean())),
        "rms_final_px": float(np.sqrt((rf**2).mean())),
        "nfev": int(result.nfev),
        "success": bool(result.success),
        "huber_pixel": huber_pixel,
    }
    return R_out, t_out, info


def cam_from_world_to_world_pos(R_cw: np.ndarray, t_cw: np.ndarray) -> np.ndarray:
    """cam_from_world → world position of camera origin: -R_cw.T @ t_cw."""
    if R_cw.ndim == 3:
        return np.einsum("nij,nj->ni", -R_cw.transpose(0, 2, 1), t_cw)
    return -R_cw.T @ t_cw
