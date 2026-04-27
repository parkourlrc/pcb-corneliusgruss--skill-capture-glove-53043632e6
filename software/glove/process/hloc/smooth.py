"""Pose-graph batch smoother for HLoc per-frame PnP measurements.

Positions and rotations are smoothed in different ways because they live on
different manifolds. Both are noise-rejection passes that take noisy per-frame
measurements and return a smooth contiguous-frame trajectory.

POSITION smoothing — Levenberg-Marquardt batch optimization:

    sum over frames i of:
        ||pos_i - pos_meas_i||² / sigma_meas_i²                       (measurement)
    sum over consecutive triplets (i-1, i, i+1):
        ρ(||pos_{i+1} - 2 pos_i + pos_{i-1}||² / sigma_smooth_pos²)   (constant-velocity)

Where ρ is a Huber loss so single-frame outliers can't drag the trajectory.
Per-frame measurement covariance comes from PnP inlier count.

ROTATION smoothing — manifold-aware moving average (NOT LM on rotvecs):

    1. Hemisphere-consistent quaternion normalization (flip sign so adjacent
       quats lie in the same hemisphere — avoids the +q ↔ −q ambiguity).
    2. Centered windowed mean using `scipy.spatial.transform.Rotation.mean`
       (chordal L2 mean over the SO(3) manifold — closed-form via SVD).

Earlier versions tried smoothing rotvecs in the same LM optimizer as positions.
That broke at the π wrap-around: adjacent frames straddling π become rotvecs
of opposite sign with magnitude ~2π, LM can't reconcile, and downstream
`from_rotvec(garbage).as_quat()` produced NaNs. The manifold-aware pass is the
standard fix.

This is Phase 1 — pure visual + temporal priors, no IMU. Phase 2 will add
GPMF IMU preintegration factors between consecutive frames as additional
constraints on the same factor graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


@dataclass
class HLocPoseObservation:
    """A single HLoc PnP result for one demo frame."""
    frame_idx: int           # frame number in the demo MP4
    pos_world: np.ndarray    # (3,) camera position in COLMAP world (post-scale, in METERS)
    quat_xyzw: np.ndarray    # (4,) camera orientation (R_world_cam)
    n_inliers: int           # PnP inlier count — proxy for measurement confidence


@dataclass
class SmoothingConfig:
    # Per-axis 1-sigma for HLoc position measurement when n_inliers is "typical"
    # (calibrated empirically from the smoke-test data — refine as we collect more).
    pos_meas_sigma_m: float = 0.020
    # Min/max sigma — clamp so a single zero-inlier frame doesn't dominate.
    pos_meas_sigma_min_m: float = 0.005
    pos_meas_sigma_max_m: float = 0.500
    rot_meas_sigma_rad: float = np.deg2rad(2.0)

    # Smoothness priors. Constant-velocity model: x_{i-1} - 2 x_i + x_{i+1} ~ N(0, sigma).
    # Tighter prior = smoother trajectory. Too tight and we wash out real motion;
    # too loose and high-frequency noise leaks through. 5mm² per (1/60)s² per
    # (1/60)s² is empirically reasonable for 60fps wrist motion.
    pos_smooth_sigma_m: float = 0.005
    rot_smooth_sigma_rad: float = np.deg2rad(0.5)

    # Robustification — Huber threshold (in standard deviations).
    huber_delta: float = 1.5

    # Max LM iterations.
    max_nfev: int = 200

    # Final Gaussian low-pass on the LM-smoothed positions. Sigma is in frames
    # at the input frame rate (so at stride-2 / 60 fps recording → 60 fps input,
    # σ=3 frames = 50 ms = ~3.2 Hz cutoff).
    #
    # Why we need this on top of LM smoothing:
    # HLoc per-frame PnP solutions are *independent* — adjacent frames see
    # nearly the same scene but solve PnP against slightly different inlier
    # subsets, producing 5–30 Hz noise that the const-velocity LM prior alone
    # leaves at ~3 mm RMS. Real hand motion on these demos lives below 2 Hz
    # (0.3 m/s pickup, direction changes on 0.5–1 s timescales), so a 3 Hz
    # low-pass cleanly separates noise from signal.
    #
    # Empirically, σ=3 frames takes a noisy demo batch from straightness
    # 0.57–0.78 / hifreq 2.6–4.9 mm to straightness 0.91–0.97 / hifreq
    # 0.87–1.17 mm while preserving trajectory extent within 1–2 %. ORB-SLAM3's
    # local Bundle Adjustment achieves similar averaging via temporal coupling
    # between keyframes; we do it explicitly with a Gaussian filter.
    #
    # σ=0 disables the pass.
    final_gaussian_sigma_frames: float = 3.0


def _inlier_to_sigma(n_inliers: int, cfg: SmoothingConfig) -> float:
    """Map PnP inlier count → measurement sigma (meters).

    Heuristic: confidence scales as 1/sqrt(N_inliers); calibrate so that
    typical-confidence frames hit `pos_meas_sigma_m`. Floor and ceiling clamp
    catastrophes from dominating.
    """
    typical = 50  # the "median" inlier count we observed empirically
    sigma = cfg.pos_meas_sigma_m * np.sqrt(typical / max(n_inliers, 1))
    return float(np.clip(sigma, cfg.pos_meas_sigma_min_m, cfg.pos_meas_sigma_max_m))


def _huber_residual(r: np.ndarray, delta: float) -> np.ndarray:
    """Huber re-weighting: scale residuals so |r| > delta are linearly penalized.

    least_squares can do this internally via loss='huber' but applying it inline
    lets us mix Huber-robustified smoothness terms with L2 measurement terms.
    """
    abs_r = np.abs(r)
    out = r.copy()
    mask = abs_r > delta
    out[mask] = np.sign(r[mask]) * np.sqrt(2 * delta * abs_r[mask] - delta**2)
    return out


def _hemisphere_consistent(quats_xyzw: np.ndarray) -> np.ndarray:
    """Flip quat sign so adjacent quats lie in the same hemisphere.

    Quaternions q and -q represent the same rotation, but mixing them across
    frames produces large angular jumps in the chordal metric and breaks both
    interpolation (slerp) and averaging (Rotation.mean). Walk the sequence and
    flip any quat whose dot product with the previous is negative.

    Operates on `(N, 4)` xyzw input.
    """
    out = quats_xyzw.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] = -out[i]
    return out


def _smooth_positions_lm(
    pos_meas: np.ndarray,           # (N, 3) — already gap-filled by interpolation
    pos_sigma: np.ndarray,          # (N,) — per-frame measurement sigma
    cfg: SmoothingConfig,
) -> tuple[np.ndarray, dict]:
    """LM batch optimization over the position coordinates."""
    N = len(pos_meas)
    x0 = pos_meas.flatten()

    def residuals(x: np.ndarray) -> np.ndarray:
        pos = x.reshape(N, 3)
        r_meas = ((pos - pos_meas) / pos_sigma[:, None]).flatten()
        d2 = pos[:-2] - 2 * pos[1:-1] + pos[2:]
        r_smooth = _huber_residual(
            (d2 / cfg.pos_smooth_sigma_m).flatten(),
            cfg.huber_delta,
        )
        return np.concatenate([r_meas, r_smooth])

    sol = least_squares(residuals, x0, method="lm", max_nfev=cfg.max_nfev)
    info = {
        "lm_iterations": int(sol.nfev),
        "lm_cost": float(sol.cost),
        "lm_status": int(sol.status),
    }
    return sol.x.reshape(N, 3), info


def _smooth_rotations_window(
    quats_xyzw: np.ndarray,       # (N, 4) — gap-filled, hemisphere-consistent
    valid: np.ndarray,            # (N,) — True where the input was a real measurement
    window: int = 5,
) -> np.ndarray:
    """Manifold-aware rotation smoothing: centered windowed mean using `Rotation.mean`.

    Rotation.mean computes the chordal L2 mean (closed-form via SVD on the
    averaged quaternion outer-product). It's the rotation analog of an arithmetic
    mean and handles ±q ambiguity correctly when fed hemisphere-consistent inputs.

    Real measurements get full weight; interpolation-filled frames get a small
    weight so they bias toward the LM-smoothed positions but don't override the
    true measurements.
    """
    N = len(quats_xyzw)
    if N < window:
        return quats_xyzw
    half = window // 2
    weights = np.where(valid, 1.0, 0.05)

    out = quats_xyzw.copy()
    for i in range(N):
        lo = max(0, i - half)
        hi = min(N, i + half + 1)
        # Only proceed if there's at least one true measurement in window
        if not valid[lo:hi].any():
            continue
        rot = Rotation.from_quat(quats_xyzw[lo:hi])
        try:
            mean_rot = rot.mean(weights=weights[lo:hi])
            out[i] = mean_rot.as_quat()
        except Exception:
            # If SVD fails (degenerate window), keep the input
            pass

    # After windowed averaging the hemispheres may have drifted; re-enforce.
    return _hemisphere_consistent(out)


def smooth_trajectory(
    obs: list[HLocPoseObservation],
    cfg: Optional[SmoothingConfig] = None,
    rotation_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Smooth a sequence of HLoc pose observations.

    Returns (positions_N×3 in meters, quats_xyzw_N×4 [R_world_cam], info dict).

    Frame indices in `obs` may be sparse — gaps are filled by linear interp /
    slerp before smoothing so the output covers `[obs[0].frame_idx, obs[-1].frame_idx]`
    contiguously.
    """
    cfg = cfg or SmoothingConfig()
    if len(obs) < 3:
        raise ValueError(f"Need at least 3 observations to smooth, got {len(obs)}")

    obs = sorted(obs, key=lambda o: o.frame_idx)
    fmin, fmax = obs[0].frame_idx, obs[-1].frame_idx
    N = fmax - fmin + 1

    pos_meas = np.full((N, 3), np.nan)
    quat_meas = np.full((N, 4), np.nan)
    inliers = np.full(N, 0, dtype=int)
    for o in obs:
        i = o.frame_idx - fmin
        pos_meas[i] = o.pos_world
        # Normalize input quat just in case
        q = np.asarray(o.quat_xyzw, dtype=np.float64)
        q = q / max(np.linalg.norm(q), 1e-12)
        quat_meas[i] = q
        inliers[i] = o.n_inliers

    valid = ~np.isnan(pos_meas[:, 0])
    idx_all = np.arange(N)

    # Fill position gaps by linear interp
    if not valid.all():
        for ax in range(3):
            pos_meas[~valid, ax] = np.interp(idx_all[~valid], idx_all[valid], pos_meas[valid, ax])

    # Fill rotation gaps via slerp on hemisphere-consistent input.
    # `Slerp` requires strictly increasing key times; adjacent quats must be
    # hemisphere-consistent or the slerp shortcut takes the long way around.
    valid_quat = quat_meas[valid]
    valid_quat = _hemisphere_consistent(valid_quat)
    quat_meas[valid] = valid_quat
    if not valid.all():
        rot_obj = Rotation.from_quat(valid_quat)
        slerp = Slerp(idx_all[valid], rot_obj)
        # Clamp to the valid range so Slerp doesn't extrapolate outside its domain.
        clamped = idx_all[~valid].clip(idx_all[valid].min(), idx_all[valid].max())
        quat_meas[~valid] = slerp(clamped).as_quat()

    # Per-frame position sigma. Filled frames get max-sigma so the prior dominates.
    pos_sigma = np.array([_inlier_to_sigma(int(inl), cfg) for inl in inliers])
    pos_sigma[~valid] = cfg.pos_meas_sigma_max_m

    # 1) Position smoothing — LM on positions only
    pos_smooth, lm_info = _smooth_positions_lm(pos_meas, pos_sigma, cfg)

    # 2) Final Gaussian low-pass to kill HLoc per-frame independence noise
    #    above ~3 Hz. See SmoothingConfig.final_gaussian_sigma_frames docstring.
    if cfg.final_gaussian_sigma_frames > 0 and N >= 3:
        from scipy.ndimage import gaussian_filter1d
        for ax in range(3):
            pos_smooth[:, ax] = gaussian_filter1d(
                pos_smooth[:, ax],
                sigma=cfg.final_gaussian_sigma_frames,
                mode="nearest",
            )

    # 3) Rotation smoothing — manifold-aware windowed mean
    quat_meas = _hemisphere_consistent(quat_meas)
    quat_smooth = _smooth_rotations_window(quat_meas, valid, window=rotation_window)

    # Sanity: no NaNs allowed in the output
    assert not np.isnan(pos_smooth).any(), "position smoothing produced NaN"
    assert not np.isnan(quat_smooth).any(), "rotation smoothing produced NaN"

    info = {
        "n_frames": N,
        "n_observations": int(valid.sum()),
        "n_filled_by_interp": int((~valid).sum()),
        "rotation_window": rotation_window,
        "frame_idx_start": int(fmin),
        "frame_idx_end": int(fmax),
        **lm_info,
    }
    return pos_smooth, quat_smooth, info
