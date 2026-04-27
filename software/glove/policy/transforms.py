"""Pose + proprio transforms for the ACT policy.

Key operations:
- Relativize a pose sequence to the body frame at a reference timestep
  (ACT observes + acts in obs-time-relative coords, not absolute world).
- Convert rotation matrices ↔ 6D continuous representation (Zhou et al. 2019),
  which is what the network regresses to avoid quaternion discontinuity.
- Normalize raw Hall / ToF to model-friendly ranges using dataset stats.

All functions are pure torch (no numpy) so they compose into the DataLoader's
transform chain and run on GPU when the batch is on device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ---------- Rotation conversions ----------

def quat_wxyz_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) [w, x, y, z] → (..., 3, 3) rotation matrix. Handles unnormalized quats."""
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-9)
    w, x, y, z = q.unbind(dim=-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrix → (..., 6) continuous rotation representation.

    Keeps the first two columns of R and flattens them. At decode time these
    are Gram-Schmidt-orthonormalized back to a valid rotation.
    """
    # Flatten column-major as [r1x, r1y, r1z, r2x, r2y, r2z]. A plain reshape
    # of R[..., :, :2] interleaves rows and makes even identity degenerate.
    return R[..., :, :2].transpose(-1, -2).reshape(*R.shape[:-2], 6)


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) → (..., 3, 3) via Gram-Schmidt. Zhou et al. (2019)."""
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-9)
    a2_proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = a2 - a2_proj
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-9)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def pose_to_matrix(pos: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """(..., 3) + (..., 4) → (..., 4, 4) homogeneous transform."""
    *batch, _ = pos.shape
    T = torch.zeros(*batch, 4, 4, dtype=pos.dtype, device=pos.device)
    T[..., :3, :3] = quat_wxyz_to_matrix(quat_wxyz)
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T


def invert_pose(T: torch.Tensor) -> torch.Tensor:
    """(..., 4, 4) rigid-transform inverse (R^T, -R^T @ t). Cheaper than torch.linalg.inv."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    inv = torch.zeros_like(T)
    inv[..., :3, :3] = R.transpose(-1, -2)
    inv[..., :3, 3] = -torch.einsum("...ij,...j->...i", R.transpose(-1, -2), t)
    inv[..., 3, 3] = 1.0
    return inv


# ---------- Relative pose / action decomposition ----------

def relativize_poses(
    poses_world: torch.Tensor,  # (..., N, 4, 4) world-frame poses for a timeline
    ref_idx: int = -1,          # index of the reference pose within each sequence
) -> torch.Tensor:
    """Express a pose sequence in the frame of poses[..., ref_idx, :, :].

    Returns (..., N, 4, 4) where the reference pose becomes identity.
    """
    T_ref_world = poses_world[..., ref_idx, :, :]
    T_world_ref = invert_pose(T_ref_world).unsqueeze(-3)  # broadcast over N
    return T_world_ref @ poses_world


def pose_matrix_to_rel_action(T_rel: torch.Tensor) -> torch.Tensor:
    """(..., 4, 4) → (..., 9) = [rel_pos (3), rot6d (6)]."""
    pos = T_rel[..., :3, 3]
    d6 = matrix_to_6d(T_rel[..., :3, :3])
    return torch.cat([pos, d6], dim=-1)


def rel_action_to_pose_matrix(action_9d: torch.Tensor) -> torch.Tensor:
    """(..., 9) [rel_pos, rot6d] → (..., 4, 4) rigid transform."""
    *batch, _ = action_9d.shape
    pos = action_9d[..., :3]
    R = rot6d_to_matrix(action_9d[..., 3:9])
    T = torch.zeros(*batch, 4, 4, dtype=action_9d.dtype, device=action_9d.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T


# ---------- Proprio / ToF normalization ----------

@dataclass
class NormStats:
    """Per-channel normalization for Hall + ToF. Computed once from training data."""

    hall_mean: np.ndarray   # (12,)
    hall_std: np.ndarray    # (12,)
    tof_mean: np.ndarray    # (2,)
    tof_std: np.ndarray     # (2,)
    pose_pos_scale: float   # single scalar — scales rel_pos to ~[-1, 1]

    def to_dict(self) -> dict:
        return {
            "hall_mean": self.hall_mean.tolist(),
            "hall_std": self.hall_std.tolist(),
            "tof_mean": self.tof_mean.tolist(),
            "tof_std": self.tof_std.tolist(),
            "pose_pos_scale": float(self.pose_pos_scale),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(
            hall_mean=np.array(d["hall_mean"], dtype=np.float32),
            hall_std=np.array(d["hall_std"], dtype=np.float32),
            tof_mean=np.array(d["tof_mean"], dtype=np.float32),
            tof_std=np.array(d["tof_std"], dtype=np.float32),
            pose_pos_scale=float(d["pose_pos_scale"]),
        )


def normalize_hall(hall_raw: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """(..., 12) int16 → (..., 12) float32 zero-mean unit-variance."""
    m = torch.as_tensor(stats.hall_mean, dtype=torch.float32, device=hall_raw.device)
    s = torch.as_tensor(stats.hall_std, dtype=torch.float32, device=hall_raw.device)
    return (hall_raw.to(torch.float32) - m) / torch.clamp(s, min=1e-3)


def normalize_tof(tof_mm: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """(..., 2) int16 (−1 = sentinel) → (..., 2) normalized float.

    Sentinel -1 samples are replaced per-channel with the channel's own mean so
    they normalize to ~0 (neutral). Handles disconnected sensors gracefully.
    """
    tof = tof_mm.to(torch.float32).clone()
    m = torch.as_tensor(stats.tof_mean, dtype=torch.float32, device=tof.device)
    s = torch.as_tensor(stats.tof_std, dtype=torch.float32, device=tof.device)
    # Broadcast m across the leading dims so masked fills get the right per-channel value.
    mask = tof < 0
    if mask.any():
        tof = torch.where(mask, m.expand_as(tof), tof)
    return (tof - m) / torch.clamp(s, min=1.0)


def scale_rel_pos(pos: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """Divide relative positions (meters) by the dataset's empirical scale."""
    return pos / max(stats.pose_pos_scale, 1e-3)


def unscale_rel_pos(pos: torch.Tensor, stats: NormStats) -> torch.Tensor:
    """Inverse of scale_rel_pos."""
    return pos * stats.pose_pos_scale
