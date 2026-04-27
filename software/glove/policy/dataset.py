"""PyTorch Dataset for ACT training on glove sessions.

Consumes the new `aligned/episode.parquet` schema:
- pose_x/y/z/qw/qx/qy/qz     — MTCP body pose in world frame
- hall_{0..3}_{bx,by,bz}     — raw Hall channels (12 int16)
- tof_top_mm, tof_bottom_mm  — ToF distance (int16; -1 = no reading)
- is_lost, is_keyframe       — SLAM state flags

For each sampled (session, t) pair this produces:
- images   : (T, 3, 224, 224) — obs_horizon past frames, ImageNet-normalized
- proprio  : (T, obs_dim=23)  — [rel_pos(3), rot6d(6), hall(12), tof(2)] per obs frame
- actions  : (H, act_dim=21)  — [rel_pos(3), rot6d(6), hall(12)] per future frame

By default, action poses are absolute future poses relativized to the
last-observed frame. With `action_pose_mode="delta"`, action poses are
per-frame SE(3) increments instead; `actions_abs` still carries the absolute
relative targets for physical-unit eval.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import Dataset

from glove.policy.augment import build_train_augment
from glove.policy.mask import load_mask
from glove.policy.transforms import (
    NormStats,
    invert_pose,
    matrix_to_6d,
    normalize_hall,
    normalize_tof,
    pose_to_matrix,
    scale_rel_pos,
)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class ACTDatasetConfig:
    """Top-level dataset config. Consumed by the DataLoader + training."""

    sessions_root: Path = Path("data/sessions")
    task: str = "pick_place"
    include: tuple[str, ...] = ()       # session-name glob include (empty = all)
    exclude: tuple[str, ...] = ()       # session-name glob exclude
    obs_horizon: int = 2
    action_horizon: int = 100
    image_size: int = 224
    val_fraction: float = 0.15
    val_seed: int = 42


def _session_iter(cfg: ACTDatasetConfig) -> list[Path]:
    """Enumerate demo-session directories matching the task + include/exclude filters."""
    import fnmatch
    root = Path(cfg.sessions_root).resolve()
    out: list[Path] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not d.name.startswith("demo_"):
            continue
        if not (d / "meta.yaml").exists() or not (d / "aligned" / "episode.parquet").exists():
            continue
        meta = yaml.safe_load((d / "meta.yaml").read_text())
        if meta.get("task") != cfg.task:
            continue
        if cfg.include and not any(fnmatch.fnmatch(d.name, p) for p in cfg.include):
            continue
        if cfg.exclude and any(fnmatch.fnmatch(d.name, p) for p in cfg.exclude):
            continue
        out.append(d)
    return out


def train_val_split(sessions: list[Path], val_fraction: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Split sessions (not frames) into train/val — avoids leakage within a single demo."""
    rng = random.Random(seed)
    shuffled = list(sessions)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return sorted(train), sorted(val)


class _SessionCache:
    """Per-session memoized state (parquet + cv2 handle + pose matrices)."""

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        tbl = pq.read_table(session_dir / "aligned" / "episode.parquet").to_pandas()
        self.n_frames = len(tbl)

        # Per-frame pose as a stacked (N, 4, 4) matrix in torch for efficient relativization
        pos = torch.tensor(tbl[["pose_x", "pose_y", "pose_z"]].to_numpy(), dtype=torch.float32)
        quat = torch.tensor(tbl[["pose_qw", "pose_qx", "pose_qy", "pose_qz"]].to_numpy(), dtype=torch.float32)
        self.poses = pose_to_matrix(pos, quat)       # (N, 4, 4)

        hall_cols = [f"hall_{i}_{ax}" for i in range(4) for ax in ("bx", "by", "bz")]
        self.hall = torch.tensor(tbl[hall_cols].to_numpy(dtype=np.int16), dtype=torch.int16)  # (N, 12)
        # ToF is int32 on new data, int16 on legacy data — int32 covers both.
        self.tof = torch.tensor(tbl[["tof_top_mm", "tof_bottom_mm"]].to_numpy(dtype=np.int32), dtype=torch.int32)
        self.is_lost = torch.tensor(tbl["is_lost"].to_numpy(), dtype=torch.bool)

        self._cap: cv2.VideoCapture | None = None

    @property
    def cap(self) -> cv2.VideoCapture:
        if self._cap is None:
            # Prefer the training-preprocessed 224×224 MP4 if present; the load-time
            # resize becomes a no-op and the file is ~15× smaller for remote upload.
            pre = self.session_dir / "raw" / "raw_video_224.mp4"
            full = self.session_dir / "raw" / "raw_video.mp4"
            self._cap = cv2.VideoCapture(str(pre if pre.exists() else full))
        return self._cap

    def read_frame(self, n: int, image_size: int) -> torch.Tensor:
        """Read a single video frame, resize to (image_size, image_size), ImageNet-normalize."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {n} of {self.session_dir.name}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0   # (3, H, W)
        return (x - IMAGENET_MEAN) / IMAGENET_STD

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def compute_norm_stats(sessions: list[Path]) -> NormStats:
    """One pass over all sessions to compute Hall/ToF mean/std + pose scale."""
    hall_accum = []
    tof_accum = []
    rel_pos_norms = []

    for sess in sessions:
        tbl = pq.read_table(sess / "aligned" / "episode.parquet").to_pandas()
        hall_cols = [f"hall_{i}_{ax}" for i in range(4) for ax in ("bx", "by", "bz")]
        hall = tbl[hall_cols].to_numpy(dtype=np.float32)
        hall_accum.append(hall)

        tof = tbl[["tof_top_mm", "tof_bottom_mm"]].to_numpy(dtype=np.float32)
        tof_valid = np.where(tof < 0, np.nan, tof)
        tof_accum.append(tof_valid)

        # Sample rel_pos norms: for a random subset of reference frames, compute
        # |rel_pos| to every other frame. Cheap but informative scale estimate.
        pos = tbl[["pose_x", "pose_y", "pose_z"]].to_numpy()
        n = len(pos)
        if n >= 2:
            ref_indices = np.random.choice(n, size=min(16, n), replace=False)
            for idx in ref_indices:
                deltas = np.linalg.norm(pos - pos[idx], axis=1)
                rel_pos_norms.append(deltas)

    hall_all = np.concatenate(hall_accum, axis=0)
    tof_all = np.concatenate(tof_accum, axis=0)
    rel_pos_all = np.concatenate(rel_pos_norms) if rel_pos_norms else np.array([0.1])

    hall_std = hall_all.std(axis=0)
    # Sanity gate: if a Hall channel's std across the whole dataset is < 5 LSB,
    # something upstream is broken. We hit this when align.py interpolated
    # frame_wall (in t_rel space) against proprio.wall_time (in epoch space):
    # out-of-range np.interp returned constants, which became int16 zeros.
    # Training proceeded fine on zero targets (L1 ≈ 0 looks great in metrics),
    # but the policy never learned finger control. Hall sensors at rest have
    # std ≫ 1 LSB even from sensor noise alone, so this floor catches the bug
    # before another round-trip through training.
    if hall_std.min() < 5.0:
        bad = [i for i, s in enumerate(hall_std) if s < 5.0]
        raise RuntimeError(
            f"compute_norm_stats: hall std too small on channels {bad}: "
            f"std={hall_std} — likely the all-zero-hall align bug. Re-run "
            f"`glove.process align <session_dir>` on every session before "
            f"training (the current align.py is fixed)."
        )

    # Some ToF channels may be entirely disconnected (all -1 → NaN stats).
    # Fall back to neutral values so normalization doesn't propagate NaN.
    tof_mean = np.nanmean(tof_all, axis=0)
    tof_std = np.nanstd(tof_all, axis=0)
    tof_mean = np.where(np.isnan(tof_mean), 500.0, tof_mean).astype(np.float32)
    tof_std = np.where(np.isnan(tof_std) | (tof_std < 1.0), 200.0, tof_std).astype(np.float32)

    return NormStats(
        hall_mean=hall_all.mean(axis=0).astype(np.float32),
        hall_std=hall_std.astype(np.float32),
        tof_mean=tof_mean,
        tof_std=tof_std,
        pose_pos_scale=float(np.quantile(rel_pos_all, 0.99)),
    )


class ACTDataset(Dataset):
    """Samples (obs window, action chunk) pairs from glove demo sessions.

    Optional per-pixel static mask + per-frame augmentation:
    - `mask_path`: PNG produced by `glove.policy build-mask` — the pixel positions
      that are always the glove body / mount. Applied in normalized-image space
      (zeros out those pixels ≈ replace with ImageNet mean).
    - `augment`: enable ACT-plus-plus-style RandomRotation + RandomResizedCrop +
      ColorJitter. **Only** pass augment=True for the training dataset, not val.
    """

    def __init__(
        self,
        sessions: list[Path],
        stats: NormStats,
        obs_horizon: int = 2,
        action_horizon: int = 100,
        image_size: int = 224,
        mask_path: Path | None = None,
        augment: bool = False,
        action_pose_mode: str = "relative",
    ):
        if action_pose_mode not in ("relative", "delta"):
            raise ValueError(f"Unsupported action_pose_mode={action_pose_mode!r}")
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.stats = stats
        self.action_pose_mode = action_pose_mode

        # Static mask (1 = static pixel to zero out). Applied BEFORE augmentation
        # so the masked region (glove body in canonical coords) moves together
        # with the rest of the frame when rotation/crop applies.
        if mask_path is not None and Path(mask_path).exists():
            m = load_mask(Path(mask_path)).astype(np.float32)    # (H, W) 0/1
            self._mask = torch.from_numpy(m).unsqueeze(0)         # (1, H, W) for broadcast
        else:
            self._mask = None

        self._augment = build_train_augment(image_size=image_size) if augment else None

        self._caches: dict[Path, _SessionCache] = {}
        self._index: list[tuple[Path, int]] = []   # (session_dir, start_frame_t)

        for sess in sessions:
            cache = _SessionCache(sess)
            self._caches[sess] = cache
            # Valid start frames: last obs at t — need t - (obs_horizon-1) >= 0
            # and obs window itself not SLAM-lost. Action window may extend
            # past episode end; we zero-pad and mark those positions in
            # `is_pad` so the training loss ignores them.
            lo = obs_horizon - 1
            for t in range(lo, cache.n_frames):
                obs_slice = cache.is_lost[t - obs_horizon + 1 : t + 1]
                if bool(obs_slice.any()):
                    continue
                self._index.append((sess, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        sess, t = self._index[idx]
        cache = self._caches[sess]
        T = self.obs_horizon
        H = self.action_horizon

        # Reference pose = pose at observation time t
        T_ref_world = cache.poses[t]                  # (4, 4)
        T_world_ref = invert_pose(T_ref_world)        # (4, 4)

        # Observation frames: [t-T+1 .. t]
        obs_start = t - T + 1
        images = torch.stack([
            cache.read_frame(obs_start + k, self.image_size) for k in range(T)
        ], dim=0)                                                  # (T, 3, H, W)

        # Static mask first (in canonical coords), then augmentation on top.
        if self._mask is not None:
            images = images * (1.0 - self._mask.unsqueeze(0))      # broadcast over T

        if self._augment is not None:
            # torchvision.v2 accepts batched tensors and samples one set of
            # aug params per call — we want the same rotation/crop across the
            # obs window (so the "temporal" signal across the T frames stays
            # consistent), so apply once to the whole stack.
            images = self._augment(images)

        obs_poses_world = cache.poses[obs_start : obs_start + T]   # (T, 4, 4)
        obs_rel = T_world_ref.unsqueeze(0) @ obs_poses_world        # (T, 4, 4)
        obs_rel_pos = scale_rel_pos(obs_rel[:, :3, 3], self.stats) # (T, 3)
        obs_rot6d = matrix_to_6d(obs_rel[:, :3, :3])                # (T, 6)

        obs_hall = normalize_hall(cache.hall[obs_start : obs_start + T], self.stats)   # (T, 12)
        obs_tof = normalize_tof(cache.tof[obs_start : obs_start + T], self.stats)      # (T, 2)

        proprio = torch.cat([obs_rel_pos, obs_rot6d, obs_hall, obs_tof], dim=-1)  # (T, 23)

        # Action frames: [t .. t+H-1]. When t+H extends past episode, take
        # what we can and zero-pad the rest; `is_pad` marks padded + lost frames.
        act_end = min(t + H, cache.n_frames)
        act_real = act_end - t

        act_poses_world = cache.poses[t:act_end]                     # (act_real, 4, 4)
        act_rel = T_world_ref.unsqueeze(0) @ act_poses_world          # (act_real, 4, 4)
        act_rel_pos = scale_rel_pos(act_rel[:, :3, 3], self.stats)   # (act_real, 3)
        act_rot6d = matrix_to_6d(act_rel[:, :3, :3])                  # (act_real, 6)
        act_hall = normalize_hall(cache.hall[t:act_end], self.stats)  # (act_real, 12)
        actions_abs_real = torch.cat([act_rel_pos, act_rot6d, act_hall], dim=-1)  # (act_real, 21)

        if self.action_pose_mode == "relative":
            actions_real = actions_abs_real
        else:
            # Step 0 is identity (current pose); later steps are local SE(3)
            # increments T_prev_current. Hall targets stay absolute normalized
            # because the policy still needs the future hand shape, not Hall velocity.
            prev_poses_world = torch.cat([
                T_ref_world.view(1, 4, 4),
                act_poses_world[:-1],
            ], dim=0)
            delta_rel = invert_pose(prev_poses_world) @ act_poses_world
            delta_pos = scale_rel_pos(delta_rel[:, :3, 3], self.stats)
            delta_rot6d = matrix_to_6d(delta_rel[:, :3, :3])
            actions_real = torch.cat([delta_pos, delta_rot6d, act_hall], dim=-1)

        if act_real < H:
            pad = torch.zeros(H - act_real, actions_real.shape[-1], dtype=actions_real.dtype)
            actions = torch.cat([actions_real, pad], dim=0)
            actions_abs = torch.cat([actions_abs_real, pad], dim=0)
        else:
            actions = actions_real
            actions_abs = actions_abs_real

        is_pad = torch.zeros(H, dtype=torch.bool)
        is_pad[act_real:] = True
        if act_real > 0:
            is_pad[:act_real] |= cache.is_lost[t:act_end]

        return {
            "images": images.float(),
            "proprio": proprio.float(),
            "actions": actions.float(),
            "actions_abs": actions_abs.float(),
            "is_pad": is_pad,
        }

    def close(self):
        for c in self._caches.values():
            c.close()


def save_norm_stats(stats: NormStats, path: Path) -> None:
    path.write_text(json.dumps(stats.to_dict(), indent=2))


def load_norm_stats(path: Path) -> NormStats:
    return NormStats.from_dict(json.loads(path.read_text()))
