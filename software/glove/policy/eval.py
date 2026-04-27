"""Offline evaluation for a trained ACT glove policy.

Two outputs:
1. Aggregate L1 error on the validation split (both CVAE-posterior and prior-z).
2. Qualitative viz: pick one val session + one query frame; show predicted vs GT
   action chunk overlaid on the session trajectory + Hall timelines. Saved PNG.

Use: `python -m glove.policy eval --checkpoint checkpoints/<run>/best.pt`
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from glove.policy.act import ACTConfig, ACTPolicy
from glove.policy.dataset import (
    ACTDataset,
    ACTDatasetConfig,
    _SessionCache,
    _session_iter,
    load_norm_stats,
    train_val_split,
)
from glove.policy.train import _pick_device, _stringify_paths, evaluate
from glove.policy.transforms import (
    NormStats,
    invert_pose,
    matrix_to_6d,
    normalize_hall,
    normalize_tof,
    rel_action_to_pose_matrix,
    scale_rel_pos,
    unscale_rel_pos,
)


def _reconstruct_world_poses(
    actions: torch.Tensor,          # (H, 21) predicted
    T_ref_world: torch.Tensor,      # (4, 4) reference pose in world
    stats: NormStats,
    action_pose_mode: str = "relative",
) -> torch.Tensor:
    """Convert predicted pose actions back to world-frame (H, 4, 4).

    `relative` actions are direct future poses in the obs frame. `delta`
    actions are chained SE(3) increments starting at the obs frame.
    """
    pos = unscale_rel_pos(actions[:, :3], stats)
    rot6d = actions[:, 3:9]
    T_step = rel_action_to_pose_matrix(torch.cat([pos, rot6d], dim=-1))  # (H, 4, 4)
    if action_pose_mode == "relative":
        T_rel = T_step
    elif action_pose_mode == "delta":
        chunks = []
        cur = torch.eye(4, dtype=T_step.dtype, device=T_step.device)
        for k in range(T_step.shape[0]):
            cur = cur @ T_step[k]
            chunks.append(cur.clone())
        T_rel = torch.stack(chunks, dim=0)
    else:
        raise ValueError(f"Unsupported action_pose_mode={action_pose_mode!r}")
    return T_ref_world.unsqueeze(0) @ T_rel


def _build_obs_sample(
    cache: _SessionCache,
    t: int,
    obs_horizon: int,
    stats: NormStats,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a single-item batch: (images, proprio, T_ref_world) for observation time t."""
    T = obs_horizon
    T_ref_world = cache.poses[t]
    T_world_ref = invert_pose(T_ref_world)

    obs_start = t - T + 1
    images = torch.stack([cache.read_frame(obs_start + k, image_size) for k in range(T)], dim=0)
    obs_rel = T_world_ref.unsqueeze(0) @ cache.poses[obs_start : obs_start + T]
    obs_rel_pos = scale_rel_pos(obs_rel[:, :3, 3], stats)
    obs_rot6d = matrix_to_6d(obs_rel[:, :3, :3])
    obs_hall = normalize_hall(cache.hall[obs_start : obs_start + T], stats)
    obs_tof = normalize_tof(cache.tof[obs_start : obs_start + T], stats)
    proprio = torch.cat([obs_rel_pos, obs_rot6d, obs_hall, obs_tof], dim=-1)

    return images.unsqueeze(0).float(), proprio.unsqueeze(0).float(), T_ref_world


def qualitative_viz(
    model: ACTPolicy,
    session_dir: Path,
    stats: NormStats,
    device: torch.device,
    output_png: Path,
    query_fraction: float = 0.35,
) -> None:
    """Pick a query frame ~one-third through the session, predict the chunk,
    overlay predicted vs GT on trajectory + Hall plots."""
    cache = _SessionCache(session_dir)
    n = cache.n_frames
    T = model.config.obs_horizon
    H = model.config.action_horizon

    t_query = int(np.clip(int(n * query_fraction), T - 1, max(T - 1, n - 1)))

    images, proprio, T_ref_world = _build_obs_sample(cache, t_query, T, stats, model.config.image_size)
    images = images.to(device)
    proprio = proprio.to(device)

    model.eval()
    with torch.no_grad():
        pred = model.predict(images, proprio).squeeze(0).cpu()      # (H, 21)

    # Reconstruct world-frame positions for predicted actions
    pred_world = _reconstruct_world_poses(
        pred, T_ref_world, stats, action_pose_mode=getattr(model.config, "action_pose_mode", "relative")
    )                                                                 # (H, 4, 4)
    pred_xy = pred_world[:, :3, 3].numpy()                            # (H, 3)

    # GT positions for the same future window. Near the end of short episodes,
    # GT may be shorter than H; plot the real prefix and let the model horizon
    # remain visible.
    gt_end = min(t_query + H, n)
    gt_world = cache.poses[t_query:gt_end, :3, 3].numpy()             # (<=H, 3)

    # Full session trajectory
    full_world = cache.poses[:, :3, 3].numpy()

    # Hall: predicted vs GT (un-normalized)
    pred_hall_norm = pred[:, 9:21].numpy()
    pred_hall = pred_hall_norm * np.clip(stats.hall_std, 1e-3, None) + stats.hall_mean
    gt_hall = cache.hall[t_query:gt_end].numpy().astype(np.float32)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"{session_dir.name}  ·  query frame {t_query}/{n}  ·  chunk H={H}")

    # Top-down trajectory
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_xy.plot(full_world[:, 0], full_world[:, 1], "-", color="lightgray", lw=1, label="full session")
    ax_xy.plot(gt_world[:, 0], gt_world[:, 1], "-", color="forestgreen", lw=2, label="GT future H")
    ax_xy.plot(pred_xy[:, 0], pred_xy[:, 1], "--", color="crimson", lw=1.8, label="predicted H")
    obs_pos = cache.poses[t_query, :3, 3].numpy()
    ax_xy.scatter([obs_pos[0]], [obs_pos[1]], c="orange", s=80, marker="o", label="obs", zorder=5)
    ax_xy.scatter([0], [0], c="blue", s=100, marker="*", label="ArUco origin", zorder=5)
    ax_xy.set_xlabel("x (m, world)"); ax_xy.set_ylabel("y (m, world)")
    ax_xy.set_title("Trajectory (top-down)")
    ax_xy.set_aspect("equal")
    ax_xy.legend(loc="best", fontsize=8)
    ax_xy.grid(alpha=0.3)

    # Hall chunk comparison (show 4 sensors × 3 channels = 12, pick Bx of each sensor for clarity)
    ax_hall = fig.add_subplot(1, 2, 2)
    h_steps = np.arange(H) / 60.0
    gt_steps = np.arange(len(gt_hall)) / 60.0
    sensor_labels = ["idx_mcp", "idx_pip", "3f_mcp", "3f_pip"]
    for i in range(4):
        c = f"C{i}"
        ax_hall.plot(gt_steps, gt_hall[:, i * 3], "-", color=c, lw=1.5, label=f"{sensor_labels[i]} Bx GT")
        ax_hall.plot(h_steps, pred_hall[:, i * 3], "--", color=c, lw=1.2, alpha=0.8, label=f"{sensor_labels[i]} Bx pred")
    ax_hall.set_xlabel("t (s, into chunk)")
    ax_hall.set_ylabel("Bx (LSB)")
    ax_hall.set_title("Hall Bx — GT vs predicted chunk")
    ax_hall.legend(loc="best", fontsize=7, ncol=2)
    ax_hall.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    cache.close()


def evaluate_session_ensembled(
    model: ACTPolicy,
    session_dir: Path,
    stats: NormStats,
    device: torch.device,
    query_stride: int = 1,
    m: float = 0.01,
) -> dict:
    """Temporal action ensembling per ACT paper (Zhao et al. 2023, §4).

    For each query frame `t_query` in the session (every `query_stride` frames),
    the model predicts an action chunk of length H. Each chunk position k
    yields a prediction for world-time `t_query + k`. Predictions for the
    same world-time τ from different queries are averaged with weights
    `exp(-m·k)` (smaller k = chunk-near-time-τ = lower variance).

    Returns per-frame world-pose L1 (vs cache.poses ground truth) for the
    ensembled predictions. Position only — rotation ensembling needs SO(3)
    averaging which we skip for eval simplicity.
    """
    cache = _SessionCache(session_dir)
    n = cache.n_frames
    H = model.config.action_horizon
    T_obs = model.config.obs_horizon
    image_size = model.config.image_size
    pose_mode = getattr(model.config, "action_pose_mode", "relative")

    # buffer[t_world] = list of (k_offset, pred_world_xyz_3vec)
    buffer: list[list[tuple[int, torch.Tensor]]] = [[] for _ in range(n)]

    model.eval()
    n_queries = 0
    for t_query in range(T_obs - 1, n, query_stride):
        images, proprio, T_ref_world = _build_obs_sample(
            cache, t_query, T_obs, stats, image_size
        )
        with torch.no_grad():
            chunk = model.predict(images.to(device), proprio.to(device)).squeeze(0).cpu()
        # Reconstruct chunk in world frame (handles relative or delta mode)
        T_pred_world = _reconstruct_world_poses(chunk, T_ref_world, stats, action_pose_mode=pose_mode)
        # T_pred_world: (H, 4, 4). Take xyz translation only.
        for k in range(H):
            t_world = t_query + k
            if t_world < n:
                buffer[t_world].append((k, T_pred_world[k, :3, 3].clone()))
        n_queries += 1

    # Aggregate per-frame ensembled predictions + per-frame L1 vs GT
    err_per_frame = []
    for t in range(n):
        if not buffer[t]:
            continue
        ks = torch.tensor([b[0] for b in buffer[t]], dtype=torch.float32)
        weights = torch.exp(-m * ks)
        weights = weights / weights.sum()
        preds = torch.stack([b[1] for b in buffer[t]], dim=0)         # (M, 3)
        ensembled = (weights.unsqueeze(-1) * preds).sum(dim=0)         # (3,)
        gt = cache.poses[t, :3, 3]                                     # (3,)
        err_per_frame.append(float((ensembled - gt).norm().item()))

    cache.close()

    if not err_per_frame:
        return {}

    arr = np.array(err_per_frame)
    return {
        "n_queries": n_queries,
        "n_frames_evaluated": len(err_per_frame),
        "pos_l1_mean_m": float(arr.mean()),
        "pos_l1_median_m": float(np.median(arr)),
        "pos_l1_p95_m": float(np.percentile(arr, 95)),
        "pos_l1_max_m": float(arr.max()),
    }


def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[ACTPolicy, NormStats]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    act_cfg = ACTConfig(**ckpt["act_config"])
    model = ACTPolicy(act_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    stats = NormStats.from_dict(ckpt["norm_stats"])
    return model, stats


def run_eval(checkpoint: Path, dataset_cfg: ACTDatasetConfig, out_dir: Path | None = None) -> dict:
    """Load checkpoint, run full val-set evaluation + qualitative viz."""
    ckpt_path = Path(checkpoint)
    device = _pick_device()
    print(f"Device: {device}")

    model, stats = load_checkpoint(ckpt_path, device)
    sessions = _session_iter(dataset_cfg)
    _, val_sess = train_val_split(sessions, dataset_cfg.val_fraction, dataset_cfg.val_seed)

    if not val_sess:
        print("No validation sessions — eval aggregate skipped.")
        return {}

    val_ds = ACTDataset(val_sess, stats,
                        obs_horizon=model.config.obs_horizon,
                        action_horizon=model.config.action_horizon,
                        image_size=model.config.image_size,
                        action_pose_mode=getattr(model.config, "action_pose_mode", "relative"))
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    print(f"Evaluating on {len(val_sess)} val sessions, {len(val_ds):,} samples…")
    metrics = evaluate(model, val_loader, device)
    print(f"  L1 (CVAE-posterior z): {metrics['l1_cvae']:.5f}")
    print(f"  L1 (prior z=0 inference): {metrics['l1_prior']:.5f}")
    if "prior_pos_l1_m" in metrics:
        print(f"  Prior pose position L1: {metrics['prior_pos_l1_m'] * 1000:.1f} mm")
        print(f"  Prior pose rotation L1: {metrics['prior_rot_l1_deg']:.2f} deg")
        print(f"  Prior Hall L1: {metrics['prior_hall_l1_raw']:.1f} raw LSB")
    print(f"  KL divergence: {metrics['kl']:.5f}")

    out_dir = out_dir or ckpt_path.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.yaml").write_text(yaml.safe_dump({
        "checkpoint": str(ckpt_path),
        "dataset": _stringify_paths(asdict(dataset_cfg)),
        "val_sessions": [p.name for p in val_sess],
        "metrics": metrics,
    }, sort_keys=False))
    for sess in val_sess[:3]:
        qualitative_viz(model, sess, stats, device, out_dir / f"viz_{sess.name}.png")
        print(f"  viz → {out_dir / f'viz_{sess.name}.png'}")

    val_ds.close()
    return metrics


def cli(args) -> int:
    exclude = tuple(args.exclude) if getattr(args, "exclude", None) else ()
    if getattr(args, "exclude_file", None):
        p = Path(args.exclude_file)
        if p.exists():
            extra = tuple(line.strip() for line in p.read_text().splitlines() if line.strip())
            exclude = exclude + extra
            print(f"Excluding {len(extra)} session names from {p}")
    include: tuple[str, ...] = ()
    if getattr(args, "include_file", None):
        p = Path(args.include_file)
        if p.exists():
            include = tuple(line.strip() for line in p.read_text().splitlines() if line.strip())
            print(f"Including {len(include)} session names from {p}")
    dataset_cfg = ACTDatasetConfig(
        sessions_root=args.sessions_root,
        task=args.task,
        val_fraction=args.val_fraction,
        val_seed=args.val_seed,
        include=include,
        exclude=exclude,
    )
    run_eval(args.checkpoint, dataset_cfg, out_dir=args.out_dir)
    return 0
