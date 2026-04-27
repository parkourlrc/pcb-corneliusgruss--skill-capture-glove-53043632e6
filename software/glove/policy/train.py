"""Training loop for the ACT glove policy.

Features:
- AMP mixed-precision on CUDA, falls back to FP32 on CPU/MPS
- AdamW with cosine-annealing LR schedule, gradient clipping
- Train/val split by SESSION (not by frame) to avoid leakage
- Per-epoch checkpointing, best-val checkpoint tracked separately
- Terminal log + CSV metrics log + final loss-curve PNG
- Graceful SIGINT — flushes a checkpoint before exit

Entry point: `run_training(config)` or `python -m glove.policy train ...`
"""

from __future__ import annotations

import csv
import json
import random
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePath


def _stringify_paths(obj):
    """Recursively replace Path-like objects with their string form.

    yaml.safe_dump and wandb.init both choke on PurePath; this makes any
    config dict safely serializable.
    """
    if isinstance(obj, dict):
        return {k: _stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_stringify_paths(v) for v in obj)
    if isinstance(obj, PurePath):
        return str(obj)
    return obj

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from glove.policy.act import ACTConfig, ACTPolicy, count_parameters
from glove.policy.dataset import (
    ACTDataset,
    ACTDatasetConfig,
    _session_iter,
    compute_norm_stats,
    load_norm_stats,
    save_norm_stats,
    train_val_split,
)
from glove.policy.transforms import NormStats, rot6d_to_matrix, unscale_rel_pos


@dataclass
class TrainConfig:
    """All training hyperparameters (overridable via YAML or CLI)."""

    run_name: str = "act_pick_place_v1"
    checkpoints_root: Path = Path("checkpoints")
    dataset: ACTDatasetConfig = field(default_factory=ACTDatasetConfig)
    model: ACTConfig = field(default_factory=ACTConfig)

    # Optimizer
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100
    grad_clip: float = 1.0

    # Validation
    val_every: int = 2
    save_every: int = 1          # write `latest.pt` every epoch — SGE may kill us at any point

    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True

    # Logging
    wandb: bool = True                       # False or WANDB_MODE=disabled skips
    wandb_project: str = "glove-act"
    wandb_entity: str | None = None          # team/org; None = your default
    seed: int = 42

    # Regularization / augmentation
    augment: bool = True                     # train-time RandomRotation+Crop+ColorJitter
    static_mask: str | None = None           # path to PNG mask (policy.mask output); optional

    # Resume from a saved checkpoint (loads model + optimizer + scheduler state,
    # continues from the saved epoch). Norm stats are taken from the run_dir if
    # they exist; otherwise recomputed.
    resume: str | None = None


def _pin_threads_single():
    """Force every thread-pool library we know of to a single thread.

    Called both in the main process and inside each DataLoader worker via
    `worker_init_fn` — env-var inheritance through fork is not reliable for
    libraries that captured their thread counts at import time in the parent.
    """
    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.set_num_threads(1)
        _torch.set_num_interop_threads(1)
    except RuntimeError:
        # interop-threads can only be set before first use; OK if it already was.
        pass


def _worker_init_fn(_worker_id: int):
    _pin_threads_single()
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_metadata() -> dict[str, str | bool | None]:
    repo_root = Path(__file__).resolve().parents[3]

    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    return {
        "commit": _run(["rev-parse", "HEAD"]),
        "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run(["status", "--porcelain"])),
    }


def _pick_device() -> torch.device:
    _pin_threads_single()

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _actions_to_relative_pose_mats(
    actions: torch.Tensor,
    stats: NormStats,
    action_pose_mode: str,
) -> torch.Tensor:
    """Convert action pose channels to T_ref_future matrices for physical metrics."""
    B, H = actions.shape[:2]
    pos = unscale_rel_pos(actions[..., :3], stats)
    R = rot6d_to_matrix(actions[..., 3:9])
    T_step = torch.zeros(B, H, 4, 4, dtype=actions.dtype, device=actions.device)
    T_step[..., :3, :3] = R
    T_step[..., :3, 3] = pos
    T_step[..., 3, 3] = 1.0

    if action_pose_mode == "relative":
        return T_step
    if action_pose_mode != "delta":
        raise ValueError(f"Unsupported action_pose_mode={action_pose_mode!r}")

    out = torch.empty_like(T_step)
    cur = torch.eye(4, dtype=actions.dtype, device=actions.device).expand(B, 4, 4).clone()
    for k in range(H):
        cur = cur @ T_step[:, k]
        out[:, k] = cur
    return out


@torch.no_grad()
def evaluate(model: ACTPolicy, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Compute validation metrics.

    The normalized L1 tracks optimization. The prior-z component metrics are
    logged in physical units so pose-label failures are visible immediately.
    """
    model.eval()
    sums = {"l1_cvae": 0.0, "l1_prior": 0.0, "kl": 0.0}
    n = 0
    real_steps = 0.0
    total_steps = 0.0
    pos_abs_m = 0.0
    rot_deg = 0.0
    hall_abs_raw = 0.0
    stats: NormStats | None = getattr(loader.dataset, "stats", None)
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        proprio = batch["proprio"].to(device, non_blocking=True)
        actions = batch["actions"].to(device, non_blocking=True)
        actions_abs = batch.get("actions_abs", batch["actions"]).to(device, non_blocking=True)
        is_pad = batch["is_pad"].to(device, non_blocking=True) if "is_pad" in batch else None

        out = model(images, proprio, actions, is_pad=is_pad)
        pred_prior = model.predict(images, proprio)

        sums["l1_cvae"] += out["l1_loss"].item() * images.size(0)
        sums["kl"] += out["kl_loss"].item() * images.size(0)
        # Prior-z eval: also mask padded positions for a fair comparison.
        if is_pad is not None:
            keep = (~is_pad).float().unsqueeze(-1)
            denom = keep.sum() * actions.shape[-1] + 1e-6
            prior_l1 = ((pred_prior - actions).abs() * keep).sum() / denom
        else:
            prior_l1 = (pred_prior - actions).abs().mean()
        sums["l1_prior"] += prior_l1.item() * images.size(0)

        keep = (~is_pad) if is_pad is not None else torch.ones(
            actions.shape[:2], dtype=torch.bool, device=device
        )
        real = keep.sum().item()
        real_steps += real
        total_steps += float(keep.numel())

        if stats is not None and real > 0:
            mode = getattr(model.config, "action_pose_mode", "relative")
            pred_T = _actions_to_relative_pose_mats(pred_prior, stats, mode)
            true_T = _actions_to_relative_pose_mats(actions_abs, stats, "relative")
            pred_pos_m = pred_T[..., :3, 3]
            true_pos_m = true_T[..., :3, 3]
            pos_abs_m += ((pred_pos_m - true_pos_m).abs() * keep.unsqueeze(-1)).sum().item()

            pred_R = pred_T[..., :3, :3]
            true_R = true_T[..., :3, :3]
            R_delta = pred_R.transpose(-1, -2) @ true_R
            trace = R_delta[..., 0, 0] + R_delta[..., 1, 1] + R_delta[..., 2, 2]
            angles = torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))
            rot_deg += (angles * keep).sum().item() * (180.0 / np.pi)

            hall_std = torch.as_tensor(stats.hall_std, dtype=torch.float32, device=device)
            pred_hall = pred_prior[..., 9:21] * torch.clamp(hall_std, min=1e-3)
            true_hall = actions_abs[..., 9:21] * torch.clamp(hall_std, min=1e-3)
            hall_abs_raw += ((pred_hall - true_hall).abs() * keep.unsqueeze(-1)).sum().item()

        n += images.size(0)

    out = {k: v / max(n, 1) for k, v in sums.items()}
    out["pad_fraction"] = 1.0 - (real_steps / max(total_steps, 1.0))
    if stats is not None and real_steps > 0:
        out["prior_pos_l1_m"] = pos_abs_m / (real_steps * 3.0)
        out["prior_rot_l1_deg"] = rot_deg / real_steps
        out["prior_hall_l1_raw"] = hall_abs_raw / (real_steps * 12.0)
    return out


def _plot_curves(csv_path: Path, png_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return
    epochs = [int(r["epoch"]) for r in rows]
    train_l1 = [float(r["train_l1"]) for r in rows]
    val_l1_prior = [float(r["val_l1_prior"]) if r.get("val_l1_prior") else np.nan for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_l1, label="train L1 (CVAE)", color="steelblue", lw=1.3)
    val_epochs = [e for e, v in zip(epochs, val_l1_prior) if not np.isnan(v)]
    val_vals = [v for v in val_l1_prior if not np.isnan(v)]
    if val_vals:
        ax.plot(val_epochs, val_vals, label="val L1 (prior z=0)", color="crimson", lw=1.5, marker="o", ms=4)
    ax.set_xlabel("epoch")
    ax.set_ylabel("L1 action error")
    ax.set_title("ACT training — action L1 loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _init_wandb(cfg: TrainConfig, run_dir: Path):
    """Initialize wandb if enabled + installed + not disabled via env. Returns run or None."""
    import os
    if not cfg.wandb or os.environ.get("WANDB_MODE") == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed — skipping cloud logging (`pip install wandb`).")
        return None
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        dir=str(run_dir),
        config=_stringify_paths({
            "model": asdict(cfg.model),
            "dataset": asdict(cfg.dataset),
            "train": {k: v for k, v in asdict(cfg).items()
                      if k not in ("model", "dataset", "checkpoints_root")},
        }),
    )
    print(f"wandb: {run.get_url()}")
    return run


def run_training(cfg: TrainConfig) -> Path:
    """Run full ACT training loop. Returns the path to the best checkpoint."""
    _seed_everything(cfg.seed)
    run_dir = Path(cfg.checkpoints_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"

    device = _pick_device()
    print(f"Device: {device}")

    wandb_run = _init_wandb(cfg, run_dir)

    # ── Dataset ────────────────────────────────────────────────────────────
    sessions = _session_iter(cfg.dataset)
    if not sessions:
        raise RuntimeError(f"No sessions matching task={cfg.dataset.task!r} in {cfg.dataset.sessions_root}")

    train_sess, val_sess = train_val_split(sessions, cfg.dataset.val_fraction, cfg.dataset.val_seed)
    print(f"Found {len(sessions)} sessions: {len(train_sess)} train + {len(val_sess)} val")
    run_manifest = {
        "run_name": cfg.run_name,
        "created_at_unix": time.time(),
        "git": _git_metadata(),
        "train_sessions": [p.name for p in train_sess],
        "val_sessions": [p.name for p in val_sess],
    }

    stats_path = run_dir / "norm_stats.json"
    if stats_path.exists():
        stats = load_norm_stats(stats_path)
        print(f"Loaded norm stats: {stats_path}")
    else:
        print("Computing norm stats over training sessions…")
        stats = compute_norm_stats(train_sess)
        save_norm_stats(stats, stats_path)
        print(f"Saved norm stats: {stats_path}")

    mask_path = Path(cfg.static_mask) if cfg.static_mask else None
    train_ds = ACTDataset(
        train_sess, stats,
        obs_horizon=cfg.model.obs_horizon,
        action_horizon=cfg.model.action_horizon,
        image_size=cfg.model.image_size,
        mask_path=mask_path,
        augment=cfg.augment,
        action_pose_mode=cfg.model.action_pose_mode,
    )
    val_ds = ACTDataset(
        val_sess, stats,
        obs_horizon=cfg.model.obs_horizon,
        action_horizon=cfg.model.action_horizon,
        image_size=cfg.model.image_size,
        mask_path=mask_path,
        augment=False,
        action_pose_mode=cfg.model.action_pose_mode,
    ) if val_sess else None

    print(f"Train samples: {len(train_ds):,}   Val samples: {len(val_ds) if val_ds else 0:,}")

    # Video I/O is the bottleneck — use workers for parallel frame decode.
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
        drop_last=True,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=max(1, cfg.num_workers // 2), pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.num_workers > 0),
        worker_init_fn=_worker_init_fn,
    ) if val_ds else None

    # ── Model + optimizer ──────────────────────────────────────────────────
    model = ACTPolicy(cfg.model).to(device)
    trainable, total = count_parameters(model)
    print(f"Model: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total params")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 1
    best_val = float("inf")
    if cfg.resume:
        ckpt_path = Path(cfg.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume target does not exist: {ckpt_path}")
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if "best_val_l1_prior" in ckpt:
            best_val = float(ckpt["best_val_l1_prior"])
        elif "val_metrics" in ckpt and "l1_prior" in ckpt["val_metrics"]:
            best_val = float(ckpt["val_metrics"]["l1_prior"])
        print(f"  resumed at epoch {ckpt.get('epoch', '?')}; "
              f"continuing from epoch {start_epoch}; "
              f"best_val_l1_prior so far: {best_val:.5f}")

    # ── Signal-handled early exit ──────────────────────────────────────────
    stop = {"flag": False}
    def _sigint(*_):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sigint)

    # Metrics CSV
    metric_fields = [
        "epoch", "train_l1", "train_kl", "val_l1_cvae", "val_l1_prior",
        "val_kl", "val_pos_l1_m", "val_rot_l1_deg", "val_hall_l1_raw",
        "val_pad_fraction", "lr", "sec",
    ]
    if csv_path.exists():
        first_line = csv_path.open().readline().strip()
        if first_line and first_line != ",".join(metric_fields):
            archived = csv_path.with_name(f"metrics_legacy_{int(time.time())}.csv")
            csv_path.rename(archived)
            print(f"Archived old metrics schema: {archived}")
    write_header = not csv_path.exists()
    csv_f = csv_path.open("a", newline="")
    writer = csv.writer(csv_f)
    if write_header:
        writer.writerow(metric_fields)

    # best_val initialized above; preserved across resume from checkpoint extras.
    best_path = run_dir / "best.pt"
    latest_path = run_dir / "latest.pt"
    (run_dir / "train_config.yaml").write_text(
        yaml.safe_dump({"train": _stringify_paths(asdict(cfg))}, sort_keys=False)
    )
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))

    def _checkpoint_payload(epoch: int, extra: dict | None = None) -> dict:
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "act_config": asdict(cfg.model),
            "train_config": _stringify_paths(asdict(cfg)),
            "norm_stats": stats.to_dict(),
            "run_manifest": run_manifest,
            "git": run_manifest["git"],
        }
        if extra:
            payload.update(extra)
        return payload

    try:
        for epoch in range(start_epoch, cfg.num_epochs + 1):
            t_epoch = time.time()
            model.train()

            train_l1_sum, train_kl_sum, n_seen = 0.0, 0.0, 0
            for step, batch in enumerate(train_loader):
                images = batch["images"].to(device, non_blocking=True)
                proprio = batch["proprio"].to(device, non_blocking=True)
                actions = batch["actions"].to(device, non_blocking=True)
                is_pad = batch["is_pad"].to(device, non_blocking=True) if "is_pad" in batch else None

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    out = model(images, proprio, actions, is_pad=is_pad)
                    loss = out["loss"]

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    cfg.grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()

                bs = images.size(0)
                train_l1_sum += out["l1_loss"].item() * bs
                train_kl_sum += out["kl_loss"].item() * bs
                n_seen += bs

                if stop["flag"]:
                    break

            scheduler.step()
            train_l1 = train_l1_sum / max(n_seen, 1)
            train_kl = train_kl_sum / max(n_seen, 1)

            # Encourage CUDA allocator to release freed blocks before the next
            # epoch's forward pass — observed to help avoid SGE cgroup kills.
            if device.type == "cuda":
                torch.cuda.empty_cache()

            val_metrics = {
                "l1_cvae": "", "l1_prior": "", "kl": "",
                "prior_pos_l1_m": "", "prior_rot_l1_deg": "",
                "prior_hall_l1_raw": "", "pad_fraction": "",
            }
            if val_loader is not None and (epoch % cfg.val_every == 0 or epoch == cfg.num_epochs):
                vm = evaluate(model, val_loader, device)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                val_metrics = {k: f"{v:.5f}" for k, v in vm.items()}
                for k in ("prior_pos_l1_m", "prior_rot_l1_deg", "prior_hall_l1_raw", "pad_fraction"):
                    val_metrics.setdefault(k, "")
                # Track best on prior-z L1 (inference-mode metric)
                if vm["l1_prior"] < best_val:
                    best_val = vm["l1_prior"]
                    torch.save(_checkpoint_payload(epoch, {
                        "best_val_l1_prior": best_val,
                        "val_metrics": vm,
                    }), best_path)

            if epoch % cfg.save_every == 0 or epoch == cfg.num_epochs:
                torch.save(_checkpoint_payload(epoch), latest_path)

            sec = time.time() - t_epoch
            lr_now = optimizer.param_groups[0]["lr"]
            writer.writerow([
                epoch, f"{train_l1:.5f}", f"{train_kl:.5f}",
                val_metrics["l1_cvae"], val_metrics["l1_prior"], val_metrics["kl"],
                val_metrics["prior_pos_l1_m"], val_metrics["prior_rot_l1_deg"],
                val_metrics["prior_hall_l1_raw"], val_metrics["pad_fraction"],
                f"{lr_now:.2e}", f"{sec:.1f}",
            ])
            csv_f.flush()

            if wandb_run is not None:
                log_dict = {
                    "train/l1": train_l1,
                    "train/kl": train_kl,
                    "train/lr": lr_now,
                    "train/epoch_sec": sec,
                    "epoch": epoch,
                }
                if val_metrics["l1_prior"] != "":
                    log_dict["val/l1_cvae"] = float(val_metrics["l1_cvae"])
                    log_dict["val/l1_prior"] = float(val_metrics["l1_prior"])
                    log_dict["val/kl"] = float(val_metrics["kl"])
                    for metric_name in (
                        "prior_pos_l1_m",
                        "prior_rot_l1_deg",
                        "prior_hall_l1_raw",
                        "pad_fraction",
                    ):
                        if val_metrics[metric_name] != "":
                            log_dict[f"val/{metric_name}"] = float(val_metrics[metric_name])
                wandb_run.log(log_dict, step=epoch)

            vstr = ""
            if val_metrics["l1_prior"] != "":
                vstr = (
                    f"  val_l1(prior)={val_metrics['l1_prior']}"
                    f"  pos={val_metrics['prior_pos_l1_m']}m"
                    f"  rot={val_metrics['prior_rot_l1_deg']}deg"
                )
            print(f"epoch {epoch:3d}/{cfg.num_epochs}  "
                  f"train_l1={train_l1:.4f}  train_kl={train_kl:.4f}{vstr}  "
                  f"lr={lr_now:.2e}  ({sec:.0f}s)")

            if stop["flag"]:
                print("SIGINT received — saving latest + exiting.")
                torch.save(_checkpoint_payload(epoch), latest_path)
                break

    finally:
        csv_f.close()
        _plot_curves(csv_path, run_dir / "loss_curve.png")
        train_ds.close()
        if val_ds is not None:
            val_ds.close()
        if wandb_run is not None:
            if best_path.exists():
                import wandb
                try:
                    wandb.save(str(best_path), base_path=str(run_dir.parent), policy="now")
                except Exception as e:
                    print(f"wandb.save failed: {e}")
            wandb_run.finish()

    print(f"Done. Best val L1 (prior z): {best_val:.5f}   → {best_path}")
    return best_path
