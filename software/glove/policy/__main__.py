"""CLI entry point for glove.policy.

Subcommands:
    train    Train an ACT policy on a task's demo sessions.
    eval     Evaluate a trained checkpoint (aggregate metrics + qualitative viz).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from glove.policy.act import ACTConfig
from glove.policy.dataset import ACTDatasetConfig
from glove.policy import eval as eval_mod
from glove.policy import mask as mask_mod
from glove.policy.train import TrainConfig, run_training


def cmd_train(args) -> int:
    # Optional include/exclude list files: one session name per line (exact match or glob).
    exclude = tuple(args.exclude) if args.exclude else ()
    if args.exclude_file:
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
    cfg = TrainConfig(
        run_name=args.run_name,
        checkpoints_root=args.checkpoints_root,
        dataset=ACTDatasetConfig(
            sessions_root=args.sessions_root,
            task=args.task,
            val_fraction=args.val_fraction,
            val_seed=args.val_seed,
            include=include,
            exclude=exclude,
        ),
        model=ACTConfig(
            obs_horizon=args.obs_horizon,
            action_horizon=args.action_horizon,
            action_pose_mode=args.action_pose_mode,
            kl_weight=args.kl_weight,
            free_bits=args.free_bits,
            endpoint_pos_weight=args.endpoint_pos_weight,
            ramp_pos_weight=args.ramp_pos_weight,
        ),
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        val_every=args.val_every,
        wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        seed=args.seed,
        augment=not args.no_augment,
        static_mask=args.static_mask,
        resume=args.resume,
    )
    run_training(cfg)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="glove.policy")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    tr = sub.add_parser("train", help="Train an ACT policy on a task's demo sessions.")
    tr.add_argument("--run-name", type=str, default="act_pick_place_v1")
    tr.add_argument("--sessions-root", type=Path, default=Path("data/sessions"))
    tr.add_argument("--task", type=str, default="pick_place")
    tr.add_argument("--val-fraction", type=float, default=0.15)
    tr.add_argument("--val-seed", type=int, default=42)
    tr.add_argument("--obs-horizon", type=int, default=2)
    tr.add_argument("--action-horizon", type=int, default=300)
    tr.add_argument("--action-pose-mode", choices=("relative", "delta"), default="relative",
                    help="Pose target for action[:, :9]. relative = v4 baseline; "
                         "delta = per-frame SE(3) increments chained at eval.")
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--num-workers", type=int, default=4)
    tr.add_argument("--val-every", type=int, default=2)
    tr.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints"))
    tr.add_argument("--no-wandb", action="store_true", help="Disable cloud logging.")
    tr.add_argument("--wandb-project", type=str, default="glove-act")
    tr.add_argument("--wandb-entity", type=str, default=None)
    tr.add_argument("--no-augment", action="store_true",
                    help="Disable train-time image augmentation (default: on).")
    tr.add_argument("--static-mask", type=str, default=None,
                    help="Path to PNG mask produced by `glove.policy build-mask`.")
    tr.add_argument("--exclude", type=str, action="append", default=[],
                    help="Session name (or glob) to exclude. Repeat for multiple.")
    tr.add_argument("--exclude-file", type=str, default=None,
                    help="File with one session name (or glob) per line to exclude.")
    tr.add_argument("--include-file", type=str, default=None,
                    help="File with one session name (or glob) per line to include "
                         "(empty = all). Applied before exclude.")
    tr.add_argument("--kl-weight", type=float, default=50.0,
                    help="CVAE KL divergence weight. Default 50 for the mug-coaster "
                         "retrain; use 10 for an ACT-paper baseline.")
    tr.add_argument("--free-bits", type=float, default=0.0,
                    help="Per-dim KL floor (nats). Standard fix for CVAE posterior "
                         "collapse: each latent dim's KL contribution is floored at "
                         "this value before kl_weight is applied. Try 0.01 for our "
                         "32-dim latent (= 0.32 nats minimum total). Default 0 (off).")
    tr.add_argument("--endpoint-pos-weight", type=float, default=0.0,
                    help="Extra weight on |pred_pos[end] - gt_pos[end]|. Fights "
                         "magnitude undershoot when CVAE collapses. Default 0 (off).")
    tr.add_argument("--ramp-pos-weight", type=float, default=0.0,
                    help="Extra weight on linspace(0,1,H)·|pred_pos - gt_pos| (up-ramp). "
                         "Pressures position L1 late in the chunk where undershoot lives. "
                         "Default 0 (off).")
    tr.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint .pt to resume from. Loads model + "
                         "optimizer + scheduler state, continues at the saved epoch.")
    tr.add_argument("--seed", type=int, default=42,
                    help="Seed Python, NumPy, PyTorch, and DataLoader workers.")
    tr.set_defaults(func=cmd_train)

    # build-mask
    bm = sub.add_parser("build-mask",
                        help="Compute a static-pixel mask from training sessions (glove body etc).")
    bm.add_argument("--sessions-root", type=Path, default=Path("data/sessions"))
    bm.add_argument("--task", type=str, default="pick_place")
    bm.add_argument("--out", type=Path, default=Path("configs/policy/static_mask.png"))
    bm.add_argument("--image-size", type=int, default=224)
    bm.add_argument("--frames-per-session", type=int, default=20)
    bm.add_argument("--percentile", type=float, default=10.0,
                    help="Mask pixels below this percentile of per-pixel std (default: 10%%).")
    bm.set_defaults(func=mask_mod.cli)

    # eval
    ev = sub.add_parser("eval", help="Evaluate a trained checkpoint.")
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--sessions-root", type=Path, default=Path("data/sessions"))
    ev.add_argument("--task", type=str, default="pick_place")
    ev.add_argument("--val-fraction", type=float, default=0.15)
    ev.add_argument("--val-seed", type=int, default=42)
    ev.add_argument("--exclude", type=str, action="append", default=[],
                    help="Session name (or glob) to exclude. Repeat for multiple.")
    ev.add_argument("--exclude-file", type=str, default=None,
                    help="File with one session name (or glob) per line to exclude.")
    ev.add_argument("--include-file", type=str, default=None,
                    help="File with one session name (or glob) per line to include "
                         "(empty = all). Must match training's --include-file to "
                         "evaluate against the same val split.")
    ev.add_argument("--out-dir", type=Path, default=None)
    ev.set_defaults(func=eval_mod.cli)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
