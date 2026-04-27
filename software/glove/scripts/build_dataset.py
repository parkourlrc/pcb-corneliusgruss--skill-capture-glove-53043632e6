"""Build a training dataset from recorded sessions.

Creates a dataset manifest (which sessions, train/val split) and computes
normalization statistics. The dataset doesn't copy data — it points to
session directories.

Usage:
    python -m glove.scripts.build_dataset \
        --name mug_coaster_v1 \
        --task mug_coaster \
        --data-dir data/sessions \
        --output data/datasets/mug_coaster_v1

    python -m glove.scripts.build_dataset \
        --dataset datasets/mug_coaster_v1.yaml \
        --output data/datasets/mug_coaster_v1
"""

import argparse
from pathlib import Path

import yaml

from glove.data.dataset import DatasetManifest
from glove.data.session import SessionLoader
from glove.data.training import compute_normalization_stats
from glove.quality.schema import (
    QualityBlock,
    ReviewBlock,
    effective_inclusion,
)


def _gate_session(session_dir: Path) -> tuple[bool, str]:
    """Return (include, tag) where tag is one of:
    auto_pass, auto_fail, needs_review, approve, reject, force_include, unscored.
    """
    meta_path = session_dir / "meta.yaml"
    if not meta_path.exists():
        return False, "unscored"
    meta = yaml.safe_load(meta_path.read_text()) or {}
    q_raw = meta.get("quality")
    r_raw = meta.get("review")
    quality = QualityBlock.from_dict(q_raw) if q_raw else None
    review = ReviewBlock.from_dict(r_raw) if r_raw else None

    include = effective_inclusion(quality, review)
    if review and review.decision:
        tag = review.decision
    elif quality:
        tag = quality.verdict
    else:
        tag = "unscored"
    return include, tag


def _apply_gate(dirs: list[Path]) -> tuple[list[Path], dict[str, int]]:
    kept: list[Path] = []
    counts: dict[str, int] = {}
    for d in dirs:
        include, tag = _gate_session(d)
        counts[tag] = counts.get(tag, 0) + 1
        if include:
            kept.append(d)
    return kept, counts


def build_dataset(
    manifest: DatasetManifest,
    output_dir: Path,
) -> None:
    """Build a training dataset from a manifest.

    1. Discover sessions matching filters
    2. Apply the quality gate (meta.yaml `quality` + `review` via effective_inclusion)
    3. Assign train/val split
    4. Compute NormalizationStats from training split
    5. Write manifest.yaml + stats.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dirs, val_dirs = manifest.get_split()

    if not train_dirs and not val_dirs:
        print(f"ERROR: No sessions found for manifest '{manifest.name}'")
        print(f"  task={manifest.task}, data_dir={manifest.data_dir}")
        return

    n_train_before, n_val_before = len(train_dirs), len(val_dirs)
    train_dirs, train_counts = _apply_gate(train_dirs)
    val_dirs, val_counts = _apply_gate(val_dirs)

    def fmt_counts(c: dict[str, int]) -> str:
        order = ["auto_pass", "approve", "force_include",
                 "needs_review", "auto_fail", "reject", "unscored"]
        parts = [f"{k}={c[k]}" for k in order if c.get(k)]
        return "  ".join(parts) if parts else "(empty)"

    print(f"Dataset: {manifest.name}")
    print(f"  Task: {manifest.task}")
    print(f"  Train: {len(train_dirs)}/{n_train_before}  gate: {fmt_counts(train_counts)}")
    print(f"  Val:   {len(val_dirs)}/{n_val_before}  gate: {fmt_counts(val_counts)}")

    # Compute normalization stats from training sessions
    print("Computing normalization statistics...")
    train_sessions = [SessionLoader(d) for d in train_dirs]
    stats = compute_normalization_stats(train_sessions)

    # Write stats.json
    stats.to_json(output_dir / "stats.json")
    print(f"  Wrote {output_dir / 'stats.json'}")

    # Write manifest.yaml
    manifest_data = {
        "name": manifest.name,
        "task": manifest.task,
        "description": manifest.description,
        "schema_version": "1.0",
        "sessions": {
            "train": [str(d) for d in train_dirs],
            "val": [str(d) for d in val_dirs],
        },
        "val_fraction": manifest.val_fraction,
        "seed": manifest.seed,
    }
    if manifest.exclude:
        manifest_data["exclude"] = manifest.exclude

    with open(output_dir / "manifest.yaml", "w") as f:
        yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False)
    print(f"  Wrote {output_dir / 'manifest.yaml'}")

    # Summary
    total_frames = sum(SessionLoader(d, require_poses=False).num_rgb_frames for d in train_dirs + val_dirs)
    print(f"\nDataset ready: {len(train_dirs) + len(val_dirs)} sessions, {total_frames} frames")
    print(f"  Train with: glove-train --dataset {output_dir / 'manifest.yaml'}")


def main():
    parser = argparse.ArgumentParser(description="Build training dataset")
    parser.add_argument("--name", default="dataset_v1", help="Dataset name")
    parser.add_argument("--task", default="", help="Filter by task label")
    parser.add_argument("--data-dir", default="data/sessions", help="Root session directory")
    parser.add_argument("--dataset", default=None, help="Load from existing manifest YAML")
    parser.add_argument("--output", required=True, help="Output directory for manifest + stats")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation fraction")
    args = parser.parse_args()

    if args.dataset:
        manifest = DatasetManifest.load(args.dataset)
    else:
        manifest = DatasetManifest(
            name=args.name,
            task=args.task,
            data_dir=Path(args.data_dir),
            val_fraction=args.val_fraction,
        )

    build_dataset(manifest, Path(args.output))


if __name__ == "__main__":
    main()
