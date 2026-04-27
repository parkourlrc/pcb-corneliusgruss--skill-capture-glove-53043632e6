"""CLI: score all aligned sessions and write quality blocks into their meta.yaml.

    python -m glove.quality backfill            # score everything under data/sessions/
    python -m glove.quality backfill --dry-run  # print what would change
    python -m glove.quality backfill --session path/to/demo_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from glove.quality.compute import score_session
from glove.quality.map_compute import score_map

DEFAULT_DATA = Path("data/sessions")


def iter_sessions(root: Path, prefix: str = ""):
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if prefix:
            if not p.name.startswith(prefix):
                continue
        elif not (p.name.startswith("demo_") or p.name.startswith("map_")):
            continue
        yield p


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score aligned sessions")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--session", type=Path, default=None,
                        help="Score a single session directory")
    parser.add_argument("--maps", action="store_true",
                        help="Score mapping sessions (map_*) instead of demos")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-score even if a quality block already exists")
    args = parser.parse_args(argv)

    if args.maps:
        return _run_maps(args)

    targets = [args.session] if args.session else list(iter_sessions(args.data_dir, prefix="demo_"))
    if not targets:
        print(f"No sessions found under {args.data_dir}")
        return 0

    n_scored = n_skipped_existing = n_skipped_unaligned = n_failed = 0
    counts = {"auto_pass": 0, "auto_fail": 0, "needs_review": 0}

    for session_dir in targets:
        meta_path = session_dir / "meta.yaml"
        episode_path = session_dir / "aligned" / "episode.parquet"
        if not episode_path.exists():
            n_skipped_unaligned += 1
            continue
        if not args.force and meta_path.exists():
            meta = yaml.safe_load(meta_path.read_text()) or {}
            if "quality" in meta:
                n_skipped_existing += 1
                counts[meta["quality"].get("verdict", "needs_review")] = (
                    counts.get(meta["quality"].get("verdict", "needs_review"), 0) + 1
                )
                continue

        try:
            block = score_session(session_dir, persist=not args.dry_run)
        except Exception as exc:
            n_failed += 1
            print(f"  FAIL {session_dir.name}: {exc}")
            continue

        if block is None:
            n_skipped_unaligned += 1
            continue

        n_scored += 1
        counts[block.verdict] = counts.get(block.verdict, 0) + 1
        marker = "(dry)" if args.dry_run else "     "
        print(
            f"  {marker} {session_dir.name:50s} {block.verdict:13s} "
            f"jump={block.metrics.pose_jump_max_mm:>6.1f}mm  "
            f"v95={block.metrics.pose_vel_p95_mps:>5.2f}m/s  "
            f"reloc={block.metrics.relocalization_suspects}"
        )

    print()
    print(f"scored:             {n_scored}")
    print(f"skipped (existing): {n_skipped_existing}")
    print(f"skipped (no align): {n_skipped_unaligned}")
    print(f"failed:             {n_failed}")
    print()
    print(f"auto_pass:          {counts.get('auto_pass', 0)}")
    print(f"needs_review:       {counts.get('needs_review', 0)}")
    print(f"auto_fail:          {counts.get('auto_fail', 0)}")
    return 0


def _run_maps(args) -> int:
    targets = [args.session] if args.session else list(iter_sessions(args.data_dir, prefix="map_"))
    if not targets:
        print(f"No mapping sessions under {args.data_dir}")
        return 0

    n_scored = n_skipped_existing = n_skipped_missing = n_failed = 0
    for session_dir in targets:
        meta_path = session_dir / "meta.yaml"
        traj_path = session_dir / "slam" / "camera_trajectory.csv"
        if not traj_path.exists():
            n_skipped_missing += 1
            continue
        if not args.force and meta_path.exists():
            meta = yaml.safe_load(meta_path.read_text()) or {}
            if "map_quality" in meta:
                n_skipped_existing += 1
                continue
        try:
            mq = score_map(session_dir, persist=not args.dry_run)
        except Exception as exc:
            n_failed += 1
            print(f"  FAIL {session_dir.name}: {exc}")
            continue
        if mq is None:
            n_skipped_missing += 1
            continue
        n_scored += 1
        marker = "(dry)" if args.dry_run else "     "
        print(
            f"  {marker} {session_dir.name:50s} "
            f"tracking={mq.mapping_tracking_ratio:.1%}  "
            f"kf={mq.n_keyframes:>5d}  "
            f"dur={mq.duration_s:>5.1f}s  "
            f"extent={mq.map_extent_m[0]:.2f}×{mq.map_extent_m[1]:.2f}×{mq.map_extent_m[2]:.2f} m"
        )

    print()
    print(f"scored:             {n_scored}")
    print(f"skipped (existing): {n_skipped_existing}")
    print(f"skipped (no traj):  {n_skipped_missing}")
    print(f"failed:             {n_failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
