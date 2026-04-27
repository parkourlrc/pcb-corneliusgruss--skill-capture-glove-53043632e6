"""Offload GoPro MP4 files from the SD card to `data/raw_video/YYYY-MM-DD/`.

Detects a mounted GoPro volume automatically (macOS `/Volumes/*` with the
standard `DCIM/100GOPRO/` layout), then copies every `GX*.MP4` not yet present
at the destination. Files are bucketed into day folders by each MP4's own
mtime so a multi-day card sorts correctly.

Use: `python -m glove.capture offload [--delete-source]`

Replaces the slow "select-all-in-Image-Capture-and-drag" ritual with a single
command that lands each file in the right spot for the ingest pipeline.
"""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path


MOUNTS_ROOT = Path("/Volumes")
GOPRO_SUBPATH = Path("DCIM")
DEST_ROOT = Path("data/raw_video")


def find_gopro_volume() -> Path | None:
    """Return the mounted GoPro card root (e.g. /Volumes/Untitled), or None."""
    if not MOUNTS_ROOT.exists():
        return None
    for vol in sorted(MOUNTS_ROOT.iterdir()):
        dcim = vol / GOPRO_SUBPATH
        if not dcim.exists():
            continue
        # Any 100GOPRO / 101GOPRO / ... subdir confirms a GoPro-formatted card
        if any(d.name.endswith("GOPRO") and d.is_dir() for d in dcim.iterdir()):
            return vol
    return None


def _iter_gopro_mp4s(volume: Path):
    """Yield MP4 paths from all 1NN GOPRO dirs, sorted."""
    for dcim_sub in sorted((volume / GOPRO_SUBPATH).iterdir()):
        if not (dcim_sub.is_dir() and dcim_sub.name.endswith("GOPRO")):
            continue
        for mp4 in sorted(dcim_sub.glob("GX*.MP4")):
            yield mp4


def _mtime_date(p: Path) -> str:
    return datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d")


def offload(
    volume: Path | None = None,
    dest_root: Path = DEST_ROOT,
    delete_source: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    """Copy every GX*.MP4 from the GoPro card into `dest_root/YYYY-MM-DD/`.

    Skips files that already exist at the destination (idempotent).

    Returns the list of destination paths (copied or already-present).
    """
    if volume is None:
        volume = find_gopro_volume()
    if volume is None or not volume.exists():
        raise FileNotFoundError(
            "No mounted GoPro volume found under /Volumes/. "
            "Plug the card in (or the camera via USB) and retry."
        )

    copied: list[Path] = []
    skipped_existing = 0
    total_bytes = 0

    for src in _iter_gopro_mp4s(volume):
        date = _mtime_date(src)
        day_dir = dest_root / date
        day_dir.mkdir(parents=True, exist_ok=True)
        dst = day_dir / src.name

        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            skipped_existing += 1
            copied.append(dst)
            continue

        size_mb = src.stat().st_size / (1024 * 1024)
        print(f"  {src.name} ({size_mb:5.0f} MB) → {dst}")
        if not dry_run:
            shutil.copy2(src, dst)
        copied.append(dst)
        total_bytes += src.stat().st_size

        if delete_source and not dry_run:
            src.unlink()

    print()
    print(f"Volume: {volume}")
    print(f"Copied {len([c for c in copied]) - skipped_existing} new files "
          f"({total_bytes / 1e9:.2f} GB) · skipped {skipped_existing} already-present")
    if delete_source and not dry_run:
        print("Source files deleted from card.")
    if dry_run:
        print("(dry-run — nothing written)")
    return copied


def cli(args) -> int:
    offload(
        volume=args.volume,
        dest_root=args.dest_root,
        delete_source=args.delete_source,
        dry_run=args.dry_run,
    )
    return 0
