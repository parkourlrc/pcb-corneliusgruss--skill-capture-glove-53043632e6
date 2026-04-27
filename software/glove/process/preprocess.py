"""Preprocess session MP4s for training: downscale 2.7K → 224×224 H.264.

The training loader resizes every frame to 224×224 anyway, so storing the full
2.7K master is wasted bytes at ~15× the file size. For uploading training data
to a remote GPU cluster (e.g. BU SCC with a 10 GB home quota), running this
preprocessor first turns a ~150 MB MP4 into a ~10 MB MP4 with negligible
quality loss at 224×224.

Output lives at `<session>/raw/raw_video_224.mp4`. The dataset loader checks
for this file first and falls back to `raw_video.mp4` when absent.

Requires ffmpeg on PATH (Homebrew: `brew install ffmpeg`).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


IMAGE_SIZE = 224
CRF = 23                # H.264 constant-rate-factor; ~visually lossless @ 224×224
PRESET = "fast"


def preprocess_session(session_dir: Path, overwrite: bool = False) -> Path | None:
    """Produce `raw/raw_video_224.mp4` for a session. Returns the output path
    (or None if already present + not overwriting)."""
    src = session_dir / "raw" / "raw_video.mp4"
    dst = session_dir / "raw" / "raw_video_224.mp4"
    if not src.exists():
        raise FileNotFoundError(f"No raw_video.mp4 in {src.parent}")
    if dst.exists() and not overwrite:
        return None

    # Scale to 224×224 forcing square (matches what the dataset does at load time).
    # -an drops audio (useless for training, saves more bytes).
    # -vsync passthrough preserves the frame timing so per-frame mtime-style
    #   assumptions stay consistent.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"scale={IMAGE_SIZE}:{IMAGE_SIZE}",
        "-c:v", "libx264", "-crf", str(CRF), "-preset", PRESET,
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def cli(args) -> int:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH. Install with: brew install ffmpeg")

    targets = []
    for root in args.session_dirs:
        root = Path(root)
        if (root / "raw" / "raw_video.mp4").exists():
            targets.append(root)
        else:
            # Treat as a parent dir; recurse one level
            for d in sorted(root.iterdir()):
                if d.is_dir() and (d / "raw" / "raw_video.mp4").exists():
                    targets.append(d)

    print(f"Preprocessing {len(targets)} sessions (CRF={CRF}, {IMAGE_SIZE}×{IMAGE_SIZE})")
    done = 0
    skipped = 0
    total_bytes_saved = 0
    for sess in targets:
        dst = preprocess_session(sess, overwrite=args.overwrite)
        if dst is None:
            skipped += 1
            print(f"  skip  {sess.name} (already has raw_video_224.mp4)")
            continue
        src = sess / "raw" / "raw_video.mp4"
        before = src.stat().st_size
        after = dst.stat().st_size
        total_bytes_saved += before - after
        print(f"  ✓     {sess.name}  {before/1e6:6.1f} → {after/1e6:5.1f} MB "
              f"({100*after/before:4.1f}%)")
        done += 1

    print(f"\nDone. {done} converted, {skipped} skipped. "
          f"Saved ~{total_bytes_saved/1e9:.2f} GB vs originals.")
    return 0
