"""Static-image mask computation + application for the ACT dataset.

A GoPro mounted on the glove always sees some pixels that don't change with
scene, task, or hand position — the glove body, arm, wrist, mount hardware.
Feeding those pixels to the policy wastes capacity (overfits to identical
patches) and hurts generalization.

We compute a **per-pixel standard-deviation map** across a random sample of
training frames. Pixels with std below a threshold are deemed "static" (and
the mask value is 1); the rest is 0. At training time the masked pixels are
zeroed in ImageNet-normalized space, which is equivalent to replacing them
with the ImageNet per-channel mean — keeping DINOv2's input distribution
natural.

Build the mask **from frames drawn across scene-varied demos** — if all demos
are on the same table, the table gets classified as static too. Rebuild after
capturing varied backgrounds.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEFAULT_FRAMES_PER_SESSION = 20
DEFAULT_STD_PERCENTILE = 10.0      # pixels below this std percentile → masked
DEFAULT_IMAGE_SIZE = 224


def _pick_video(session_dir: Path) -> Path:
    pre = session_dir / "raw" / "raw_video_224.mp4"
    if pre.exists():
        return pre
    return session_dir / "raw" / "raw_video.mp4"


def compute_static_mask(
    session_dirs: list[Path],
    image_size: int = DEFAULT_IMAGE_SIZE,
    frames_per_session: int = DEFAULT_FRAMES_PER_SESSION,
    std_percentile: float = DEFAULT_STD_PERCENTILE,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample frames, compute per-pixel std, threshold → static mask.

    Returns (mask, std_map) where `mask` is uint8 (H, W) with 1 = static,
    and `std_map` is float32 (H, W) = mean-over-RGB std for inspection.
    """
    rng = random.Random(rng_seed)
    frames: list[np.ndarray] = []

    for sess in session_dirs:
        vid = _pick_video(sess)
        if not vid.exists():
            continue
        cap = cv2.VideoCapture(str(vid))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_total == 0:
            cap.release()
            continue
        idxs = sorted(rng.sample(range(n_total), k=min(frames_per_session, n_total)))
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame.shape[0] != image_size or frame.shape[1] != image_size:
                frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()

    if not frames:
        raise RuntimeError("No frames sampled — check that sessions have raw_video(_224).mp4")

    stack = np.stack(frames, axis=0).astype(np.float32)     # (N, H, W, 3)
    # Average the per-channel std so a pixel is "static" only if ALL three
    # channels stay steady (a pixel that's steady in R but changes in B still
    # gets unmasked).
    std_map = stack.std(axis=0).mean(axis=-1)               # (H, W)
    threshold = float(np.percentile(std_map, std_percentile))
    mask = (std_map <= threshold).astype(np.uint8)          # (H, W) 1 = static
    return mask, std_map


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    """Save mask as a white-on-black PNG (255 = static, 0 = dynamic) for quick eyeballing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = (mask.astype(np.uint8) * 255)
    Image.fromarray(img, mode="L").save(path)


def save_std_map_png(std_map: np.ndarray, path: Path) -> None:
    """Save the std map as a colorized PNG to debug threshold choice."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normed = std_map - std_map.min()
    if normed.max() > 0:
        normed = normed / normed.max()
    colored = (normed * 255).astype(np.uint8)
    Image.fromarray(colored, mode="L").save(path)


def load_mask(path: Path) -> np.ndarray:
    """Load a mask PNG saved by `save_mask_png` → uint8 (H, W) with 1 = static."""
    img = np.array(Image.open(path).convert("L"))
    return (img > 127).astype(np.uint8)


def cli(args) -> int:
    import yaml
    import fnmatch
    sessions_root = Path(args.sessions_root).resolve()
    targets = []
    for d in sorted(sessions_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("demo_"):
            continue
        if not (d / "aligned" / "episode.parquet").exists():
            continue
        meta = yaml.safe_load((d / "meta.yaml").read_text())
        if meta.get("task") != args.task:
            continue
        targets.append(d)

    if not targets:
        raise RuntimeError(f"No aligned demo sessions matching task={args.task!r} "
                           f"under {sessions_root}")

    print(f"Computing static mask over {len(targets)} sessions, "
          f"{args.frames_per_session} frames each, percentile={args.percentile}…")
    mask, std_map = compute_static_mask(
        targets,
        image_size=args.image_size,
        frames_per_session=args.frames_per_session,
        std_percentile=args.percentile,
    )

    out = Path(args.out)
    save_mask_png(mask, out)
    debug_out = out.parent / (out.stem + "_stdmap.png")
    save_std_map_png(std_map, debug_out)

    coverage = 100 * mask.mean()
    print(f"Static pixels: {coverage:.1f}% of frame")
    print(f"  mask:    {out}")
    print(f"  std map: {debug_out}")
    return 0
