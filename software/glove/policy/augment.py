"""Image augmentation for ACT training.

Mirrors ACT-plus-plus's recipe (`act-plus-plus/utils.py:113`):
- RandomResizedCrop(scale=0.95) — mild zoom + shift
- RandomRotation(±5°) — small viewpoint jitter
- ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)

Applied train-side only. Crucially applied AFTER the static-image mask (see
`policy.mask`) so the glove-body pixels get zeroed in canonical coords before
any geometric transform moves them around.

Uses torchvision.transforms.v2 because it accepts tensor-of-floats directly
(v1 API is stricter and slower on GPU).
"""

from __future__ import annotations

from torchvision.transforms import v2 as T


def build_train_augment(image_size: int = 224) -> T.Transform:
    """Return a per-frame augment chain suitable for a (3, H, W) ImageNet-normalized tensor.

    Each call to the returned transform samples fresh parameters. Safe to apply
    identically across all frames in an observation window (they just get
    different random shifts, which provides implicit temporal augmentation).
    """
    return T.Compose([
        # Rotation first (keeps masked glove-body region moving with the image).
        T.RandomRotation(degrees=5, fill=0.0),
        # Slight random-resized crop; 95% keeps most of the frame while
        # forcing pixel-level invariance to camera shake.
        T.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.9, 1.0), ratio=(0.95, 1.05), antialias=True,
        ),
        # Color jitter applied on normalized tensors — jitter values are in
        # normalized space but the relative effect is what matters. Keep
        # moderate to avoid shifting semantics for DINOv2.
        T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.3),
    ])
