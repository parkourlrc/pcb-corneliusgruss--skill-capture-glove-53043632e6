"""Session thumbnail generation.

Creates a side-by-side stereo thumbnail (320x100) from the first frame
of each camera's MP4 video. Stored as thumbnail.jpg inside the session directory.
"""

import logging
from pathlib import Path
from typing import Optional

import av
from PIL import Image

logger = logging.getLogger(__name__)

THUMB_W = 160
THUMB_H = 100
THUMB_QUALITY = 80


def _first_frame_from_mp4(video_path: Path) -> Optional[Image.Image]:
    """Extract the first frame from an MP4 as a PIL Image."""
    if not video_path.exists():
        return None
    try:
        container = av.open(str(video_path))
        for frame in container.decode(video=0):
            img = frame.to_image()
            container.close()
            return img
        container.close()
    except Exception as e:
        logger.warning("Failed to read %s: %s", video_path.name, e)
    return None


def generate_thumbnail(session_dir: Path) -> Optional[Path]:
    """Generate a stereo thumbnail from the first frame of each camera MP4.

    Parameters
    ----------
    session_dir : Path
        Path to the session directory (must contain videos/cam_top.mp4).

    Returns
    -------
    Path or None
        Path to the generated thumbnail, or None if no video found.
    """
    left_img = _first_frame_from_mp4(session_dir / "videos" / "cam_top.mp4")
    right_img = _first_frame_from_mp4(session_dir / "videos" / "cam_bottom.mp4")

    if not left_img and not right_img:
        return None

    try:
        if left_img and right_img:
            left_thumb = left_img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
            right_thumb = right_img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
            combined = Image.new("RGB", (THUMB_W * 2, THUMB_H))
            combined.paste(left_thumb, (0, 0))
            combined.paste(right_thumb, (THUMB_W, 0))
        elif left_img:
            combined = left_img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
        else:
            combined = right_img.resize((THUMB_W, THUMB_H), Image.LANCZOS)

        thumb_path = session_dir / "thumbnail.jpg"
        combined.save(thumb_path, "JPEG", quality=THUMB_QUALITY)
        return thumb_path
    except Exception as e:
        logger.warning("Failed to generate thumbnail for %s: %s", session_dir.name, e)
        return None


def ensure_thumbnail(session_dir: Path) -> Optional[Path]:
    """Return existing thumbnail or generate a new one."""
    thumb_path = session_dir / "thumbnail.jpg"
    if thumb_path.exists():
        return thumb_path
    return generate_thumbnail(session_dir)
