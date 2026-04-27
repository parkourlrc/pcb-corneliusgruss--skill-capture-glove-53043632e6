"""Per-session JPEG frame cache for frame-accurate scrubbing.

GoPro H.264 has sparse keyframes (~every 2-3s), so `<video>.currentTime` seek
lands far from the target frame. For inspecting a specific pose jump we need
frame-accurate access, which means pre-extracted JPEGs on disk.

ffmpeg bulk extraction is ~7s for a 22s demo (1327 frames) at 640w, ~50MB
on disk. Far cheaper than per-frame PyAV random-access (2s/frame).

Cache layout:
    <session>/cache/scrub_640/
        frame_00001.jpg   # ffmpeg uses 1-based %05d naming
        frame_00002.jpg
        ...
        .complete         # sentinel — exists once extraction succeeded

Extraction runs in a background thread; callers trigger it with
`ensure_extracting(session_dir)` and poll `status(session_dir)` for progress.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

SCRUB_WIDTH = 640
CACHE_SUBDIR = f"cache/scrub_{SCRUB_WIDTH}"

# Track in-flight extractions. Serial to avoid CPU-storming when a user
# navigates between demos rapidly (each ffmpeg burst pegs all cores).
_active: set[str] = set()
_active_lock = threading.Lock()
_queue_lock = threading.Lock()  # serializes actual ffmpeg runs


def cache_dir(session_dir: Path) -> Path:
    return session_dir / CACHE_SUBDIR


def _sentinel(session_dir: Path) -> Path:
    return cache_dir(session_dir) / ".complete"


def is_ready(session_dir: Path) -> bool:
    return _sentinel(session_dir).exists()


def frame_path(session_dir: Path, index: int) -> Path:
    # ffmpeg emits 1-based filenames; frame 0 → frame_00001.jpg
    return cache_dir(session_dir) / f"frame_{index + 1:05d}.jpg"


def get_cached_frame(session_dir: Path, index: int) -> bytes | None:
    p = frame_path(session_dir, index)
    if p.exists():
        try:
            return p.read_bytes()
        except OSError:
            return None
    return None


def status(session_dir: Path) -> dict:
    """Progress snapshot. Caller also needs n_total from episode.parquet."""
    cd = cache_dir(session_dir)
    n_done = 0
    if cd.exists():
        n_done = sum(1 for _ in cd.glob("frame_*.jpg"))
    with _active_lock:
        running = session_dir.name in _active
    return {
        "ready": is_ready(session_dir),
        "running": running,
        "n_done": n_done,
    }


def ensure_extracting(session_dir: Path) -> None:
    """Spawn background ffmpeg extraction if not ready and not already running.

    Safe to call multiple times / on every page render.
    """
    if is_ready(session_dir):
        return
    sid = session_dir.name
    with _active_lock:
        if sid in _active:
            return
        _active.add(sid)

    t = threading.Thread(target=_extract, args=(session_dir,), daemon=True,
                         name=f"scrub-extract-{sid}")
    t.start()


def _extract(session_dir: Path) -> None:
    sid = session_dir.name
    try:
        video = session_dir / "raw" / "raw_video.mp4"
        if not video.exists():
            logger.warning("scrub_cache: no raw_video.mp4 in %s", session_dir)
            return

        cd = cache_dir(session_dir)
        # Serialize ffmpeg runs (one at a time) so multiple queued sessions
        # don't starve each other or pin the box.
        with _queue_lock:
            if is_ready(session_dir):
                return  # another path completed it while we waited
            cd.mkdir(parents=True, exist_ok=True)

            cmd = [
                "ffmpeg", "-v", "error", "-y",
                "-i", str(video),
                "-vf", f"scale={SCRUB_WIDTH}:-2",
                "-q:v", "5",
                str(cd / "frame_%05d.jpg"),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except FileNotFoundError:
                logger.error("scrub_cache: ffmpeg not found in PATH")
                return
            except subprocess.CalledProcessError as exc:
                logger.error("scrub_cache: ffmpeg failed for %s: %s",
                             sid, exc.stderr[:400] if exc.stderr else exc)
                return

            _sentinel(session_dir).write_text("ok\n")
            logger.info("scrub_cache: extracted %s", sid)
    except Exception as exc:
        logger.exception("scrub_cache: unexpected failure for %s: %s", sid, exc)
    finally:
        with _active_lock:
            _active.discard(sid)


def purge(session_dir: Path) -> None:
    """Delete a session's scrub cache (useful for tests / forced re-extract)."""
    cd = cache_dir(session_dir)
    if cd.exists():
        shutil.rmtree(cd, ignore_errors=True)
