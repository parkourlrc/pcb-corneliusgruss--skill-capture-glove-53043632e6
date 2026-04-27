"""On-demand JPEG extraction from session MP4s.

Used by the review-detail route as a fallback when the on-disk scrub cache
hasn't yet warmed for the requested frame index.
"""

import io
from pathlib import Path

import av


def _jpeg_from_av_frame(frame) -> bytes:
    buf = io.BytesIO()
    frame.to_image().save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _frame_from_mp4_linear(video_path: Path, index: int) -> bytes | None:
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == index:
                out = _jpeg_from_av_frame(frame)
                container.close()
                return out
        container.close()
    except Exception:
        pass
    return None


def frame_from_mp4(video_path: Path, index: int) -> bytes | None:
    """JPEG from MP4 frame index: fast seek + PTS match for large indices."""
    if not video_path.exists() or index < 0:
        return None

    if index < 64:
        return _frame_from_mp4_linear(video_path, index)

    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate and float(stream.average_rate) > 1e-6 else 30.0
        tb = stream.time_base
        tb_f = float(tb)
        start_off = (
            float(stream.start_time * tb)
            if stream.start_time is not None
            else 0.0
        )

        def frame_display_index(frame) -> int | None:
            if frame.pts is None:
                return None
            if stream.start_time is not None:
                t = float((frame.pts - stream.start_time) * tb)
            else:
                t = float(frame.pts * tb) - start_off
            return int(round(t * fps))

        target_sec = start_off + index / fps
        seek_sec = max(start_off, target_sec - 1.0)
        seek_pts = int(seek_sec / tb_f)
        try:
            container.seek(seek_pts, stream=stream, backward=True)
        except Exception:
            container.close()
            return _frame_from_mp4_linear(video_path, index)

        best = None
        best_d = 10**9
        for frame in container.decode(stream):
            fi = frame_display_index(frame)
            if fi is None:
                continue
            d = abs(fi - index)
            if d < best_d:
                best_d = d
                best = frame
            if fi >= index:
                break
        container.close()
        if best is not None and best_d <= 4:
            return _jpeg_from_av_frame(best)
    except Exception:
        pass
    return _frame_from_mp4_linear(video_path, index)
