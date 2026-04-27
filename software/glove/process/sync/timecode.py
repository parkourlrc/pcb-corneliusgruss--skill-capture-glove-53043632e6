"""GoPro MP4 wall-clock extraction.

Ported from UMI (umi/common/timecode_util.py, Cheng Chi et al. 2024).

The trick: GoPro writes two metadata fields we combine for sub-frame-precision
wall-clock on the first video frame:
- creation_time — ISO datetime with millisecond resolution
- timecode      — "HH:MM:SS:FF" (frames within day) at the recording frame rate

Combined they give the absolute datetime of frame 0. Subsequent frames are at
stream PTS * time_base offsets from there. This is the anchor for sync-ing an
external sensor stream whose samples carry wall-clock timestamps.
"""

from __future__ import annotations

import datetime
from fractions import Fraction

import av


def timecode_to_seconds(
    timecode: str, frame_rate: int | float | Fraction
) -> float | Fraction:
    """Convert SMPTE non-drop-frame timecode to seconds since midnight.

    Treats `frame_rate` as if it were the nearest integer for the frame index
    arithmetic (29.97 → 30, 59.94 → 60), then divides by the true rate for
    the fractional-second result. This matches the convention GoPro writes.
    """
    int_rate = round(frame_rate)
    h, m, s, f = [int(x) for x in timecode.split(":")]
    frames = (3600 * h + 60 * m + s) * int_rate + f
    return frames / frame_rate


def stream_get_start_datetime(
    stream: av.stream.Stream, tz_offset_hours: float = 0.0
) -> datetime.datetime:
    """Absolute datetime of the first video frame.

    `tz_offset_hours` is added to the parsed time. GoPros write local time stamped
    with a `Z` (UTC) suffix — if the camera is set to local time and the recording
    laptop runs on UTC epoch (`time.time()`), pass tz_offset_hours = +N where N is
    the laptop's local offset from UTC (e.g. EDT → +4, CET → +1).
    """
    rate = stream.average_rate
    tc = stream.metadata["timecode"]
    creation = stream.metadata["creation_time"]

    seconds_since_midnight = float(timecode_to_seconds(tc, rate))
    day = datetime.datetime.strptime(creation, r"%Y-%m-%dT%H:%M:%S.%fZ").replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=datetime.timezone.utc
    )
    return day + datetime.timedelta(seconds=seconds_since_midnight + tz_offset_hours * 3600.0)


def mp4_get_start_datetime(
    mp4_path: str, tz_offset_hours: float = 0.0
) -> datetime.datetime:
    """Wall-clock of the first video frame for an MP4 file.

    See `stream_get_start_datetime` for `tz_offset_hours` semantics.
    """
    with av.open(str(mp4_path)) as container:
        return stream_get_start_datetime(container.streams.video[0],
                                         tz_offset_hours=tz_offset_hours)
