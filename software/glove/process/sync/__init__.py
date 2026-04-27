"""Time sync between GoPro video and external sensor streams.

Primary path: GoPro Labs firmware + UTC precision-time QR
(https://gopro.github.io/labs/control/precisiontime_utc/) — sets the GoPro RTC
so MP4 timecodes match the laptop's `time.time()` UTC epoch directly.
"""

from glove.process.sync.timecode import (
    mp4_get_start_datetime,
    stream_get_start_datetime,
    timecode_to_seconds,
)

__all__ = [
    "mp4_get_start_datetime",
    "stream_get_start_datetime",
    "timecode_to_seconds",
]
