"""Master-tape recorder for continuous DAQ sensor streams.

Opens the STM32 DAQ over USB CDC and appends every IMU / proprio frame to
parquet files on disk. Wall-clock timestamps on every sample let post-session
ingestion slice the stream into per-demo windows matching each GoPro MP4.

Two parquet files per recording:
- imu.parquet      — ~1 kHz QMI8658C (accel + gyro in SI units)
- proprio.parquet  — ~30 Hz MLX90393 Hall ×4 + VL53L1X ToF ×2 + sync metadata

Both carry `wall_time` (float64 epoch seconds) as the sync-to-GoPro anchor,
and `stm32_time_us` (int64 boot-relative microseconds) for jitter analysis.

Flushes to disk every `flush_interval_s` seconds so a hard kill loses at most
that much data. Writes `meta.yaml` at start and updates it on clean exit.
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from glove.data.usb_reader import DaqReader, ImuFrame, ProprioFrame


IMU_SCHEMA = pa.schema([
    ("wall_time", pa.float64()),
    ("stm32_time_us", pa.int64()),
    ("accel_x", pa.float32()),
    ("accel_y", pa.float32()),
    ("accel_z", pa.float32()),
    ("gyro_x", pa.float32()),
    ("gyro_y", pa.float32()),
    ("gyro_z", pa.float32()),
])

PROPRIO_SCHEMA = pa.schema([
    ("wall_time", pa.float64()),
    ("stm32_time_us", pa.int64()),
    *[(f"hall_{i}_{ax}", pa.int16()) for i in range(4) for ax in ("bx", "by", "bz")],
    # int32 for ToF: VL53L1X occasionally emits glitched readings >32767 that
    # aren't the exact 0xFFFF sentinel. int32 preserves the -1 sentinel AND any
    # raw-value glitches for later analysis. Downstream code treats >4500 as invalid.
    ("tof_top_mm", pa.int32()),
    ("tof_bottom_mm", pa.int32()),
    ("sync_timestamp_us", pa.int64()),
    ("sync_seq", pa.uint16()),
    ("status_flags", pa.uint8()),
])


@dataclass
class _Buffer:
    """Column-oriented accumulator that flushes to a ParquetWriter."""

    schema: pa.Schema
    path: Path
    _writer: Optional[pq.ParquetWriter] = None
    _cols: dict[str, list] = field(default_factory=dict)

    def __post_init__(self):
        self._cols = {name: [] for name in self.schema.names}
        self._writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")

    def append(self, row: dict) -> None:
        for name in self.schema.names:
            self._cols[name].append(row[name])

    def flush(self) -> int:
        n = len(next(iter(self._cols.values()))) if self._cols else 0
        if n == 0:
            return 0
        table = pa.table(self._cols, schema=self.schema)
        self._writer.write_table(table)
        for k in self._cols:
            self._cols[k].clear()
        return n

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _imu_row(f: ImuFrame) -> dict:
    a = f.accel_ms2
    g = f.gyro_rads
    return {
        "wall_time": f.local_time,
        "stm32_time_us": f.timestamp_us,
        "accel_x": a[0], "accel_y": a[1], "accel_z": a[2],
        "gyro_x": g[0], "gyro_y": g[1], "gyro_z": g[2],
    }


def _proprio_row(f: ProprioFrame) -> dict:
    row = {
        "wall_time": f.local_time,
        "stm32_time_us": f.timestamp_us,
        "tof_top_mm": f.tof_mm[0],
        "tof_bottom_mm": f.tof_mm[1],
        "sync_timestamp_us": f.sync_timestamp_us,
        "sync_seq": f.sync_seq,
        "status_flags": f.status_flags,
    }
    for i, (bx, by, bz) in enumerate(f.hall_raw):
        row[f"hall_{i}_bx"] = bx
        row[f"hall_{i}_by"] = by
        row[f"hall_{i}_bz"] = bz
    return row


class MasterTapeRecorder:
    """Continuous DAQ-stream writer. Run a single instance per recording session."""

    def __init__(
        self,
        out_dir: Path,
        daq_port: Optional[str] = None,
        flush_interval_s: float = 5.0,
        skip_imu: bool = False,
    ):
        self.out_dir = Path(out_dir)
        self.daq_port = daq_port
        self.flush_interval_s = flush_interval_s
        self.skip_imu = skip_imu

        self._stop = False
        self._imu_count = 0
        self._proprio_count = 0
        self._start_wall: Optional[float] = None

    def _install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            self._stop = True
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _write_meta(self, end_wall: Optional[float] = None) -> None:
        meta = {
            "kind": "raw_stream",
            "daq_port": self.daq_port,
            "started_at_wall": self._start_wall,
            "ended_at_wall": end_wall,
            "imu_samples": self._imu_count,
            "proprio_samples": self._proprio_count,
        }
        if self._start_wall and end_wall:
            meta["duration_s"] = end_wall - self._start_wall
        (self.out_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    def run(self, status_cb=None) -> None:
        """Record until SIGINT/SIGTERM. Call `status_cb(stats_dict)` ~1 Hz if given."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._install_signal_handlers()

        imu_buf = None if self.skip_imu else _Buffer(IMU_SCHEMA, self.out_dir / "imu.parquet")
        pro_buf = _Buffer(PROPRIO_SCHEMA, self.out_dir / "proprio.parquet")

        self._start_wall = time.time()
        last_flush = self._start_wall
        last_status = self._start_wall
        last_imu_count = 0
        last_proprio_count = 0

        try:
            self._write_meta()  # initial meta so killed runs still leave a breadcrumb
            with DaqReader(port=self.daq_port) as daq:
                for frame in daq:
                    if self._stop:
                        break

                    try:
                        if isinstance(frame, ImuFrame):
                            if imu_buf is not None:
                                imu_buf.append(_imu_row(frame))
                                self._imu_count += 1
                        else:
                            pro_buf.append(_proprio_row(frame))
                            self._proprio_count += 1
                    except Exception as e:
                        # A single malformed frame must not kill the whole session.
                        # Log to stderr and keep recording.
                        import sys
                        print(f"\n[recorder] dropped bad frame: {e}", file=sys.stderr)

                    now = time.time()
                    if now - last_flush >= self.flush_interval_s:
                        try:
                            if imu_buf is not None:
                                imu_buf.flush()
                            pro_buf.flush()
                        except Exception as e:
                            import sys
                            print(f"\n[recorder] flush error: {e} — continuing",
                                  file=sys.stderr)
                        last_flush = now

                    if status_cb is not None and now - last_status >= 1.0:
                        dt = now - last_status
                        status_cb({
                            "elapsed_s": now - self._start_wall,
                            "imu_total": self._imu_count,
                            "proprio_total": self._proprio_count,
                            "imu_hz": (self._imu_count - last_imu_count) / dt,
                            "proprio_hz": (self._proprio_count - last_proprio_count) / dt,
                        })
                        last_status = now
                        last_imu_count = self._imu_count
                        last_proprio_count = self._proprio_count
        finally:
            if imu_buf is not None:
                imu_buf.close()
            pro_buf.close()
            self._write_meta(end_wall=time.time())
