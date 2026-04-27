"""Create a demo session directory from a GoPro MP4 + master-tape raw stream.

Slices `imu.parquet` + `proprio.parquet` from the master tape to the wall-clock
window covering the MP4, creating a fresh session directory with the new schema:

    session_dir/
    ├── meta.yaml
    ├── raw/
    │   ├── raw_video.mp4
    │   ├── imu_daq.parquet      (our QMI8658, sliced from master tape; optional)
    │   └── proprio.parquet      (Hall + ToF, sliced from master tape)
    ├── slam/                    (populated later by build_map/localize)
    └── logs/

Does NOT run SLAM — that's a downstream step (`glove.process extract-gpmf` →
`build-map` / `localize` → `align`).

Sync assumption: the GoPro is Labs-synced to UTC via
https://gopro.github.io/labs/control/precisiontime_utc/ so its MP4 timestamps
match the laptop's `time.time()` UTC epoch directly. The `--mp4-tz-offset-hours`
and `--mp4-wall-offset-seconds` flags exist for edge cases (un-synced cameras,
residual drift) — normal use leaves them at 0.
"""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path
from typing import Optional

import av
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from glove.process.sync.timecode import mp4_get_start_datetime


def _mp4_duration_seconds(mp4_path: Path) -> float:
    with av.open(str(mp4_path)) as container:
        s = container.streams.video[0]
        return float(s.duration * s.time_base)


def _slice_parquet(src: Path, dst: Path, start_wall: float, end_wall: float) -> int:
    """Read src parquet, filter `wall_time` ∈ [start, end], write dst. Returns row count."""
    table = pq.read_table(src)
    mask = pc.and_(
        pc.greater_equal(table["wall_time"], start_wall),
        pc.less_equal(table["wall_time"], end_wall),
    )
    filtered = table.filter(mask)
    pq.write_table(filtered, dst, compression="zstd")
    return len(filtered)


def _next_ordinal(parent: Path, prefix: str) -> int:
    existing = sorted(parent.glob(f"{prefix}*"))
    return len(existing) + 1


def ingest(
    mp4_path: Path,
    stream_dir: Optional[Path],
    sessions_root: Path = Path("data/sessions"),
    task: str = "demo",
    ordinal: Optional[int] = None,
    kind: str = "demo",
    mp4_tz_offset_hours: float = 0.0,
    mp4_wall_offset_seconds: float = -0.333,
) -> Path:
    """Create a session dir from one MP4 + the corresponding master-tape window.

    `mp4_tz_offset_hours=0.0` assumes Labs-synced UTC GoPro.
    `mp4_wall_offset_seconds=-0.333` corrects for the Hero 10's empirically
    measured ~333 ms encoder/metadata pipeline delay (the MP4 timecode is
    stamped after the shutter). Re-measure with a spatial sync target (an
    ArUco sticker, or a hardware pulse if you have one) if swapping cameras.
    """
    mp4_path = Path(mp4_path).resolve()
    stream_dir = Path(stream_dir).resolve() if stream_dir is not None else None
    sessions_root = Path(sessions_root).resolve()
    if stream_dir is None and kind != "mapping":
        raise ValueError("--stream-dir is required for demo sessions; only mapping clips may omit it.")

    # MP4 wall-clock bounds (offsets applied)
    start_dt = mp4_get_start_datetime(mp4_path, tz_offset_hours=mp4_tz_offset_hours)
    start_dt = start_dt + datetime.timedelta(seconds=mp4_wall_offset_seconds)
    duration_s = _mp4_duration_seconds(mp4_path)
    end_dt = start_dt + datetime.timedelta(seconds=duration_s)
    start_wall = start_dt.timestamp()
    end_wall = end_dt.timestamp()

    # Session directory name
    day = start_dt.date().isoformat()
    prefix = f"{kind}_{day}_{task}_"
    if ordinal is None:
        ordinal = _next_ordinal(sessions_root, prefix)
    session_name = f"{kind}_{day}_{task}_{ordinal:04d}"
    session_dir = sessions_root / session_name

    if session_dir.exists():
        raise FileExistsError(f"Session already exists: {session_dir}")

    raw = session_dir / "raw"
    raw.mkdir(parents=True)
    (session_dir / "slam").mkdir()
    (session_dir / "logs").mkdir()

    # Video: copy (APFS CoW on same volume = near-instant)
    shutil.copy2(mp4_path, raw / "raw_video.mp4")

    # Slice master tape. IMU is optional (recorder may have run with --no-imu).
    # Mapping clips may have no master-tape pairing at all — proprio isn't used
    # during build-map, so we skip slicing in that case.
    if stream_dir is not None:
        imu_src = stream_dir / "imu.parquet"
        proprio_src = stream_dir / "proprio.parquet"
        if not proprio_src.exists():
            raise FileNotFoundError(f"Master tape missing proprio.parquet in {stream_dir}")
        n_proprio = _slice_parquet(proprio_src, raw / "proprio.parquet", start_wall, end_wall)
        n_imu = (_slice_parquet(imu_src, raw / "imu_daq.parquet", start_wall, end_wall)
                 if imu_src.exists() else 0)
    else:
        n_proprio = 0
        n_imu = 0

    # Meta
    meta = {
        "session_id": session_name,
        "kind": kind,
        "task": task,
        "ordinal": ordinal,
        "created_at": start_dt.isoformat(),
        "video": {
            "wall_start": start_wall,
            "wall_end": end_wall,
            "duration_s": duration_s,
            "source_file": str(mp4_path),
        },
        "master_tape": {
            "source_dir": str(stream_dir) if stream_dir is not None else None,
            "imu_samples": n_imu,
            "proprio_samples": n_proprio,
        },
        "mp4_tz_offset_hours": mp4_tz_offset_hours,
        "mp4_wall_offset_seconds": mp4_wall_offset_seconds,
        "hardware": {
            "glove": "v3",
            "cameras": ["gopro_hero10_maxlens"],
        },
    }
    (session_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    return session_dir


def _fmt(n: float) -> str:
    return f"{n:,.1f}" if n >= 100 else f"{n:.2f}"


def cli(args) -> int:
    sess = ingest(
        mp4_path=args.mp4,
        stream_dir=args.stream_dir,
        sessions_root=args.sessions_root,
        task=args.task,
        ordinal=args.ordinal,
        kind=args.kind,
        mp4_tz_offset_hours=args.mp4_tz_offset_hours,
        mp4_wall_offset_seconds=args.mp4_wall_offset_seconds,
    )
    meta = yaml.safe_load((sess / "meta.yaml").read_text())
    dur = meta["video"]["duration_s"]
    print(f"Created {sess}")
    print(f"  video: {_fmt(dur)}s")
    print(f"  imu_daq: {meta['master_tape']['imu_samples']:,} samples "
          f"(~{meta['master_tape']['imu_samples']/max(dur, 1e-6):.0f} Hz)")
    print(f"  proprio: {meta['master_tape']['proprio_samples']:,} samples "
          f"(~{meta['master_tape']['proprio_samples']/max(dur, 1e-6):.0f} Hz)")
    print(f"\nInspect via:  python -m glove.dashboard --data-dir {sess.parent}")
    return 0
