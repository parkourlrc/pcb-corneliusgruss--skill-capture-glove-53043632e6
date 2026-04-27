"""CLI entry point for glove.capture.

Usage:
    python -m glove.capture record [--out PATH] [--port /dev/tty.usbmodem...]
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

from glove.capture import offload as offload_mod
from glove.capture.recorder import MasterTapeRecorder


def _print_status(stats: dict) -> None:
    # Clear to end of line, carriage-return back — gives a single updating row.
    line = (
        f"[{stats['elapsed_s']:7.1f}s] "
        f"IMU {stats['imu_hz']:6.1f} Hz ({stats['imu_total']:>8d}) · "
        f"proprio {stats['proprio_hz']:5.1f} Hz ({stats['proprio_total']:>6d})"
    )
    sys.stdout.write("\r\x1b[2K" + line)
    sys.stdout.flush()


def cmd_record(args: argparse.Namespace) -> int:
    if args.out is None:
        day = datetime.date.today().isoformat()
        out = Path("data/raw_stream") / day
        # Avoid clobbering an earlier run the same day — suffix with HHMMSS if needed.
        if out.exists() and any(out.iterdir()):
            stamp = datetime.datetime.now().strftime("%H%M%S")
            out = Path("data/raw_stream") / f"{day}_{stamp}"
    else:
        out = Path(args.out)

    print(f"Recording to {out}")
    print("Press Ctrl-C to stop.\n")

    recorder = MasterTapeRecorder(out_dir=out, daq_port=args.port, skip_imu=args.no_imu)
    crashed = False
    try:
        recorder.run(status_cb=_print_status)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        crashed = True
        print()
        banner = "!" * 72
        print(f"\n\033[1;31m{banner}")
        print(f"RECORDER CRASHED — MASTER TAPE STOPPED AT {recorder._proprio_count} PROPRIO SAMPLES")
        print(f"Error: {type(exc).__name__}: {exc}")
        print("Any demos recorded after this point have NO proprio data.")
        print(f"{banner}\033[0m\n")
        raise

    print()  # newline after the updating status row
    if not crashed:
        print(f"Stopped. Wrote {out}/imu.parquet and {out}/proprio.parquet")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="glove.capture")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="Record a continuous DAQ master tape.")
    rec.add_argument("--out", type=str, default=None,
                     help="Output directory (default: data/raw_stream/YYYY-MM-DD[_HHMMSS]).")
    rec.add_argument("--port", type=str, default=None,
                     help="Serial port (default: auto-detect STM32 CDC).")
    rec.add_argument("--no-imu", action="store_true",
                     help="Skip writing IMU to disk (still read from wire). Useful when the "
                          "policy stack doesn't need DAQ IMU and you want lower CPU load.")
    rec.set_defaults(func=cmd_record)

    off = sub.add_parser("offload",
                         help="Copy GX*.MP4 from a mounted GoPro SD card into data/raw_video/YYYY-MM-DD/.")
    off.add_argument("--volume", type=Path, default=None,
                     help="GoPro card root (auto-detect if omitted).")
    off.add_argument("--dest-root", type=Path, default=Path("data/raw_video"),
                     help="Destination parent dir (date subfolders created underneath).")
    off.add_argument("--delete-source", action="store_true",
                     help="Delete files from the card after copying. Off by default.")
    off.add_argument("--dry-run", action="store_true")
    off.set_defaults(func=offload_mod.cli)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
