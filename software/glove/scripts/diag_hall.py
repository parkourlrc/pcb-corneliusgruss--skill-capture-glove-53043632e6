"""Live (Bx, By) scatter + Bz strip chart for all 4 MLX90393 Hall sensors.

Smoke test for finger-joint magnet calibration and force-sensitivity check.

Top row — (Bx, By) per sensor:
- Rotate one joint at a time. A clean circular arc (offset from origin) means
  the magnet sweep is well-conditioned. Joint angle is atan2 around the arc's
  *center*, not around the electrical origin — see `glove.data.hall_calibration`
  for the circle-fit recovery.

Bottom row — Bz over time per sensor:
- Bz tracks magnet-sensor distance along the axial direction. Hold a finger
  at a fixed bend, then squeeze the fingertip — if Bz shifts proportional to
  squeeze force, the linkage transmits force to the magnet and Bz can be used
  as a force proxy in the policy. If Bz barely moves, the linkage is too
  stiff and force estimation needs a different sensor.

Usage:
    python -m glove.scripts.diag_hall
    python -m glove.scripts.diag_hall --port /dev/cu.usbmodem<N>
    python -m glove.scripts.diag_hall --trail-seconds 5

Cannot run while `glove.capture record` is running — both want the USB port.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from glove.data.usb_reader import DaqReader, ProprioFrame


SENSOR_LABELS = ["index_mcp", "index_pip", "3f_mcp", "3f_pip"]
PROPRIO_HZ = 30  # nominal


class ReaderState:
    """Shared state between reader thread and plot thread."""

    def __init__(self, trail_len: int) -> None:
        # Each entry: (t_rel, bx, by, bz)
        self.trails: list[deque] = [deque(maxlen=trail_len) for _ in range(4)]
        self.frame_count: int = 0
        self.t0: float | None = None
        self.error: str | None = None
        self.stop = threading.Event()
        self.lock = threading.Lock()


def reader_thread(port: str | None, state: ReaderState) -> None:
    """Stream proprio frames into per-sensor (t, Bx, By, Bz) deques until stop is set."""
    try:
        with DaqReader(port=port) as daq:
            for frame in daq:
                if state.stop.is_set():
                    return
                if not isinstance(frame, ProprioFrame):
                    continue
                now = time.time()
                with state.lock:
                    if state.t0 is None:
                        state.t0 = now
                    t_rel = now - state.t0
                    for i, (bx, by, bz) in enumerate(frame.hall_raw):
                        state.trails[i].append((t_rel, bx, by, bz))
                    state.frame_count += 1
                    if state.frame_count <= 3:
                        raw = [t[-1] for t in state.trails]
                        print(
                            f"[reader] frame {state.frame_count}: "
                            + "  ".join(
                                f"{SENSOR_LABELS[i]}=({raw[i][1]:+6d},{raw[i][2]:+6d},{raw[i][3]:+6d})"
                                for i in range(4)
                            )
                        )
    except Exception as e:
        with state.lock:
            state.error = f"{type(e).__name__}: {e}"
        print(f"[reader] FATAL: {state.error}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=None, help="DAQ serial port (auto-detect if omitted)")
    parser.add_argument("--trail-seconds", type=float, default=5.0)
    args = parser.parse_args()

    trail_len = max(1, int(args.trail_seconds * PROPRIO_HZ))
    state = ReaderState(trail_len)

    t = threading.Thread(target=reader_thread, args=(args.port, state), daemon=True)
    t.start()

    deadline = time.time() + 3.0
    while time.time() < deadline:
        with state.lock:
            if state.frame_count > 0 or state.error:
                break
        time.sleep(0.05)
    with state.lock:
        if state.error:
            print(f"ERROR: reader failed to start: {state.error}", file=sys.stderr)
            return
        if state.frame_count == 0:
            print(
                "WARNING: no proprio frames received in 3s. "
                "Is the DAQ connected and not in use by another process?",
                file=sys.stderr,
            )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        "Top: (Bx, By) raw — sweep joint, look for clean arc.   "
        "Bottom: Bz over time — squeeze fingertip, watch for shift."
    )

    xy_panels = []
    for col in range(4):
        ax = axes[0, col]
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("Bx (LSB)")
        ax.set_ylabel("By (LSB)")
        trail_line, = ax.plot([], [], "-", color="C0", alpha=0.4, lw=1)
        head_dot, = ax.plot([], [], "o", color="C3", ms=10)
        xy_panels.append((ax, trail_line, head_dot, col))

    bz_panels = []
    for col in range(4):
        ax = axes[1, col]
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("Bz (LSB)")
        bz_line, = ax.plot([], [], "-", color="C2", lw=1)
        bz_panels.append((ax, bz_line, col))

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(_frame_idx):
        with state.lock:
            if state.error:
                fig.suptitle(f"READER ERROR: {state.error}", color="red")
                return []
            snap = [list(state.trails[i]) for i in range(4)]

        for ax, trail_line, head_dot, i in xy_panels:
            pts = snap[i]
            if not pts:
                ax.set_title(f"{SENSOR_LABELS[i]} — no data")
                continue
            xs = [p[1] for p in pts]
            ys = [p[2] for p in pts]
            trail_line.set_data(xs, ys)
            head_dot.set_data([xs[-1]], [ys[-1]])
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            half = max(200.0, 0.6 * max(x_max - x_min, y_max - y_min))
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_title(
                f"{SENSOR_LABELS[i]}  Bx={xs[-1]:+6d}  By={ys[-1]:+6d}"
            )

        for ax, bz_line, i in bz_panels:
            pts = snap[i]
            if not pts:
                continue
            ts = [p[0] for p in pts]
            bzs = [p[3] for p in pts]
            bz_line.set_data(ts, bzs)
            ax.set_xlim(max(0.0, ts[-1] - args.trail_seconds), ts[-1] + 0.1)
            bz_min, bz_max = min(bzs), max(bzs)
            pad = max(200.0, 0.1 * (bz_max - bz_min))
            ax.set_ylim(bz_min - pad, bz_max + pad)
            ax.set_title(f"Bz={bzs[-1]:+6d}  span={bz_max - bz_min:5d}")

        return []

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        state.stop.set()
        t.join(timeout=1.0)
    _ = ani


if __name__ == "__main__":
    main()
