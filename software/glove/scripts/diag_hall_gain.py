"""Send the 'G' command to the DAQ and print the live GAIN_SEL readback.

Use this to verify whether boot-time gain configuration actually took effect
on the MLX90393 chips. Non-destructive — does not reset/reinit the sensors.

Usage:
    python -m glove.scripts.diag_hall_gain
    python -m glove.scripts.diag_hall_gain --port /dev/cu.usbmodem<N>

Cannot run while `glove.capture record` or `diag_hall` is using the port.
"""

from __future__ import annotations

import argparse
import sys
import time

import serial
import serial.tools.list_ports


def find_stm32_port() -> str | None:
    for port in serial.tools.list_ports.comports():
        if port.vid == 0x0483:
            return port.device
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default=None)
    args = ap.parse_args()

    port = args.port or find_stm32_port()
    if not port:
        print("Error: STM32 DAQ not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to {port}...")
    ser = serial.Serial(port, baudrate=115200, timeout=0.2)
    time.sleep(0.3)
    ser.reset_input_buffer()

    print("Sending 'G' (gain readback)...")
    ser.write(b"G")
    ser.flush()

    # Drain text response — interleaved with binary sensor packets.
    # Strategy: collect bytes for ~2s; pull printable text out of the mix.
    deadline = time.time() + 2.0
    buf = bytearray()
    while time.time() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf.extend(chunk)
        if b"=== END GAIN ===" in buf:
            break

    ser.close()

    # Extract just the GAIN diag block (between BEGIN/END markers).
    text = buf.decode("ascii", errors="replace")
    start = text.find("=== Hall GAIN_SEL ===")
    end = text.find("=== END GAIN ===")
    if start >= 0 and end >= 0:
        print("\n" + text[start : end + len("=== END GAIN ===")])
    else:
        print("\n[no GAIN diag block found in response]")
        # Print any printable ASCII for debugging
        printable = "".join(c for c in text if c.isprintable() or c in "\r\n")
        if printable.strip():
            print("Raw printable bytes received:")
            print(printable)


if __name__ == "__main__":
    main()
