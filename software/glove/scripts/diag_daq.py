"""Diagnostic tool for STM32 DAQ board bring-up.

Reads the boot diagnostic text output from the DAQ, then streams
sensor data with live stats. Useful for verifying I2C bus, mux,
and individual sensors during hardware bring-up.

Usage:
    python -m glove.scripts.diag_daq [port]
    python -m glove.scripts.diag_daq /dev/cu.usbmodem<N>
"""

import sys
import time
import struct
from collections import defaultdict

import serial
import serial.tools.list_ports


def find_stm32_port() -> str | None:
    for port in serial.tools.list_ports.comports():
        if port.vid == 0x0483:
            return port.device
    return None


def read_boot_diag(ser: serial.Serial, timeout: float = 60.0) -> str:
    """Read boot diagnostic text until END DIAG marker or timeout.

    Default timeout is 60s so the user has time to physically unplug+replug
    the USB cable, which power-cycles the board and re-fires the boot diag.
    """
    text = bytearray()
    start = time.time()
    deadline = start + timeout
    seen_start = False
    last_countdown = 0

    while time.time() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            text.extend(chunk)
            decoded = text.decode("ascii", errors="replace")
            if "=== END DIAG ===" in decoded:
                return decoded
            if "=== Glove DAQ" in decoded:
                seen_start = True

        # Periodic countdown so user knows there's time to replug.
        elapsed = int(time.time() - start)
        if elapsed >= last_countdown + 5 and not seen_start:
            remaining = int(deadline - time.time())
            print(f"  ...waiting for boot diag ({remaining}s left — replug USB to trigger)")
            last_countdown = elapsed

    decoded = text.decode("ascii", errors="replace")
    if seen_start:
        return decoded + "\n[timeout - END DIAG not received]"
    return "[no boot diagnostic received - board may already be streaming]"


def stream_sensor_data(ser: serial.Serial):
    """Stream binary packets and print live stats."""
    from glove.data.usb_reader import (
        PACKET_IMU, PACKET_PROPRIO, PACKET_SIZES,
        parse_imu, parse_proprio, ImuFrame, ProprioFrame,
    )

    imu_count = 0
    proprio_count = 0
    last_report = time.time()
    last_hall = ((0, 0, 0),) * 4
    last_accel = [0.0, 0.0, 0.0]

    print("\n--- Live Sensor Stream (Ctrl+C to stop) ---")
    print(f"{'Time':>6s}  {'IMU/s':>6s}  {'Prop/s':>6s}  "
          f"{'Enc0':>6s} {'Enc1':>6s} {'Enc2':>6s} {'Enc3':>6s}  "
          f"{'ax':>7s} {'ay':>7s} {'az':>7s}")

    while True:
        byte = ser.read(1)
        if not byte:
            continue

        ptype = byte[0]
        if ptype not in PACKET_SIZES:
            continue

        remaining = PACKET_SIZES[ptype] - 1
        payload = ser.read(remaining)
        if len(payload) < remaining:
            continue

        packet = byte + payload
        now = time.time()

        if ptype == PACKET_IMU:
            imu_count += 1
            frame = parse_imu(packet, local_time=now)
            last_accel = frame.accel_ms2
        elif ptype == PACKET_PROPRIO:
            proprio_count += 1
            frame = parse_proprio(packet, local_time=now)
            last_hall = frame.hall_raw

        # Report every second
        if now - last_report >= 1.0:
            dt = now - last_report
            enc_strs = []
            for bx, by, _bz in last_hall:
                if bx == 0 and by == 0:
                    enc_strs.append("  ---")
                else:
                    import math
                    enc_strs.append(f"{math.degrees(math.atan2(by, bx)):+6.1f}°")

            print(f"{now - last_report:6.1f}  "
                  f"{imu_count / dt:6.0f}  {proprio_count / dt:6.0f}  "
                  f"{enc_strs[0]:>6s} {enc_strs[1]:>6s} "
                  f"{enc_strs[2]:>6s} {enc_strs[3]:>6s}  "
                  f"{last_accel[0]:+7.2f} {last_accel[1]:+7.2f} {last_accel[2]:+7.2f}")

            imu_count = 0
            proprio_count = 0
            last_report = now


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else find_stm32_port()
    if not port:
        print("Error: STM32 DAQ not found. Specify port as argument.")
        sys.exit(1)

    print(f"Connecting to {port}...")

    # Open with short timeout for boot diag reading
    ser = serial.Serial(port, baudrate=115200, timeout=0.5)
    time.sleep(0.3)
    ser.reset_input_buffer()

    print("Waiting for boot diagnostics — physically replug the USB cable now.")
    print("(60s timeout; will give up and stream sensors after that)\n")
    diag = read_boot_diag(ser, timeout=60.0)
    print(diag)

    try:
        stream_sensor_data(ser)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
