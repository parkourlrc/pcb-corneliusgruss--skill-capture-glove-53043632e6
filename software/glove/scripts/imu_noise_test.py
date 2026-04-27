"""Quick static IMU noise test for QMI8658C on STM32 DAQ.

Place the board flat and still on a table, then run:
    python3 software/glove/scripts/imu_noise_test.py

Standalone — only needs pyserial and numpy (no full stack install).
"""

import struct
import sys
import time

import numpy as np
import serial
import serial.tools.list_ports

# --- Constants from usb_reader.py ---
PACKET_IMU = 0x01
PACKET_PROPRIO = 0x02
IMU_PKT_SIZE = 17
PROPRIO_PKT_SIZE = 46
ACCEL_LSB_TO_MS2 = 9.80665 / 16384.0
GYRO_LSB_TO_RADS = (1.0 / 128.0) * (3.14159265358979 / 180.0)


def find_stm32():
    for port in serial.tools.list_ports.comports():
        if port.vid == 0x0483:
            return port.device
    return None


def main():
    duration = 10

    print("=== QMI8658C Static Noise Test ===")
    print("Board must be FLAT and STILL on a table.")
    print(f"Collecting {duration}s of IMU data...\n")

    port = find_stm32()
    if not port:
        print("ERROR: No STM32 DAQ found. Is it plugged in?")
        sys.exit(1)

    ser = serial.Serial(port, baudrate=115200, timeout=1.0)
    time.sleep(0.5)
    ser.reset_input_buffer()
    print(f"Connected to {port}")

    accel_data = []
    gyro_data = []
    t0 = time.time()
    count = 0

    try:
        while (time.time() - t0) < duration:
            # Find valid packet type
            b = ser.read(1)
            if len(b) == 0:
                continue
            ptype = b[0]

            if ptype == PACKET_IMU:
                payload = ser.read(IMU_PKT_SIZE - 1)
                if len(payload) < IMU_PKT_SIZE - 1:
                    continue
                ts, ax, ay, az, gx, gy, gz = struct.unpack_from("<I3h3h", payload, 0)
                accel_data.append([ax * ACCEL_LSB_TO_MS2, ay * ACCEL_LSB_TO_MS2, az * ACCEL_LSB_TO_MS2])
                gyro_data.append([gx * GYRO_LSB_TO_RADS, gy * GYRO_LSB_TO_RADS, gz * GYRO_LSB_TO_RADS])
                count += 1
                if count % 400 == 0:
                    elapsed = time.time() - t0
                    print(f"  {elapsed:.1f}s — {count} frames ({count/elapsed:.0f} Hz)")
            elif ptype == PACKET_PROPRIO:
                ser.read(PROPRIO_PKT_SIZE - 1)  # skip
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

    if count < 100:
        print(f"\nOnly got {count} frames — is the board streaming?")
        sys.exit(1)

    accel = np.array(accel_data)
    gyro = np.array(gyro_data)
    dt = time.time() - t0
    rate = count / dt

    print(f"\n{'='*60}")
    print(f"Collected {count} frames in {dt:.1f}s ({rate:.0f} Hz)")
    print(f"{'='*60}")

    # Accel
    accel_mean = accel.mean(axis=0)
    accel_std = accel.std(axis=0)
    gravity_mag = np.linalg.norm(accel_mean)

    print(f"\n--- Accelerometer ---")
    print(f"Mean [m/s²]:  X={accel_mean[0]:+8.4f}  Y={accel_mean[1]:+8.4f}  Z={accel_mean[2]:+8.4f}")
    print(f"Std  [m/s²]:  X={accel_std[0]:8.4f}  Y={accel_std[1]:8.4f}  Z={accel_std[2]:8.4f}")
    print(f"|g| = {gravity_mag:.4f} m/s² (expected 9.8066)")

    accel_nd_ug = (accel_std / np.sqrt(rate)) / 9.80665 * 1e6

    print(f"\nNoise density [µg/√Hz]:")
    print(f"  X={accel_nd_ug[0]:.1f}  Y={accel_nd_ug[1]:.1f}  Z={accel_nd_ug[2]:.1f}")

    # Gyro
    gyro_mean = gyro.mean(axis=0)
    gyro_std = gyro.std(axis=0)
    gyro_mean_dps = np.degrees(gyro_mean)
    gyro_std_dps = np.degrees(gyro_std)

    print(f"\n--- Gyroscope ---")
    print(f"Mean [°/s]:   X={gyro_mean_dps[0]:+8.4f}  Y={gyro_mean_dps[1]:+8.4f}  Z={gyro_mean_dps[2]:+8.4f}")
    print(f"Std  [°/s]:   X={gyro_std_dps[0]:8.4f}  Y={gyro_std_dps[1]:8.4f}  Z={gyro_std_dps[2]:8.4f}")

    gyro_nd_dps = np.degrees(gyro_std / np.sqrt(rate))

    print(f"\nNoise density [°/s/√Hz]:")
    print(f"  X={gyro_nd_dps[0]:.4f}  Y={gyro_nd_dps[1]:.4f}  Z={gyro_nd_dps[2]:.4f}")

    # Comparison
    accel_worst = accel_nd_ug.max()
    gyro_worst = gyro_nd_dps.max()

    print(f"\n{'='*60}")
    print(f"COMPARISON (worst axis)")
    print(f"{'='*60}")

    specs = [
        ("QMI8658C datasheet", 120, 0.01),
        ("ICM-20948 (V2 glove)", 230, 0.015),
        ("ICM-42688-P (intended)", 70, 0.0028),
        ("THIS CHIP (measured)", accel_worst, gyro_worst),
    ]

    print(f"\n{'IMU':<28s} {'Accel [µg/√Hz]':>16s} {'Gyro [°/s/√Hz]':>16s}")
    print("-" * 62)
    for name, a, g in specs:
        marker = " <--" if name.startswith("THIS") else ""
        print(f"{name:<28s} {a:>14.1f}   {g:>14.4f}{marker}")

    print(f"\n--- Verdict ---")
    if accel_worst < 300 and gyro_worst < 0.02:
        print("GOOD ENOUGH for OKVIS2. Noise is within usable range.")
        if accel_worst < 150 and gyro_worst < 0.012:
            print("  Close to datasheet spec — chip is legit.")
        else:
            print("  Noisier than datasheet but workable.")
        print(f"\n  Suggested OKVIS2 config update (sigma = measured std):")
        print(f"    sigma_a_c: {accel_std.mean():.4f}   # [m/s²/√Hz]")
        print(f"    sigma_g_c: {gyro_std.mean():.6f}   # [rad/s/√Hz]")
    else:
        print("NOISY — may cause issues with OKVIS2.")
        if accel_worst > 500:
            print("  Accel very high — was the board truly still?")
        if gyro_worst > 0.05:
            print("  Gyro very high — possible defective chip.")


if __name__ == "__main__":
    main()
