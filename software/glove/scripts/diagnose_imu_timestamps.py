"""Script to diagnose IMU timestamp jitter caused by serial buffering.

Usage:
    python -m glove.scripts.diagnose_imu_timestamps --session data/sessions/demo_<date>_<task>_NNNN
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run(session_dir: Path):
    imu_csv = session_dir / "imu0" / "data.csv"
    if not imu_csv.exists():
        print(f"No IMU data found at {imu_csv}")
        return

    # Load EuRoC CSV (which uses the Python host timestamps)
    host_ts_ns = []
    with open(imu_csv) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 1:
                host_ts_ns.append(int(row[0]))
                
    host_ts = np.array(host_ts_ns) / 1e9
    
    if len(host_ts) < 2:
        print("Not enough IMU data.")
        return

    # Calculate dt between consecutive readings
    dt_ms = np.diff(host_ts) * 1000.0
    
    duration = host_ts[-1] - host_ts[0]
    rate = len(host_ts) / duration
    
    print(f"--- IMU Timestamp Diagnostics ({session_dir.name}) ---")
    print(f"Total readings: {len(host_ts)}")
    print(f"Expected dt: {1000.0/rate:.1f} ms ({rate:.0f}Hz)")
    print(f"Mean dt: {np.mean(dt_ms):.2f} ms")
    print(f"Median dt: {np.median(dt_ms):.2f} ms")
    print(f"Min dt: {np.min(dt_ms):.2f} ms")
    print(f"Max dt: {np.max(dt_ms):.2f} ms")
    print(f"Std dev: {np.std(dt_ms):.2f} ms")
    
    # Count how many readings are "clumped" (arrived in same millisecond)
    clumped = np.sum(dt_ms < 1.0)
    print(f"Clumped readings (<1ms apart): {clumped} ({(clumped/len(dt_ms))*100:.1f}%)")
    
    # Count how many readings are "delayed" (arrived >10ms apart)
    delayed = np.sum(dt_ms > 10.0)
    print(f"Delayed readings (>10ms apart): {delayed} ({(delayed/len(dt_ms))*100:.1f}%)")

    # Plot histogram of dt
    plt.figure(figsize=(10, 5))
    plt.hist(dt_ms, bins=100, range=(0, 20), color='blue', alpha=0.7)
    plt.axvline(1000.0/rate, color='red', linestyle='dashed', linewidth=2, label=f'Expected ({1000.0/rate:.1f}ms)')
    plt.title('IMU dt Distribution (Host Timestamps)')
    plt.xlabel('dt (ms)')
    plt.ylabel('Count')
    plt.legend()
    
    plot_path = session_dir / "diagnostic_imu_dt.png"
    plt.savefig(plot_path)
    print(f"\nSaved dt histogram to {plot_path}")
    
    print("\nNote: uniform dt = MCU hardware timestamps were used (good).")
    print("High variance / clumping = host-side timestamps were used (bad).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, required=True)
    args = parser.parse_args()
    run(Path(args.session))
