"""
Visualize pose pipeline output for a session.

Plots:
1. 3D trajectory (metric axes in cm, colored by time)
2. Top-down XY view (most intuitive for tabletop validation)
3. Per-axis displacement (X/Y/Z position over time)
4. IMU sanity check (accel magnitude ≈ 9.81, gyro near zero when static)
5. Ruler test overlay (optional: annotate start/end, show measured vs expected distance)

Usage:
    python -m glove.scripts.visualize_poses data/sessions/demo_<date>_<task>_NNNN
    python -m glove.scripts.visualize_poses data/sessions/demo_<date>_<task>_NNNN --expected-distance 0.313
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("ERROR: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


def load_poses(session_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load poses.json → timestamps (s), positions (N×3 in meters)."""
    poses_file = session_dir / "poses.json"
    if not poses_file.exists():
        raise FileNotFoundError(f"No poses.json in {session_dir}")

    with open(poses_file) as f:
        poses = json.load(f)

    if not poses:
        raise ValueError(f"poses.json is empty in {session_dir}")

    timestamps = []
    positions = []
    for p in poses:
        T = np.array(p["transform"])
        timestamps.append(p["timestamp"])
        positions.append(T[:3, 3] if T.shape == (4, 4) else T[:3])

    return np.array(timestamps), np.array(positions)


def load_imu(session_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load imu0/data.csv → timestamps (s), gyro (N×3), accel (N×3)."""
    imu_file = session_dir / "imu0" / "data.csv"
    if not imu_file.exists():
        return np.array([]), np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    timestamps = []
    gyro = []
    accel = []

    with open(imu_file) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            try:
                ts_ns = int(row[0])
                timestamps.append(ts_ns / 1e9)
                gyro.append([float(row[1]), float(row[2]), float(row[3])])
                accel.append([float(row[4]), float(row[5]), float(row[6])])
            except (ValueError, IndexError):
                continue

    return np.array(timestamps), np.array(gyro), np.array(accel)


def plot_session(session_dir: Path, expected_distance: float = None):
    """Generate visualization plots for a session."""
    session_dir = Path(session_dir)
    name = session_dir.name

    # Load data
    try:
        ts_pose, positions = load_poses(session_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return

    ts_imu, gyro, accel = load_imu(session_dir)
    has_imu = len(ts_imu) > 0

    # Convert to cm for display
    pos_cm = positions * 100

    # Time relative to start
    t0 = ts_pose[0]
    t_pose = ts_pose - t0
    t_imu = ts_imu - t0 if has_imu else np.array([])

    # Compute trajectory stats
    total_dist = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    displacement = np.linalg.norm(positions[-1] - positions[0])
    duration = t_pose[-1] - t_pose[0]

    print(f"\n  Session: {name}")
    print(f"  Duration: {duration:.1f}s, Frames: {len(positions)}")
    print(f"  Total path: {total_dist*100:.1f} cm")
    print(f"  Net displacement: {displacement*100:.1f} cm")
    if expected_distance is not None:
        error_pct = abs(displacement - expected_distance) / expected_distance * 100
        print(f"  Expected: {expected_distance*100:.1f} cm, Error: {error_pct:.1f}%")

    # Create figure
    n_rows = 3 if has_imu else 2
    fig = plt.figure(figsize=(16, 5 * n_rows))
    gs = GridSpec(n_rows, 2, figure=fig, hspace=0.3, wspace=0.3)

    # --- 1. 3D trajectory ---
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    colors = plt.cm.viridis(np.linspace(0, 1, len(pos_cm)))
    ax1.scatter(pos_cm[:, 0], pos_cm[:, 1], pos_cm[:, 2], c=colors, s=1)
    ax1.plot(pos_cm[:, 0], pos_cm[:, 1], pos_cm[:, 2], "k-", alpha=0.2, linewidth=0.5)
    ax1.scatter(*pos_cm[0], c="green", s=50, marker="o", label="Start", zorder=5)
    ax1.scatter(*pos_cm[-1], c="red", s=50, marker="x", label="End", zorder=5)
    ax1.set_xlabel("X (cm)")
    ax1.set_ylabel("Y (cm)")
    ax1.set_zlabel("Z (cm)")
    ax1.set_title("3D Trajectory")
    ax1.legend()

    # --- 2. Top-down XY view ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(pos_cm[:, 0], pos_cm[:, 1], c=t_pose, cmap="viridis", s=2)
    ax2.plot(pos_cm[0, 0], pos_cm[0, 1], "go", markersize=10, label="Start")
    ax2.plot(pos_cm[-1, 0], pos_cm[-1, 1], "rx", markersize=10, label="End")
    if expected_distance is not None:
        ax2.annotate(
            f"Measured: {displacement*100:.1f} cm\nExpected: {expected_distance*100:.1f} cm",
            xy=(0.02, 0.98), xycoords="axes fraction", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    ax2.set_xlabel("X (cm)")
    ax2.set_ylabel("Y (cm)")
    ax2.set_title("Top-Down View (XY)")
    ax2.set_aspect("equal")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- 3. Per-axis displacement ---
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(t_pose, pos_cm[:, 0], label="X", alpha=0.8)
    ax3.plot(t_pose, pos_cm[:, 1], label="Y", alpha=0.8)
    ax3.plot(t_pose, pos_cm[:, 2], label="Z", alpha=0.8)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Position (cm)")
    ax3.set_title("Per-Axis Position Over Time")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- 4. IMU sanity check ---
    if has_imu:
        ax4 = fig.add_subplot(gs[2, 0])
        accel_mag = np.linalg.norm(accel, axis=1)
        ax4.plot(t_imu, accel_mag, "b-", linewidth=0.5, alpha=0.5, label="|accel|")
        ax4.axhline(9.81, color="r", linestyle="--", alpha=0.5, label="9.81 m/s²")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Accel magnitude (m/s²)")
        ax4.set_title(f"Accelerometer (mean |a|={np.mean(accel_mag):.2f} m/s²)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[2, 1])
        gyro_mag = np.linalg.norm(gyro, axis=1)
        ax5.plot(t_imu, gyro[:, 0], label="gx", alpha=0.5, linewidth=0.5)
        ax5.plot(t_imu, gyro[:, 1], label="gy", alpha=0.5, linewidth=0.5)
        ax5.plot(t_imu, gyro[:, 2], label="gz", alpha=0.5, linewidth=0.5)
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Angular rate (rad/s)")
        ax5.set_title(f"Gyroscope (mean |ω|={np.mean(gyro_mag):.3f} rad/s)")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    fig.suptitle(f"{name}  |  {duration:.1f}s  |  {total_dist*100:.1f} cm path  |  "
                 f"{len(positions)} poses", fontsize=12)

    # Save
    out_path = session_dir / "pose_viz.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize pose pipeline output"
    )
    parser.add_argument("session", type=str,
                        help="Path to session directory")
    parser.add_argument("--expected-distance", type=float, default=None,
                        help="Expected displacement in meters (for ruler test validation)")
    args = parser.parse_args()

    plot_session(Path(args.session), expected_distance=args.expected_distance)


if __name__ == "__main__":
    main()
