"""Plot trajectory X/Y/Z over time for debugging pose pipeline.

Usage:
    python -m glove.scripts.plot_trajectory --session data/sessions/demo_<date>_<task>_NNNN
    python -m glove.scripts.plot_trajectory --session data/sessions/demo_<date>_<task>_NNNN --seed
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


def plot_session(session_dir: Path, show_seed: bool = False):
    poses_path = session_dir / "data" / "poses.parquet"
    if not poses_path.exists():
        print(f"No poses.parquet in {session_dir}")
        return

    table = pq.read_table(poses_path)
    pos_x = table.column("pose_tx").to_numpy()
    pos_y = table.column("pose_ty").to_numpy()
    pos_z = table.column("pose_tz").to_numpy()
    N = len(pos_x)
    frames = np.arange(N)
    time_s = frames / 30.0  # approximate

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"{session_dir.name} — Trajectory", fontsize=14)

    # X over time
    axes[0].plot(time_s, pos_x, 'b-', linewidth=0.8)
    axes[0].set_ylabel("X (m)")
    axes[0].axhline(0, color='gray', linewidth=0.3)
    axes[0].grid(True, alpha=0.3)

    # Y over time
    axes[1].plot(time_s, pos_y, 'g-', linewidth=0.8)
    axes[1].set_ylabel("Y (m)")
    axes[1].axhline(0, color='gray', linewidth=0.3)
    axes[1].grid(True, alpha=0.3)

    # Z over time
    axes[2].plot(time_s, pos_z, 'r-', linewidth=0.8)
    axes[2].set_ylabel("Z (m)")
    axes[2].axhline(0, color='gray', linewidth=0.3)
    axes[2].grid(True, alpha=0.3)

    # XY top-down view
    axes[3].plot(pos_x, pos_y, 'k-', linewidth=0.5, alpha=0.5)
    axes[3].plot(pos_x[0], pos_y[0], 'go', markersize=8, label='start')
    axes[3].plot(pos_x[-1], pos_y[-1], 'rs', markersize=8, label='end')
    axes[3].set_xlabel("X (m)")
    axes[3].set_ylabel("Y (m)")
    axes[3].set_aspect('equal')
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    # Stats
    span = np.array([pos_x.max() - pos_x.min(),
                     pos_y.max() - pos_y.min(),
                     pos_z.max() - pos_z.min()])
    steps = np.sqrt(np.diff(pos_x)**2 + np.diff(pos_y)**2 + np.diff(pos_z)**2)
    path = steps.sum()
    horiz = np.sqrt(span[0]**2 + span[1]**2)

    axes[0].set_title(
        f"span=[{span[0]:.3f}, {span[1]:.3f}, {span[2]:.3f}]m  "
        f"horiz={horiz:.3f}m  path={path:.3f}m  "
        f"step: med={np.median(steps)*1000:.1f}mm max={steps.max()*1000:.1f}mm",
        fontsize=10, loc='left')

    plt.tight_layout()
    out_path = session_dir / "data" / "trajectory_debug.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--seed", action="store_true",
                        help="Plot seed poses (run with --seed-only first)")
    args = parser.parse_args()
    plot_session(Path(args.session))


if __name__ == "__main__":
    main()
