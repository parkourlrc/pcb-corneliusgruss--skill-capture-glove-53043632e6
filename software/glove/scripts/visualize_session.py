#!/usr/bin/env python3
"""
Visualize a capture session.

Usage:
    python -m glove.scripts.visualize_session <session_path>
    python -m glove.scripts.visualize_session <session_path> --save
"""

import argparse
import sys
from pathlib import Path

from glove.viz.session_viz import visualize_session


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize a capture session")
    parser.add_argument("session_path", type=str, help="Path to session directory")
    parser.add_argument("--save", action="store_true", help="Save to session_path/viz/")

    args = parser.parse_args()

    session_path = Path(args.session_path)
    if not session_path.exists():
        print(f"Error: Session not found: {session_path}")
        return 1

    visualize_session(str(session_path), save=args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
