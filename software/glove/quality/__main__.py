"""Entry point: `python -m glove.quality <command>`."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="python -m glove.quality")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("backfill", add_help=False,
                   help="Score all aligned sessions (use --help after the command)")

    args, rest = parser.parse_known_args()

    if args.cmd == "backfill":
        from glove.quality.backfill import cli as backfill_cli
        sys.exit(backfill_cli(rest))
    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
