"""Run the whole analysis: python scripts/run_analysis.py

Writes outputs/tables/*.csv, outputs/figures/*.svg and outputs/results.json,
then prints the digest.  Everything in README.md is taken from these outputs.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from insight.report import main  # noqa: E402


def cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--threshold",
        type=float,
        default=19.0,
        help="deployment threshold in minutes (default 19, the sweep peak)",
    )
    args = ap.parse_args()
    main(threshold=args.threshold)


if __name__ == "__main__":
    cli()
