#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.config import DEFAULT_SEEDS
from core.split import run_split_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 01: split aligned train tables into multiseed folds."
    )
    parser.add_argument(
        "--mb-train",
        default=str(PACKAGE_ROOT / "mb_otu_filter0right_train.csv"),
        help="Microbe training CSV path",
    )
    parser.add_argument(
        "--mt-train",
        default=str(PACKAGE_ROOT / "metabolites_train.csv"),
        help="Metabolite training CSV path",
    )
    parser.add_argument(
        "--out-root",
        default=str(SCRIPT_DIR / "multiround_raw"),
        help="Output split root",
    )
    parser.add_argument("--label-column", default=None, help="Optional label column in CSV")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help=f"Random seeds. Default: {DEFAULT_SEEDS}",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Fold count")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_split_pipeline(
        mb_train=args.mb_train,
        mt_train=args.mt_train,
        out_root=args.out_root,
        label_column=args.label_column,
        seeds=args.seeds,
        n_splits=args.n_splits,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
