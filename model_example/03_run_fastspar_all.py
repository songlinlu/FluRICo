#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.fastspar import run_fastspar_from_split_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 03: run FastSpar for all task dirs.")
    parser.add_argument("--root", default=str(SCRIPT_DIR), help="model_example root")
    parser.add_argument(
        "--task-dirs",
        nargs="+",
        default=["hc_vs_dis", "mia_vs_ia"],
        help="Task directories under --root",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional seed subset")
    parser.add_argument("--folds", nargs="+", type=int, default=None, help="Optional fold subset")
    parser.add_argument("--fastspar-bin", default="fastspar", help="FastSpar executable")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel FastSpar tasks")
    parser.add_argument("--threads", type=int, default=None, help="Threads per FastSpar process")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()

    for task_dir in args.task_dirs:
        task_root = root / task_dir / "multiround"
        if not task_root.exists():
            print(f"[WARN] Skip missing task root: {task_root}")
            continue
        print(f"[RUN] FastSpar task_dir={task_dir}")
        completed = run_fastspar_from_split_root(
            root=task_root,
            seeds=args.seeds,
            folds=args.folds,
            fastspar_bin=args.fastspar_bin,
            jobs=args.jobs,
            threads=args.threads,
        )
        print(f"[DONE] FastSpar task_dir={task_dir} jobs={len(completed)}")


if __name__ == "__main__":
    main()
