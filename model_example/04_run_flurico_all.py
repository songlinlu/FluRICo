#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.config import RunConfig
from core.flurico_runner import run_flurico_from_task_root


def detect_seeds(task_root: Path) -> list[int]:
    return sorted(
        int(path.name.split("seed_", 1)[1])
        for path in task_root.glob("seed_*")
        if path.is_dir() and path.name.startswith("seed_")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 04: run FluRiCo for all task dirs.")
    parser.add_argument("--root", default=str(SCRIPT_DIR), help="model_example root")
    parser.add_argument(
        "--task-dirs",
        nargs="+",
        default=["hc_vs_dis", "mia_vs_ia"],
        help="Task directories under --root",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional seed subset")
    parser.add_argument("--n-folds", type=int, default=5, help="Fold count")
    parser.add_argument("--flurico-n-jobs", type=int, default=1, help="FluRiCo n_jobs")
    parser.add_argument("--flurico-use-fast", action="store_true", default=False, help="Use fast FluRiCo")
    parser.add_argument("--mb-top-n", type=int, default=50, help="Top-N microbe features")
    parser.add_argument("--mt-top-n", type=int, default=20, help="Top-N metabolite features")
    parser.add_argument(
        "--flurico-log-microbe",
        dest="flurico_log_microbe",
        action="store_true",
        default=True,
        help="Use log1p microbe data for FluRiCo (default: enabled)",
    )
    parser.add_argument(
        "--no-flurico-log-microbe",
        dest="flurico_log_microbe",
        action="store_false",
        help="Disable log1p microbe data",
    )
    parser.add_argument(
        "--flurico-log-metabolite",
        dest="flurico_log_metabolite",
        action="store_true",
        default=True,
        help="Use log1p metabolite data for FluRiCo (default: enabled)",
    )
    parser.add_argument(
        "--no-flurico-log-metabolite",
        dest="flurico_log_metabolite",
        action="store_false",
        help="Disable log1p metabolite data",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()

    for task_dir in args.task_dirs:
        task_out_root = root / task_dir
        task_root = task_out_root / "multiround"
        if not task_root.exists():
            print(f"[WARN] Skip missing task root: {task_root}")
            continue

        seeds = list(args.seeds) if args.seeds else detect_seeds(task_root)
        if not seeds:
            raise ValueError(f"No seeds found under {task_root}")

        run_config = RunConfig(
            out_dir=task_out_root,
            flurico_n_jobs=args.flurico_n_jobs,
            flurico_use_fast=bool(args.flurico_use_fast),
            mb_top_n=args.mb_top_n,
            mt_top_n=args.mt_top_n,
            flurico_log_microbe=bool(args.flurico_log_microbe),
            flurico_log_metabolite=bool(args.flurico_log_metabolite),
        )
        run_config.validate()
        print(f"[RUN] FluRiCo task_dir={task_dir} seeds={seeds}")
        outputs = run_flurico_from_task_root(
            task_root=task_root,
            seeds=seeds,
            n_folds=args.n_folds,
            run_config=run_config,
        )
        print(f"[DONE] FluRiCo task_dir={task_dir}: {outputs}")


if __name__ == "__main__":
    main()
