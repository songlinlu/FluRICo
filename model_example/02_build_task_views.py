#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.config import TaskSpec
from core.task_view import build_task_views_from_split, prepare_task_test_tables


TASK_SPECS = {
    "hc_vs_dis": TaskSpec(
        task_name="HC_vs_Disease",
        labels=["HC", "IA", "MIA", "Disease", "Dis"],
        label_map={"IA": "Disease", "MIA": "Disease", "Dis": "Disease"},
        positive_label="Disease",
    ),
    "mia_vs_ia": TaskSpec(
        task_name="MIA_vs_IA",
        labels=["MIA", "IA"],
        positive_label="IA",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 02: build task views + zero-prevalence filtering + optional task test tables."
    )
    parser.add_argument(
        "--source-root",
        default=str(SCRIPT_DIR / "multiround_raw"),
        help="Source split root from Step 01",
    )
    parser.add_argument(
        "--out-root",
        default=str(SCRIPT_DIR),
        help="Output root that will contain hc_vs_dis/ and mia_vs_ia/",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_SPECS),
        default=["hc_vs_dis", "mia_vs_ia"],
        help="Task dirs to build",
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Fold count")
    parser.add_argument(
        "--mb-zero-max-fraction",
        type=float,
        default=0.8,
        help="Zero-prevalence threshold for microbe features",
    )
    parser.add_argument(
        "--mb-test",
        default=str(PACKAGE_ROOT / "mb_otu_filter0right_test.csv"),
        help="Microbe test CSV path",
    )
    parser.add_argument(
        "--mt-test",
        default=str(PACKAGE_ROOT / "metabolites_test.csv"),
        help="Metabolite test CSV path",
    )
    parser.add_argument("--label-column", default=None, help="Optional label column in CSV")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        default=False,
        help="Skip task_test table preparation",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.source_root).resolve()
    out_root = Path(args.out_root).resolve()
    mb_test = Path(args.mb_test).resolve()
    mt_test = Path(args.mt_test).resolve()

    for task_dir in args.tasks:
        task_spec = TASK_SPECS[task_dir]
        task_out_root = out_root / task_dir
        multiround_out = task_out_root / "multiround"
        multiround_out.mkdir(parents=True, exist_ok=True)

        info = build_task_views_from_split(
            source_root=source_root,
            out_root=multiround_out,
            task_spec=task_spec,
            n_folds=args.n_folds,
            zero_max_fraction=args.mb_zero_max_fraction,
        )
        (task_out_root / "task_spec.json").write_text(
            json.dumps(task_spec.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[DONE] Built task view: {task_dir}")
        for key, value in info.items():
            print(f"  {key}: {value}")

        if args.skip_test:
            continue
        if not mb_test.exists() or not mt_test.exists():
            print(
                f"[WARN] Skip task_test for {task_dir}: missing test files "
                f"mb={mb_test.exists()} mt={mt_test.exists()}"
            )
            continue

        test_info = prepare_task_test_tables(
            mb_test=mb_test,
            mt_test=mt_test,
            out_dir=task_out_root / "task_test",
            task_spec=task_spec,
            label_column=args.label_column,
        )
        print(f"[DONE] Prepared task_test for {task_dir}: {test_info}")


if __name__ == "__main__":
    main()
