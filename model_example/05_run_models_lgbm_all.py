#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.config import RunConfig, TaskSpec
from core.modeling import run_modeling_from_task_root


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


def detect_seeds(task_root: Path) -> list[int]:
    return sorted(
        int(path.name.split("seed_", 1)[1])
        for path in task_root.glob("seed_*")
        if path.is_dir() and path.name.startswith("seed_")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 05: run LGBM modeling (all + selected) and export raw_fold_param_metrics.csv"
    )
    parser.add_argument("--root", default=str(SCRIPT_DIR), help="model_example root")
    parser.add_argument(
        "--task-dirs",
        nargs="+",
        choices=sorted(TASK_SPECS),
        default=["hc_vs_dis", "mia_vs_ia"],
        help="Task directories to run",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Optional seed subset")
    parser.add_argument("--n-splits", type=int, default=5, help="Fold count")
    parser.add_argument("--n-jobs", type=int, default=128, help="Parallel budget for param/fold evaluation")
    parser.add_argument(
        "--lgbm-profile",
        choices=["lgbm_grid", "lgbm_grid_small_sample"],
        default="lgbm_grid",
        help="LGBM profile",
    )
    parser.add_argument("--tie-eps", type=float, default=1e-6, help="Tie threshold on val ROC-AUC")
    parser.add_argument(
        "--full-prefix",
        default="modeling_all_features_lgbm_final",
        help="Output prefix for all-features pipeline",
    )
    parser.add_argument(
        "--selected-prefix",
        default="modeling_selected_features_lgbm_final",
        help="Output prefix for selected-features pipeline",
    )
    parser.add_argument(
        "--log-microbe",
        dest="log_microbe",
        action="store_true",
        default=True,
        help="Apply log1p to microbe for modeling (default: enabled)",
    )
    parser.add_argument(
        "--no-log-microbe",
        dest="log_microbe",
        action="store_false",
        help="Disable log1p on microbe for modeling",
    )
    parser.add_argument(
        "--log-metabolite",
        dest="log_metabolite",
        action="store_true",
        default=True,
        help="Apply log1p to metabolite for modeling (default: enabled)",
    )
    parser.add_argument(
        "--no-log-metabolite",
        dest="log_metabolite",
        action="store_false",
        help="Disable log1p on metabolite for modeling",
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
            seeds=seeds,
            n_splits=args.n_splits,
            n_jobs=args.n_jobs,
            lgbm_profile=args.lgbm_profile,
            tie_eps=args.tie_eps,
            full_prefix=args.full_prefix,
            selected_prefix=args.selected_prefix,
            model_log_microbe=bool(args.log_microbe),
            model_log_metabolite=bool(args.log_metabolite),
        )
        run_config.validate()

        print(f"[RUN] Modeling task_dir={task_dir} seeds={seeds} profile={args.lgbm_profile}")
        outputs = run_modeling_from_task_root(
            task_root=task_root,
            output_root=task_out_root,
            task_spec=TASK_SPECS[task_dir],
            run_config=run_config,
        )
        print(f"[DONE] Modeling task_dir={task_dir}: {outputs}")


if __name__ == "__main__":
    main()
