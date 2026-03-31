from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib as mpl

mpl.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLOR_FULL = "#E69F00"
COLOR_SELECTED = "#489FA7"
PIPELINE_SPECS = {
    "full": {"display": "All", "color": COLOR_FULL},
    "selected": {"display": "FluRiCo", "color": COLOR_SELECTED},
}


def _parse_seed_from_dir(seed_dir: Path) -> int:
    token = seed_dir.name
    if not token.startswith("seed_"):
        raise ValueError(f"Invalid seed directory name: {seed_dir}")
    return int(token.split("seed_", 1)[1])


def _require_columns(df: pd.DataFrame, required: Sequence[str], path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def _format_label(mean_v: float, std_v: float) -> str:
    if np.isnan(mean_v):
        return ""
    if np.isnan(std_v):
        return f"{mean_v:.3f}±NA"
    return f"{mean_v:.3f}±{std_v:.3f}"


def _discover_task_name(root: Path, pipeline_prefix: str, task_name: Optional[str]) -> str:
    if task_name:
        return task_name
    candidates = sorted(
        path.name.split("task_", 1)[1]
        for path in (root / pipeline_prefix).glob("task_*")
        if path.is_dir()
    )
    if len(candidates) != 1:
        raise ValueError(
            "Failed to infer task_name automatically. Pass task_name explicitly. "
            f"Discovered task directories: {candidates}"
        )
    return candidates[0]


def _collect_pipeline_records(
    *,
    root: Path,
    task_name: str,
    pipeline_key: str,
    pipeline_prefix: str,
    seed_filter: Optional[List[int]],
) -> pd.DataFrame:
    task_root = root / pipeline_prefix / f"task_{task_name}"
    if not task_root.exists():
        raise FileNotFoundError(f"Missing task root: {task_root}")

    raw_files = sorted(task_root.glob("omics_*/seed_*/raw_fold_param_metrics.csv"))
    if not raw_files:
        raise FileNotFoundError(f"No raw_fold_param_metrics.csv files found under {task_root}")

    records = []
    for raw_path in raw_files:
        seed_dir = raw_path.parent
        omics = raw_path.parent.parent.name.split("omics_", 1)[1]
        seed = _parse_seed_from_dir(seed_dir)
        if seed_filter is not None and seed not in seed_filter:
            continue
        raw_df = pd.read_csv(raw_path)
        _require_columns(
            raw_df,
            [
                "fold",
                "model_type",
                "param_id",
                "val_primary_metric",
                "test_primary_metric",
                "n_features",
                "n_train",
                "n_val",
                "n_test",
            ],
            raw_path,
        )
        raw_df = raw_df.copy()
        raw_df["fold"] = pd.to_numeric(raw_df["fold"], errors="coerce")
        raw_df["val_primary_metric"] = pd.to_numeric(raw_df["val_primary_metric"], errors="coerce")
        raw_df["test_primary_metric"] = pd.to_numeric(raw_df["test_primary_metric"], errors="coerce")
        raw_df = raw_df[
            (raw_df["model_type"].astype(str) == "lgbm")
            & raw_df["fold"].notna()
            & raw_df["val_primary_metric"].notna()
        ].copy()
        if raw_df.empty:
            continue
        for fold, group in raw_df.groupby("fold", dropna=False):
            best_val = float(group["val_primary_metric"].max())
            tied = group[np.isclose(group["val_primary_metric"], best_val, atol=1e-12)].copy()
            test_mean = float(tied["test_primary_metric"].mean()) if tied["test_primary_metric"].notna().any() else float("nan")
            records.append(
                {
                    "task_name": task_name,
                    "omics": omics,
                    "seed": int(seed),
                    "fold": int(fold),
                    "pipeline": pipeline_key,
                    "best_val_metric": best_val,
                    "test_metric_mean": test_mean,
                    "n_features": int(pd.to_numeric(tied["n_features"], errors="coerce").dropna().iloc[0]),
                    "n_tied_params": int(tied.shape[0]),
                    "n_train": int(pd.to_numeric(tied["n_train"], errors="coerce").dropna().iloc[0]),
                    "n_val": int(pd.to_numeric(tied["n_val"], errors="coerce").dropna().iloc[0]),
                    "n_test": int(pd.to_numeric(tied["n_test"], errors="coerce").dropna().iloc[0]),
                    "tied_param_ids_json": json.dumps([str(item) for item in tied["param_id"].tolist()], ensure_ascii=True),
                }
            )
    return pd.DataFrame(records)


def _build_fold_table(full_df: pd.DataFrame, selected_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["task_name", "omics", "seed", "fold"]
    merged = full_df.merge(
        selected_df,
        on=key_cols,
        how="outer",
        suffixes=("_full", "_selected"),
    )
    return merged.rename(
        columns={
            "best_val_metric_full": "full_best_val_metric",
            "test_metric_mean_full": "full_test_metric_mean",
            "n_features_full": "full_n_features",
            "n_tied_params_full": "full_n_tied_params",
            "tied_param_ids_json_full": "full_tied_param_ids_json",
            "best_val_metric_selected": "selected_best_val_metric",
            "test_metric_mean_selected": "selected_test_metric_mean",
            "n_features_selected": "selected_n_features",
            "n_tied_params_selected": "selected_n_tied_params",
            "tied_param_ids_json_selected": "selected_tied_param_ids_json",
        }
    )


def _plot_metric(summary_df: pd.DataFrame, metric_column: str, out_path: Path, title: str) -> None:
    if summary_df.empty:
        return
    ordered = summary_df.sort_values(["omics", "pipeline"]).reset_index(drop=True)
    labels = [
        f"{PIPELINE_SPECS[row['pipeline']]['display']}-{row['omics'].capitalize()}"
        for _, row in ordered.iterrows()
    ]
    means = ordered[f"{metric_column}_mean"].to_numpy(dtype=float)
    stds = ordered[f"{metric_column}_std"].to_numpy(dtype=float)
    colors = [PIPELINE_SPECS[row["pipeline"]]["color"] for _, row in ordered.iterrows()]

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels)), 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric_column)
    ax.set_title(title)
    ymax = np.nanmax(means + np.nan_to_num(stds, nan=0.0)) if means.size else 1.0
    ax.set_ylim(0.0, min(1.05, max(1.0, ymax + 0.08)))

    for idx, (_, row) in enumerate(ordered.iterrows()):
        label = _format_label(row[f"{metric_column}_mean"], row[f"{metric_column}_std"])
        if label:
            ax.text(idx, means[idx] + 0.02, label, ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_bars_lgbm_valmax_testtieavg(
    *,
    root: str | Path,
    task_name: Optional[str] = None,
    full_prefix: str = "modeling_all_features_lgbm",
    selected_prefix: str = "modeling_selected_features_lgbm",
    out_dir: Optional[str | Path] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, str]:
    root = Path(root).resolve()
    task_name = _discover_task_name(root, full_prefix, task_name)
    out_dir = Path(out_dir).resolve() if out_dir is not None else root / "bars_lgbm_valmax_testtieavg"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_filter = sorted({int(seed) for seed in seeds}) if seeds is not None else None

    full_df = _collect_pipeline_records(
        root=root,
        task_name=task_name,
        pipeline_key="full",
        pipeline_prefix=full_prefix,
        seed_filter=seed_filter,
    )
    selected_df = _collect_pipeline_records(
        root=root,
        task_name=task_name,
        pipeline_key="selected",
        pipeline_prefix=selected_prefix,
        seed_filter=seed_filter,
    )
    if full_df.empty or selected_df.empty:
        raise ValueError("Missing usable LGBM rows for one or both pipelines.")

    fold_table = _build_fold_table(full_df, selected_df)
    fold_table_path = out_dir / "fold_table_lgbm_valmax_testtieavg.csv"
    fold_table.to_csv(fold_table_path, index=False)

    summary_rows = []
    for metric_column in ["best_val_metric", "test_metric_mean"]:
        base_df = pd.concat([full_df, selected_df], axis=0, ignore_index=True)
        for (omics, pipeline), group in base_df.groupby(["omics", "pipeline"], dropna=False):
            values = pd.to_numeric(group[metric_column], errors="coerce")
            summary_rows.append(
                {
                    "task_name": task_name,
                    "omics": str(omics),
                    "pipeline": str(pipeline),
                    "metric": metric_column,
                    f"{metric_column}_mean": float(values.mean()),
                    f"{metric_column}_std": float(values.std()),
                    "n_rows": int(group.shape[0]),
                    "n_seeds": int(group["seed"].nunique()),
                    "n_folds": int(group["fold"].nunique()),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "bar_summary_lgbm_valmax_testtieavg.csv"
    summary_df.to_csv(summary_path, index=False)

    val_summary = summary_df[summary_df["metric"] == "best_val_metric"].copy()
    _plot_metric(
        val_summary,
        metric_column="best_val_metric",
        out_path=out_dir / "val_auc_barplot.svg",
        title=f"{task_name}: validation ROC-AUC",
    )

    test_summary = summary_df[summary_df["metric"] == "test_metric_mean"].copy()
    test_plot_path = None
    if not test_summary.empty and test_summary["test_metric_mean_mean"].notna().any():
        test_plot_path = out_dir / "test_auc_barplot.svg"
        _plot_metric(
            test_summary,
            metric_column="test_metric_mean",
            out_path=test_plot_path,
            title=f"{task_name}: test ROC-AUC",
        )

    return {
        "task_name": task_name,
        "fold_table": str(fold_table_path),
        "summary": str(summary_path),
        "val_plot": str(out_dir / "val_auc_barplot.svg"),
        "test_plot": str(test_plot_path) if test_plot_path is not None else "",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic LGBM bar summary using val-max / test-tie-average")
    parser.add_argument("--root", required=True, help="Training output root")
    parser.add_argument("--task-name", default=None, help="Task name. Auto-detected when omitted.")
    parser.add_argument("--full-prefix", default="modeling_all_features_lgbm")
    parser.add_argument("--selected-prefix", default="modeling_selected_features_lgbm")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = plot_bars_lgbm_valmax_testtieavg(
        root=args.root,
        task_name=args.task_name,
        full_prefix=args.full_prefix,
        selected_prefix=args.selected_prefix,
        out_dir=args.out_dir,
        seeds=args.seeds,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
