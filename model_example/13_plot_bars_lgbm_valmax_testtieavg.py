#!/usr/bin/env python3
"""Plot fold-aggregated LGBM bar charts using strict fold-wise val max.

For each pipeline/task/omics/seed/fold:
- best_val_auc = max(val_roc_auc) among LGBM rows
- test_auc_mean = mean(test_roc_auc) across all rows tied at best_val_auc

Outputs:
- bars_lgbm_valmax_testtieavg/val_auc_barplot.svg
- bars_lgbm_valmax_testtieavg/test_auc_barplot.svg
- bars_lgbm_valmax_testtieavg/fold_table_lgbm_valmax_testtieavg.csv
- bars_lgbm_valmax_testtieavg/bar_summary_lgbm_valmax_testtieavg.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl

mpl.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


TASK_SPECS = [
    ("hc_vs_dis", "HC_vs_Disease"),
    ("mia_vs_ia", "MIA_vs_IA"),
]
OMICS_SPECS = [
    ("microbe", "Microbe"),
    ("metabolite", "Metabolite"),
]
COLOR_FULL = "#E69F00"
COLOR_SELECTED = "#489fa7"
HATCH_MICROBE = ""
HATCH_METABOLITE = "///"
Y_BREAK = 0.4
Y_TOP = 1.03
ERRORBAR_TYPE = "sd"
PIPELINE_SPECS = {
    "full": {
        "label": "All",
        "display": "All",
        "color": COLOR_FULL,
    },
    "selected": {
        "label": "FluRiCo",
        "display": "FluRiCo",
        "color": COLOR_SELECTED,
    },
}
BAR_SPECS = [
    ("microbe__all", "All-Microbe", "microbe", "full"),
    ("microbe__flurico", "FluRiCo Microbes", "microbe", "selected"),
    ("metabolite__all", "All-Metabolite", "metabolite", "full"),
    ("metabolite__flurico", "FluRiCo Metabolites", "metabolite", "selected"),
]
KEY_COLUMNS = ["task_dir", "task", "omics", "seed", "fold"]
FOLD_TABLE_COLUMNS = [
    "task_dir",
    "task",
    "omics",
    "seed",
    "fold",
    "n_train",
    "n_val",
    "n_test",
    "full_n_features",
    "full_n_tied_params",
    "full_best_val_auc",
    "full_test_auc_mean",
    "full_tied_param_ids_json",
    "selected_n_features",
    "selected_n_tied_params",
    "selected_best_val_auc",
    "selected_test_auc_mean",
    "selected_tied_param_ids_json",
]
RAW_REQUIRED_COLUMNS = [
    "fold",
    "model_type",
    "param_id",
    "val_roc_auc",
    "test_roc_auc",
    "n_features",
    "n_train",
    "n_val",
    "n_test",
]
SUMMARY_COLUMNS = [
    "metric",
    "task_dir",
    "task",
    "bar_order",
    "series_key",
    "bar_label",
    "bar_name",
    "pipeline",
    "omics",
    "n_rows",
    "n_seeds",
    "n_folds",
    "auc_mean",
    "auc_std",
    "mean_metric",
    "std_metric",
    "errorbar_type",
    "error_value",
    "error",
    "label_mean_std",
    "seed_filter",
]
EXPECTED_FOLDS = [0, 1, 2, 3, 4]


@dataclass(frozen=True)
class FoldRecord:
    task_dir: str
    task: str
    omics: str
    seed: int
    fold: int
    pipeline: str
    n_features: int
    n_tied_params: int
    best_val_auc: float
    test_auc_mean: float
    tied_param_ids_json: str
    n_train: int
    n_val: int
    n_test: int


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Plot strict val-max / tied-test-mean LGBM bar charts from raw_fold_param_metrics.csv."
        )
    )
    parser.add_argument("--root", default=str(script_dir), help="final_log_compare root. Default: script directory")
    parser.add_argument(
        "--full-prefix",
        default="modeling_all_features_lgbm_final",
        help="All-feature pipeline prefix.",
    )
    parser.add_argument(
        "--selected-prefix",
        default="modeling_selected_features_lgbm_final",
        help="Selected-feature pipeline prefix.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: <root>/bars_lgbm_valmax_testtieavg",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional seed subset. If provided, only these seeds are used.",
    )
    return parser.parse_args()


def normalize_seed_list(seeds: Optional[Sequence[int]]) -> Optional[List[int]]:
    if seeds is None:
        return None
    return sorted({int(x) for x in seeds})


def pretty_task(task: str) -> str:
    return task.replace("_vs_", " vs. ").replace("_", " ")


def format_mean_std_label(mean_v: float, std_v: float) -> str:
    if np.isnan(mean_v):
        return ""
    if np.isnan(std_v):
        return f"{mean_v:.3f}±NA"
    return f"{mean_v:.3f}±{std_v:.3f}"


def require_columns(df: pd.DataFrame, required: Sequence[str], path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def parse_seed_from_dir(seed_dir: Path) -> int:
    token = seed_dir.name
    if not token.startswith("seed_"):
        raise ValueError(f"Invalid seed directory name: {seed_dir}")
    return int(token.split("seed_", 1)[1])


def resolve_single_int(g: pd.DataFrame, column: str, context: str) -> int:
    vals = pd.to_numeric(g[column], errors="coerce").dropna()
    uniq = sorted({int(x) for x in vals.tolist()})
    if not uniq:
        raise ValueError(f"{context}: column '{column}' has no usable values")
    if len(uniq) != 1:
        raise ValueError(f"{context}: column '{column}' is inconsistent: {uniq}")
    return int(uniq[0])


def collect_pipeline_records(
    *,
    root: Path,
    task_dir: str,
    task_name: str,
    omics: str,
    pipeline: str,
    pipeline_prefix: str,
    seed_filter: Optional[List[int]],
) -> pd.DataFrame:
    omics_root = root / task_dir / "multiround" / pipeline_prefix / f"task_{task_name}" / f"omics_{omics}"
    if not omics_root.exists():
        raise FileNotFoundError(f"Missing omics root: {omics_root}")

    seed_dirs = sorted([p for p in omics_root.glob("seed_*") if p.is_dir()])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories found under: {omics_root}")

    records: List[FoldRecord] = []
    seen_requested: set[int] = set()

    for seed_dir in seed_dirs:
        seed = parse_seed_from_dir(seed_dir)
        if seed_filter is not None and seed not in seed_filter:
            continue

        raw_path = seed_dir / "raw_fold_param_metrics.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw metrics file: {raw_path}")

        raw_df = pd.read_csv(raw_path)
        require_columns(raw_df, RAW_REQUIRED_COLUMNS, raw_path)
        raw_df = raw_df.copy()
        raw_df["fold_num"] = pd.to_numeric(raw_df["fold"], errors="coerce")
        raw_df["val_roc_auc"] = pd.to_numeric(raw_df["val_roc_auc"], errors="coerce")
        raw_df["test_roc_auc"] = pd.to_numeric(raw_df["test_roc_auc"], errors="coerce")
        raw_df = raw_df[
            (raw_df["model_type"].astype(str) == "lgbm")
            & raw_df["fold_num"].notna()
            & raw_df["val_roc_auc"].notna()
            & raw_df["test_roc_auc"].notna()
        ].copy()
        if raw_df.empty:
            raise ValueError(f"No usable LGBM rows after filtering: {raw_path}")

        for fold_val, g in raw_df.groupby("fold_num", dropna=False):
            fold = int(fold_val)
            context = (
                f"pipeline={pipeline} task={task_name} omics={omics} seed={seed} fold={fold}"
            )
            best_val_auc = float(g["val_roc_auc"].max())
            tie = g[g["val_roc_auc"] == best_val_auc].copy().sort_values(["param_id"], ascending=[True])
            if tie.empty:
                raise ValueError(f"{context}: empty tie set at best val auc")

            records.append(
                FoldRecord(
                    task_dir=task_dir,
                    task=task_name,
                    omics=omics,
                    seed=seed,
                    fold=fold,
                    pipeline=pipeline,
                    n_features=resolve_single_int(g, "n_features", context),
                    n_tied_params=int(tie.shape[0]),
                    best_val_auc=best_val_auc,
                    test_auc_mean=float(tie["test_roc_auc"].mean()),
                    tied_param_ids_json=json.dumps(
                        [str(x) for x in tie["param_id"].astype(str).tolist()],
                        ensure_ascii=True,
                    ),
                    n_train=resolve_single_int(g, "n_train", context),
                    n_val=resolve_single_int(g, "n_val", context),
                    n_test=resolve_single_int(g, "n_test", context),
                )
            )
        seen_requested.add(seed)

    if seed_filter is not None:
        missing = [s for s in seed_filter if s not in seen_requested]
        if missing:
            raise ValueError(
                f"Missing requested seeds for pipeline={pipeline} task={task_name} omics={omics}: {missing}"
            )

    out = pd.DataFrame([r.__dict__ for r in records])
    if out.empty:
        raise ValueError(
            f"No fold-level rows collected for pipeline={pipeline} task={task_name} omics={omics}"
        )
    return out.sort_values(["seed", "fold"], kind="stable").reset_index(drop=True)


def assert_complete_fold_coverage(df: pd.DataFrame) -> None:
    issues: List[str] = []
    for (task_dir, task, omics, seed), g in df.groupby(["task_dir", "task", "omics", "seed"], dropna=False):
        present = sorted({int(x) for x in pd.to_numeric(g["fold"], errors="coerce").dropna().tolist()})
        if present != EXPECTED_FOLDS:
            issues.append(
                f"task_dir={task_dir} task={task} omics={omics} seed={seed} present_folds={present}"
            )
    if issues:
        raise ValueError("Incomplete fold coverage detected:\n" + "\n".join(issues))


def merge_sample_size_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for base_col in ["n_train", "n_val", "n_test"]:
        full_col = f"full_{base_col}"
        selected_col = f"selected_{base_col}"
        merged: List[float] = []
        for _, row in out[[full_col, selected_col]].iterrows():
            full_val = row[full_col]
            sel_val = row[selected_col]
            if pd.notna(full_val) and pd.notna(sel_val):
                if int(full_val) != int(sel_val):
                    raise ValueError(
                        f"Sample size mismatch for {base_col}: full={full_val}, selected={sel_val}"
                    )
                merged.append(int(full_val))
            elif pd.notna(full_val):
                merged.append(int(full_val))
            elif pd.notna(sel_val):
                merged.append(int(sel_val))
            else:
                merged.append(np.nan)
        out[base_col] = merged
        out = out.drop(columns=[full_col, selected_col])
    return out


def build_fold_table(root: Path, full_prefix: str, selected_prefix: str, seed_filter: Optional[List[int]]) -> pd.DataFrame:
    full_parts: List[pd.DataFrame] = []
    selected_parts: List[pd.DataFrame] = []

    for task_dir, task_name in TASK_SPECS:
        for omics, _ in OMICS_SPECS:
            full_parts.append(
                collect_pipeline_records(
                    root=root,
                    task_dir=task_dir,
                    task_name=task_name,
                    omics=omics,
                    pipeline="full",
                    pipeline_prefix=full_prefix,
                    seed_filter=seed_filter,
                )
            )
            selected_parts.append(
                collect_pipeline_records(
                    root=root,
                    task_dir=task_dir,
                    task_name=task_name,
                    omics=omics,
                    pipeline="selected",
                    pipeline_prefix=selected_prefix,
                    seed_filter=seed_filter,
                )
            )

    full_df = pd.concat(full_parts, axis=0, ignore_index=True)
    selected_df = pd.concat(selected_parts, axis=0, ignore_index=True)
    assert_complete_fold_coverage(full_df)
    assert_complete_fold_coverage(selected_df)

    full_df = full_df.rename(
        columns={
            "n_features": "full_n_features",
            "n_tied_params": "full_n_tied_params",
            "best_val_auc": "full_best_val_auc",
            "test_auc_mean": "full_test_auc_mean",
            "tied_param_ids_json": "full_tied_param_ids_json",
            "n_train": "full_n_train",
            "n_val": "full_n_val",
            "n_test": "full_n_test",
        }
    ).drop(columns=["pipeline"])
    selected_df = selected_df.rename(
        columns={
            "n_features": "selected_n_features",
            "n_tied_params": "selected_n_tied_params",
            "best_val_auc": "selected_best_val_auc",
            "test_auc_mean": "selected_test_auc_mean",
            "tied_param_ids_json": "selected_tied_param_ids_json",
            "n_train": "selected_n_train",
            "n_val": "selected_n_val",
            "n_test": "selected_n_test",
        }
    ).drop(columns=["pipeline"])

    merged = full_df.merge(selected_df, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    full_keys = set(tuple(row) for row in full_df[KEY_COLUMNS].itertuples(index=False, name=None))
    selected_keys = set(tuple(row) for row in selected_df[KEY_COLUMNS].itertuples(index=False, name=None))
    if full_keys != selected_keys:
        only_full = sorted(full_keys - selected_keys)
        only_selected = sorted(selected_keys - full_keys)
        raise ValueError(
            "full/selected fold keys do not align.\n"
            f"only_full={only_full[:10]}\n"
            f"only_selected={only_selected[:10]}"
        )
    merged = merge_sample_size_columns(merged)
    merged = merged.reindex(columns=FOLD_TABLE_COLUMNS)
    if merged.isna().any().any():
        na_cols = merged.columns[merged.isna().any()].tolist()
        raise ValueError(f"Merged fold table contains unexpected missing values in columns: {na_cols}")
    return merged.sort_values(["task", "omics", "seed", "fold"], kind="stable").reset_index(drop=True)


def build_bar_summary(fold_table: pd.DataFrame, seed_filter: Optional[List[int]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    seed_label = "ALL" if seed_filter is None else ",".join(str(x) for x in seed_filter)

    metric_to_col = {
        ("val", "full"): "full_best_val_auc",
        ("val", "selected"): "selected_best_val_auc",
        ("test", "full"): "full_test_auc_mean",
        ("test", "selected"): "selected_test_auc_mean",
    }

    for metric in ["val", "test"]:
        for task_dir, task_name in TASK_SPECS:
            task_df = fold_table[fold_table["task"] == task_name].copy()
            if task_df.empty:
                raise ValueError(f"No fold rows found for task={task_name}")

            for bar_order, (series_key, bar_label, omics, pipeline) in enumerate(BAR_SPECS, start=1):
                sub = task_df[task_df["omics"] == omics].copy()
                col = metric_to_col[(metric, pipeline)]
                values = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
                if values.size == 0:
                    raise ValueError(
                        f"No values for metric={metric} task={task_name} omics={omics} pipeline={pipeline}"
                    )

                mean_v = float(values.mean())
                std_v = float(values.std(ddof=1)) if values.size > 1 else 0.0
                rows.append(
                    {
                        "metric": metric,
                        "task_dir": task_dir,
                        "task": task_name,
                        "bar_order": bar_order,
                        "series_key": series_key,
                        "bar_label": bar_label,
                        "bar_name": bar_label,
                        "pipeline": pipeline,
                        "omics": omics,
                        "n_rows": int(values.size),
                        "n_seeds": int(sub["seed"].nunique()),
                        "n_folds": int(sub["fold"].nunique()),
                        "auc_mean": mean_v,
                        "auc_std": std_v,
                        "mean_metric": mean_v,
                        "std_metric": std_v,
                        "errorbar_type": ERRORBAR_TYPE,
                        "error_value": std_v,
                        "error": std_v,
                        "label_mean_std": format_mean_std_label(mean_v, std_v),
                        "seed_filter": seed_label,
                    }
                )

    out = pd.DataFrame(rows)
    return out.reindex(columns=SUMMARY_COLUMNS)


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, which="major", color="#E5E7EB", linewidth=0.8, alpha=1.0)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color("#6B7280")


def style_for_bar(omics: str, pipeline: str) -> Tuple[str, str]:
    color = PIPELINE_SPECS[pipeline]["color"]
    hatch = HATCH_MICROBE if omics == "microbe" else HATCH_METABOLITE
    return color, hatch


def apply_broken_y_axis(ax_top: plt.Axes, ax_bottom: plt.Axes, y_break: float = Y_BREAK) -> None:
    ax_top.set_ylim(y_break, Y_TOP)
    ax_bottom.set_ylim(0.0, y_break)

    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_bottom.xaxis.tick_bottom()

    ax_bottom.set_yticks([0.0, 0.2])
    ax_top.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    kwargs = dict(
        marker=[(-1, -1), (1, 1)],
        markersize=10,
        linestyle="none",
        color="#6B7280",
        mec="#6B7280",
        mew=1.0,
        clip_on=False,
    )
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)


def annotate_bar_label(
    ax_top: plt.Axes,
    ax_bottom: plt.Axes,
    *,
    x: float,
    y: float,
    label: str,
    color: str,
) -> None:
    if not label or np.isnan(y):
        return
    target_ax = ax_top if y >= Y_BREAK else ax_bottom
    text_dy = 5
    va = "bottom"
    if target_ax is ax_top and y >= (Y_TOP - 0.04):
        text_dy = -4
        va = "top"
    target_ax.annotate(
        label,
        xy=(x, y),
        xytext=(0, text_dy),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=9,
        color=color,
        clip_on=False,
        zorder=6,
    )


def summary_value(summary_df: pd.DataFrame, metric: str, task_name: str, omics: str, pipeline: str) -> Dict[str, object]:
    sub = summary_df[
        (summary_df["metric"].astype(str) == str(metric))
        & (summary_df["task"].astype(str) == str(task_name))
        & (summary_df["omics"].astype(str) == str(omics))
        & (summary_df["pipeline"].astype(str) == str(pipeline))
    ].copy()
    if sub.shape[0] != 1:
        raise ValueError(
            f"Expected exactly one summary row for metric={metric} task={task_name} omics={omics} pipeline={pipeline}, got {sub.shape[0]}"
        )
    return sub.iloc[0].to_dict()


def plot_metric(summary_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(14.0, 10.0))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[5.2, 1.0], hspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
    style_axis(ax_top)
    style_axis(ax_bottom)

    x = np.arange(len(TASK_SPECS), dtype=float)
    width = 0.19
    offsets = np.array([-0.30, -0.10, 0.10, 0.30], dtype=float)

    for task_idx, (_, task_name) in enumerate(TASK_SPECS):
        for bar_idx, (_series_key, bar_label, omics, pipeline) in enumerate(BAR_SPECS):
            rec = summary_value(summary_df, metric=metric, task_name=task_name, omics=omics, pipeline=pipeline)
            xpos = float(x[task_idx] + offsets[bar_idx])
            mean_v = float(rec["mean_metric"])
            err_v = float(rec["error"])
            color, hatch = style_for_bar(omics=omics, pipeline=pipeline)

            for ax_ in (ax_top, ax_bottom):
                ax_.bar(
                    xpos,
                    mean_v,
                    width=width,
                    color=color,
                    edgecolor="#6B7280",
                    linewidth=0.7,
                    alpha=0.95,
                    hatch=hatch,
                    label=bar_label,
                    zorder=3,
                )
                ax_.errorbar(
                    xpos,
                    mean_v,
                    yerr=err_v,
                    fmt="none",
                    ecolor="#6B7280",
                    elinewidth=1.0,
                    capsize=3,
                    capthick=1.0,
                    zorder=4,
                )
            annotate_bar_label(
                ax_top,
                ax_bottom,
                x=xpos,
                y=mean_v,
                label=str(rec["label_mean_std"]),
                color=color,
            )

    ax_top.set_ylabel(f"ROC-AUC (Mean ± {ERRORBAR_TYPE.upper()})", fontsize=15)
    ax_top.set_title(
        "Validation ROC-AUC" if metric == "val" else "Test ROC-AUC",
        fontsize=16,
        pad=8,
    )
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels([pretty_task(task_name) for _, task_name in TASK_SPECS], fontsize=17)
    ax_top.tick_params(axis="y", labelsize=13)
    ax_bottom.tick_params(axis="y", labelsize=13)
    apply_broken_y_axis(ax_top=ax_top, ax_bottom=ax_bottom, y_break=Y_BREAK)

    legend_handles = []
    for _series_key, bar_label, omics, pipeline in BAR_SPECS:
        color, hatch = style_for_bar(omics=omics, pipeline=pipeline)
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor="#6B7280",
                hatch=hatch,
                label=bar_label,
            )
        )
    ax_top.legend(
        handles=legend_handles,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#D1D5DB",
        fontsize=11,
        ncol=1,
        loc="upper left",
    )

    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"--root does not exist: {root}")

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (root / "bars_lgbm_valmax_testtieavg")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_filter = normalize_seed_list(args.seeds)

    fold_table = build_fold_table(
        root=root,
        full_prefix=str(args.full_prefix),
        selected_prefix=str(args.selected_prefix),
        seed_filter=seed_filter,
    )
    summary_df = build_bar_summary(fold_table, seed_filter=seed_filter)

    fold_table_path = out_dir / "fold_table_lgbm_valmax_testtieavg.csv"
    summary_path = out_dir / "bar_summary_lgbm_valmax_testtieavg.csv"
    val_svg_path = out_dir / "val_auc_barplot.svg"
    test_svg_path = out_dir / "test_auc_barplot.svg"

    save_csv(fold_table, fold_table_path)
    save_csv(summary_df, summary_path)
    plot_metric(summary_df, metric="val", out_path=val_svg_path)
    plot_metric(summary_df, metric="test", out_path=test_svg_path)

    print(f"[DONE] Fold table: {fold_table_path}")
    print(f"[DONE] Bar summary: {summary_path}")
    print(f"[DONE] Validation plot: {val_svg_path}")
    print(f"[DONE] Test plot: {test_svg_path}")


if __name__ == "__main__":
    main()
