from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import TaskSpec
from .data_io import assert_aligned, load_aligned_tables, write_json


def transform_one_split(df: pd.DataFrame, task_spec: TaskSpec) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    raw_labels = df.index.astype(str).to_numpy()
    keep_mask = np.isin(raw_labels, task_spec.labels)
    keep_pos = np.where(keep_mask)[0]
    mapped_labels = np.array([task_spec.label_map.get(label, label) for label in raw_labels[keep_pos]], dtype=object)
    out = df.iloc[keep_pos].copy()
    out.index = pd.Index(mapped_labels.astype(str), name="label")
    return out, keep_pos.astype(int), mapped_labels.astype(str)


def ensure_required_labels(labels_arr: np.ndarray, required: List[str], tag: str) -> None:
    present = set(labels_arr.astype(str).tolist())
    missing = [label for label in required if label not in present]
    if missing:
        raise ValueError(f"[{tag}] missing required labels after mapping/filtering: {missing}")


def save_index_csv(path: Path, indices: np.ndarray, labels: np.ndarray) -> None:
    pd.DataFrame({"position": indices.astype(int), "label": labels.astype(str)}).to_csv(path, index=False)


def filter_features_by_zero_prevalence(df: pd.DataFrame, max_zero_fraction: float) -> pd.DataFrame:
    zero_fraction = (df == 0).sum(axis=0) / max(df.shape[0], 1)
    keep_cols = zero_fraction[zero_fraction < max_zero_fraction].index.tolist()
    if not keep_cols:
        raise ValueError("All microbe features were removed by zero-prevalence filtering.")
    return df.loc[:, keep_cols].copy()


def export_fastspar_inputs(mb_subtrain: pd.DataFrame, labels: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_features = mb_subtrain.shape[1]
    idx_labels = mb_subtrain.index.astype(str).to_numpy()
    for label in labels:
        rows = np.where(idx_labels == label)[0]
        if rows.size == 0:
            raise ValueError(f"Subtrain has 0 samples for label '{label}', cannot build FastSpar input.")
        otu = mb_subtrain.iloc[rows].T.copy()
        otu.columns = [f"sample_{i + 1:05d}" for i in range(otu.shape[1])]
        otu.to_csv(
            out_dir / f"{label}_absolute_{n_features}.tsv",
            sep="\t",
            index=True,
            index_label="#OTU ID",
        )


def write_log1p_copy(df: pd.DataFrame, out_path: Path) -> None:
    log_df = pd.DataFrame(
        np.log1p(df.to_numpy(dtype=float)),
        index=df.index.copy(),
        columns=df.columns.copy(),
    )
    log_df.to_csv(out_path)


def prepare_task_test_tables(
    *,
    mb_test: str | Path,
    mt_test: str | Path,
    out_dir: str | Path,
    task_spec: TaskSpec,
    label_column: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    if mb_test is None or mt_test is None:
        return None

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mb_raw, mt_raw = load_aligned_tables(
        mb_test,
        mt_test,
        label_column=label_column,
        context="task_test",
    )

    mb_task, keep_mb, labels_mb = transform_one_split(mb_raw, task_spec)
    mt_task, keep_mt, labels_mt = transform_one_split(mt_raw, task_spec)
    if not np.array_equal(keep_mb, keep_mt):
        raise ValueError("Task test keep positions differ between microbe and metabolite.")
    if not np.array_equal(labels_mb, labels_mt):
        raise ValueError("Task test labels differ between microbe and metabolite after mapping.")

    mb_task.to_csv(out_dir / "mb_test.csv")
    mt_task.to_csv(out_dir / "metabolites_test.csv")
    write_log1p_copy(mb_task, out_dir / "mb_test_log1p.csv")
    write_log1p_copy(mt_task, out_dir / "metabolites_test_log1p.csv")
    write_json(
        out_dir / "task_test_metadata.json",
        {
            "task_name": task_spec.task_name,
            "labels": task_spec.final_labels(),
            "n_test": int(mb_task.shape[0]),
            "label_counts": {
                str(label): int(count)
                for label, count in pd.Series(labels_mb).value_counts(sort=False).items()
            },
        },
    )
    return {
        "mb_test": str(out_dir / "mb_test.csv"),
        "mt_test": str(out_dir / "metabolites_test.csv"),
    }


def build_task_views_from_split(
    *,
    source_root: str | Path,
    out_root: str | Path,
    task_spec: TaskSpec,
    n_folds: int = 5,
    zero_max_fraction: Optional[float] = 0.8,
) -> Dict[str, object]:
    task_spec.validate()
    source_root = Path(source_root).resolve()
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    final_labels = task_spec.final_labels()
    seeds = sorted(
        int(path.name.split("seed_", 1)[1])
        for path in source_root.glob("seed_*")
        if path.is_dir() and path.name.startswith("seed_")
    )
    if not seeds:
        raise ValueError(f"No seed directories found under {source_root}")

    split_stats_rows = []
    filter_rows = []
    for seed in seeds:
        for fold in range(n_folds):
            src_fold = source_root / f"seed_{seed}" / f"fold_{fold}"
            if not src_fold.exists():
                print(f"[WARN] Missing source fold, skip: {src_fold}")
                continue

            src_data = src_fold / "data"
            src_idx = src_fold / "indices"
            mb_sub_raw = pd.read_csv(src_data / "mb_subtrain.csv", index_col=0)
            mt_sub_raw = pd.read_csv(src_data / "metabolites_subtrain.csv", index_col=0)
            mb_val_raw = pd.read_csv(src_data / "mb_val.csv", index_col=0)
            mt_val_raw = pd.read_csv(src_data / "metabolites_val.csv", index_col=0)
            assert_aligned(mb_sub_raw, mt_sub_raw, f"task_view seed={seed} fold={fold} subtrain")
            assert_aligned(mb_val_raw, mt_val_raw, f"task_view seed={seed} fold={fold} val")

            mb_sub, keep_sub_pos, sub_labels = transform_one_split(mb_sub_raw, task_spec)
            mt_sub, keep_sub_pos_mt, _ = transform_one_split(mt_sub_raw, task_spec)
            mb_val, keep_val_pos, val_labels = transform_one_split(mb_val_raw, task_spec)
            mt_val, keep_val_pos_mt, _ = transform_one_split(mt_val_raw, task_spec)
            if not np.array_equal(keep_sub_pos, keep_sub_pos_mt):
                raise ValueError(f"seed={seed} fold={fold} subtrain keep positions mismatch across omics")
            if not np.array_equal(keep_val_pos, keep_val_pos_mt):
                raise ValueError(f"seed={seed} fold={fold} val keep positions mismatch across omics")

            ensure_required_labels(sub_labels, final_labels, f"seed={seed} fold={fold} subtrain")
            ensure_required_labels(val_labels, final_labels, f"seed={seed} fold={fold} val")

            n_features_before = int(mb_sub.shape[1])
            if zero_max_fraction is not None:
                mb_sub = filter_features_by_zero_prevalence(mb_sub, max_zero_fraction=float(zero_max_fraction))
                mb_val = mb_val.loc[:, mb_sub.columns].copy()
            n_features_after = int(mb_sub.shape[1])
            removed = int(n_features_before - n_features_after)
            removed_fraction = float(removed / n_features_before) if n_features_before > 0 else 0.0

            src_train_indices = np.load(src_idx / "train_indices.npy")
            src_val_indices = np.load(src_idx / "val_indices.npy")
            train_indices = src_train_indices[keep_sub_pos]
            val_indices = src_val_indices[keep_val_pos]

            dst_fold = out_root / f"seed_{seed}" / f"fold_{fold}"
            dst_data = dst_fold / "data"
            dst_idx = dst_fold / "indices"
            dst_fastspar_in = dst_fold / "fastspar" / "input"
            dst_fastspar_out = dst_fold / "fastspar" / "output"
            dst_data.mkdir(parents=True, exist_ok=True)
            dst_idx.mkdir(parents=True, exist_ok=True)
            dst_fastspar_in.mkdir(parents=True, exist_ok=True)
            dst_fastspar_out.mkdir(parents=True, exist_ok=True)

            mb_sub.to_csv(dst_data / "mb_subtrain.csv")
            mb_sub.to_csv(dst_data / "mb_subtrain_zero_filtered.csv")
            mt_sub.to_csv(dst_data / "metabolites_subtrain.csv")
            mb_val.to_csv(dst_data / "mb_val.csv")
            mb_val.to_csv(dst_data / "mb_val_zero_filtered.csv")
            mt_val.to_csv(dst_data / "metabolites_val.csv")
            write_log1p_copy(mb_sub, dst_data / "mb_subtrain_log1p.csv")
            write_log1p_copy(mt_sub, dst_data / "metabolites_subtrain_log1p.csv")
            write_log1p_copy(mb_val, dst_data / "mb_val_log1p.csv")
            write_log1p_copy(mt_val, dst_data / "metabolites_val_log1p.csv")

            np.save(dst_idx / "train_indices.npy", train_indices.astype(int))
            np.save(dst_idx / "val_indices.npy", val_indices.astype(int))
            save_index_csv(dst_idx / "train_indices.csv", train_indices, sub_labels)
            save_index_csv(dst_idx / "val_indices.csv", val_indices, val_labels)
            export_fastspar_inputs(mb_sub, final_labels, dst_fastspar_in)

            filter_payload = {
                "task_name": task_spec.task_name,
                "seed": int(seed),
                "fold": int(fold),
                "threshold": float(zero_max_fraction) if zero_max_fraction is not None else None,
                "n_samples_subtrain": int(mb_sub.shape[0]),
                "n_features_before": n_features_before,
                "n_features_after": n_features_after,
                "n_features_removed": removed,
                "removed_fraction": removed_fraction,
            }
            write_json(dst_data / "microbe_zero_prevalence_filter_stats.json", filter_payload)

            write_json(
                dst_fold / "labels.json",
                {
                    "labels": final_labels,
                    "task_name": task_spec.task_name,
                    "positive_label": task_spec.positive_label,
                    "seed": int(seed),
                    "fold": int(fold),
                    "n_splits": int(n_folds),
                    "n_train": int(mb_sub.shape[0]),
                    "n_val": int(mb_val.shape[0]),
                    "mb_zero_prevalence_filter": filter_payload,
                    "source_fold_dir": str(src_fold),
                },
            )

            for split_name, split_labels in (("train", sub_labels), ("val", val_labels)):
                counts = pd.Series(split_labels).value_counts(sort=False)
                for label in final_labels:
                    split_stats_rows.append(
                        {
                            "task_name": task_spec.task_name,
                            "seed": int(seed),
                            "fold": int(fold),
                            "split": split_name,
                            "label": label,
                            "count": int(counts.get(label, 0)),
                        }
                    )
            filter_rows.append(filter_payload)
            print(
                f"[OK] task_view seed={seed} fold={fold} "
                f"train={mb_sub.shape[0]} val={mb_val.shape[0]} -> {dst_fold}"
            )

    pd.DataFrame(split_stats_rows).to_csv(out_root / "split_stats.csv", index=False)
    pd.DataFrame(filter_rows).to_csv(out_root / "microbe_zero_prevalence_filter_stats.csv", index=False)
    return {
        "out_root": str(out_root),
        "task_name": task_spec.task_name,
        "labels": final_labels,
        "seeds": seeds,
        "n_splits": int(n_folds),
    }
