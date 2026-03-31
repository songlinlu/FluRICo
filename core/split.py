from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import DEFAULT_SEEDS
from .data_io import load_aligned_tables, ordered_label_counts, write_json


def save_index_table(path: Path, idx: np.ndarray, labels: np.ndarray) -> None:
    pd.DataFrame({"position": idx.astype(int), "label": labels[idx].astype(str)}).to_csv(path, index=False)


def export_fastspar_input_tables(mb_subtrain: pd.DataFrame, labels: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_features = mb_subtrain.shape[1]
    idx_labels = mb_subtrain.index.astype(str).to_numpy()

    for label in labels:
        rows = np.where(idx_labels == label)[0]
        if rows.size == 0:
            raise ValueError(f"Label '{label}' has 0 samples in subtrain; cannot export FastSpar input.")
        otu = mb_subtrain.iloc[rows].T.copy()
        otu.columns = [f"sample_{i + 1:05d}" for i in range(otu.shape[1])]
        otu.to_csv(
            out_dir / f"{label}_absolute_{n_features}.tsv",
            sep="\t",
            index=True,
            index_label="#OTU ID",
        )


def run_split_pipeline(
    *,
    mb_train: str | Path,
    mt_train: str | Path,
    out_root: str | Path,
    label_column: Optional[str] = None,
    seeds: Optional[Iterable[int]] = None,
    n_splits: int = 5,
) -> Dict[str, object]:
    mb_df, mt_df = load_aligned_tables(
        mb_train,
        mt_train,
        label_column=label_column,
        context="split_train",
    )

    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    labels = mb_df.index.astype(str).to_numpy()
    label_order = list(pd.Index(labels).unique().astype(str))
    counts = ordered_label_counts(labels)
    too_small = {label: count for label, count in counts.items() if count < n_splits}
    if too_small:
        raise ValueError(
            f"Some labels have fewer than n_splits={n_splits} samples: {too_small}"
        )

    split_stats_rows = []
    seeds = [int(seed) for seed in (seeds or DEFAULT_SEEDS)]
    for seed in seeds:
        seed_dir = out_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(labels.shape[0]), labels)):
            fold_dir = seed_dir / f"fold_{fold}"
            data_dir = fold_dir / "data"
            idx_dir = fold_dir / "indices"
            fastspar_in = fold_dir / "fastspar" / "input"
            fastspar_out = fold_dir / "fastspar" / "output"
            data_dir.mkdir(parents=True, exist_ok=True)
            idx_dir.mkdir(parents=True, exist_ok=True)
            fastspar_in.mkdir(parents=True, exist_ok=True)
            fastspar_out.mkdir(parents=True, exist_ok=True)

            mb_subtrain = mb_df.iloc[train_idx].copy()
            mb_val = mb_df.iloc[val_idx].copy()
            mt_subtrain = mt_df.iloc[train_idx].copy()
            mt_val = mt_df.iloc[val_idx].copy()

            mb_subtrain.to_csv(data_dir / "mb_subtrain.csv")
            mb_val.to_csv(data_dir / "mb_val.csv")
            mt_subtrain.to_csv(data_dir / "metabolites_subtrain.csv")
            mt_val.to_csv(data_dir / "metabolites_val.csv")

            np.save(idx_dir / "train_indices.npy", train_idx.astype(int))
            np.save(idx_dir / "val_indices.npy", val_idx.astype(int))
            save_index_table(idx_dir / "train_indices.csv", train_idx, labels)
            save_index_table(idx_dir / "val_indices.csv", val_idx, labels)
            export_fastspar_input_tables(mb_subtrain, label_order, fastspar_in)

            write_json(
                fold_dir / "labels.json",
                {
                    "original_labels": label_order,
                    "seed": int(seed),
                    "fold": int(fold),
                    "n_splits": int(n_splits),
                    "n_train": int(train_idx.size),
                    "n_val": int(val_idx.size),
                },
            )

            for split_name, idx in (("train", train_idx), ("val", val_idx)):
                value_counts = pd.Series(labels[idx]).value_counts(sort=False)
                for label in label_order:
                    split_stats_rows.append(
                        {
                            "seed": int(seed),
                            "fold": int(fold),
                            "split": split_name,
                            "label": label,
                            "count": int(value_counts.get(label, 0)),
                        }
                    )

            print(
                f"[OK] split seed={seed} fold={fold} train={train_idx.size} "
                f"val={val_idx.size} -> {fold_dir}"
            )

    pd.DataFrame(split_stats_rows).to_csv(out_root / "split_stats.csv", index=False)
    return {
        "out_root": str(out_root),
        "original_labels": label_order,
        "seeds": seeds,
        "n_splits": int(n_splits),
    }
