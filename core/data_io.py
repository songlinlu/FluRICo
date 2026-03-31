from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ordered_label_counts(labels: np.ndarray) -> Dict[str, int]:
    series = pd.Series(labels.astype(str))
    return {str(key): int(value) for key, value in series.value_counts(sort=False).items()}


def _coerce_numeric_features(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    keep_cols = [col for col in out.columns if out[col].notna().any()]
    if not keep_cols:
        raise ValueError(f"No numeric feature columns found in {path}")
    dropped = [str(col) for col in out.columns if col not in keep_cols]
    if dropped:
        print(f"[WARN] Dropping non-numeric columns from {path.name}: {dropped}")
    out = out.loc[:, keep_cols].copy()
    out = out.fillna(0.0)
    return out


def read_feature_table(path: str | Path, label_column: Optional[str] = None) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")

    if label_column is None:
        raw = pd.read_csv(path, index_col=0)
        labels = raw.index.astype(str)
        features = _coerce_numeric_features(raw, path)
        features.index = pd.Index(labels, name="label")
        return features

    raw = pd.read_csv(path)
    if label_column not in raw.columns:
        raise ValueError(f"label_column '{label_column}' not found in {path}")

    labels = raw[label_column].astype(str)
    features = raw.drop(columns=[label_column])
    features = _coerce_numeric_features(features, path)
    features.index = pd.Index(labels, name="label")
    return features


def assert_aligned(mb_df: pd.DataFrame, mt_df: pd.DataFrame, context: str) -> None:
    if mb_df.shape[0] != mt_df.shape[0]:
        raise ValueError(
            f"[{context}] row count mismatch: microbe={mb_df.shape[0]} metabolite={mt_df.shape[0]}"
        )
    mb_labels = mb_df.index.astype(str).to_numpy()
    mt_labels = mt_df.index.astype(str).to_numpy()
    if not np.array_equal(mb_labels, mt_labels):
        mismatch = np.flatnonzero(mb_labels != mt_labels)
        first_bad = int(mismatch[0]) if mismatch.size else -1
        raise ValueError(
            f"[{context}] row labels not aligned between microbe and metabolite, "
            f"first_mismatch_position={first_bad}"
        )


def load_aligned_tables(
    mb_path: str | Path,
    mt_path: str | Path,
    label_column: Optional[str] = None,
    context: str = "input",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mb_df = read_feature_table(mb_path, label_column=label_column)
    mt_df = read_feature_table(mt_path, label_column=label_column)
    assert_aligned(mb_df, mt_df, context=context)
    return mb_df, mt_df
