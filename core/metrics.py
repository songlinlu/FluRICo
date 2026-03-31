from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


PRIMARY_METRIC_NAME = "roc_auc"


def encode_labels(labels: Sequence[str], label_order: Sequence[str]) -> np.ndarray:
    mapping = {str(label): idx for idx, label in enumerate(label_order)}
    try:
        return np.array([mapping[str(label)] for label in labels], dtype=int)
    except KeyError as exc:
        raise ValueError(f"Unknown label encountered during encoding: {exc}") from exc


def resolve_positive_label(label_order: Sequence[str], positive_label: Optional[str]) -> str:
    if positive_label is not None:
        if positive_label not in label_order:
            raise ValueError(f"positive_label '{positive_label}' not found in label_order={list(label_order)}")
        return positive_label
    return str(label_order[-1])


def _safe_metric(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def compute_classification_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_order: Sequence[str],
    positive_label: Optional[str] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    n_classes = len(label_order)

    if y_true.size == 0:
        return {
            "primary_metric": float("nan"),
            "roc_auc": float("nan"),
            "auprc": float("nan"),
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "f1_macro": float("nan"),
            "logloss": float("nan"),
        }

    if n_classes == 2:
        pos_label = resolve_positive_label(label_order, positive_label)
        pos_index = list(label_order).index(pos_label)
        pos_scores = y_proba[:, pos_index]
        y_true_binary = (y_true == pos_index).astype(int)
        y_pred_binary = (y_pred == pos_index).astype(int)
        roc_auc = _safe_metric(roc_auc_score, y_true_binary, pos_scores)
        auprc = _safe_metric(average_precision_score, y_true_binary, pos_scores)
    else:
        roc_auc = _safe_metric(
            roc_auc_score,
            y_true,
            y_proba,
            labels=list(range(n_classes)),
            multi_class="ovr",
            average="macro",
        )
        y_true_onehot = label_binarize(y_true, classes=list(range(n_classes)))
        auprc = _safe_metric(average_precision_score, y_true_onehot, y_proba, average="macro")

    return {
        "primary_metric": roc_auc,
        "roc_auc": roc_auc,
        "auprc": auprc,
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
        "f1_macro": _safe_metric(f1_score, y_true, y_pred, average="macro"),
        "logloss": _safe_metric(log_loss, y_true, y_proba, labels=list(range(n_classes))),
    }
