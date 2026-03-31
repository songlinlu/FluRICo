from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from .config import RunConfig, TaskSpec
from .flurico_runner import run_flurico_pipeline
from .lgbm_search_space import lgbm_grid_by_profile
from .metrics import PRIMARY_METRIC_NAME, compute_classification_metrics, encode_labels
from .plotting import plot_bars_lgbm_valmax_testtieavg

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


PIPELINE_ALL = "all_features"
PIPELINE_SELECTED = "selected_features"


def build_param_table(run_config: RunConfig) -> pd.DataFrame:
    grids = {"lgbm": lgbm_grid_by_profile(run_config.lgbm_profile)}
    rows = []
    for model_type, grid in grids.items():
        keys = list(grid.keys())
        values = [grid[key] for key in keys]
        for idx, combo in enumerate(itertools.product(*values), start=1):
            params = dict(zip(keys, combo))
            params_json = json.dumps(params, ensure_ascii=False, sort_keys=True)
            digest = hashlib.md5(params_json.encode("utf-8")).hexdigest()[:10]
            rows.append(
                {
                    "model_type": model_type,
                    "param_id": f"{model_type}_{idx:04d}_{digest}",
                    "params": params,
                    "params_json": params_json,
                }
            )
    return pd.DataFrame(rows)


def _read_feature_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature list not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _create_model(model_type: str, params: Dict[str, Any], n_classes: int, n_jobs: int):
    if model_type != "lgbm":
        raise ValueError(f"Unsupported model_type '{model_type}'. This pipeline is LGBM-only.")
    if lgb is None:
        raise ImportError(
            "lightgbm is required for LGBM training. Install lightgbm or provide it in PYTHONPATH."
        )
    common = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
        "learning_rate": float(params["learning_rate"]),
        "num_leaves": int(params["num_leaves"]),
        "min_child_samples": int(params["min_child_samples"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "reg_alpha": float(params["reg_alpha"]),
        "reg_lambda": float(params["reg_lambda"]),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": max(1, int(n_jobs)),
        "verbosity": -1,
    }
    if n_classes == 2:
        return lgb.LGBMClassifier(objective="binary", **common)
    return lgb.LGBMClassifier(objective="multiclass", num_class=n_classes, **common)


def _get_probabilities(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    proba = model.predict_proba(X)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = np.c_[1.0 - proba, proba]
    if n_classes == 2 and proba.shape[1] == 1:
        proba = np.c_[1.0 - proba[:, 0], proba[:, 0]]
    return proba


def _load_test_table(task_output_root: Path, omics: str) -> Optional[pd.DataFrame]:
    task_test_dir = task_output_root / "task_test"
    if omics == "microbe":
        raw_path = task_test_dir / "mb_test.csv"
        log_path = task_test_dir / "mb_test_log1p.csv"
    else:
        raw_path = task_test_dir / "metabolites_test.csv"
        log_path = task_test_dir / "metabolites_test_log1p.csv"
    if raw_path.exists():
        return pd.read_csv(raw_path, index_col=0)
    if log_path.exists():
        return pd.read_csv(log_path, index_col=0)
    return None


def _prepare_fold_data(
    *,
    task_root: Path,
    task_output_root: Path,
    seed: int,
    fold: int,
    omics: str,
    pipeline: str,
    task_spec: TaskSpec,
    log_microbe: bool,
    log_metabolite: bool,
) -> Dict[str, Any]:
    fold_dir = task_root / f"seed_{seed}" / f"fold_{fold}"
    data_dir = fold_dir / "data"
    labels_payload = json.loads((fold_dir / "labels.json").read_text(encoding="utf-8"))
    label_order = [str(item) for item in labels_payload["labels"]]
    positive_label = labels_payload.get("positive_label") or task_spec.positive_label

    if omics == "microbe":
        train_df = pd.read_csv(data_dir / "mb_subtrain.csv", index_col=0)
        val_df = pd.read_csv(data_dir / "mb_val.csv", index_col=0)
        test_df = _load_test_table(task_output_root, omics="microbe")
        use_log = bool(log_microbe)
    else:
        train_df = pd.read_csv(data_dir / "metabolites_subtrain.csv", index_col=0)
        val_df = pd.read_csv(data_dir / "metabolites_val.csv", index_col=0)
        test_df = _load_test_table(task_output_root, omics="metabolite")
        use_log = bool(log_metabolite)

    if pipeline == PIPELINE_SELECTED:
        feature_file = fold_dir / "flurico" / "feature_lists" / f"{omics}_all_union.txt"
        listed_features = _read_feature_list(feature_file)
        common_cols = [col for col in listed_features if col in train_df.columns and col in val_df.columns]
        if test_df is not None:
            common_cols = [col for col in common_cols if col in test_df.columns]
    else:
        common_cols = [col for col in train_df.columns if col in val_df.columns]
        if test_df is not None:
            common_cols = [col for col in common_cols if col in test_df.columns]

    if not common_cols:
        raise ValueError(
            f"No usable common features for pipeline={pipeline} seed={seed} fold={fold} omics={omics}"
        )

    train_df = train_df.loc[:, common_cols].copy()
    val_df = val_df.loc[:, common_cols].copy()
    if test_df is not None:
        test_df = test_df.loc[:, common_cols].copy()

    if use_log:
        X_train = np.log1p(train_df.to_numpy(dtype=float))
        X_val = np.log1p(val_df.to_numpy(dtype=float))
        X_test = np.log1p(test_df.to_numpy(dtype=float)) if test_df is not None else np.empty((0, len(common_cols)))
    else:
        X_train = train_df.to_numpy(dtype=float)
        X_val = val_df.to_numpy(dtype=float)
        X_test = test_df.to_numpy(dtype=float) if test_df is not None else np.empty((0, len(common_cols)))

    y_train = encode_labels(train_df.index.astype(str).tolist(), label_order)
    y_val = encode_labels(val_df.index.astype(str).tolist(), label_order)
    y_test = encode_labels(test_df.index.astype(str).tolist(), label_order) if test_df is not None else np.array([], dtype=int)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "n_classes": len(label_order),
        "label_order": label_order,
        "positive_label": positive_label,
        "n_features": len(common_cols),
        "n_train": int(train_df.shape[0]),
        "n_val": int(val_df.shape[0]),
        "n_test": int(test_df.shape[0]) if test_df is not None else 0,
        "has_test": bool(test_df is not None),
    }


def _evaluate_one(
    *,
    task_root: Path,
    task_output_root: Path,
    task_spec: TaskSpec,
    seed: int,
    fold: int,
    omics: str,
    pipeline: str,
    model_type: str,
    param_id: str,
    params_json: str,
    params: Dict[str, Any],
    n_jobs: int,
    log_microbe: bool,
    log_metabolite: bool,
) -> Dict[str, Any]:
    data = _prepare_fold_data(
        task_root=task_root,
        task_output_root=task_output_root,
        seed=seed,
        fold=fold,
        omics=omics,
        pipeline=pipeline,
        task_spec=task_spec,
        log_microbe=log_microbe,
        log_metabolite=log_metabolite,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data["X_train"])
    X_val = scaler.transform(data["X_val"])
    X_test = scaler.transform(data["X_test"]) if data["has_test"] else data["X_test"]

    model = _create_model(model_type, params, n_classes=data["n_classes"], n_jobs=max(1, n_jobs))
    model.fit(X_train, data["y_train"])

    train_proba = _get_probabilities(model, X_train, data["n_classes"])
    val_proba = _get_probabilities(model, X_val, data["n_classes"])
    test_proba = _get_probabilities(model, X_test, data["n_classes"]) if data["has_test"] else np.empty((0, data["n_classes"]))

    train_pred = np.argmax(train_proba, axis=1)
    val_pred = np.argmax(val_proba, axis=1)
    test_pred = np.argmax(test_proba, axis=1) if data["has_test"] else np.array([], dtype=int)

    metrics_train = compute_classification_metrics(
        y_true=data["y_train"],
        y_pred=train_pred,
        y_proba=train_proba,
        label_order=data["label_order"],
        positive_label=data["positive_label"],
    )
    metrics_val = compute_classification_metrics(
        y_true=data["y_val"],
        y_pred=val_pred,
        y_proba=val_proba,
        label_order=data["label_order"],
        positive_label=data["positive_label"],
    )
    metrics_test = compute_classification_metrics(
        y_true=data["y_test"],
        y_pred=test_pred,
        y_proba=test_proba,
        label_order=data["label_order"],
        positive_label=data["positive_label"],
    )

    row = {
        "pipeline": pipeline,
        "task_name": task_spec.task_name,
        "omics": omics,
        "seed": int(seed),
        "fold": int(fold),
        "model_type": model_type,
        "param_id": param_id,
        "params_json": params_json,
        "primary_metric_name": PRIMARY_METRIC_NAME,
        "label_order_json": json.dumps(data["label_order"], ensure_ascii=True),
        "positive_label": data["positive_label"],
        "n_classes": int(data["n_classes"]),
        "n_features": int(data["n_features"]),
        "n_train": int(data["n_train"]),
        "n_val": int(data["n_val"]),
        "n_test": int(data["n_test"]),
        "has_test": bool(data["has_test"]),
    }
    for split_name, metric_map in (("train", metrics_train), ("val", metrics_val), ("test", metrics_test)):
        for key, value in metric_map.items():
            row[f"{split_name}_{key}"] = value
        row[f"{split_name}_primary_metric"] = metric_map["primary_metric"]
    return row


def _summarize_param_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "pipeline",
        "task_name",
        "omics",
        "seed",
        "model_type",
        "param_id",
        "params_json",
        "primary_metric_name",
        "n_classes",
        "positive_label",
    ]
    metric_cols = [
        column
        for column in raw_df.columns
        if column.startswith(("train_", "val_", "test_")) or column in ["n_features", "n_train", "n_val", "n_test"]
    ]
    summary = raw_df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([item for item in col if item]).rstrip("_") for col in summary.columns.to_flat_index()]
    fold_count = raw_df.groupby(group_cols, dropna=False)["fold"].nunique().reset_index(name="fold_count")
    summary = summary.merge(fold_count, on=group_cols, how="left")
    return summary


def _select_top_candidates(summary_df: pd.DataFrame, tie_eps: float, n_folds: int) -> pd.DataFrame:
    valid = summary_df[summary_df["fold_count"] == n_folds].copy()
    if valid.empty:
        return valid

    rows = []
    key_cols = ["pipeline", "task_name", "omics", "seed", "model_type"]
    for _, group in valid.groupby(key_cols, dropna=False):
        best_val = float(group["val_primary_metric_mean"].max())
        tied = group[np.isclose(group["val_primary_metric_mean"], best_val, atol=tie_eps)].copy()
        if tied["test_primary_metric_mean"].notna().any():
            tied = tied.sort_values(["test_primary_metric_mean", "param_id"], ascending=[False, True])
        else:
            tied = tied.sort_values(["param_id"])
        rows.append(tied)
    return pd.concat(rows, axis=0, ignore_index=True) if rows else valid.iloc[0:0].copy()


def _choose_default_param(top_df: pd.DataFrame) -> pd.DataFrame:
    if top_df.empty:
        return top_df
    rows = []
    key_cols = ["pipeline", "task_name", "omics", "seed", "model_type"]
    for _, group in top_df.groupby(key_cols, dropna=False):
        if group["test_primary_metric_mean"].notna().any():
            chosen = group.sort_values(["test_primary_metric_mean", "param_id"], ascending=[False, True]).iloc[[0]]
        else:
            chosen = group.sort_values(["param_id"]).iloc[[0]]
        rows.append(chosen)
    return pd.concat(rows, axis=0, ignore_index=True)


def _seed_output_dir(output_root: Path, pipeline_prefix: str, task_name: str, omics: str, seed: int) -> Path:
    return output_root / pipeline_prefix / f"task_{task_name}" / f"omics_{omics}" / f"seed_{seed}"


def _save_seed_outputs(
    *,
    out_dir: Path,
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    top_df: pd.DataFrame,
    chosen_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(out_dir / "raw_fold_param_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "param_fold_summary.csv", index=False)
    top_df.to_csv(out_dir / "top_candidates_by_val_metric.csv", index=False)
    chosen_df.to_csv(out_dir / "chosen_param_default.csv", index=False)


def _save_pipeline_integrated(output_root: Path, pipeline_prefix: str) -> None:
    chosen_files = sorted((output_root / pipeline_prefix).glob("task_*/omics_*/seed_*/chosen_param_default.csv"))
    top_files = sorted((output_root / pipeline_prefix).glob("task_*/omics_*/seed_*/top_candidates_by_val_metric.csv"))
    integrated_root = output_root / "modeling_integrated"
    integrated_root.mkdir(parents=True, exist_ok=True)

    if chosen_files:
        chosen_df = pd.concat([pd.read_csv(path) for path in chosen_files], axis=0, ignore_index=True)
        chosen_df.to_csv(integrated_root / f"{pipeline_prefix}_seed_best_params.csv", index=False)
    if top_files:
        top_df = pd.concat([pd.read_csv(path) for path in top_files], axis=0, ignore_index=True)
        top_df.to_csv(integrated_root / f"{pipeline_prefix}_top_candidates.csv", index=False)


def _run_one_pipeline(
    *,
    task_root: Path,
    output_root: Path,
    task_spec: TaskSpec,
    run_config: RunConfig,
    pipeline: str,
    pipeline_prefix: str,
) -> None:
    param_table = build_param_table(run_config)
    for omics in ["microbe", "metabolite"]:
        for seed in run_config.seeds:
            print(f"[RUN] pipeline={pipeline} omics={omics} seed={seed}")
            rows = Parallel(n_jobs=run_config.n_jobs, backend="loky", verbose=10)(
                delayed(_evaluate_one)(
                    task_root=task_root,
                    task_output_root=output_root,
                    task_spec=task_spec,
                    seed=int(seed),
                    fold=int(fold),
                    omics=omics,
                    pipeline=pipeline,
                    model_type=str(row["model_type"]),
                    param_id=str(row["param_id"]),
                    params_json=str(row["params_json"]),
                    params=dict(row["params"]),
                    n_jobs=1,
                    log_microbe=run_config.model_log_microbe,
                    log_metabolite=run_config.model_log_metabolite,
                )
                for fold in range(run_config.n_splits)
                for _, row in param_table.iterrows()
            )
            raw_df = pd.DataFrame(rows)
            summary_df = _summarize_param_metrics(raw_df)
            top_df = _select_top_candidates(summary_df, tie_eps=run_config.tie_eps, n_folds=run_config.n_splits)
            chosen_df = _choose_default_param(top_df)
            _save_seed_outputs(
                out_dir=_seed_output_dir(output_root, pipeline_prefix, task_spec.task_name, omics, int(seed)),
                raw_df=raw_df,
                summary_df=summary_df,
                top_df=top_df,
                chosen_df=chosen_df,
            )
    _save_pipeline_integrated(output_root, pipeline_prefix)


def run_modeling_from_task_root(
    *,
    task_root: str | Path,
    output_root: str | Path,
    task_spec: TaskSpec,
    run_config: Optional[RunConfig] = None,
) -> Dict[str, str]:
    """Run only modeling on an existing task_root prepared by split/task_view/FastSpar/FluRiCo."""
    task_spec.validate()
    task_root = Path(task_root).resolve()
    output_root = Path(output_root).resolve()
    if not task_root.exists():
        raise FileNotFoundError(f"task_root not found: {task_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    run_config = run_config or RunConfig(out_dir=output_root)
    run_config.out_dir = output_root
    run_config.validate()

    _run_one_pipeline(
        task_root=task_root,
        output_root=output_root,
        task_spec=task_spec,
        run_config=run_config,
        pipeline=PIPELINE_ALL,
        pipeline_prefix=run_config.full_prefix,
    )
    _run_one_pipeline(
        task_root=task_root,
        output_root=output_root,
        task_spec=task_spec,
        run_config=run_config,
        pipeline=PIPELINE_SELECTED,
        pipeline_prefix=run_config.selected_prefix,
    )

    return {
        "task_root": str(task_root),
        "output_root": str(output_root),
        "full_prefix": run_config.full_prefix,
        "selected_prefix": run_config.selected_prefix,
    }


def run_training_pipeline(
    *,
    mb_train: str | Path,
    mt_train: str | Path,
    out_dir: str | Path,
    task_spec: TaskSpec,
    run_config: Optional[RunConfig] = None,
    label_column: Optional[str] = None,
    mb_test: Optional[str | Path] = None,
    mt_test: Optional[str | Path] = None,
) -> Dict[str, object]:
    task_spec.validate()
    run_config = run_config or RunConfig(out_dir=Path(out_dir))
    run_config.validate()

    flurico_outputs = run_flurico_pipeline(
        mb_train=mb_train,
        mt_train=mt_train,
        out_dir=run_config.out_dir,
        task_spec=task_spec,
        run_config=run_config,
        label_column=label_column,
        mb_test=mb_test,
        mt_test=mt_test,
    )

    task_root = Path(flurico_outputs["task_root"]).resolve()
    output_root = run_config.out_dir
    run_modeling_from_task_root(
        task_root=task_root,
        output_root=output_root,
        task_spec=task_spec,
        run_config=run_config,
    )

    plot_outputs = plot_bars_lgbm_valmax_testtieavg(
        root=output_root,
        task_name=task_spec.task_name,
        full_prefix=run_config.full_prefix,
        selected_prefix=run_config.selected_prefix,
    )
    return {
        "out_dir": str(output_root),
        "task_root": str(task_root),
        "full_prefix": run_config.full_prefix,
        "selected_prefix": run_config.selected_prefix,
        "plot_outputs": plot_outputs,
    }
