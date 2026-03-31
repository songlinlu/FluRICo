from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .config import RunConfig, TaskSpec, ordered_unique
from .data_io import write_json
from .fastspar import run_fastspar_from_split_root
from .score_names import (
    CROSS_OMICS_CORRELATION,
    INTRA_OMICS_CORRELATION,
    LINK_DETAILS,
    NEGATIVE_LINKS,
    POSITIVE_LINKS,
    VARIANCE_HETEROGENEITY,
    is_change_effect_score_name,
    safe_score_filename,
)
from .split import run_split_pipeline
from .task_view import build_task_views_from_split, prepare_task_test_tables

STRUCTURAL_SCORE_NAMES = [
    INTRA_OMICS_CORRELATION,
    CROSS_OMICS_CORRELATION,
]


def _read_labels(fold_dir: Path) -> List[str]:
    payload = pd.read_json(fold_dir / "labels.json", typ="series")
    labels = [str(item) for item in payload["labels"]]
    if not labels:
        raise ValueError(f"No labels found in {fold_dir / 'labels.json'}")
    return labels


def _load_fastspar_correlations(fold_dir: Path, labels: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    fastspar_output = fold_dir / "fastspar" / "output"
    for label in labels:
        corr_path = fastspar_output / f"{label}_median_correlation.tsv"
        if not corr_path.exists():
            raise FileNotFoundError(f"Missing FastSpar correlation file: {corr_path}")
        out[label] = pd.read_csv(corr_path, sep="\t", index_col=0)
    return out


def _top_features_from_column(df: pd.DataFrame, column: str, top_n: int) -> List[str]:
    if column not in df.columns or top_n <= 0:
        return []
    return df.nlargest(top_n, column, keep="all").index.astype(str).tolist()


def _write_list_txt(path: Path, items: List[str]) -> None:
    path.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")


def _write_top_by_label_score_csv(path: Path, top_by_label_score: Dict[str, Dict[str, List[str]]], labels: List[str]) -> None:
    rows = []
    for label in labels:
        score_map = top_by_label_score.get(label, {})
        for score_name, features in score_map.items():
            for rank, feature in enumerate(features, start=1):
                rows.append(
                    {
                        "label": label,
                        "score_name": score_name,
                        "rank": rank,
                        "feature": feature,
                    }
                )
    pd.DataFrame(rows, columns=["label", "score_name", "rank", "feature"]).to_csv(path, index=False)


def _write_score_union_csv(path: Path, score_union: Dict[str, List[str]], score_name_order: List[str]) -> None:
    rows = []
    for score_name in score_name_order:
        for rank, feature in enumerate(score_union.get(score_name, []), start=1):
            rows.append(
                {
                    "score_name": score_name,
                    "rank": rank,
                    "feature": feature,
                }
            )
    pd.DataFrame(rows, columns=["score_name", "rank", "feature"]).to_csv(path, index=False)


def _write_category_union_csv(path: Path, category_union: Dict[str, List[str]]) -> None:
    rows = []
    for category in ["dynamic_union", "structural_union", "all_union"]:
        for rank, feature in enumerate(category_union.get(category, []), start=1):
            rows.append({"category": category, "rank": rank, "feature": feature})
    pd.DataFrame(rows, columns=["category", "rank", "feature"]).to_csv(path, index=False)


def _build_score_feature_lists(score_union: Dict[str, List[str]], score_name_order: List[str]) -> Dict[str, List[str]]:
    payload: Dict[str, List[str]] = {}
    for score_name in score_name_order:
        payload[score_name] = list(score_union.get(score_name, []))
    return payload


def _write_score_feature_lists_csv(
    path: Path,
    score_feature_lists: Dict[str, List[str]],
    score_name_order: List[str],
) -> None:
    rows = []
    for score_name in score_name_order:
        for rank, feature in enumerate(score_feature_lists.get(score_name, []), start=1):
            rows.append(
                {
                    "score_name": score_name,
                    "rank": rank,
                    "feature": feature,
                }
            )
    pd.DataFrame(rows, columns=["score_name", "rank", "feature"]).to_csv(path, index=False)


def _write_per_score_txts(
    out_dir: Path,
    score_feature_lists: Dict[str, List[str]],
    score_name_order: List[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for score_name in score_name_order:
        features = [str(item) for item in score_feature_lists.get(score_name, [])]
        _write_list_txt(out_dir / f"{safe_score_filename(score_name)}.txt", features)


def _extract_feature_lists_for_data_type(analyzer, labels: List[str], data_type: str, top_n: int) -> Dict[str, object]:
    top_by_label_score: Dict[str, Dict[str, List[str]]] = {}
    all_change_effect_scores: List[str] = []
    dynamic_category_scores: List[str] = []
    structural_scores: List[str] = []

    for label in labels:
        summary_df = analyzer._build_summary_table_for_group(data_type, label)
        change_effect_scores = [column for column in summary_df.columns if is_change_effect_score_name(column)]
        all_change_effect_scores = ordered_unique(all_change_effect_scores + change_effect_scores)
        label_scores: Dict[str, List[str]] = {}

        if VARIANCE_HETEROGENEITY in summary_df.columns:
            dynamic_category_scores = ordered_unique(dynamic_category_scores + [VARIANCE_HETEROGENEITY])
            label_scores[VARIANCE_HETEROGENEITY] = _top_features_from_column(summary_df, VARIANCE_HETEROGENEITY, top_n)

        for score_name in change_effect_scores:
            label_scores[score_name] = _top_features_from_column(summary_df, score_name, top_n)

        for score_name in STRUCTURAL_SCORE_NAMES:
            if score_name in summary_df.columns:
                structural_scores = ordered_unique(structural_scores + [score_name])
                label_scores[score_name] = _top_features_from_column(summary_df, score_name, top_n)

        for score_name in [POSITIVE_LINKS, NEGATIVE_LINKS]:
            if score_name in summary_df.columns:
                dynamic_category_scores = ordered_unique(dynamic_category_scores + [score_name])
                label_scores[score_name] = _top_features_from_column(summary_df, score_name, top_n)

        top_by_label_score[label] = label_scores

    dynamic_score_name_order = ordered_unique(
        [VARIANCE_HETEROGENEITY] + all_change_effect_scores + [POSITIVE_LINKS, NEGATIVE_LINKS]
    )
    dynamic_score_name_order = [score_name for score_name in dynamic_score_name_order if score_name in dynamic_category_scores or is_change_effect_score_name(score_name)]
    structural_score_name_order = [score_name for score_name in STRUCTURAL_SCORE_NAMES if score_name in structural_scores]
    score_name_order = dynamic_score_name_order + structural_score_name_order

    score_union: Dict[str, List[str]] = {}
    for score_name in score_name_order:
        score_union[score_name] = ordered_unique(
            [
                feature
                for label in labels
                for feature in top_by_label_score.get(label, {}).get(score_name, [])
            ]
        )

    dynamic_union = ordered_unique(
        [feature for score_name in dynamic_score_name_order for feature in score_union.get(score_name, [])]
    )
    structural_union = ordered_unique(
        [feature for score_name in structural_score_name_order for feature in score_union.get(score_name, [])]
    )
    all_union = ordered_unique(dynamic_union + structural_union)

    return {
        "top_by_label_score": top_by_label_score,
        "score_union": score_union,
        "score_name_order": score_name_order,
        "category_union": {
            "dynamic_union": dynamic_union,
            "structural_union": structural_union,
            "all_union": all_union,
        },
    }


def _save_feature_lists(feature_lists_dir: Path, prefix: str, labels: List[str], extracted: Dict[str, object]) -> None:
    feature_lists_dir.mkdir(parents=True, exist_ok=True)
    top_by_label_score = extracted["top_by_label_score"]
    score_union = extracted["score_union"]
    score_name_order = extracted["score_name_order"]
    category_union = extracted["category_union"]
    score_feature_lists = _build_score_feature_lists(score_union=score_union, score_name_order=score_name_order)

    write_json(feature_lists_dir / f"{prefix}_top_by_label_score.json", top_by_label_score)
    _write_top_by_label_score_csv(
        feature_lists_dir / f"{prefix}_top_by_label_score.csv",
        top_by_label_score=top_by_label_score,
        labels=labels,
    )

    write_json(feature_lists_dir / f"{prefix}_score_union.json", score_union)
    _write_score_union_csv(
        feature_lists_dir / f"{prefix}_score_union.csv",
        score_union=score_union,
        score_name_order=score_name_order,
    )

    write_json(feature_lists_dir / f"{prefix}_score_feature_lists.json", score_feature_lists)
    _write_score_feature_lists_csv(
        feature_lists_dir / f"{prefix}_score_feature_lists.csv",
        score_feature_lists=score_feature_lists,
        score_name_order=score_name_order,
    )
    _write_per_score_txts(
        feature_lists_dir / f"{prefix}_per_score_lists",
        score_feature_lists=score_feature_lists,
        score_name_order=score_name_order,
    )

    write_json(feature_lists_dir / f"{prefix}_category_union.json", category_union)
    _write_category_union_csv(feature_lists_dir / f"{prefix}_category_union.csv", category_union)
    _write_list_txt(feature_lists_dir / f"{prefix}_dynamic_union.txt", category_union["dynamic_union"])
    _write_list_txt(feature_lists_dir / f"{prefix}_structural_union.txt", category_union["structural_union"])
    _write_list_txt(feature_lists_dir / f"{prefix}_all_union.txt", category_union["all_union"])


def export_feature_lists_for_analyzer(
    *,
    analyzer,
    labels: Iterable[str],
    out_dir: str | Path,
    mb_top_n: int = 50,
    mt_top_n: int = 20,
) -> Dict[str, str]:
    resolved_labels = [str(item) for item in labels]
    feature_lists_dir = Path(out_dir).resolve()
    feature_lists_dir.mkdir(parents=True, exist_ok=True)

    mb_extracted = _extract_feature_lists_for_data_type(analyzer, resolved_labels, "microbe", mb_top_n)
    mt_extracted = _extract_feature_lists_for_data_type(analyzer, resolved_labels, "metabolite", mt_top_n)
    _save_feature_lists(feature_lists_dir, "microbe", resolved_labels, mb_extracted)
    _save_feature_lists(feature_lists_dir, "metabolite", resolved_labels, mt_extracted)

    return {
        "feature_lists_dir": str(feature_lists_dir),
        "microbe_score_feature_lists": str(feature_lists_dir / "microbe_score_feature_lists.csv"),
        "metabolite_score_feature_lists": str(feature_lists_dir / "metabolite_score_feature_lists.csv"),
    }


def _run_single_fold(
    *,
    task_root: Path,
    seed: int,
    fold: int,
    n_jobs: int,
    use_fast: bool,
    mb_top_n: int,
    mt_top_n: int,
    flurico_log_microbe: bool,
    flurico_log_metabolite: bool,
    save_pair_detail: bool = False,
    worker_logs: bool = True,
) -> None:
    if use_fast:
        from .flurico_fast import FluRiCoAnalysisFast as AnalyzerClass
    else:
        from .flurico import FluRiCoAnalysis as AnalyzerClass

    fold_dir = task_root / f"seed_{seed}" / f"fold_{fold}"
    data_dir = fold_dir / "data"
    labels = _read_labels(fold_dir)
    mb_path = data_dir / ("mb_subtrain_log1p.csv" if flurico_log_microbe else "mb_subtrain.csv")
    mt_path = data_dir / ("metabolites_subtrain_log1p.csv" if flurico_log_metabolite else "metabolites_subtrain.csv")
    if not mb_path.exists() or not mt_path.exists():
        raise FileNotFoundError(f"Missing FluRiCo input files in {data_dir}")

    mb_df = pd.read_csv(mb_path, index_col=0)
    mt_df = pd.read_csv(mt_path, index_col=0)
    mb_corr = _load_fastspar_correlations(fold_dir, labels)
    results_dir = fold_dir / "flurico" / "results"
    feature_lists_dir = fold_dir / "flurico" / "feature_lists"
    log_path = fold_dir / "flurico" / "logs" / f"seed_{seed}_fold_{fold}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _do_work() -> None:
        analyzer = AnalyzerClass(
            microbe_data=mb_df,
            metabolite_data=mt_df,
            group_labels=labels,
            microbe_coab_data=mb_corr,
            n_jobs=n_jobs,
        )
        if use_fast:
            setattr(analyzer, "save_remote_trend_detail", bool(save_pair_detail))
            analyzer.calculate_all_scores(n_jobs=n_jobs, parallel_sections=True)
        else:
            analyzer.calculate_all_scores(n_jobs=n_jobs)
            if not save_pair_detail:
                analyzer.results[LINK_DETAILS] = None

        results_dir.mkdir(parents=True, exist_ok=True)
        analyzer.save_results(output_dir=str(results_dir))
        export_feature_lists_for_analyzer(
            analyzer=analyzer,
            labels=labels,
            out_dir=feature_lists_dir,
            mb_top_n=mb_top_n,
            mt_top_n=mt_top_n,
        )

    if worker_logs:
        with log_path.open("w", encoding="utf-8") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            _do_work()
    else:
        _do_work()

    print(f"[OK] FluRiCo seed={seed} fold={fold} -> {results_dir}")


def run_flurico_from_task_root(
    *,
    task_root: str | Path,
    seeds: Iterable[int],
    n_folds: int,
    run_config: RunConfig,
) -> Dict[str, object]:
    task_root = Path(task_root).resolve()
    for seed in seeds:
        for fold in range(n_folds):
            _run_single_fold(
                task_root=task_root,
                seed=int(seed),
                fold=int(fold),
                n_jobs=run_config.flurico_n_jobs,
                use_fast=run_config.flurico_use_fast,
                mb_top_n=run_config.mb_top_n,
                mt_top_n=run_config.mt_top_n,
                flurico_log_microbe=run_config.flurico_log_microbe,
                flurico_log_metabolite=run_config.flurico_log_metabolite,
            )
    return {"task_root": str(task_root), "seeds": [int(seed) for seed in seeds], "n_folds": int(n_folds)}


def run_flurico_pipeline(
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

    out_dir = run_config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "task_spec.json", task_spec.to_dict())

    raw_split_root = out_dir / "multiround_raw"
    task_root = out_dir / "multiround"
    split_info = run_split_pipeline(
        mb_train=mb_train,
        mt_train=mt_train,
        out_root=raw_split_root,
        label_column=label_column,
        seeds=run_config.seeds,
        n_splits=run_config.n_splits,
    )
    task_info = build_task_views_from_split(
        source_root=raw_split_root,
        out_root=task_root,
        task_spec=task_spec,
        n_folds=run_config.n_splits,
        zero_max_fraction=run_config.zero_max_fraction,
    )
    test_info = None
    if mb_test is not None and mt_test is not None:
        test_info = prepare_task_test_tables(
            mb_test=mb_test,
            mt_test=mt_test,
            out_dir=out_dir / "task_test",
            task_spec=task_spec,
            label_column=label_column,
        )

    run_fastspar_from_split_root(
        root=task_root,
        seeds=run_config.seeds,
        folds=range(run_config.n_splits),
        fastspar_bin=run_config.fastspar_bin,
        jobs=run_config.fastspar_jobs,
        threads=run_config.fastspar_threads,
    )
    flurico_info = run_flurico_from_task_root(
        task_root=task_root,
        seeds=run_config.seeds,
        n_folds=run_config.n_splits,
        run_config=run_config,
    )

    return {
        "out_dir": str(out_dir),
        "raw_split_root": str(raw_split_root),
        "task_root": str(task_root),
        "split_info": split_info,
        "task_info": task_info,
        "test_info": test_info,
        "flurico_info": flurico_info,
    }
