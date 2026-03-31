#!/usr/bin/env python3
"""Fast drop-in FluRiCo implementation.

This module keeps FluRiCo scoring definitions unchanged and only optimizes
execution strategy for Cross-omics Correlation.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

from .flurico import FluRiCoAnalysis
from .score_names import (
    CROSS_OMICS_CORRELATION,
    LINK_DETAILS,
    NEGATIVE_LINKS,
    POSITIVE_LINKS,
)


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """Resolve n_jobs with joblib-like semantics."""
    cpu = os.cpu_count() or 1
    if n_jobs is None:
        return cpu
    if n_jobs == 0:
        raise ValueError("n_jobs=0 is invalid")
    if n_jobs < 0:
        # joblib style: -1 => all CPUs, -2 => all but one, etc.
        return max(1, cpu + 1 + n_jobs)
    return max(1, n_jobs)


def _iter_chunks(total: int, chunk_size: int) -> Iterator[Tuple[int, int]]:
    start = 0
    while start < total:
        end = min(total, start + chunk_size)
        yield start, end
        start = end


def _wmc_spearman_fdr_exact(source_col: np.ndarray, target_matrix: np.ndarray) -> float:
    """Exact scoring logic copied from FluRiCoAnalysis._calculate_wmc_spearman_fdr."""
    corrs = []
    x_all = source_col.astype(float)

    for target_idx in range(target_matrix.shape[1]):
        y_all = target_matrix[:, target_idx].astype(float)

        if x_all.shape[0] != y_all.shape[0]:
            continue

        mask = np.isfinite(x_all) & np.isfinite(y_all)
        if mask.sum() < 3:
            continue

        x = x_all[mask]
        y = y_all[mask]

        try:
            rho, pval = spearmanr(x, y)
        except Exception:
            rho, pval = np.nan, np.nan

        corrs.append((rho, pval))

    if not corrs:
        return 0.0

    rhos, pvals = zip(*corrs)
    rhos = np.array(rhos, dtype=float)
    pvals = np.array(pvals, dtype=float)

    non_nan = ~np.isnan(pvals)
    p_fdr = np.full_like(pvals, np.nan, dtype=float)
    if non_nan.sum() > 0:
        _, p_corr, _, _ = multipletests(pvals[non_nan], method="fdr_bh")
        p_fdr[non_nan] = p_corr

    scores = []
    for rho, pf in zip(rhos, p_fdr):
        if np.isnan(rho) or np.isnan(pf):
            continue

        pf_clip = min(max(pf, 1e-16), 1.0)
        score = abs(rho) * (-np.log10(pf_clip))
        scores.append(score)

    if not scores:
        return 0.0

    return float(np.mean(scores))


def _score_chunk_exact(
    source_matrix: np.ndarray,
    target_matrix: np.ndarray,
    start: int,
    end: int,
) -> Tuple[int, List[float]]:
    """Compute scores for source feature indices in [start, end)."""
    scores: List[float] = []
    for source_idx in range(start, end):
        scores.append(_wmc_spearman_fdr_exact(source_matrix[:, source_idx], target_matrix))
    return start, scores


class FluRiCoAnalysisFast(FluRiCoAnalysis):
    """Drop-in FluRiCo class with faster Cross-omics Correlation execution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default to keep original behavior; caller can switch off for speed.
        self.save_remote_trend_detail = True

    def _should_save_remote_trend_detail(self) -> bool:
        return bool(getattr(self, "save_remote_trend_detail", True))

    @staticmethod
    def _extract_fc_vector(fc_df: pd.DataFrame, features: List[str]) -> np.ndarray:
        if fc_df is None or fc_df.empty:
            return np.zeros(len(features), dtype=float)

        first_rows = fc_df.drop_duplicates(subset=["feature"], keep="first")
        fc_series = pd.to_numeric(first_rows.set_index("feature")["log2FC"], errors="coerce")
        return fc_series.reindex(features).fillna(0.0).to_numpy(dtype=float)

    def _score_direction_chunked(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        n_jobs: int,
        chunk_size: int | None = None,
        verbose: int = 0,
    ) -> List[float]:
        source_matrix = np.ascontiguousarray(source_df.to_numpy(dtype=float, copy=True))
        target_matrix = np.ascontiguousarray(target_df.to_numpy(dtype=float, copy=True))

        n_sources = source_matrix.shape[1]
        if n_sources == 0:
            return []

        resolved_jobs = _resolve_n_jobs(n_jobs)
        if resolved_jobs == 1 or n_sources == 1:
            return [
                _wmc_spearman_fdr_exact(source_matrix[:, source_idx], target_matrix)
                for source_idx in range(n_sources)
            ]

        if chunk_size is None:
            # Heuristic: a few chunks per worker to reduce scheduling and pickle overhead.
            chunk_size = max(16, int(math.ceil(n_sources / (resolved_jobs * 4))))
        else:
            chunk_size = max(1, int(chunk_size))

        chunks = list(_iter_chunks(n_sources, chunk_size))
        if len(chunks) == 1:
            return [
                _wmc_spearman_fdr_exact(source_matrix[:, source_idx], target_matrix)
                for source_idx in range(n_sources)
            ]

        chunk_outputs = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            batch_size=1,
            max_nbytes="50K",
            verbose=int(verbose),
        )(
            delayed(_score_chunk_exact)(source_matrix, target_matrix, start, end)
            for start, end in chunks
        )

        chunk_outputs.sort(key=lambda x: x[0])
        all_scores: List[float] = []
        for _, scores in chunk_outputs:
            all_scores.extend(scores)

        if len(all_scores) != n_sources:
            raise RuntimeError(
                f"Remote interaction score size mismatch: got {len(all_scores)}, expected {n_sources}"
            )
        return all_scores

    def calculate_remote_interaction(
        self,
        data_type: str = "both",
        n_jobs: int | None = None,
        chunk_size: int | None = None,
        verbose: int = 0,
    ):
        """Cross-omics Correlation with faster scheduling."""
        print(f"Computing {CROSS_OMICS_CORRELATION}...")
        if n_jobs is None:
            n_jobs = self.n_jobs

        for label in self.group_labels:
            mb_data = self.microbe_data.loc[self.microbe_data.index == label]
            met_data = self.metabolite_data.loc[self.metabolite_data.index == label]

            if mb_data.shape[0] == 0 or met_data.shape[0] == 0:
                print(f"  Group {label}: microbe or metabolite sample count is 0; skipping {CROSS_OMICS_CORRELATION}.")
                continue

            if data_type in ["microbe", "both"]:
                print(f"  [microbe] Group {label}: microbe -> metabolite {CROSS_OMICS_CORRELATION}...")
                mb_remote = self._score_direction_chunked(
                    source_df=mb_data,
                    target_df=met_data,
                    n_jobs=n_jobs,
                    chunk_size=chunk_size,
                    verbose=verbose,
                )
                self.results["microbe"][CROSS_OMICS_CORRELATION][label] = pd.Series(
                    mb_remote, index=mb_data.columns
                )

            if data_type in ["metabolite", "both"]:
                print(f"  [metabolite] Group {label}: metabolite -> microbe {CROSS_OMICS_CORRELATION}...")
                met_remote = self._score_direction_chunked(
                    source_df=met_data,
                    target_df=mb_data,
                    n_jobs=n_jobs,
                    chunk_size=chunk_size,
                    verbose=verbose,
                )
                self.results["metabolite"][CROSS_OMICS_CORRELATION][label] = pd.Series(
                    met_remote, index=met_data.columns
                )

        if data_type == "microbe":
            return self.results["microbe"][CROSS_OMICS_CORRELATION]
        if data_type == "metabolite":
            return self.results["metabolite"][CROSS_OMICS_CORRELATION]
        return {
            "microbe": self.results["microbe"][CROSS_OMICS_CORRELATION],
            "metabolite": self.results["metabolite"][CROSS_OMICS_CORRELATION],
        }

    def _calculate_trend_cosine_method(self, fc_mb, fc_met, mb_features, met_features, tau_count):
        """Vectorized equivalent of FluRiCoAnalysis._calculate_trend_cosine_method."""
        print("  [step 2/4] Building trend vectors...")
        mb_trend_dict = self._build_trend_vectors_from_fc_detail(fc_mb)
        met_trend_dict = self._build_trend_vectors_from_fc_detail(fc_met)

        mb_vecs = np.vstack(
            [mb_trend_dict.get(f, np.zeros(len(self.fc_pairs), dtype=float)) for f in mb_features]
        )
        met_vecs = np.vstack(
            [met_trend_dict.get(f, np.zeros(len(self.fc_pairs), dtype=float)) for f in met_features]
        )

        print("  [step 3/4] Computing the cosine similarity matrix...")
        mb_norms = np.linalg.norm(mb_vecs, axis=1)
        met_norms = np.linalg.norm(met_vecs, axis=1)
        mb_norms_safe = np.where(mb_norms > 0, mb_norms, 1.0)
        met_norms_safe = np.where(met_norms > 0, met_norms, 1.0)

        mb_unit = mb_vecs / mb_norms_safe[:, None]
        met_unit = met_vecs / met_norms_safe[:, None]
        cos_matrix = np.clip(mb_unit.dot(met_unit.T), -1.0, 1.0)

        print(f"    Cosine matrix shape: {cos_matrix.shape[0]} microbes x {cos_matrix.shape[1]} metabolites")
        print("  [step 4/4] Counting Positive Links and Negative Links...")

        mb_pos_count = np.sum(cos_matrix >= tau_count, axis=1).astype(int)
        mb_neg_count = np.sum(cos_matrix <= -tau_count, axis=1).astype(int)
        met_pos_count = np.sum(cos_matrix >= tau_count, axis=0).astype(int)
        met_neg_count = np.sum(cos_matrix <= -tau_count, axis=0).astype(int)

        self.results["microbe"][POSITIVE_LINKS] = pd.Series(mb_pos_count, index=mb_features)
        self.results["microbe"][NEGATIVE_LINKS] = pd.Series(mb_neg_count, index=mb_features)
        self.results["metabolite"][POSITIVE_LINKS] = pd.Series(met_pos_count, index=met_features)
        self.results["metabolite"][NEGATIVE_LINKS] = pd.Series(met_neg_count, index=met_features)

        if self._should_save_remote_trend_detail():
            detail_df = pd.DataFrame(
                {
                    "microbe": np.repeat(np.asarray(mb_features, dtype=object), len(met_features)),
                    "metabolite": np.tile(np.asarray(met_features, dtype=object), len(mb_features)),
                    "cosine_trend": cos_matrix.ravel(order="C"),
                }
            )
            self.results[LINK_DETAILS] = detail_df
        else:
            self.results[LINK_DETAILS] = None

    def _calculate_trend_two_label_method(self, fc_mb, fc_met, mb_features, met_features):
        """Vectorized equivalent of FluRiCoAnalysis._calculate_trend_two_label_method."""
        print("  [step 2/4] Extracting a single FC value per feature...")
        mb_fc = self._extract_fc_vector(fc_mb, mb_features)
        met_fc = self._extract_fc_vector(fc_met, met_features)

        print("  [step 3/4] Computing sign matching and effect-size products...")
        product = np.abs(np.outer(mb_fc, met_fc))
        same_sign = (mb_fc[:, None] >= 0) == (met_fc[None, :] >= 0)
        diff_sign = ~same_sign

        print("  [step 4/4] Counting Top 20% occurrences...")
        conc_vals = product[same_sign]
        disc_vals = product[diff_sign]
        conc_threshold = np.percentile(conc_vals, 80) if conc_vals.size else np.inf
        disc_threshold = np.percentile(disc_vals, 80) if disc_vals.size else np.inf

        pos_mask = same_sign & (product >= conc_threshold)
        neg_mask = diff_sign & (product >= disc_threshold)

        mb_pos_count = np.sum(pos_mask, axis=1).astype(int)
        mb_neg_count = np.sum(neg_mask, axis=1).astype(int)
        met_pos_count = np.sum(pos_mask, axis=0).astype(int)
        met_neg_count = np.sum(neg_mask, axis=0).astype(int)

        self.results["microbe"][POSITIVE_LINKS] = pd.Series(mb_pos_count, index=mb_features)
        self.results["microbe"][NEGATIVE_LINKS] = pd.Series(mb_neg_count, index=mb_features)
        self.results["metabolite"][POSITIVE_LINKS] = pd.Series(met_pos_count, index=met_features)
        self.results["metabolite"][NEGATIVE_LINKS] = pd.Series(met_neg_count, index=met_features)

        if self._should_save_remote_trend_detail():
            sign_concordance = np.where(same_sign, 1.0, -1.0)
            detail_df = pd.DataFrame(
                {
                    "microbe": np.repeat(np.asarray(mb_features, dtype=object), len(met_features)),
                    "metabolite": np.tile(np.asarray(met_features, dtype=object), len(mb_features)),
                    "sign_concordance": sign_concordance.ravel(order="C"),
                }
            )
            self.results[LINK_DETAILS] = detail_df
        else:
            self.results[LINK_DETAILS] = None

    def calculate_all_scores(
        self,
        data_type: str = "both",
        tau_count: float = 0.5,
        n_jobs: int | None = None,
        parallel_sections: bool = True,
        remote_chunk_size: int | None = None,
        remote_verbose: int = 0,
    ):
        """Calculate all scores; optionally overlap independent sections.

        Notes
        -----
        - Mathematical definitions are unchanged.
        - `parallel_sections=True` overlaps `Intra-omics Correlation` and
          `Cross-omics Correlation`
          after `Fluctuation` is complete.
        """
        print("==== Starting full FluRiCo score computation ====")
        if n_jobs is None:
            n_jobs = self.n_jobs

        self.calculate_fluctuation(data_type=data_type)

        if parallel_sections:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                fut_coord = executor.submit(self.calculate_coordination, data_type=data_type)
                fut_remote = executor.submit(
                    self.calculate_remote_interaction,
                    data_type=data_type,
                    n_jobs=n_jobs,
                    chunk_size=remote_chunk_size,
                    verbose=remote_verbose,
                )
                _ = fut_coord.result()
                _ = fut_remote.result()
        else:
            self.calculate_coordination(data_type=data_type)
            self.calculate_remote_interaction(
                data_type=data_type,
                n_jobs=n_jobs,
                chunk_size=remote_chunk_size,
                verbose=remote_verbose,
            )

        self.calculate_remote_trend_concordance(tau_count=tau_count, n_jobs=n_jobs)
        print("==== Full FluRiCo score computation completed ====")
        return self.results
