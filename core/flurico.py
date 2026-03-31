import concurrent.futures
import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr, levene, mannwhitneyu
from tqdm import tqdm
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from .score_names import (
    CHANGE_EFFECT_DETAIL,
    CHANGE_EFFECT_SCORES,
    CROSS_OMICS_CORRELATION,
    INTRA_OMICS_CORRELATION,
    LINK_DETAILS,
    NEGATIVE_LINKS,
    POSITIVE_LINKS,
    VARIANCE_HETEROGENEITY,
    VARIANCE_HETEROGENEITY_STATS,
    build_change_effect_score_name,
)


class FluRiCoAnalysis:
    """
    FluRiCo analysis class for three major dimensions and their sub-concepts.
    All scores remain independent and are not combined into a single total score.

    - Fluctuation
      * `Variance Heterogeneity`
      * Change Effect scores, stored directly as `<group1> vs. <group2> Change Effect`

    - Intra-omics Correlation
      * `Intra-omics Correlation`

    - Remote
      * `Cross-omics Correlation`
      * `Positive Links`
      * `Negative Links`
    """

    def __init__(self, microbe_data=None, metabolite_data=None,
                 group_labels=None, microbe_coab_data=None,n_jobs=-1):

        self.microbe_data = microbe_data
        self.metabolite_data = metabolite_data
        self.group_labels = group_labels or []
        self.microbe_coab_data = microbe_coab_data
        self.n_jobs = n_jobs
        self._validate_data()

        # Determine Change Effect pairs dynamically
        self.fc_pairs = self._determine_fc_pairs()

        self.results = {
            'microbe': {
                # Fluctuation
                VARIANCE_HETEROGENEITY: None,
                VARIANCE_HETEROGENEITY_STATS: None,
                CHANGE_EFFECT_SCORES: {},
                CHANGE_EFFECT_DETAIL: None,

                # Intra-omics Correlation
                INTRA_OMICS_CORRELATION: {},

                # Cross-omics Correlation
                CROSS_OMICS_CORRELATION: {},

                # Remote Trend: Positive Links / Negative Links
                POSITIVE_LINKS: None,
                NEGATIVE_LINKS: None,
            },
            'metabolite': {
                VARIANCE_HETEROGENEITY: None,
                VARIANCE_HETEROGENEITY_STATS: None,
                CHANGE_EFFECT_SCORES: {},
                CHANGE_EFFECT_DETAIL: None,
                INTRA_OMICS_CORRELATION: {},
                CROSS_OMICS_CORRELATION: {},
                POSITIVE_LINKS: None,
                NEGATIVE_LINKS: None,
            },
            # Detailed table of all microbe-metabolite pairs for Positive/Negative Links
            LINK_DETAILS: None,
        }

    # ===================== Basic validation =====================

    def _validate_data(self):
        if self.microbe_data is None or self.metabolite_data is None:
            print("Warning: microbe_data or metabolite_data is missing; later computations will fail.")
            return

        # Derive actual labels from the data
        if not self.group_labels:
            mb_labels = set(self.microbe_data.index.unique())
            met_labels = set(self.metabolite_data.index.unique())
            self.group_labels = sorted(list(mb_labels & met_labels))
            print(f"Automatically detected labels: {self.group_labels}")

        for label in self.group_labels:
            if not (self.microbe_data.index == label).any():
                print(f"Warning: group label '{label}' was not found in the microbe data index")
            if not (self.metabolite_data.index == label).any():
                print(f"Warning: group label '{label}' was not found in the metabolite data index")

        if self.microbe_coab_data is not None:
            if not isinstance(self.microbe_coab_data, dict):
                print("Error: microbe_coab_data must be a dict mapping group names to co-abundance DataFrames")
            else:
                missing = set(self.group_labels) - set(self.microbe_coab_data.keys())
                if missing:
                    print(f"Warning: microbe_coab_data is missing co-abundance matrices for: {missing}")

    def _determine_fc_pairs(self):
        """Determine the Change Effect pairs to compute from the number of labels."""
        if len(self.group_labels) < 2:
            return []
        
        pairs = []
        # Adjacent pairs
        for i in range(len(self.group_labels) - 1):
            g1, g2 = self.group_labels[i], self.group_labels[i + 1]
            pairs.append((g1, g2, build_change_effect_score_name(g1, g2)))
        
        # First/last pair when there are at least three labels
        if len(self.group_labels) >= 3:
            g1, g2 = self.group_labels[0], self.group_labels[-1]
            pairs.append((g1, g2, build_change_effect_score_name(g1, g2)))
        
        return pairs

    def load_data(self, microbe_data, metabolite_data, microbe_coab_data=None, group_labels=None):
        self.microbe_data = microbe_data
        self.metabolite_data = metabolite_data
        self.microbe_coab_data = microbe_coab_data
        if group_labels is not None:
            self.group_labels = group_labels
        self._validate_data()
        self.fc_pairs = self._determine_fc_pairs()

    # ===================== Fluctuation =====================

    def calculate_fluctuation(self, data_type='both'):
        print("Computing Fluctuation...")

        if data_type in ['microbe', 'both'] and self.microbe_data is not None:
            print("  [microbe] Computing Variance Heterogeneity and Change Effect scores...")
            lev_score, lev_pvals = self._calculate_levene_score(self.microbe_data)
            fc_dict, fc_detail = self._calculate_fold_change_score(self.microbe_data)

            self.results['microbe'][VARIANCE_HETEROGENEITY] = lev_score
            self.results['microbe'][VARIANCE_HETEROGENEITY_STATS] = lev_pvals
            self.results['microbe'][CHANGE_EFFECT_SCORES] = fc_dict
            self.results['microbe'][CHANGE_EFFECT_DETAIL] = fc_detail

        if data_type in ['metabolite', 'both'] and self.metabolite_data is not None:
            print("  [metabolite] Computing Variance Heterogeneity and Change Effect scores...")
            lev_score, lev_pvals = self._calculate_levene_score(self.metabolite_data)
            fc_dict, fc_detail = self._calculate_fold_change_score(self.metabolite_data)

            self.results['metabolite'][VARIANCE_HETEROGENEITY] = lev_score
            self.results['metabolite'][VARIANCE_HETEROGENEITY_STATS] = lev_pvals
            self.results['metabolite'][CHANGE_EFFECT_SCORES] = fc_dict
            self.results['metabolite'][CHANGE_EFFECT_DETAIL] = fc_detail

        if data_type == 'microbe':
            return {
                VARIANCE_HETEROGENEITY: self.results['microbe'][VARIANCE_HETEROGENEITY],
                VARIANCE_HETEROGENEITY_STATS: self.results['microbe'][VARIANCE_HETEROGENEITY_STATS],
                CHANGE_EFFECT_SCORES: self.results['microbe'][CHANGE_EFFECT_SCORES],
                CHANGE_EFFECT_DETAIL: self.results['microbe'][CHANGE_EFFECT_DETAIL],
            }
        elif data_type == 'metabolite':
            return {
                VARIANCE_HETEROGENEITY: self.results['metabolite'][VARIANCE_HETEROGENEITY],
                VARIANCE_HETEROGENEITY_STATS: self.results['metabolite'][VARIANCE_HETEROGENEITY_STATS],
                CHANGE_EFFECT_SCORES: self.results['metabolite'][CHANGE_EFFECT_SCORES],
                CHANGE_EFFECT_DETAIL: self.results['metabolite'][CHANGE_EFFECT_DETAIL],
            }
        else:
            return self.results

    def _calculate_levene_score(self, df):
        dv_score = {}
        p_values = []
        stats = []
        feature_names = []
        error_list = []

        for feature in tqdm(df.columns, desc="  Levene test (multi-group)"):
            group_vals = []
            for g in self.group_labels:
                mask = (df.index == g)
                if mask.sum() > 1:
                    group_data = df.loc[mask, feature]
                    if group_data.nunique() > 1:
                        group_vals.append(group_data.values)

            if len(group_vals) >= 2:
                try:
                    stat, p = levene(*group_vals, center='median')
                    dv_score[feature] = stat * (-np.log10(p)) if p > 0 else stat * 50
                    p_values.append(p)
                    stats.append(stat)
                    feature_names.append(feature)
                except Exception as e:
                    print(f"    Levene test failed for feature {feature}: {e}")
                    dv_score[feature] = np.nan
                    p_values.append(np.nan)
                    stats.append(np.nan)
                    feature_names.append(feature)
                    error_list.append(feature)
            else:
                dv_score[feature] = np.nan
                p_values.append(np.nan)
                stats.append(np.nan)
                feature_names.append(feature)
                error_list.append(feature)

        if error_list:
            print(
                f"  Variance Heterogeneity: {len(error_list)} features did not have enough samples "
                "or variability for testing"
            )

        p_values = np.array(p_values, dtype=float)
        stats = np.array(stats, dtype=float)
        non_nan = ~np.isnan(p_values)
        corrected_p = np.full_like(p_values, np.nan, dtype=float)

        if non_nan.sum() > 0:
            _, p_corr, _, _ = multipletests(p_values[non_nan], method='fdr_bh')
            corrected_p[non_nan] = p_corr

        corrected_dv_score = {}
        fdr_results = {}
        for i, feature in enumerate(feature_names):
            if np.isnan(corrected_p[i]) or np.isnan(stats[i]):
                corrected_dv_score[feature] = 0.0
                fdr_results[feature] = {
                    'original_p': np.nan,
                    'fdr_p': np.nan,
                    'levene_stat': np.nan
                }
            else:
                p_fdr_clip = max(corrected_p[i], 1e-16)
                corrected_dv_score[feature] = stats[i] * (-np.log10(p_fdr_clip))
                fdr_results[feature] = {
                    'original_p': p_values[i],
                    'fdr_p': corrected_p[i],
                    'levene_stat': stats[i]
                }

        score_series = pd.Series(corrected_dv_score).fillna(0.0)
        fdr_df = pd.DataFrame(fdr_results).T
        return score_series, fdr_df

    def _calculate_fold_change_score(self, df, eps=1e-9):
        """
        Compute all required Change Effect scores based on the current label set.
        Returns: fc_dict (dict[pair_name -> Series]), fc_detail (DataFrame)
        """
        records = []
        features = list(df.columns)

        # === 1. Compute Change Effect statistics for all pairs ===
        for i in range(len(self.group_labels)):
            for j in range(i + 1, len(self.group_labels)):
                g1, g2 = self.group_labels[i], self.group_labels[j]
                mask1 = (df.index == g1)
                mask2 = (df.index == g2)
                if mask1.sum() == 0 or mask2.sum() == 0:
                    continue

                for feat in features:
                    x1 = df.loc[mask1, feat].values
                    x2 = df.loc[mask2, feat].values
                    if x1.size == 0 or x2.size == 0:
                        continue

                    mean1 = float(np.mean(x1))
                    mean2 = float(np.mean(x2))

                    fc = (mean2 + eps) / (mean1 + eps)
                    log2fc = np.log2(fc)

                    try:
                        if np.all(x1 == x2):
                            p_raw = 1.0
                        else:
                            _, p_raw = mannwhitneyu(x1, x2, alternative='two-sided')
                    except Exception:
                        p_raw = np.nan

                    records.append({
                        'feature': feat,
                        'group1': g1,
                        'group2': g2,
                        f'mean_{g1}': mean1,
                        f'mean_{g2}': mean2,
                        'log2FC': log2fc,
                        'p_raw': p_raw
                    })

        if not records:
            return {}, pd.DataFrame()

        fc_df = pd.DataFrame(records)

        # === 2. FDR correction ===
        pvals = fc_df['p_raw'].values.astype(float)
        mask = ~np.isnan(pvals)
        fc_df['p_fdr'] = np.nan
        if mask.sum() > 0:
            _, p_corr, _, _ = multipletests(pvals[mask], method='fdr_bh')
            fc_df.loc[mask, 'p_fdr'] = p_corr

        # === 3. Extract scores for the predefined pairs ===
        fc_dict = {}
        for g1, g2, pair_name in self.fc_pairs:
            fc_scores = {feat: 0.0 for feat in features}
            pair_set = frozenset([g1, g2])
            
            for _, row in fc_df.iterrows():
                feat = row['feature']
                curr_pair = frozenset([row['group1'], row['group2']])
                if curr_pair != pair_set:
                    continue
                
                log2fc = float(row['log2FC'])
                p_fdr = row['p_fdr']
                
                if np.isnan(p_fdr):
                    continue
                
                p_eff = max(float(p_fdr), 1e-300)
                score = abs(log2fc) * (-np.log10(p_eff))
                fc_scores[feat] = score
            
            fc_dict[pair_name] = pd.Series(fc_scores).reindex(features).fillna(0.0)

        return fc_dict, fc_df

    # =============== Trend vector construction (for Positive/Negative Links) ===============

    def _build_trend_vectors_from_fc_detail(self, fc_df):
        """
        Build a trend vector for each feature from Change Effect Detail.
        - If there are n labels, the vector dimension is len(self.fc_pairs)
        - Each dimension is the signed value:
          log2FC * (-log10(FDR)) for the corresponding pair
        """
        if fc_df is None or fc_df.empty:
            return {}

        n_dims = len(self.fc_pairs)
        trend_dict = {}
        for feat in fc_df['feature'].unique():
            trend_dict[feat] = np.zeros(n_dims, dtype=float)

        for idx, (g1, g2, pair_name) in enumerate(self.fc_pairs):
            pair_set = frozenset([g1, g2])
            
            for _, row in fc_df.iterrows():
                feat = row['feature']
                curr_pair = frozenset([row['group1'], row['group2']])
                if curr_pair != pair_set:
                    continue
                
                log2fc = row['log2FC']
                p_fdr = row['p_fdr']
                
                if pd.isna(log2fc) or pd.isna(p_fdr):
                    continue
                
                pf = float(p_fdr)
                pf = max(min(pf, 1.0), 1e-16)
                val = float(log2fc) * (-np.log10(pf))
                
                # Keep the larger-magnitude value
                current = trend_dict[feat][idx]
                if abs(val) > abs(current):
                    trend_dict[feat][idx] = val

        return trend_dict

    # ===================== Intra-omics Correlation =====================

    def calculate_coordination(self, data_type='both'):
        print(f"Computing {INTRA_OMICS_CORRELATION}...")

        for label in self.group_labels:
            # Microbe coordination -> Intra-omics Correlation from the precomputed co-abundance matrix
            if data_type in ['microbe', 'both'] and self.microbe_coab_data is not None:
                if label not in self.microbe_coab_data:
                    print(f"  [microbe] Group {label} is missing from microbe_coab_data; skipping.")
                else:
                    print(f"  [microbe] Group {label}: computing {INTRA_OMICS_CORRELATION} from the co-abundance matrix...")
                    mb_data = self.microbe_coab_data[label]
                    mb_corr = mb_data.abs()
                    np.fill_diagonal(mb_corr.values, np.nan)
                    mb_coord = mb_corr.mean(axis=1, skipna=True)
                    self.results['microbe'][INTRA_OMICS_CORRELATION][label] = mb_coord

            # Metabolite coordination -> Intra-omics Correlation from within-group Spearman correlations
            if data_type in ['metabolite', 'both'] and self.metabolite_data is not None:
                mask = (self.metabolite_data.index == label)
                if mask.sum() < 2:
                    print(f"  [metabolite] Group {label} has fewer than 2 samples; cannot compute correlations.")
                else:
                    print(f"  [metabolite] Group {label}: computing metabolite self-correlation for {INTRA_OMICS_CORRELATION}...")
                    met_data = self.metabolite_data.loc[mask]
                    met_corr = met_data.corr(method='spearman').abs()
                    np.fill_diagonal(met_corr.values, np.nan)
                    met_coord = met_corr.mean(axis=1, skipna=True)
                    self.results['metabolite'][INTRA_OMICS_CORRELATION][label] = met_coord

        if data_type == 'microbe':
            return self.results['microbe'][INTRA_OMICS_CORRELATION]
        elif data_type == 'metabolite':
            return self.results['metabolite'][INTRA_OMICS_CORRELATION]
        else:
            return {
                'microbe': self.results['microbe'][INTRA_OMICS_CORRELATION],
                'metabolite': self.results['metabolite'][INTRA_OMICS_CORRELATION],
            }

    # ===================== Remote Interaction =====================

    def calculate_remote_interaction(self, data_type='both', n_jobs=None):
        """
        Cross-omics Correlation score.
        """
        print(f"Computing {CROSS_OMICS_CORRELATION}...")
        # Use the instance default when not specified
        if n_jobs is None:
            n_jobs = self.n_jobs
        for label in self.group_labels:
            mb_data = self.microbe_data.loc[self.microbe_data.index == label]
            met_data = self.metabolite_data.loc[self.metabolite_data.index == label]

            if mb_data.shape[0] == 0 or met_data.shape[0] == 0:
                print(f"  Group {label}: microbe or metabolite sample count is 0; skipping {CROSS_OMICS_CORRELATION}.")
                continue

            if data_type in ['microbe', 'both']:
                print(f"  [microbe] Group {label}: microbe -> metabolite {CROSS_OMICS_CORRELATION}...")
                mb_remote = Parallel(n_jobs=n_jobs)(
                    delayed(self._calculate_wmc_spearman_fdr)(
                        mb_data[microbe], met_data
                    )
                    for microbe in tqdm(mb_data.columns, desc=f"{label} - microbe")
                )
                self.results['microbe'][CROSS_OMICS_CORRELATION][label] = pd.Series(
                    mb_remote, index=mb_data.columns
                )

            if data_type in ['metabolite', 'both']:
                print(f"  [metabolite] Group {label}: metabolite -> microbe {CROSS_OMICS_CORRELATION}...")
                met_remote = Parallel(n_jobs=n_jobs)(
                    delayed(self._calculate_wmc_spearman_fdr)(
                        met_data[metabolite], mb_data
                    )
                    for metabolite in tqdm(met_data.columns, desc=f"{label} - metabolite")
                )
                self.results['metabolite'][CROSS_OMICS_CORRELATION][label] = pd.Series(
                    met_remote, index=met_data.columns
                )

        if data_type == 'microbe':
            return self.results['microbe'][CROSS_OMICS_CORRELATION]
        elif data_type == 'metabolite':
            return self.results['metabolite'][CROSS_OMICS_CORRELATION]
        else:
            return {
                'microbe': self.results['microbe'][CROSS_OMICS_CORRELATION],
                'metabolite': self.results['metabolite'][CROSS_OMICS_CORRELATION],
            }

    def _calculate_wmc_spearman_fdr(self, feature_series, df_target):
        """
        Compute Spearman correlations between one feature and all columns in
        df_target, and return a continuous score that combines effect size
        (|rho|) and significance (-log10 p_fdr).
        """
        corrs = []
        x_all = feature_series.values.astype(float)

        for target in df_target.columns:
            y_all = df_target[target].values.astype(float)

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
            _, p_corr, _, _ = multipletests(pvals[non_nan], method='fdr_bh')
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

    # =============== Remote Trend (Positive Links / Negative Links) ===============

    def calculate_remote_trend_concordance(self, tau_count=0.5, n_jobs=None):
        """
        Compute Remote Trend from FC vectors and keep only Positive Links
        and Negative Links counts.

        - If the number of labels is >= 3: use cosine similarity
        - If the number of labels is 2: use sign matching + top 20%
          effect-size product

        Parameters
        ----------
        tau_count : float
            Cosine similarity threshold used only when the number of labels
            is >= 3
        """
        print("Computing Remote Trend...")
        # Use the instance default when not specified
        if n_jobs is None:
            n_jobs = self.n_jobs
        # 1. Ensure Change Effect Detail is available
        print("  [step 1/4] Checking whether Change Effect Detail is available...")
        if (
            self.results['microbe'][CHANGE_EFFECT_DETAIL] is None
            or self.results['metabolite'][CHANGE_EFFECT_DETAIL] is None
        ):
            print("  Remote Trend depends on Change Effect Detail; computing Fluctuation first...")
            self.calculate_fluctuation(data_type='both')

        fc_mb = self.results['microbe'][CHANGE_EFFECT_DETAIL]
        fc_met = self.results['metabolite'][CHANGE_EFFECT_DETAIL]

        if fc_mb is None or fc_mb.empty or fc_met is None or fc_met.empty:
            print("  Warning: FC_Detail is empty; Remote Trend cannot be computed, so all outputs are set to 0.")
            self._set_remote_trend_to_zero()
            return

        mb_features = list(self.microbe_data.columns)
        met_features = list(self.metabolite_data.columns)

        if len(mb_features) == 0 or len(met_features) == 0:
            print("  Microbe or metabolite feature lists are empty; Remote Trend cannot be computed.")
            self._set_remote_trend_to_zero()
            return

        # 2. Select the method based on the number of labels
        n_labels = len(self.group_labels)
        
        if n_labels >= 3:
            # Use the cosine-similarity method
            print(f"  Detected {n_labels} labels; using the cosine-similarity method...")
            self._calculate_trend_cosine_method(
                fc_mb, fc_met, mb_features, met_features, tau_count
            )
        elif n_labels == 2:
            # Use the sign-matching + top-20% method
            print("  Detected 2 labels; using the sign-matching + Top 20% method...")
            self._calculate_trend_two_label_method(
                fc_mb, fc_met, mb_features, met_features
            )
        else:
            print(f"  Not enough labels ({n_labels}) to compute Remote Trend.")
            self._set_remote_trend_to_zero()
            return

        print("  Remote Trend computation completed.")

    def _calculate_trend_cosine_method(self, fc_mb, fc_met, mb_features, met_features, tau_count):
        """Multi-label case: use cosine similarity for Positive/Negative Links."""
        print("  [step 2/4] Building trend vectors...")
        mb_trend_dict = self._build_trend_vectors_from_fc_detail(fc_mb)
        met_trend_dict = self._build_trend_vectors_from_fc_detail(fc_met)

        mb_vecs = np.vstack([mb_trend_dict.get(f, np.zeros(len(self.fc_pairs), dtype=float)) 
                            for f in mb_features])
        met_vecs = np.vstack([met_trend_dict.get(f, np.zeros(len(self.fc_pairs), dtype=float)) 
                             for f in met_features])

        print("  [step 3/4] Computing the cosine similarity matrix...")
        mb_norms = np.linalg.norm(mb_vecs, axis=1)
        met_norms = np.linalg.norm(met_vecs, axis=1)

        mb_norms_safe = np.where(mb_norms > 0, mb_norms, 1.0)
        met_norms_safe = np.where(met_norms > 0, met_norms, 1.0)

        mb_unit = mb_vecs / mb_norms_safe[:, None]
        met_unit = met_vecs / met_norms_safe[:, None]

        cos_matrix = mb_unit.dot(met_unit.T)
        cos_matrix = np.clip(cos_matrix, -1.0, 1.0)

        M, K = cos_matrix.shape
        print(f"    Cosine matrix shape: {M} microbes x {K} metabolites")

        print("  [step 4/4] Counting Positive Links and Negative Links...")
        # Microbe-centered Positive/Negative Links
        mb_pos_count = np.sum(cos_matrix >= tau_count, axis=1)
        mb_neg_count = np.sum(cos_matrix <= -tau_count, axis=1)

        # Metabolite-centered Positive/Negative Links
        met_pos_count = np.sum(cos_matrix >= tau_count, axis=0)
        met_neg_count = np.sum(cos_matrix <= -tau_count, axis=0)

        # Store outputs
        self.results['microbe'][POSITIVE_LINKS] = pd.Series(mb_pos_count, index=mb_features)
        self.results['microbe'][NEGATIVE_LINKS] = pd.Series(mb_neg_count, index=mb_features)
        self.results['metabolite'][POSITIVE_LINKS] = pd.Series(met_pos_count, index=met_features)
        self.results['metabolite'][NEGATIVE_LINKS] = pd.Series(met_neg_count, index=met_features)

        # Pair-level detail table used to derive Positive/Negative Links
        records = []
        for i, m in enumerate(mb_features):
            for j, met in enumerate(met_features):
                records.append({
                    'microbe': m,
                    'metabolite': met,
                    'cosine_trend': float(cos_matrix[i, j])
                })
        self.results[LINK_DETAILS] = pd.DataFrame(records)

    def _calculate_trend_two_label_method(self, fc_mb, fc_met, mb_features, met_features):
        """Two-label case: use sign matching + top-20% effect-size product for Positive/Negative Links."""
        print("  [step 2/4] Extracting a single FC value per feature...")
        
        # There is only one pair
        pair_name = self.fc_pairs[0][2]
        
        # Extract the signed log2FC for each feature
        mb_fc_dict = {}
        for feat in mb_features:
            feat_rows = fc_mb[fc_mb['feature'] == feat]
            if len(feat_rows) > 0:
                mb_fc_dict[feat] = float(feat_rows.iloc[0]['log2FC'])
            else:
                mb_fc_dict[feat] = 0.0
        
        met_fc_dict = {}
        for feat in met_features:
            feat_rows = fc_met[fc_met['feature'] == feat]
            if len(feat_rows) > 0:
                met_fc_dict[feat] = float(feat_rows.iloc[0]['log2FC'])
            else:
                met_fc_dict[feat] = 0.0

        print("  [step 3/4] Computing sign matching and effect-size products...")
        
        # Compute signs and products for all pairs
        concordance_pairs = []  # (mb_idx, met_idx, product)
        discordance_pairs = []  # (mb_idx, met_idx, product)
        
        records = []
        for i, mb in enumerate(mb_features):
            mb_fc = mb_fc_dict[mb]
            for j, met in enumerate(met_features):
                met_fc = met_fc_dict[met]
                
                product = abs(mb_fc * met_fc)  # Effect size
                same_sign = (mb_fc >= 0) == (met_fc >= 0)  # Whether the signs match
                
                records.append({
                    'microbe': mb,
                    'metabolite': met,
                    'sign_concordance': 1.0 if same_sign else -1.0
                })
                
                if same_sign:
                    concordance_pairs.append((i, j, product))
                else:
                    discordance_pairs.append((i, j, product))

        print("  [step 4/4] Counting Top 20% occurrences...")
        
        # Compute the 80th percentile separately for positive and negative link candidates
        if concordance_pairs:
            conc_products = [p[2] for p in concordance_pairs]
            conc_threshold = np.percentile(conc_products, 80)
        else:
            conc_threshold = np.inf
        
        if discordance_pairs:
            disc_products = [p[2] for p in discordance_pairs]
            disc_threshold = np.percentile(disc_products, 80)
        else:
            disc_threshold = np.inf

        # Count how often each microbe/metabolite appears among the strongest positive or negative links
        mb_pos_count = np.zeros(len(mb_features), dtype=int)
        mb_neg_count = np.zeros(len(mb_features), dtype=int)
        met_pos_count = np.zeros(len(met_features), dtype=int)
        met_neg_count = np.zeros(len(met_features), dtype=int)

        for i, j, product in concordance_pairs:
            if product >= conc_threshold:
                mb_pos_count[i] += 1
                met_pos_count[j] += 1

        for i, j, product in discordance_pairs:
            if product >= disc_threshold:
                mb_neg_count[i] += 1
                met_neg_count[j] += 1

        # Store outputs
        self.results['microbe'][POSITIVE_LINKS] = pd.Series(mb_pos_count, index=mb_features)
        self.results['microbe'][NEGATIVE_LINKS] = pd.Series(mb_neg_count, index=mb_features)
        self.results['metabolite'][POSITIVE_LINKS] = pd.Series(met_pos_count, index=met_features)
        self.results['metabolite'][NEGATIVE_LINKS] = pd.Series(met_neg_count, index=met_features)
        self.results[LINK_DETAILS] = pd.DataFrame(records)

    def _set_remote_trend_to_zero(self):
        """Set all Positive Links / Negative Links outputs to zero."""
        mb_features = list(self.microbe_data.columns) if self.microbe_data is not None else []
        met_features = list(self.metabolite_data.columns) if self.metabolite_data is not None else []

        zero_mb = pd.Series(0, index=mb_features, dtype=int)
        zero_met = pd.Series(0, index=met_features, dtype=int)

        self.results['microbe'][POSITIVE_LINKS] = zero_mb
        self.results['microbe'][NEGATIVE_LINKS] = zero_mb
        self.results['metabolite'][POSITIVE_LINKS] = zero_met
        self.results['metabolite'][NEGATIVE_LINKS] = zero_met
        self.results[LINK_DETAILS] = pd.DataFrame()

    # ===================== Combined execution =====================

    def calculate_all_scores(self, data_type='both', tau_count=0.5, n_jobs=None):
        print("==== Starting full FluRiCo score computation ====")
        # Use the instance default when not specified
        if n_jobs is None:
            n_jobs = self.n_jobs
        self.calculate_fluctuation(data_type=data_type)
        self.calculate_coordination(data_type=data_type)
        self.calculate_remote_interaction(data_type=data_type, n_jobs=n_jobs)
        self.calculate_remote_trend_concordance(tau_count=tau_count, n_jobs=n_jobs)
        print("==== Full FluRiCo score computation completed ====")
        return self.results

    # ===================== Build normalized summaries =====================

    def _build_summary_table_for_group(self, data_type, label):
        """
        Summary table for one group with all primary scores normalized to 0-1.
        """
        if data_type == 'microbe':
            features = list(self.microbe_data.columns)
        elif data_type == 'metabolite':
            features = list(self.metabolite_data.columns)
        else:
            raise ValueError("data_type must be 'microbe' or 'metabolite'")

        res = self.results[data_type]
        summary = pd.DataFrame(index=features)

        # 1. Variance Heterogeneity
        if res[VARIANCE_HETEROGENEITY] is not None:
            summary[VARIANCE_HETEROGENEITY] = res[VARIANCE_HETEROGENEITY].reindex(features)

        # 2. Dynamic Change Effect columns
        change_effect_scores = res[CHANGE_EFFECT_SCORES]
        for score_name, score_series in change_effect_scores.items():
            summary[score_name] = score_series.reindex(features)

        # 3. Intra-omics Correlation
        coord_dict = res[INTRA_OMICS_CORRELATION]
        if label in coord_dict and coord_dict[label] is not None:
            summary[INTRA_OMICS_CORRELATION] = coord_dict[label].reindex(features)

        # 4. Cross-omics Correlation
        ri_dict = res[CROSS_OMICS_CORRELATION]
        if label in ri_dict and ri_dict[label] is not None:
            summary[CROSS_OMICS_CORRELATION] = ri_dict[label].reindex(features)

        # 5. Positive Links
        if res[POSITIVE_LINKS] is not None:
            summary[POSITIVE_LINKS] = res[POSITIVE_LINKS].reindex(features)

        # 6. Negative Links
        if res[NEGATIVE_LINKS] is not None:
            summary[NEGATIVE_LINKS] = res[NEGATIVE_LINKS].reindex(features)

        # Normalize all columns
        for col in summary.columns:
            col_data = summary[col].astype(float)
            values = col_data.values.copy()
            finite_mask = np.isfinite(values)

            if not finite_mask.any():
                summary[col] = 0.0
                continue

            finite_vals = values[finite_mask]
            v_min = np.nanmin(finite_vals)
            v_max = np.nanmax(finite_vals)

            if np.isfinite(v_min) and np.isfinite(v_max) and v_max > v_min:
                norm_vals = (col_data - v_min) / (v_max - v_min)
                norm_vals = norm_vals.fillna(0.0)
                summary[col] = norm_vals
            else:
                summary[col] = 0.0

        # Sort by Variance Heterogeneity
        if VARIANCE_HETEROGENEITY in summary.columns:
            summary = summary.sort_values(VARIANCE_HETEROGENEITY, ascending=False)

        return summary

    # ===================== Save outputs =====================

    def save_results(self, output_dir="results"):
        """Save all outputs."""
        os.makedirs(output_dir, exist_ok=True)

        for data_type in ['microbe', 'metabolite']:
            res = self.results[data_type]

            # Group-level summary tables
            for label in self.group_labels:
                summary_df = self._build_summary_table_for_group(data_type, label)
                file_path = os.path.join(output_dir, f"{data_type}_summary_{label}.csv")
                summary_df.to_csv(file_path)
                print(f"{data_type}, summary scores for group {label} were saved to {file_path}")

            # Variance Heterogeneity p/FDR table
            if res[VARIANCE_HETEROGENEITY_STATS] is not None:
                p_value_df = res[VARIANCE_HETEROGENEITY_STATS]
                file_path_p = os.path.join(output_dir, f"{data_type}_variance_heterogeneity_p_values.csv")
                p_value_df.to_csv(file_path_p)
                print(f"{data_type} Variance Heterogeneity p/FDR values were saved to {file_path_p}")

            # Change Effect details
            if res[CHANGE_EFFECT_DETAIL] is not None and not res[CHANGE_EFFECT_DETAIL].empty:
                fc_detail_path = os.path.join(output_dir, f"{data_type}_change_effect_detail.csv")
                res[CHANGE_EFFECT_DETAIL].to_csv(fc_detail_path, index=False)
                print(f"{data_type} Change Effect details were saved to {fc_detail_path}")

        # Positive/Negative Links pair details
        if self.results[LINK_DETAILS] is not None:
            detail_path = os.path.join(output_dir, "link_details.csv")
            self.results[LINK_DETAILS].to_csv(detail_path, index=False)
            print(f"Remote Trend microbe-metabolite pair details were saved to {detail_path}")

        print(f"All outputs were saved to {output_dir}")
