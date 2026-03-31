from scipy.spatial.distance import braycurtis, pdist, squareform
from scipy.stats import spearmanr
from scipy.spatial import procrustes
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from skbio.stats.distance import mantel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def prepare_common_data(df_mb, df_mt, structure_mb, structure_mt, cond):
    # Get common microbiome features
    common_microbes = structure_mb[cond]
    common_microbes = [m for m in common_microbes if m in df_mb.columns]
    df_mb_common = df_mb[common_microbes]

    # Get common metabolite features
    common_metabolite = structure_mt[cond]
    common_metabolite = [m for m in common_metabolite if m in df_mt.columns]
    df_mt_common = df_mt[common_metabolite]

    print(f"  Microbiome feature count: {len(common_microbes)}")
    print(f"  Metabolite feature count: {len(common_metabolite)}")
    print(f"  df_mb_common shape: {df_mb_common.shape}")
    print(f"  df_mt_common shape: {df_mt_common.shape}")

    # Ensure aligned sample size
    min_len = min(len(df_mb_common), len(df_mt_common))
    df_mb_common_p = df_mb_common.iloc[:min_len].copy()
    df_mt_common_p = df_mt_common.iloc[:min_len].copy()

    # Re-index
    df_mb_common_p.index = [f"S{i}" for i in range(len(df_mb_common_p))]
    df_mt_common_p.index = df_mb_common_p.index

    # Add group labels
    df_mb_common_p['Group'] = df_mb_common.iloc[:min_len].index.to_list()
    df_mt_common_p['Group'] = df_mt_common.iloc[:min_len].index.to_list()

    return df_mb_common_p, df_mt_common_p, common_microbes, common_metabolite


def standardize_group_data(df_mb_group, df_mt_group):
    scaler_mb = StandardScaler()
    df_mb_group_z = pd.DataFrame(
        scaler_mb.fit_transform(df_mb_group),
        index=df_mb_group.index,
        columns=df_mb_group.columns
    )

    scaler_mt = StandardScaler()
    df_mt_group_z = pd.DataFrame(
        scaler_mt.fit_transform(df_mt_group),
        index=df_mt_group.index,
        columns=df_mt_group.columns
    )

    return df_mb_group_z, df_mt_group_z


def compute_distance_matrices(df_mb_group_z, df_mt_group_z):
    dist_mb = squareform(pdist(df_mb_group_z.values, metric='braycurtis'))
    dist_mt = squareform(pdist(df_mt_group_z.values, metric='braycurtis'))
    return dist_mb, dist_mt


def compute_feature_correlation_and_fdr(df_mb_group_z, df_mt_group_z):
    n_mb = len(df_mb_group_z.columns)
    n_mt = len(df_mt_group_z.columns)

    corr_matrix = np.zeros((n_mb, n_mt))
    pval_matrix = np.zeros((n_mb, n_mt))

    for i, mb in enumerate(df_mb_group_z.columns):
        for j, mt in enumerate(df_mt_group_z.columns):
            x = df_mb_group_z[mb].values.flatten()
            y = df_mt_group_z[mt].values.flatten()

            mask = ~(np.isnan(x) | np.isnan(y))
            x_clean = x[mask]
            y_clean = y[mask]

            if len(x_clean) > 3:
                result = spearmanr(x_clean, y_clean)
                if hasattr(result, 'correlation'):
                    corr_matrix[i, j] = result.correlation
                    pval_matrix[i, j] = result.pvalue
                elif hasattr(result, 'statistic'):
                    corr_matrix[i, j] = result.statistic
                    pval_matrix[i, j] = result.pvalue
                else:
                    corr_matrix[i, j] = result[0]
                    pval_matrix[i, j] = result[1]
            else:
                corr_matrix[i, j] = np.nan
                pval_matrix[i, j] = 1.0

    # FDR correction for internal multiple testing
    pval_flat = pval_matrix.flatten()
    valid_mask = ~np.isnan(pval_flat)

    if valid_mask.sum() > 0:
        pval_valid = pval_flat[valid_mask]
        rejected, pval_corrected_valid, _, _ = multipletests(pval_valid, method='fdr_bh')

        pval_corrected = np.ones_like(pval_flat)
        pval_corrected[valid_mask] = pval_corrected_valid
        pval_fdr_matrix = pval_corrected.reshape(pval_matrix.shape)
    else:
        pval_fdr_matrix = np.ones_like(pval_matrix)

    return corr_matrix, pval_matrix, pval_fdr_matrix


def plot_mantel_heatmap(ax, corr_matrix, pval_fdr_matrix, df_mb_group_z, df_mt_group_z,
                        group, mantel_r, mantel_p, n_samples):
    n_mb = len(df_mb_group_z.columns)
    n_mt = len(df_mt_group_z.columns)

    im = ax.imshow(
        corr_matrix,
        cmap='RdBu_r',
        aspect='auto',
        vmin=-1,
        vmax=1,
        interpolation='nearest'
    )

    # Mark significant pairs
    for i in range(n_mb):
        for j in range(n_mt):
            if not np.isnan(corr_matrix[i, j]):
                if pval_fdr_matrix[i, j] < 0.01:
                    ax.text(
                        j, i + 0.5, '**',
                        ha='center', va='center',
                        fontsize=6, color='black', fontweight='bold',
                        transform=ax.transData
                    )
                elif pval_fdr_matrix[i, j] < 0.05:
                    ax.text(
                        j, i + 0.5, '*',
                        ha='center', va='center',
                        fontsize=6, color='black',
                        transform=ax.transData
                    )

    # Set labels
    ax.set_xlabel("Metabolite Features", fontsize=11, fontweight='bold')
    ax.set_ylabel("Microbiome Features", fontsize=11, fontweight='bold')

    sig_marker = '***' if mantel_p < 0.001 else '**' if mantel_p < 0.01 else '*' if mantel_p < 0.05 else 'ns'
    ax.set_title(
        f"{group} (n={n_samples})\nMantel r = {mantel_r:.3f} ({sig_marker})",
        fontsize=12,
        fontweight='bold'
    )

    # Feature name labels
    if n_mb <= 20:
        ax.set_yticks(range(n_mb))
        ax.set_yticklabels(df_mb_group_z.columns, fontsize=8)
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"Microbiome Features (n={n_mb})", fontsize=11, fontweight='bold')

    if n_mt <= 20:
        ax.set_xticks(range(n_mt))
        ax.set_xticklabels(df_mt_group_z.columns, fontsize=8, rotation=90)
    else:
        ax.set_xticks([])
        ax.set_xlabel(f"Metabolite Features (n={n_mt})", fontsize=11, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Spearman Correlation', fontsize=10)

    return im


def run_mantel_analysis(cond, df_mb_common_p, df_mt_common_p, group_list, results_collector):
    print(f"\n{'─'*60}")
    print("【Mantel Test Analysis】")
    print(f"{'─'*60}")

    fig1, axs1 = plt.subplots(
        1, len(group_list),
        figsize=(6 * len(group_list), 5),
        constrained_layout=True,
        dpi=500
    )

    for idx, group in enumerate(group_list):
        df_mb_group = df_mb_common_p[df_mb_common_p['Group'] == group].drop(columns='Group')
        df_mt_group = df_mt_common_p[df_mt_common_p['Group'] == group].drop(columns='Group')

        print(f"\n{group} group: n={len(df_mb_group)}")

        if len(df_mb_group) < 4:
            print(f"  ⚠️ Insufficient sample size, skipped")
            continue

        # Standardization
        df_mb_group_z, df_mt_group_z = standardize_group_data(df_mb_group, df_mt_group)

        # Distance matrices
        dist_mb, dist_mt = compute_distance_matrices(df_mb_group_z, df_mt_group_z)

        # Mantel test
        mantel_r, mantel_p, _ = mantel(
            dist_mb, dist_mt,
            method='spearman',
            permutations=999,
            alternative='two-sided',
            seed=42
        )

        # Save results without outer FDR correction
        results_collector.append({
            'Condition': cond,
            'Group': group,
            'Method': 'Mantel',
            'Statistic': mantel_r,
            'P_value': mantel_p,
            'Sample_size': len(df_mb_group)
        })

        # Compute feature-feature correlations with internal FDR
        corr_matrix, pval_matrix, pval_fdr_matrix = compute_feature_correlation_and_fdr(
            df_mb_group_z, df_mt_group_z
        )

        # Plot heatmap
        ax = axs1[idx] if len(group_list) > 1 else axs1
        plot_mantel_heatmap(
            ax, corr_matrix, pval_fdr_matrix,
            df_mb_group_z, df_mt_group_z,
            group, mantel_r, mantel_p, len(df_mb_group)
        )

        # Summary statistics
        valid_tests = ~np.isnan(corr_matrix)
        sig_pairs_fdr_005 = ((pval_fdr_matrix < 0.05) & valid_tests).sum()
        sig_pairs_fdr_001 = ((pval_fdr_matrix < 0.01) & valid_tests).sum()
        total_pairs = valid_tests.sum()

        print(f"  Total tests: {total_pairs}")
        print(f"  FDR<0.05: {sig_pairs_fdr_005} ({100 * sig_pairs_fdr_005 / total_pairs:.1f}%)")
        print(f"  FDR<0.01: {sig_pairs_fdr_001} ({100 * sig_pairs_fdr_001 / total_pairs:.1f}%)")

    plt.suptitle(f'Mantel Test - {cond} Structure Features', fontsize=14, fontweight='bold', y=1.03)
    plt.show()


def run_procrustes_analysis(cond, df_mb_common_p, df_mt_common_p, group_list, results_collector):
    print(f"\n{'─'*60}")
    print("【Procrustes Analysis】")
    print(f"{'─'*60}")

    fig2, axs2 = plt.subplots(
        1, len(group_list),
        figsize=(6 * len(group_list), 6),
        constrained_layout=True,
        dpi=500
    )

    for idx, group in enumerate(group_list):
        df_mb_group = df_mb_common_p[df_mb_common_p['Group'] == group].drop(columns='Group')
        df_mt_group = df_mt_common_p[df_mt_common_p['Group'] == group].drop(columns='Group')

        if len(df_mb_group) < 4:
            continue

        # Standardization
        df_mb_group_z, df_mt_group_z = standardize_group_data(df_mb_group, df_mt_group)

        # Distance matrices + MDS
        dist_mb, dist_mt = compute_distance_matrices(df_mb_group_z, df_mt_group_z)

        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords_mb = mds.fit_transform(dist_mb)
        coords_mt = mds.fit_transform(dist_mt)

        # Procrustes alignment + permutation test
        aligned_mb, aligned_mt, m2 = procrustes(coords_mb, coords_mt)

        n_perm = 999
        rng = np.random.default_rng(42)
        perm_m2 = np.zeros(n_perm)
        for i in range(n_perm):
            perm_idx = rng.permutation(aligned_mt.shape[0])
            perm_m2[i] = procrustes(coords_mb, coords_mt[perm_idx])[2]

        p_val = (np.sum(perm_m2 <= m2) + 1) / (n_perm + 1)

        # Save results without outer FDR correction
        results_collector.append({
            'Condition': cond,
            'Group': group,
            'Method': 'Procrustes',
            'Statistic': m2,
            'P_value': p_val,
            'Sample_size': len(df_mb_group)
        })

        print(f"\n{group} group: m²={m2:.3f}, p={p_val:.3f}")

        # Plot
        ax = axs2[idx] if len(group_list) > 1 else axs2

        for i in range(aligned_mb.shape[0]):
            ax.plot(
                [aligned_mb[i, 0], aligned_mt[i, 0]],
                [aligned_mb[i, 1], aligned_mt[i, 1]],
                color='gray', alpha=0.5, linewidth=0.8
            )

        ax.scatter(
            aligned_mb[:, 0], aligned_mb[:, 1],
            c='skyblue', label='Microbiome',
            marker='o', s=60, edgecolors='navy', linewidths=0.5
        )
        ax.scatter(
            aligned_mt[:, 0], aligned_mt[:, 1],
            c='lightcoral', label='Metabolite',
            marker='x', s=60, linewidths=1.5
        )

        sig_marker = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax.set_title(
            f"{group} (n={len(df_mb_group)})\nm² = {m2:.3f} ({sig_marker})",
            fontsize=12,
            fontweight='bold'
        )
        ax.set_xlabel("Dimension 1", fontsize=11)
        ax.set_ylabel("Dimension 2", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(f'Procrustes Analysis - {cond} Structure Features', fontsize=14, fontweight='bold', y=1.03)
    plt.show()


def summarize_cross_connection_results(results_collector):
    print(f"\n{'='*80}")
    print("【Summary Results】")
    print(f"{'='*80}\n")

    results_df = pd.DataFrame(results_collector)

    # Add significance labels
    results_df['Significant'] = results_df['P_value'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    )

    # Sort by condition and method
    results_df = results_df.sort_values(
        ['Condition', 'Method', 'Group']
    ).reset_index(drop=True)

    print(results_df.to_string(index=False))
    return results_df

