import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.patches import Ellipse
from scipy.stats import chi2, gaussian_kde
import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
import seaborn as sns
from umap import UMAP

def select_reducer(method, n_components, random_state, labels=None):
    method = method.upper()
    if method == 'PCA':
        return PCA(n_components=n_components, random_state=random_state)
    elif method == 'LDA':
        if labels is None:
            raise ValueError("LDA requires labels.")
        return LinearDiscriminantAnalysis(n_components=n_components)
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'PCA' or 'LDA'.")


def plot_confidence_ellipse(X, ax, color, confidence_level=0.95):
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.arctan2(*eigvecs[:, 0][::-1])
    scale = np.sqrt(chi2.ppf(confidence_level, df=2))
    width, height = 2 * scale * np.sqrt(eigvals)

    ellipse_fill = Ellipse(
        xy=mean, width=width, height=height, angle=np.degrees(angle),
        facecolor=color, edgecolor='none', alpha=0.08
    )
    ax.add_patch(ellipse_fill)

    ellipse_border = Ellipse(
        xy=mean, width=width, height=height, angle=np.degrees(angle),
        facecolor='none', edgecolor=color, linewidth=2, alpha=1.0
    )
    ax.add_patch(ellipse_border)


def plot_kde_density(data, ax, orientation='horizontal', color='blue', scale=0.1, alpha=0.08, fill=True):
    if len(data) <= 1:
        return
    kde = gaussian_kde(data)
    if orientation == 'horizontal':
        x_range = np.linspace(data.min() - 0.5 * data.std(), data.max() + 0.5 * data.std(), 200)
        density = kde(x_range) * scale
        ax.fill_between(x_range, 0, density, alpha=alpha, color=color) if fill else ax.plot(x_range, density, color=color, linewidth=2, alpha=1, linestyle='--')
    else:
        y_range = np.linspace(data.min() - 0.5 * data.std(), data.max() + 0.5 * data.std(), 200)
        density = kde(y_range) * scale
        ax.fill_betweenx(y_range, 0, density, alpha=alpha, color=color) if fill else ax.plot(density, y_range, color=color, linewidth=2, alpha=1, linestyle='--')


def get_feature_list(group_name, data_type, common_features, mode=None):
    """
    Retrieve a feature list from `common_features` for a given group and data type.

    Parameters
    ----------
    group_name : str
        One of {'HC','MIA','IA'}.
    data_type : str
        One of {'microbe','metabolite'}.
    common_features : dict
        Nested dict containing keys 'common_to_all', 'unique_to_each', and 'pairwise_common'.
    mode : str or None
        One of {'common_to_all','unique_to_each','pairwise_common'}.

    Returns
    -------
    list
        Combined feature list.
    """
    pairwise_keys = {
        'HC': ['HC_and_MIA', 'HC_and_IA'],
        'MIA': ['HC_and_MIA', 'MIA_and_IA'],
        'IA': ['HC_and_IA', 'MIA_and_IA'],
    }

    feature_list = []
    if mode == 'pairwise_common':
        for key in pairwise_keys.get(group_name, []):
            feature_list += list(common_features[data_type]['pairwise_common'].get(key, []))
    elif mode == 'unique_to_each':
        feature_list += list(common_features[data_type]['unique_to_each'].get(group_name, []))
    elif mode == 'common_to_all':
        feature_list += list(common_features[data_type].get('common_to_all', []))
    return feature_list


def _build_feature_view_table(df_data, data_type, common_features, groups=('HC','MIA','IA'), mode='unique_to_each'):
    """
    Construct the feature-view table:
    rows are features (core per group), columns are samples, index set to group labels.
    """
    vis_df = pd.DataFrame()
    for cond in groups:
        feats = get_feature_list(cond, data_type, common_features, mode=mode)
        feats = [f for f in feats if f in df_data.columns]
        if len(feats) == 0:
            continue
        df_common = df_data[feats]
        df_tmp = df_common.T
        df_tmp['label'] = [cond] * len(df_tmp)
        vis_df = pd.concat([vis_df, df_tmp], axis=0)
    if 'label' not in vis_df.columns or vis_df.empty:
        raise ValueError("No features found after filtering; check inputs or mode.")
    vis_df = vis_df.set_index('label')
    return vis_df


def visualize_common_features_samples(
    df_query,
    common_features=None,
    feature_type='microbe',
    group=('HC', 'MIA', 'IA'),
    method='PCA',
    random_state=42,
    colors=None,
    confidence_level=0.95,
    use_all_features=False,
    save_path=None
):
    """
    Plot sample scores using shared (or all) features with PCA or LDA.

    Parameters
    ----------
    df_query : pd.DataFrame
        Feature table with samples as rows indexed by group labels (e.g., 'HC', 'MIA', 'IA').
    common_features : dict or iterable or None
        If dict, expects keys like common_features[feature_type]['common_to_all'].
        If iterable, treated as a list of feature names directly.
        If None and use_all_features=False, raises an error.
    feature_type : str
        One of {'microbe','metabolite'} or a key present in `common_features` when dict.
    group : tuple or list
        Ordered group labels to plot.
    method : str
        'PCA' or 'LDA'.
    random_state : int
        Random state for reproducibility (PCA).
    colors : list or None
        Optional list of colors aligned with `group`.
    confidence_level : float
        Confidence level for the ellipse.
    use_all_features : bool
        If True, use all columns of df_query; otherwise use features from `common_features`.
    save_path : str or None
        If provided, save the figure; otherwise show the figure.
    """
    method = method.upper()

    if use_all_features:
        df_common = df_query.copy()
    else:
        try:
            feats = list(common_features[feature_type]['common_to_all'])
        except (KeyError, TypeError):
            if common_features is None:
                raise ValueError("common_features is required when use_all_features=False.")
            feats = list(common_features)
        df_common = df_query[feats].copy()

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_common)
    if feature_type == 'microbe':
        data_scaled = df_common

    reducer = select_reducer(method, 2, random_state, labels=df_common.index if method == 'LDA' else None)

    if method == 'LDA':
        labels = df_common.index.to_numpy()
        reduced = reducer.fit_transform(data_scaled, labels)
        xlabel, ylabel = "LD1", "LD2"
    else:
        reduced = reducer.fit_transform(data_scaled)
        var_ratio = reducer.explained_variance_ratio_
        xlabel = f"PC1 ({var_ratio[0]*100:.1f}%)"
        ylabel = f"PC2 ({var_ratio[1]*100:.1f}%)"

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.6, 4], width_ratios=[4, 0.6], hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    ax_main.set_facecolor('white')
    ax_main.grid(False, linestyle='--', color='gray', alpha=0.5)

    for idx, g in enumerate(group):
        indices = [i for i, label in enumerate(df_common.index) if label == g]
        if indices:
            color = colors[idx] if colors else plt.cm.tab10(idx)
            ax_main.scatter(reduced[indices, 0], reduced[indices, 1], label=g, color=color, alpha=0.9, s=80)
            plot_confidence_ellipse(reduced[indices], ax_main, color=color, confidence_level=confidence_level)
            if len(indices) > 1:
                plot_kde_density(reduced[indices, 0], ax_top, orientation='horizontal', color=color, alpha=0.25)
                plot_kde_density(reduced[indices, 1], ax_right, orientation='vertical', color=color, alpha=0.25)

    ax_main.set_xlabel(xlabel, fontsize=15)
    ax_main.set_ylabel(ylabel, fontsize=15)
    if method == 'LDA':
        ax_main.set_title("LDA", fontsize=20)
    ax_main.legend(loc='lower right')

    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.tick_params(axis='y', labelleft=False)
    ax_top.set_facecolor('white')
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(True)
    ax_top.spines['bottom'].set_visible(True)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.grid(False)

    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.tick_params(axis='x', labelbottom=False)
    ax_right.set_facecolor('white')
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(True)
    ax_right.spines['bottom'].set_visible(True)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.grid(False)

    if method in ['PCA', 'LDA']:
        scores_df = pd.DataFrame(reduced[:, :2], columns=["Dim1", "Dim2"])
        scores_df["Group"] = df_common.index.values
        maov = MANOVA.from_formula('Dim1 + Dim2 ~ Group', data=scores_df)
        p_value = maov.mv_test().results['Group']['stat']["Pr > F"]["Wilks' lambda"]
        text = f"MANOVA P = {p_value:.2f}" if p_value >= 0.05 else ("MANOVA P < 0.05" if p_value >= 0.01 else "MANOVA P < 0.01")
        ax_main.text(0.05, 0.95, text, transform=ax_main.transAxes, fontsize=15, color='black', ha='left', va='top')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    else:
        plt.show()


def visualize_common_features_features(
    df_mb=None,
    df_mt=None,
    common_features=None,
    data_type='microbe',
    group=('HC', 'MIA', 'IA'),
    method='PCA',
    random_state=42,
    colors=None,
    confidence_level=0.95,
    core_mode='unique_to_each',
    save_path=None
):
    """
    Plot feature-wise distributions across all samples using core features per cohort.

    Parameters
    ----------
    df_mb : pd.DataFrame or None
        Microbiome table with samples as rows and microbes as columns.
    df_mt : pd.DataFrame or None
        Metabolite table with samples as rows and metabolites as columns.
    common_features : dict
        Nested dict containing keys 'common_to_all', 'unique_to_each', and 'pairwise_common'.
    data_type : str
        One of {'microbe','metabolite'}; selects which table to use internally.
    group : tuple or list
        Ordered group labels to plot.
    method : str
        'PCA' or 'LDA'.
    random_state : int
        Random state for reproducibility (PCA).
    colors : list or None
        Optional list of colors aligned with `group`.
    confidence_level : float
        Confidence level for the ellipse.
    core_mode : str
        One of {'unique_to_each','pairwise_common','common_to_all'} controlling feature selection.
    save_path : str or None
        If provided, save the figure; otherwise show the figure.
    """
    method = method.upper()
    if data_type not in {'microbe', 'metabolite'}:
        raise ValueError("data_type must be 'microbe' or 'metabolite'.")

    if data_type == 'microbe':
        if df_mb is None:
            raise ValueError("df_mb is required when data_type='microbe'.")
        df_common = _build_feature_view_table(df_mb, 'microbe', common_features, groups=group, mode=core_mode)
    else:
        if df_mt is None:
            raise ValueError("df_mt is required when data_type='metabolite'.")
        df_common = _build_feature_view_table(df_mt, 'metabolite', common_features, groups=group, mode=core_mode)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_common)
    if data_type == 'microbe':
        data_scaled = df_common

    reducer = select_reducer(method, 2, random_state, labels=df_common.index if method == 'LDA' else None)

    if method == 'LDA':
        labels = df_common.index.to_numpy()
        reduced = reducer.fit_transform(data_scaled, labels)
        xlabel, ylabel = "LD1", "LD2"
    else:
        reduced = reducer.fit_transform(data_scaled)
        var_ratio = reducer.explained_variance_ratio_
        xlabel = f"PC1 ({var_ratio[0]*100:.1f}%)"
        ylabel = f"PC2 ({var_ratio[1]*100:.1f}%)"

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.6, 4], width_ratios=[4, 0.6], hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    ax_main.set_facecolor('white')
    ax_main.grid(False, linestyle='--', color='gray', alpha=0.5)

    for idx, g in enumerate(group):
        indices = [i for i, label in enumerate(df_common.index) if label == g]
        if indices:
            color = colors[idx] if colors else plt.cm.tab10(idx)
            ax_main.scatter(reduced[indices, 0], reduced[indices, 1], label=g, color=color, alpha=0.9, s=50, marker='s')
            plot_confidence_ellipse(reduced[indices], ax_main, color=color, confidence_level=confidence_level)
            if len(indices) > 1:
                plot_kde_density(reduced[indices, 0], ax_top, orientation='horizontal', color=color, alpha=0.25)
                plot_kde_density(reduced[indices, 1], ax_right, orientation='vertical', color=color, alpha=0.25)

    ax_main.set_xlabel(xlabel, fontsize=15)
    ax_main.set_ylabel(ylabel, fontsize=15)
    if method == 'LDA':
        ax_main.set_title("LDA", fontsize=20)
    ax_main.legend(loc='lower right')

    ax_top.tick_params(axis='x', labelbottom=False)
    ax_top.tick_params(axis='y', labelleft=False)
    ax_top.set_facecolor('white')
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(True)
    ax_top.spines['bottom'].set_visible(True)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.grid(False)

    ax_right.tick_params(axis='y', labelleft=False)
    ax_right.tick_params(axis='x', labelbottom=False)
    ax_right.set_facecolor('white')
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(True)
    ax_right.spines['bottom'].set_visible(True)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.grid(False)

    if method in ['PCA', 'LDA']:
        scores_df = pd.DataFrame(reduced[:, :2], columns=["Dim1", "Dim2"])
        scores_df["Group"] = df_common.index.values
        maov = MANOVA.from_formula('Dim1 + Dim2 ~ Group', data=scores_df)
        p_value = maov.mv_test().results['Group']['stat']["Pr > F"]["Wilks' lambda"]
        text = f"MANOVA P = {p_value:.2f}" if p_value >= 0.05 else ("MANOVA P < 0.05" if p_value >= 0.01 else "MANOVA P < 0.01")
        ax_main.text(0.05, 0.95, text, transform=ax_main.transAxes, fontsize=15, color='black', ha='left', va='top')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    else:
        plt.show()