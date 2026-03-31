import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import MDS
from matplotlib.patches import Ellipse
from scipy.stats import chi2, gaussian_kde
from scipy.spatial.distance import pdist, squareform
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from skbio.stats.distance import permanova
    from skbio import DistanceMatrix
    SKBIO_AVAILABLE = True
except ImportError:
    SKBIO_AVAILABLE = False
    
def permanova_test_fast(
    X, labels, n_permutations=999, random_state=None, n_jobs=-1, metric="euclidean"
):
    labels = np.asarray(labels)
    n = len(labels)

    # 1) Compute the distance matrix only once
    dist = squareform(pdist(X, metric=metric))
    D2 = dist * dist

    # 2) Original group information
    unique_labels, counts = np.unique(labels, return_counts=True)
    group_sizes = counts.tolist()
    df_between = len(unique_labels) - 1
    df_within = n - len(unique_labels)

    # 3) TSS / WSS / BSS
    tss = np.sum(D2) / n

    # Original WSS: compute directly by label-based slicing
    wss = 0.0
    for g in unique_labels:
        idx = np.where(labels == g)[0]
        ng = len(idx)
        if ng > 1:
            wss += np.sum(D2[np.ix_(idx, idx)]) / ng

    bss = tss - wss
    if tss <= 0 or df_within <= 0 or wss <= 0:
        return (bss / tss if tss > 0 else 0.0), 1.0

    pseudo_f = (bss / df_between) / (wss / df_within)

    # 4) Permutation: keep group sizes fixed for speed
    #    Split shuffled indices according to group_sizes
    def wss_from_shuffled_indices(shuffled_idx):
        start = 0
        s = 0.0
        for ng in group_sizes:
            block = shuffled_idx[start:start + ng]
            start += ng
            if ng > 1:
                s += np.sum(D2[np.ix_(block, block)]) / ng
        return s

    # 5) Parallel permutation + reproducible RNG
    if random_state is None:
        ss = np.random.SeedSequence()
    else:
        ss = np.random.SeedSequence(int(random_state))
    child_seeds = ss.spawn(n_permutations)

    def one_perm(seedseq):
        rng = np.random.default_rng(seedseq)
        perm_idx = rng.permutation(n)
        perm_wss = wss_from_shuffled_indices(perm_idx)
        perm_bss = tss - perm_wss
        if perm_wss <= 0:
            return -np.inf
        return (perm_bss / df_between) / (perm_wss / df_within)

    perm_f = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(one_perm)(s) for s in child_seeds
    )
    perm_f = np.asarray(perm_f)

    p_value = (np.sum(perm_f >= pseudo_f) + 1) / (n_permutations + 1)
    r_squared = bss / tss
    return r_squared, p_value

def permanova_test(X, labels, n_permutations=999, random_state=None):
    
    if SKBIO_AVAILABLE:
        # Use skbio PERMANOVA (recommended)
        dist_matrix = squareform(pdist(X, metric='euclidean'))
        dm = DistanceMatrix(dist_matrix)
        
        # If random_state is provided, set the numpy random seed
        if random_state is not None:
            np.random.seed(random_state)
            
        result = permanova(dm, labels, permutations=n_permutations)
        return result['test statistic'], result['p-value']
    else:
        # Manual PERMANOVA implementation
        dist_matrix = squareform(pdist(X, metric='euclidean'))
        n = len(labels)
        
        # Compute total sum of squares (TSS)
        tss = np.sum(dist_matrix ** 2) / n
        
        # Compute within-group sum of squares (WSS)
        unique_labels = np.unique(labels)
        wss = 0
        for label in unique_labels:
            idx = np.where(labels == label)[0]
            n_group = len(idx)
            if n_group > 1:
                group_dist = dist_matrix[np.ix_(idx, idx)]
                wss += np.sum(group_dist ** 2) / n_group
        
        # Compute between-group sum of squares (BSS)
        bss = tss - wss
        
        # R²
        r_squared = bss / tss if tss > 0 else 0
        
        # Degrees of freedom
        df_between = len(unique_labels) - 1
        df_within = n - len(unique_labels)
        
        if df_within == 0 or wss == 0:
            return r_squared, 1.0
        
        # Pseudo-F statistic
        pseudo_f = (bss / df_between) / (wss / df_within)
        
        # Permutation test with random_state
        perm_f_stats = []
        if random_state is not None:
            np.random.seed(random_state)
        
        for _ in range(n_permutations):
            perm_labels = np.random.permutation(labels)
            
            perm_wss = 0
            for label in unique_labels:
                idx = np.where(perm_labels == label)[0]
                n_group = len(idx)
                if n_group > 1:
                    group_dist = dist_matrix[np.ix_(idx, idx)]
                    perm_wss += np.sum(group_dist ** 2) / n_group
            
            perm_bss = tss - perm_wss
            
            if perm_wss > 0:
                perm_f = (perm_bss / df_between) / (perm_wss / df_within)
                perm_f_stats.append(perm_f)
        
        # Compute p-value
        p_value = (np.sum(np.array(perm_f_stats) >= pseudo_f) + 1) / (n_permutations + 1)
        return r_squared, p_value


def compute_pcoa_variance_explained(dist_matrix):
    """Compute variance explained ratio for PCoA"""
    # Double-center the distance matrix
    n = dist_matrix.shape[0]
    D_squared = dist_matrix ** 2
    
    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n
    
    # Gower matrix
    B = -0.5 * H @ D_squared @ H
    
    # Eigen decomposition
    eigenvalues, _ = np.linalg.eigh(B)
    
    # Sort in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Keep only positive eigenvalues
    positive_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    # Compute variance explained ratio
    if len(positive_eigenvalues) > 0:
        var_explained = positive_eigenvalues / np.sum(positive_eigenvalues)
        return var_explained
    else:
        return np.array([0.0])
        

def format_p_for_plot(p):
    """Format p-value for LaTeX rendering"""
    if p < 0.01:
        exp = int(np.floor(np.log10(p)))
        mant = p / (10 ** exp)
        return f"${mant:.2f} \\times 10^{{{exp}}}$"
    else:
        return f"{p:.3f}"


def get_scaler(scale_method):
    """Return scaler according to the method name"""
    if scale_method is None or scale_method.lower() == 'none':
        return None
    elif scale_method.lower() == 'standard':
        return StandardScaler()
    elif scale_method.lower() == 'minmax':
        return MinMaxScaler()
    elif scale_method.lower() == 'robust':
        return RobustScaler()
    else:
        raise ValueError(f"Unsupported scale method: {scale_method}. "
                        f"Choose from: 'standard', 'minmax', 'robust', 'none'")


def select_reducer(method, n_components, random_state, labels=None):
    """Select dimensionality reduction method"""
    method = method.upper()
    if method == 'PCA':
        return PCA(n_components=n_components, random_state=random_state)
    elif method == 'PLSDA':
        return PLSRegression(n_components=n_components)
    elif method == 'LDA':
        if labels is None:
            raise ValueError("LDA requires labels.")
        return LinearDiscriminantAnalysis(n_components=n_components)
    elif method == 'PCOA':
        return MDS(n_components=n_components, dissimilarity='precomputed', 
                   random_state=random_state)
    elif method == 'UMAP':
        if not UMAP_AVAILABLE:
            raise ValueError("UMAP is not installed. Install it using: pip install umap-learn")
        return umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=13, min_dist=0.1)
    else:
        raise ValueError(f"Unsupported method: {method}")


def plot_confidence_ellipse(X, ax, color, confidence_level=0.95):
    """Plot confidence ellipse"""
    if len(X) < 3:
        return
        
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    
    if np.linalg.det(cov) == 0:
        return
        
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.arctan2(*eigvecs[:, 0][::-1])
    
    scale = np.sqrt(chi2.ppf(confidence_level, df=2))
    width, height = 2 * scale * np.sqrt(np.abs(eigvals))

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


def plot_kde_density(data, ax, orientation='horizontal', color='blue', 
                     scale=0.1, alpha=0.08):
    """Plot kernel density"""
    if len(data) <= 1:
        return
    
    kde = gaussian_kde(data)
    
    if orientation == 'horizontal':
        x_range = np.linspace(data.min() - 0.5*data.std(), 
                             data.max() + 0.5*data.std(), 200)
        density = kde(x_range) * scale
        ax.fill_between(x_range, 0, density, alpha=alpha, color=color)
        ax.plot(x_range, density, color=color, linewidth=2, alpha=1, linestyle='--')
    else:
        y_range = np.linspace(data.min() - 0.5*data.std(), 
                             data.max() + 0.5*data.std(), 200)
        density = kde(y_range) * scale
        ax.fill_betweenx(y_range, 0, density, alpha=alpha, color=color)
        ax.plot(density, y_range, color=color, linewidth=2, alpha=1, linestyle='--')

def visualize_common_features_samples(df_query, 
                                      features=None,
                                      group=['HC', 'MIA', 'IA'], 
                                      method='PCA',
                                      scale='standard',
                                      random_state=42, 
                                      colors=None, 
                                      alphas=0.9,  # New parameter: default is 0.9
                                      confidence_level=0.95,
                                      test_on_original=True,
                                      n_permutations=9999,
                                      save_path=None,
                                      figsize=(12, 10),
                                      marker='s',
                                      transparent=True,
                                      show_stats=True):
    """
    Dimensionality reduction visualization function (enhanced version)
    
    Parameters:
    -----------
    ... (existing parameters) ...
    colors : list, dict or None
        - list: color list corresponding to the group order.
        - dict: {'GroupA': 'red', 'GroupB': 'blue'}, must include all keys in group.
        - None: use default Tab10 palette.
        
    alphas : float, list, dict
        Control scatter transparency (0.0 ~ 1.0).
        - float: same transparency for all groups (default 0.9).
        - list: transparency list corresponding to the group order.
        - dict: {'GroupA': 1.0, 'GroupB': 0.1}, must include all keys in group.
    """
    method = method.upper()

    # LDA check
    if len(group) < 2 and method == 'LDA':
        raise ValueError("LDA requires at least 2 groups.")

    # Feature selection
    if features is not None:
        df_common = df_query[features].copy()
    else:
        df_common = df_query.copy()

    # Group filtering
    df_common = df_common[df_common.index.isin(group)]
    if len(df_common) == 0:
        raise ValueError("No samples match the specified groups")

    # =========================================
    # 1. Handle marker parameter
    # =========================================
    if isinstance(marker, dict):
        missing_groups = set(group) - set(marker.keys())
        relevant_marker_keys = [g for g in group if g in marker]
        if not relevant_marker_keys and len(group) > 0: 
             print(f"Warning: Marker dictionary might not match current groups.")
        marker_dict = marker
    elif isinstance(marker, str):
        marker_dict = {g: marker for g in group}
    else:
        raise TypeError("marker must be str or dict")

    # =========================================
    # 2. Handle colors parameter (list or dict)
    # =========================================
    color_map = {}
    if colors is None:
        # Default palette
        cmap = plt.cm.tab10
        color_map = {g: cmap(i) for i, g in enumerate(group)}
    elif isinstance(colors, dict):
        # Validate dictionary keys
        missing_colors = set(group) - set(colors.keys())
        if missing_colors:
            raise ValueError(f"Colors dictionary missing keys for: {missing_colors}")
        color_map = colors
    elif isinstance(colors, list):
        # Validate list length
        if len(colors) < len(group):
            raise ValueError(f"Colors list length ({len(colors)}) is less than groups length ({len(group)}).")
        color_map = {g: colors[i] for i, g in enumerate(group)}
    else:
        raise TypeError("colors must be a list, a dictionary, or None.")

    # =========================================
    # 3. Handle alphas parameter (float, list or dict)
    # =========================================
    alpha_map = {}
    if isinstance(alphas, (float, int)):
        alpha_map = {g: float(alphas) for g in group}
    elif isinstance(alphas, dict):
        missing_alphas = set(group) - set(alphas.keys())
        if missing_alphas:
            raise ValueError(f"Alphas dictionary missing keys for: {missing_alphas}")
        alpha_map = alphas
    elif isinstance(alphas, list):
        if len(alphas) < len(group):
            raise ValueError(f"Alphas list length ({len(alphas)}) is less than groups length ({len(group)}).")
        alpha_map = {g: alphas[i] for i, g in enumerate(group)}
    else:
        raise TypeError("alphas must be a float, a list, or a dictionary.")

    # =========================================
    # Data preprocessing and dimensionality reduction
    # =========================================
    # Standardization
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    def get_scaler_local(method_name):
        if method_name is None or method_name.lower() == 'none': return None
        if method_name.lower() == 'standard': return StandardScaler()
        if method_name.lower() == 'minmax': return MinMaxScaler()
        if method_name.lower() == 'robust': return RobustScaler()
        return None

    scaler = get_scaler_local(scale)
    if scaler is not None:
        data_scaled = scaler.fit_transform(df_common)
    else:
        data_scaled = df_common.values
    
    labels = df_common.index.to_numpy()
    unique_labels = np.unique(labels)

    # Auto-disable stats for single group
    if len(unique_labels) < 2:
        if show_stats:
            print("Notice: Only 1 group detected. Statistical test (PERMANOVA) skipped.")
        show_stats = False

    # Reducer selection
    reducer = select_reducer(method, 2, random_state, labels=labels if method == 'LDA' else None)

    # Fit transform
    if method == 'LDA':
        reduced = reducer.fit_transform(data_scaled, labels)
        xlabel, ylabel = "LD1", "LD2"
    elif method == 'PLSDA':
        labels_dummy = pd.get_dummies(labels)
        reducer.fit(data_scaled, labels_dummy)
        reduced = reducer.x_scores_
        x_var = np.var(reduced, axis=0)
        x_var_ratio = x_var / np.sum(x_var)
        xlabel = f"LV1 ({x_var_ratio[0]*100:.1f}%)"
        ylabel = f"LV2 ({x_var_ratio[1]*100:.1f}%)"
    elif method == 'PCOA':
        dist_matrix = squareform(pdist(data_scaled, metric='euclidean'))
        reduced = reducer.fit_transform(dist_matrix)
        # Compute PCoA variance explained
        var_explained = compute_pcoa_variance_explained(dist_matrix) 
        if len(var_explained) >= 2:
            xlabel = f"PCoA1 ({var_explained[0]*100:.1f}%)"
            ylabel = f"PCoA2 ({var_explained[1]*100:.1f}%)"
        else:
            xlabel, ylabel = "PCoA1", "PCoA2"
    elif method == 'UMAP':
        reduced = reducer.fit_transform(data_scaled)
        xlabel, ylabel = "UMAP1", "UMAP2"
    else:  # PCA
        reduced = reducer.fit_transform(data_scaled)
        var_ratio = reducer.explained_variance_ratio_
        xlabel = f"PC1 ({var_ratio[0]*100:.1f}%)"
        ylabel = f"PC2 ({var_ratio[1]*100:.1f}%)"

    # Statistical test
    p_value = None
    if show_stats:
        if test_on_original:
            _, p_value = permanova_test_fast(data_scaled, labels, n_permutations=n_permutations, random_state=random_state)
        else:
            _, p_value = permanova_test_fast(reduced, labels, n_permutations=n_permutations, random_state=random_state)

    # =========================================
    # Plotting logic
    # =========================================
    fig = plt.figure(figsize=figsize, dpi=300)
    
    if not transparent:
        fig.patch.set_facecolor('white')
    else:
        fig.patch.set_alpha(0.0)
    
    gs = fig.add_gridspec(2, 2, height_ratios=[0.6, 4], width_ratios=[4, 0.6], 
                          hspace=0.05, wspace=0.05)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    for ax in [ax_main, ax_top, ax_right]:
        if not transparent:
            ax.set_facecolor('white')
        else:
            ax.patch.set_alpha(0.0)
    
    ax_main.grid(False, linestyle='--', color='gray', alpha=0.5)

    # Loop through each group for plotting
    for idx, g in enumerate(group):
        indices = [i for i, label in enumerate(labels) if label == g]
        if not indices:
            continue
            
        # Get processed color and alpha
        current_color = color_map[g]
        current_alpha = alpha_map[g]
        current_marker = marker_dict.get(g, 'o')
        
        # Scatter plot (apply custom alpha)
        ax_main.scatter(reduced[indices, 0], reduced[indices, 1], 
                       label=g, 
                       color=current_color, 
                       alpha=current_alpha,
                       s=50, 
                       marker=current_marker)
        
        # Auxiliary graphics (confidence ellipse and density curves)
        # If scatter alpha is extremely low, it may indicate that the user wants to hide the group,
        # but the current logic still draws the auxiliary layers
        
        # Confidence ellipse
        if len(indices) > 2:
            plot_confidence_ellipse(reduced[indices], ax_main, 
                                    color=current_color, confidence_level=confidence_level)
        
        # KDE density
        if len(indices) > 1:
            plot_kde_density(reduced[indices, 0], ax_top, 
                           orientation='horizontal', color=current_color, 
                           scale=0.1, alpha=0.25)
            plot_kde_density(reduced[indices, 1], ax_right, 
                           orientation='vertical', color=current_color, 
                           scale=0.1, alpha=0.25)

    ax_main.set_xlabel(xlabel, fontsize=15)
    ax_main.set_ylabel(ylabel, fontsize=15)

    # Legend
    legend = ax_main.legend(loc='lower right', fontsize=14, markerscale=1.5, 
                            frameon=True, fancybox=True)
    if transparent:
        legend.get_frame().set_alpha(0.8)

    # P-value text
    if show_stats and p_value is not None:
        p_text = f"$\it{{p}}$ = {format_p_for_plot(p_value)}"
        bbox_props = dict(boxstyle='round', facecolor='white', edgecolor='grey',
        linewidth=1.2, alpha=0.8 if not transparent else 0.9)
        ax_main.text(0.05, 0.95, p_text, transform=ax_main.transAxes,
                    fontsize=13, color='black', ha='left', va='top',
                    bbox=bbox_props)

    # Axis alignment and cleanup
    xlim = ax_main.get_xlim()
    ylim = ax_main.get_ylim()
    ax_top.set_xlim(xlim)
    ax_right.set_ylim(ylim)

    for ax in [ax_top, ax_right]:
        ax.tick_params(axis='both', labelbottom=False, labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight', transparent=transparent)
        print(f"✓ Image saved to: {save_path}")
    else:
        plt.show()

    # return fig