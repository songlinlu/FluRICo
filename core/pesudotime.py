import numpy as np
import pandas as pd
import omicverse as ov


def process_group_corr_mats(
    R_all,
    R_groups,
    use_fisher: bool = True,
    eps: float = 1e-6,
    diag_value: float = 0.0,
    symmetrize: bool = True,
    standardize: str | None = None,   # None | "zscore" | "robust_z" | "rank_normal"
    # --- New: shuffle after stacking ---
    shuffle_after_stack: bool = False,
    shuffle_seed = 42,
):
    """
    Modified version:
      - The returned matrix index now uses the format "{taxon}_{group}"
        (for example, "Bacteroides_HC").
      - This preserves row identity even when shuffle_after_stack=True.
    """

    def _to_df(M, taxa):
        if isinstance(M, pd.DataFrame):
            df = M.copy()
            if df.columns is None or len(df.columns) != len(taxa):
                if df.shape[1] != len(taxa):
                    raise ValueError("Matrix column count does not match taxa length; alignment failed.")
                df.columns = list(taxa)
            else:
                if set(df.columns) != set(taxa):
                    raise ValueError("Grouped matrix columns do not match the global matrix columns; alignment failed.")
                df = df.loc[:, taxa]

            if df.shape[0] != len(taxa):
                raise ValueError("Matrix row count does not match taxa length; cannot assign a square index.")
            df.index = list(taxa)
            return df

        arr = np.asarray(M)
        if arr.shape[0] != arr.shape[1]:
            raise ValueError("Input ndarray is not square.")
        if arr.shape[0] != len(taxa):
            raise ValueError("Input ndarray size does not match the global taxa length.")
        return pd.DataFrame(arr, index=list(taxa), columns=list(taxa))

    def _symmetrize(df):
        A = df.values.astype(float)
        A = 0.5 * (A + A.T)
        return pd.DataFrame(A, index=df.index, columns=df.columns)

    def _fisher(df):
        A = df.values.astype(float)
        A = np.clip(A, -1 + eps, 1 - eps)
        return pd.DataFrame(np.arctanh(A), index=df.index, columns=df.columns)

    def _set_diag(df, val):
        A = df.values
        np.fill_diagonal(A, val)
        return df

    def _standardize_offdiag(df, mode: str):
        A = df.values.astype(float)
        n = A.shape[0]
        mask = ~np.eye(n, dtype=bool)
        x = A[mask]

        if mode == "zscore":
            mu = float(np.mean(x))
            sd = float(np.std(x)) + 1e-12
            A[mask] = (x - mu) / sd

        elif mode == "robust_z":
            med = float(np.median(x))
            mad = float(np.median(np.abs(x - med))) + 1e-12
            A[mask] = (x - med) / mad

        elif mode == "rank_normal":
            from scipy.stats import rankdata, norm
            r = rankdata(x, method="average")
            u = (r - 0.5) / len(x)
            A[mask] = norm.ppf(u)

        else:
            raise ValueError("standardize must be one of None/'zscore'/'robust_z'/'rank_normal'.")

        return pd.DataFrame(A, index=df.index, columns=df.columns)

    # --- Taxa reference: prefer R_all.columns; otherwise use 0..N-1 ---
    if isinstance(R_all, pd.DataFrame) and R_all.columns is not None and len(R_all.columns) == R_all.shape[1]:
        taxa = list(map(str, R_all.columns))
    else:
        n = np.asarray(R_all).shape[0]
        taxa = list(map(str, range(n)))

    R_all_df = _to_df(R_all, taxa)
    if symmetrize:
        R_all_df = _symmetrize(R_all_df)

    if isinstance(R_groups, dict):
        keys = list(R_groups.keys())
        mats = {k: R_groups[k] for k in keys}
    else:
        mats = {f"G{i}": M for i, M in enumerate(R_groups)}

    Z_all = _fisher(R_all_df) if use_fisher else R_all_df

    D_list = []
    for k, M in mats.items():
        Rg = _to_df(M, taxa)
        if symmetrize:
            Rg = _symmetrize(Rg)

        Zg = _fisher(Rg) if use_fisher else Rg
        D = Zg - Z_all
        D = _set_diag(D, diag_value)

        if standardize is not None:
            D = _standardize_offdiag(D, standardize)

        # Updated: use the "{taxon}_{group}" index format instead of repeating only the group name.
        D.index = [f"{bac}_{k}" for bac in taxa]
        
        D_list.append(D)

    # --- Stack into (kN, N) ---
    df_stack = pd.concat(D_list, axis=0)

    # --- Shuffle after stacking (physical row order; index moves with rows) ---
    if shuffle_after_stack:
        rng = np.random.default_rng(shuffle_seed)
        perm = rng.permutation(df_stack.shape[0])
        df_stack = df_stack.iloc[perm].copy()

    return df_stack


import numpy as np
import anndata
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

def run_trajectory_analysis_ultimate(
    df,
    full_name_dic: dict,
    group_col: str = "group",
    root_group: str = "HC",
    # --- Matrix type and grouping controls ---
    is_symmetric: bool = True,
    group_labels_input: list = None,
    # --- Core analysis parameters ---
    n_comps: int = 50,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    use_pca: bool = False,
    root_strategy: str = "centroid",
    # --- Diagonal handling ---
    mask_self_corr: bool = True,
    self_corr_value: float = 0.0,
    self_corr_min_match_rate: float = 0.8,
    # --- Figure 3 (microbe only): most abundant taxa by count (TopN) ---
    tax_level: str = "f__",
    top_n: int = 10,
    target_taxa: list = None,
    custom_colors: list = None,
    # --- Figure 4: largest pseudotime changes ---
    plot_change: bool = True,
    top_n_change: int = 10,
    change_tax_level: str = None,          # Used for microbe; ignored for metabolite
    comparison_groups: tuple = ("HC", "IA"),  # By default compare only HC and IA
    change_metric: str = "abs_delta",      # 'delta' | 'abs_delta' | 'fold'
    # --- Plotting parameters ---
    plot: bool = True,
    show_taxonomy_panel: bool = False,
    show_change_panel: bool = False,
    transparent: bool = True,
    violin_order=("HC", "MIA", "IA"),
    group_colors=("#6B98C4", "#FFBC80", "#F5867F"),
    cmap: str = "Reds",
    paga_threshold: float = 0.01,
    # --- Data type ---
    data_type: str = "microbe",            # "microbe" | "metabolite"
    savepath = None,
    figsize: tuple[float, float] | None = None,
):
    """
    Requirements implemented:
    1) Keep Figures 1 and 2 unchanged in terms of analysis, dimensionality
       reduction, and pseudotime workflow.
    2) Figure 3 shows the most abundant taxa (microbe only; taxonomy count TopN).
    3) Figure 4 shows the largest pseudotime changes (HC -> IA only, selecting
       the top changing items).
       - For microbe: aggregate by taxonomy and keep the original logic.
       - For metabolite: rank by Entity directly without taxonomy.
    4) Figures 3 and 4 are still computed, but can be hidden from rendering.
       By default, only Figures 1 and 2 are displayed.
    5) Additional plotting enhancements:
       - In Figure 4, each top-changing feature is connected with dashed lines
         and arrows in HC -> MIA -> IA order.
         * For microbe: features are taxonomy labels, using group-wise UMAP centroids.
         * For metabolite: features are Entity names, using group-wise UMAP centroids.
       - Figure 1 uses HC -> MIA -> IA arrows between centroids and hides the
         original PAGA edges.
       - Figure 1 uses larger centroids, and arrows start outside the centroid radius.
       - Figure 4 uses lower arrow opacity, avoids covering points, points to
         point edges, keeps point alpha at 0.9, and draws top points on top.
       - Figure 4 uses vivid discrete colors for top features.
    """

    # ============================
    # Internal plotting helpers: arrows and dashed trajectories
    # ============================
    def _draw_arrow(ax, p0, p1, color="k", lw=1.8, ls="-", alpha=0.9, ms=12, z=10, shrink_start=0, shrink_end=0):
        """Draw a p0->p1 arrow on ax; dashed lines are supported."""
        if p0 is None or p1 is None:
            return
        arrow = FancyArrowPatch(
            posA=(float(p0[0]), float(p0[1])),
            posB=(float(p1[0]), float(p1[1])),
            arrowstyle="-|>",
            mutation_scale=ms,
            lw=lw,
            linestyle=ls,
            color=color,
            alpha=alpha,
            shrinkA=shrink_start,  # Offset from the start point.
            shrinkB=shrink_end,    # Offset from the end point.
            zorder=z,
        )
        ax.add_patch(arrow)

    def _draw_group_flow_arrows(ax, group_to_xy, order, color="k", lw=2.2, ms=16, z=12, shrink=15):
        """Draw centroid arrows in the specified order (for example HC->MIA->IA)."""
        for a, b in zip(order[:-1], order[1:]):
            if a in group_to_xy and b in group_to_xy:
                _draw_arrow(ax, group_to_xy[a], group_to_xy[b], color=color, lw=lw, ls="-", 
                           alpha=0.95, ms=ms, z=z, shrink_start=shrink, shrink_end=shrink)

    def _draw_feature_trajectories(ax, df_centroids, order, color="k", lw=1.2, ms=10, alpha=0.5, shrink=8):
        """
        df_centroids columns: [feature, group, x, y]
        For each feature, connect group centroids in the specified order with
        dashed lines and arrows, while shrinking arrow endpoints away from points.
        """
        feats = df_centroids["feature"].unique().tolist()
        for feat in feats:
            sub = df_centroids[df_centroids["feature"] == feat]
            # Map each group to its centroid.
            g2p = {r["group"]: (r["x"], r["y"]) for _, r in sub.iterrows()}
            for a, b in zip(order[:-1], order[1:]):
                if a in g2p and b in g2p:
                    # Draw a lighter dashed connector first.
                    ax.plot(
                        [g2p[a][0], g2p[b][0]],
                        [g2p[a][1], g2p[b][1]],
                        linestyle="--",
                        linewidth=lw,
                        color=color,
                        alpha=alpha * 0.5,  # Keep the line lighter than the arrow.
                        zorder=8,  # Keep lines below highlighted points.
                    )
                    # Add the directional arrow.
                    _draw_arrow(ax, g2p[a], g2p[b], color=color, lw=lw, ls="--", 
                               alpha=alpha, ms=ms, z=9, shrink_start=shrink, shrink_end=shrink)

    # 0. Environment hotfix
    if not hasattr(matplotlib.rcParams, "_get"):
        def _get_compatibility(key, default=None):
            return matplotlib.rcParams.get(key, default)
        matplotlib.rcParams._get = _get_compatibility

    # 1. Data cleaning
    if df.isnull().values.any():
        print("Detected NaN values; filling them with 0.")
        df = df.fillna(0)

    n_obs, n_vars = df.shape
    raw_index_names = df.index.astype(str).tolist()

    print(f"\n{'='*60}")
    print(f"data_type: {data_type}")
    print(f"matrix type: {'symmetric / stacked correlation matrices' if is_symmetric else 'asymmetric matrix (samples x features)'}")
    print(f"data shape: {n_obs} rows x {n_vars} columns")
    print(f"{'='*60}\n")

    # ----------------------------
    # A) Group labels: parse from index without relying on row order
    # ----------------------------
    valid_groups = set(violin_order)

    def _parse_group_from_index(idx_str: str) -> str:
        s = str(idx_str).strip()
        if s in valid_groups:
            return s
        for g in violin_order:
            if s.endswith(f"_{g}") or s.endswith(f"-{g}") or s.endswith(f"|{g}"):
                return g
        return "Unknown"

    if group_labels_input is not None:
        if len(group_labels_input) != n_obs:
            raise ValueError(f"group_labels_input length ({len(group_labels_input)}) must match the number of rows ({n_obs}).")
        group_labels = list(map(str, group_labels_input))
        print("✓ Using manually provided group labels from group_labels_input.")
    else:
        group_labels = [_parse_group_from_index(x) for x in raw_index_names]
        unknown = [raw_index_names[i] for i, g in enumerate(group_labels) if g == "Unknown"]
        if len(unknown) > 0:
            raise ValueError(
                f"Failed to parse group labels from df.index: {len(unknown)}/{n_obs} rows were labeled as Unknown.\n"
                f"Examples: {unknown[:10]}\n"
                f"Make sure the index is exactly one of {violin_order}, or follows patterns such as "
                f"xxx_HC / xxx-MIA / xxx|IA, or pass group_labels_input manually."
            )
        print("✓ Parsed group labels from df.index without relying on row order.")

    print(f"  group counts: {dict(pd.Series(group_labels).value_counts())}")

    # ----------------------------
    # B) Self-correlation masking
    # ----------------------------
    def _canon_entity_name(s: str) -> str:
        s = str(s)
        if "|" in s:
            s = s.split("|")[0]
        for suf in ["_MIA", "_IA", "_HC", "-MIA", "-IA", "-HC"]:
            if s.endswith(suf):
                s = s[: -len(suf)]
                break
        return s

    if mask_self_corr:
        col_names = df.columns.astype(str).tolist()
        col_index = {c: j for j, c in enumerate(col_names)}
        hit = 0
        for i, rn in enumerate(raw_index_names):
            ent = _canon_entity_name(rn)
            j = col_index.get(ent, None)
            if j is not None:
                df.iat[i, j] = self_corr_value
                hit += 1
        hit_rate = hit / max(1, n_obs)
        print(f"✓ Self-corr masking: matched {hit}/{n_obs} rows ({hit_rate:.1%}), set to {self_corr_value}.")
        if hit_rate < self_corr_min_match_rate:
            print(
                "⚠ Self-corr match rate is low, which suggests that entity identities in df.index "
                "may not align with df.columns.\n"
                "  If your index contains only HC/MIA/IA without entity names, consider disabling mask_self_corr."
            )

    # ----------------------------
    # C) Build AnnData
    # ----------------------------
    obs_names = [f"{raw_index_names[i]}__{group_labels[i]}__{i}" for i in range(n_obs)]
    adata = anndata.AnnData(X=df.values)
    adata.obs_names = obs_names
    adata.var_names = df.columns.astype(str)
    adata.obs[group_col] = pd.Categorical(group_labels, categories=violin_order, ordered=True)

    default_map = {"HC": group_colors[0], "MIA": group_colors[1], "IA": group_colors[2]}
    ordered_groups = [g for g in violin_order if g in set(group_labels)]
    adata.uns[f"{group_col}_colors"] = [default_map.get(g, "#808080") for g in ordered_groups]

    # Normalize entity names (microbe = taxon name; metabolite = metabolite name).
    adata.obs["Entity"] = [_canon_entity_name(rn) for rn in raw_index_names]

    # ----------------------------
    # D) Microbe taxonomy (used by the microbe branches of Figures 3 and 4)
    # ----------------------------
    def extract_tax_level(full_str, level_prefix):
        if not isinstance(full_str, str):
            return "Others"
        parts = full_str.split(";")
        for p in parts:
            p = p.strip()
            if p.startswith(level_prefix):
                name_body = p.replace(level_prefix, "")
                if not name_body or name_body.lower() in ["unclassified", "unknown", ""]:
                    return "Others"
                return p
        return "Others"

    if data_type == "microbe":
        print(f"Parsing taxonomy at level: {tax_level} ...")
        row_taxs_raw = []
        for rn in raw_index_names:
            bac = _canon_entity_name(rn)
            full_name = full_name_dic.get(bac, "")
            row_taxs_raw.append(extract_tax_level(full_name, tax_level) if full_name else "Others")
        adata.obs["Taxonomy_Raw"] = row_taxs_raw

        # Figure 3: most abundant taxa (TopN by Taxonomy_Raw count)
        if target_taxa:
            plot_names = list(target_taxa)
            print(f"Figure 3 mode A: using a custom taxa list ({len(plot_names)} entries).")
        else:
            counts = adata.obs["Taxonomy_Raw"].value_counts()
            if "Others" in counts:
                counts = counts.drop("Others")
            plot_names = counts.nlargest(top_n).index.tolist()
            print(f"Figure 3 mode B: automatically selecting the Top {top_n} most abundant taxa.")

        adata.obs["Taxonomy_Plot"] = adata.obs["Taxonomy_Raw"].apply(lambda x: x if x in plot_names else "Others")
        categories = plot_names + ["Others"]
        adata.obs["Taxonomy_Plot"] = pd.Categorical(adata.obs["Taxonomy_Plot"], categories=categories, ordered=True)

        if custom_colors and len(custom_colors) >= len(plot_names):
            palette_list = list(custom_colors)[: len(plot_names)]
        else:
            palette_list = sns.color_palette("husl", max(1, len(plot_names))).as_hex()[: len(plot_names)]
        palette_dict = {name: color for name, color in zip(plot_names, palette_list)}
        palette_dict["Others"] = "#e0e0e0"
        adata.uns["Taxonomy_Plot_colors"] = [palette_dict[cat] for cat in categories]
    else:
        print("data_type=metabolite: skipping taxonomy (Figure 3 is disabled and Figure 4 does not depend on taxonomy).")

    # ----------------------------
    # E) Dimensionality reduction and pseudotime (analysis flow unchanged)
    # ----------------------------
    print("Computing dimensionality reduction and pseudotime...")

    if use_pca:
        sc.pp.pca(adata, n_comps=n_comps, random_state=0)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=0)
    else:
        print("Note: using the raw matrix to compute neighbors (PCA skipped).")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X", random_state=0)

    print("Computing Diffusion Map...")
    sc.tl.diffmap(adata)

    groups_series = adata.obs[group_col].astype(str)
    root_indices = np.flatnonzero(groups_series.values == str(root_group))
    if len(root_indices) == 0:
        raise ValueError(f"Root group '{root_group}' not found in data.")

    if root_strategy == "centroid":
        root_diffmap = adata.obsm["X_diffmap"][root_indices]
        centroid = np.mean(root_diffmap, axis=0)
        distances = np.linalg.norm(root_diffmap - centroid, axis=1)
        closest_idx = root_indices[np.argmin(distances)]
        adata.uns["iroot"] = int(closest_idx)
        print(f"✓ Root strategy: using the centroid-nearest sample in group {root_group} (index: {closest_idx}).")
        print(f"  -> sample: {raw_index_names[closest_idx]} | Entity: {adata.obs['Entity'].iloc[closest_idx]}")
    else:
        target_ent = str(root_strategy).strip()
        print(f"Searching for the root sample: Entity='{target_ent}', group='{root_group}' ...")
        found_idx = -1
        for idx in root_indices:
            if adata.obs["Entity"].iloc[idx] == target_ent:
                found_idx = idx
                break
        if found_idx == -1:
            raise ValueError(
                f"Root strategy failed: entity '{target_ent}' was not found in group {root_group}.\n"
                f"Please check whether the provided name matches df.index after suffix removal."
            )
        adata.uns["iroot"] = int(found_idx)
        print(f"✓ Root strategy: using the specified entity '{target_ent}' ({root_group}) as the starting point (index: {found_idx}).")

    sc.tl.dpt(adata)
    sc.tl.umap(adata, random_state=0)

    # ----------------------------
    # F) PAGA (keep the computation, but do not draw the original edges)
    # ----------------------------
    print("Computing PAGA topology...")
    sc.tl.paga(adata, groups=group_col)

    if "paga" in adata.uns and "connectivities" in adata.uns["paga"]:
        cats = list(adata.obs[group_col].cat.categories)
        try:
            hc_idx = cats.index("HC")
            ia_idx = cats.index("IA")
            conns = adata.uns["paga"]["connectivities"].todense()
            conns[hc_idx, ia_idx] = 0
            conns[ia_idx, hc_idx] = 0
            from scipy import sparse
            adata.uns["paga"]["connectivities"] = sparse.csr_matrix(conns)
            print("✓ PAGA: removed the direct HC-IA edge.")
        except ValueError:
            pass

    # Group centroids (mean UMAP coordinates)
    group_to_xy = {}
    for g in violin_order:
        mask = (adata.obs[group_col].astype(str) == str(g)).values
        if np.sum(mask) > 0:
            mu = np.mean(adata.obsm["X_umap"][mask], axis=0)
            group_to_xy[g] = (float(mu[0]), float(mu[1]))

    # Keep paga_pos available in adata.uns to preserve the existing data structure.
    paga_pos = np.zeros((len(adata.obs[group_col].cat.categories), 2))
    for i, cat in enumerate(adata.obs[group_col].cat.categories):
        mask = (adata.obs[group_col] == cat)
        if np.sum(mask) > 0:
            paga_pos[i] = np.mean(adata.obsm["X_umap"][mask], axis=0)
    adata.uns["paga"]["pos"] = paga_pos

    # ----------------------------
    # G) Figure 4: largest pseudotime changes (HC -> IA only, ranked by abs_delta by default)
    # ----------------------------
    change_taxa_info = None   # For microbe
    change_entity_info = None # For metabolite

    # Centroid trajectories used for dashed lines and arrows.
    traj_centroids_df = None  # columns: feature, group, x, y

    if plot_change:
        start_group, end_group = comparison_groups
        print(f"✓ Figure 4: computing pseudotime change ({start_group} -> {end_group}) using the {change_metric} metric.")

        if data_type == "microbe":
            analysis_level = change_tax_level if change_tax_level is not None else tax_level

            row_taxs_change = []
            for rn in raw_index_names:
                bac = _canon_entity_name(rn)
                full_name = full_name_dic.get(bac, "")
                row_taxs_change.append(extract_tax_level(full_name, analysis_level) if full_name else "Others")
            adata.obs["Taxonomy_Change"] = row_taxs_change

            df_pt = pd.DataFrame({
                "taxonomy": adata.obs["Taxonomy_Change"].values,
                "group": adata.obs[group_col].astype(str).values,
                "pseudotime": adata.obs["dpt_pseudotime"].values.astype(float),
                "entity": adata.obs["Entity"].values,  # Preserve the original feature names.
            })
            df_pt = df_pt[df_pt["taxonomy"] != "Others"]

            if df_pt.shape[0] == 0:
                print("⚠ Figure 4 skipped: Taxonomy_Change contains only 'Others' (index lacks taxon names or full_name_dic has no match).")
                plot_change = False
            else:
                group_mean = df_pt.groupby(["taxonomy", "group"])["pseudotime"].mean().reset_index()
                pivot = group_mean.pivot(index="taxonomy", columns="group", values="pseudotime")
                # Add the original_features column listing all original taxa mapped to each taxonomy.
                entity_map = df_pt.groupby("taxonomy")["entity"].apply(
                    lambda x: "; ".join(sorted(set(x)))
                ).to_dict()
                pivot["original_features"] = pivot.index.map(entity_map)

                if start_group not in pivot.columns or end_group not in pivot.columns:
                    print(f"⚠ Figure 4 skipped: comparison groups {comparison_groups} are incomplete in the data.")
                    plot_change = False
                else:
                    # Compute only the HC -> IA change.
                    if change_metric == "delta":
                        pivot["change"] = pivot[end_group] - pivot[start_group]
                    elif change_metric == "abs_delta":
                        pivot["change"] = (pivot[end_group] - pivot[start_group]).abs()
                    elif change_metric == "fold":
                        pivot["change"] = pivot[end_group] / (pivot[start_group] + 1e-10)
                    else:
                        raise ValueError(f"Unknown change_metric: {change_metric}")

                    pivot = pivot.dropna(subset=["change"])
                    if pivot.shape[0] == 0:
                        print("⚠ Figure 4 skipped: no valid taxonomy change entries were found.")
                        plot_change = False
                    else:
                        # Select the top-changing entries.
                        if change_metric == "fold":
                            pivot["sort_key"] = (pivot["change"] - 1).abs()
                        else:
                            pivot["sort_key"] = pivot["change"].abs()
                        
                        top_change = pivot.nlargest(top_n_change, "sort_key")
                        top_list = top_change.index.tolist()
                        print(top_change)
                        adata.obs["Taxonomy_Change_Plot"] = adata.obs["Taxonomy_Change"].apply(
                            lambda x: x if x in top_list else "Others"
                        )
                        sorted_taxa = top_change.sort_values("change", ascending=False).index.tolist()
                        categories_change = sorted_taxa + ["Others"]
                        adata.obs["Taxonomy_Change_Plot"] = pd.Categorical(
                            adata.obs["Taxonomy_Change_Plot"],
                            categories=categories_change,
                            ordered=True,
                        )

                        # Use vivid discrete colors (husl palette).
                        colors_change = sns.color_palette("husl", n_colors=len(sorted_taxa)).as_hex()
                        pal = {name: color for name, color in zip(sorted_taxa, colors_change)}
                        pal["Others"] = "#e0e0e0"
                        adata.uns["Taxonomy_Change_Plot_colors"] = [pal[cat] for cat in categories_change]

                        change_taxa_info = {
                            "top": top_change,
                            "sorted": sorted_taxa,
                            "start_group": start_group,
                            "end_group": end_group,
                            "metric": change_metric,
                            "level": analysis_level,
                        }

                        # Prepare taxonomy trajectory centroids in HC -> MIA -> IA order for Figure 4.
                        rows = []
                        umap_xy = adata.obsm["X_umap"]
                        obs_group = adata.obs[group_col].astype(str).values
                        obs_tax = adata.obs["Taxonomy_Change"].values
                        for tax in sorted_taxa:
                            for g in violin_order:  # Keep the HC -> MIA -> IA order.
                                mask = (obs_group == str(g)) & (obs_tax == tax)
                                if np.sum(mask) > 0:
                                    mu = np.mean(umap_xy[mask], axis=0)
                                    rows.append({"feature": tax, "group": str(g), "x": float(mu[0]), "y": float(mu[1])})
                        traj_centroids_df = pd.DataFrame(rows)

        else:
            # For metabolite, aggregate by Entity and ignore taxonomy entirely.
            df_pt = pd.DataFrame({
                "entity": adata.obs["Entity"].values,
                "group": adata.obs[group_col].astype(str).values,
                "pseudotime": adata.obs["dpt_pseudotime"].values.astype(float),
            })
            group_mean = df_pt.groupby(["entity", "group"])["pseudotime"].mean().reset_index()
            pivot = group_mean.pivot(index="entity", columns="group", values="pseudotime")

            if start_group not in pivot.columns or end_group not in pivot.columns:
                print(f"⚠ Figure 4 skipped: comparison groups {comparison_groups} are incomplete in the data.")
                plot_change = False
            else:
                # Compute only the HC -> IA change.
                if change_metric == "delta":
                    pivot["change"] = pivot[end_group] - pivot[start_group]
                elif change_metric == "abs_delta":
                    pivot["change"] = (pivot[end_group] - pivot[start_group]).abs()
                elif change_metric == "fold":
                    pivot["change"] = pivot[end_group] / (pivot[start_group] + 1e-10)
                else:
                    raise ValueError(f"Unknown change_metric: {change_metric}")

                pivot = pivot.dropna(subset=["change"])
                if pivot.shape[0] == 0:
                    print("⚠ Figure 4 skipped: no valid entity change entries were found.")
                    plot_change = False
                else:
                    # Select the top-changing entries.
                    if change_metric == "fold":
                        pivot["sort_key"] = (pivot["change"] - 1).abs()
                    else:
                        pivot["sort_key"] = pivot["change"].abs()
                    
                    top_change = pivot.nlargest(top_n_change, "sort_key")
                    top_list = top_change.index.tolist()
                    print(top_change)
                    adata.obs["Change_Plot"] = adata.obs["Entity"].apply(lambda x: x if x in top_list else "Others")
                    sorted_entities = top_change.sort_values("change", ascending=False).index.tolist()
                    categories_change = sorted_entities + ["Others"]
                    adata.obs["Change_Plot"] = pd.Categorical(
                        adata.obs["Change_Plot"],
                        categories=categories_change,
                        ordered=True,
                    )
                    
                    # Use vivid discrete colors (husl palette).
                    colors_change = sns.color_palette("husl", n_colors=len(sorted_entities)).as_hex()
                    pal = {name: color for name, color in zip(sorted_entities, colors_change)}
                    pal["Others"] = "#e0e0e0"
                    adata.uns["Change_Plot_colors"] = [pal[cat] for cat in categories_change]

                    change_entity_info = {
                        "top": top_change,
                        "sorted": sorted_entities,
                        "start_group": start_group,
                        "end_group": end_group,
                        "metric": change_metric,
                    }

                    # Prepare entity trajectory centroids in HC -> MIA -> IA order for Figure 4.
                    rows = []
                    umap_xy = adata.obsm["X_umap"]
                    obs_group = adata.obs[group_col].astype(str).values
                    obs_ent = adata.obs["Entity"].values
                    for ent in sorted_entities:
                        for g in violin_order:  # Keep the HC -> MIA -> IA order.
                            mask = (obs_group == str(g)) & (obs_ent == ent)
                            if np.sum(mask) > 0:
                                mu = np.mean(umap_xy[mask], axis=0)
                                rows.append({"feature": ent, "group": str(g), "x": float(mu[0]), "y": float(mu[1])})
                    traj_centroids_df = pd.DataFrame(rows)

    # ----------------------------
    # H) Legend helpers (microbe only, original logic preserved)
    # ----------------------------
    RANK_FULL_NAMES = {
        "k__": "Kingdom", "p__": "Phylum", "c__": "Class", "o__": "Order",
        "f__": "Family", "g__": "Genus", "s__": "Species"
    }

    def get_rank_full_name(prefix: str) -> str:
        return RANK_FULL_NAMES.get(prefix, prefix)

    def _pick_level_from_full_tax(full: str, prefer_prefix: str = None) -> str:
        if full is None:
            return ""
        full = str(full).strip()
        if not full or full == "Others":
            return "Others"
        parts = [p.strip() for p in full.split(";") if p.strip()]
        if not parts:
            return ""
        if prefer_prefix:
            for p in parts:
                if p.startswith(prefer_prefix):
                    return p
        return parts[-1]

    def _strip_rank_prefix(token: str) -> str:
        if token is None:
            return ""
        token = str(token).strip()
        for pref in ["k__", "p__", "c__", "o__", "f__", "g__", "s__"]:
            if token.startswith(pref):
                return token[len(pref):].strip()
        return token

    def _clean_tax_label(full_tax: str) -> str:
        token = _pick_level_from_full_tax(full_tax, prefer_prefix=tax_level)
        name = _strip_rank_prefix(token)
        if not name or name.lower() in {"unclassified", "unknown", "na", "nan"}:
            return "Others"
        return name.replace("_", " ").strip()

    def _should_italic(full_tax: str, current_level: str) -> bool:
        if full_tax == "Others":
            return False
        return current_level in ["s__", "g__"]

    # ----------------------------
    # I) Plotting
    # ----------------------------
    if plot:
        print("\nRendering plots...")

        draw_taxonomy_panel = bool(show_taxonomy_panel) and data_type == "microbe"
        draw_change_panel = bool(show_change_panel) and (
            (data_type == "microbe" and plot_change and change_taxa_info is not None)
            or (data_type != "microbe" and plot_change and change_entity_info is not None)
        )

        n_plots = 2 + int(draw_taxonomy_panel) + int(draw_change_panel)
        resolved_figsize = figsize if figsize is not None else (5.3 * n_plots, 4.2)

        fig, axs = plt.subplots(1, n_plots, figsize=resolved_figsize, dpi=500)
        if n_plots == 1:
            axs = [axs]
        else:
            axs = list(np.ravel(axs))

        if transparent:
            fig.patch.set_alpha(0.0)
            for ax in axs:
                ax.patch.set_alpha(0.0)

        trajectory_ax = axs[0]
        pseudotime_ax = axs[1]
        next_ax_idx = 2
        taxonomy_ax = None
        change_ax = None

        if draw_taxonomy_panel:
            taxonomy_ax = axs[next_ax_idx]
            next_ax_idx += 1
        if draw_change_panel:
            change_ax = axs[next_ax_idx]

        # Figure 1: trajectory (larger centroids, arrows avoid centroids)
        sc.pl.umap(
            adata, color=group_col, ax=trajectory_ax, show=False,
            alpha=0.5, legend_loc="on data", title="Trajectory Inference", frameon=False
        )

        # Centroid markers (larger size)
        centroid_size = 150
        for g in violin_order:
            if g in group_to_xy:
                trajectory_ax.scatter(
                    [group_to_xy[g][0]],
                    [group_to_xy[g][1]],
                    s=centroid_size,
                    c=default_map.get(g, "k"),
                    edgecolors="white",
                    linewidths=1.2,
                    zorder=13,
                )

        # Draw only arrows (HC -> MIA -> IA), shrinking them away from the centroids.
        _draw_group_flow_arrows(
            trajectory_ax,
            group_to_xy=group_to_xy,
            order=violin_order,
            color="black",
            lw=2.4,
            ms=18,
            z=14,
            shrink=18,
        )

        # Figure 2: pseudotime (unchanged)
        sc.pl.umap(
            adata, color="dpt_pseudotime", color_map=cmap, ax=pseudotime_ax, show=False,
            title=f"Pseudotime (Root: {root_group})", frameon=False
        )

        if data_type == "microbe":
            # Figure 3: most abundant taxa (original logic preserved)
            if draw_taxonomy_panel and taxonomy_ax is not None:
                rank_name = get_rank_full_name(tax_level)
                title3 = f"Top {top_n} Most Abundant {rank_name}"
                sc.pl.umap(
                    adata, color="Taxonomy_Plot", ax=taxonomy_ax, show=False, alpha=0.5,
                    title=title3, frameon=False
                )
                leg3 = taxonomy_ax.get_legend()
                if leg3 is not None:
                    for txt in leg3.get_texts():
                        old = txt.get_text()
                        txt.set_text(_clean_tax_label(old))
                        txt.set_fontstyle("italic" if _should_italic(old, tax_level) else "normal")
                    leg3.set_bbox_to_anchor((1.0, 0.5))
                    leg3._loc = 6
                    leg3.set_frame_on(False)

            # Figure 4: largest pseudotime changes with brighter colors and HC -> MIA -> IA arrows.
            if draw_change_panel and change_ax is not None and plot_change and change_taxa_info is not None:
                change_level = change_taxa_info["level"]
                change_rank_name = get_rank_full_name(change_level)
                if change_metric == "fold":
                    change_word = "Fold Change"
                elif change_metric == "abs_delta":
                    change_word = "Absolute Change"
                else:
                    change_word = "Pseudotime Change"
                title4 = f"Top {top_n_change} {change_rank_name} by {change_word}"

                # Draw "Others" first as the background layer.
                mask_others = (adata.obs["Taxonomy_Change_Plot"] == "Others").values
                if np.sum(mask_others) > 0:
                    change_ax.scatter(
                        adata.obsm["X_umap"][mask_others, 0],
                        adata.obsm["X_umap"][mask_others, 1],
                        c="#e0e0e0",
                        s=10,
                        alpha=0.3,
                        zorder=1,
                    )

                # Draw the top entries on top with alpha=0.9.
                sorted_taxa = change_taxa_info["sorted"]
                color_map = {cat: adata.uns["Taxonomy_Change_Plot_colors"][i] 
                            for i, cat in enumerate(adata.obs["Taxonomy_Change_Plot"].cat.categories)}
                
                for tax in sorted_taxa:
                    mask_tax = (adata.obs["Taxonomy_Change_Plot"] == tax).values
                    if np.sum(mask_tax) > 0:
                        change_ax.scatter(
                            adata.obsm["X_umap"][mask_tax, 0],
                            adata.obsm["X_umap"][mask_tax, 1],
                            c=color_map.get(tax, "#000000"),
                            s=15,
                            alpha=0.9,  # Higher opacity for highlighted points.
                            zorder=10,  # Keep highlighted points on top.
                            label=tax,
                        )

                change_ax.set_title(title4)
                change_ax.axis("off")

                # Draw trajectory arrows in HC -> MIA -> IA order without covering the points.
                if traj_centroids_df is not None and traj_centroids_df.shape[0] > 0:
                    _draw_feature_trajectories(
                        change_ax,
                        df_centroids=traj_centroids_df,
                        order=violin_order,  # Keep the HC -> MIA -> IA order.
                        color="black",
                        lw=1.0,
                        ms=8,
                        alpha=0.35,
                        shrink=10,
                    )

                # Manually build the legend with circular markers.
                from matplotlib.lines import Line2D
                legend_elements = []
                top_tbl = change_taxa_info["top"]
                for tax in sorted_taxa:
                    new = _strip_rank_prefix(tax).replace("_", " ").strip()
                    if tax in top_tbl.index:
                        v = float(top_tbl.loc[tax, "change"])
                        label = f"{new} ({v:.2f}x)" if change_metric == "fold" else f"{new} ({v:+.3f})"
                    else:
                        label = new
                    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                                 markerfacecolor=color_map.get(tax, "#000000"), 
                                                 markersize=8, label=label))

                
                leg4 = change_ax.legend(
                    handles=legend_elements,
                    bbox_to_anchor=(1.0, 0.5),
                    loc="center left",
                    frameon=False,
                    title=f"Δ Pseudotime: {change_taxa_info['end_group']}-{change_taxa_info['start_group']}"
                        if change_metric != "fold"
                        else f"Change: {change_taxa_info['start_group']}→{change_taxa_info['end_group']} (fold)"
                )
                leg4.get_title().set_fontweight("bold")
                leg4.get_title().set_fontsize(9)

        else:
            # Metabolite: draw only 1/2/4, since Figure 4 is taxonomy-independent.
            if draw_change_panel and change_ax is not None and plot_change and change_entity_info is not None:
                if change_metric == "fold":
                    change_word = "Fold Change"
                elif change_metric == "abs_delta":
                    change_word = "Absolute Change"
                else:
                    change_word = "Pseudotime Change"

                title4 = f"Top {top_n_change} Metabolites by {change_word}"

                # Draw "Others" first as the background layer.
                mask_others = (adata.obs["Change_Plot"] == "Others").values
                if np.sum(mask_others) > 0:
                    change_ax.scatter(
                        adata.obsm["X_umap"][mask_others, 0],
                        adata.obsm["X_umap"][mask_others, 1],
                        c="#e0e0e0",
                        s=10,
                        alpha=0.3,
                        zorder=1,
                    )

                # Draw the top entries on top with alpha=0.9.
                sorted_entities = change_entity_info["sorted"]
                color_map = {cat: adata.uns["Change_Plot_colors"][i] 
                            for i, cat in enumerate(adata.obs["Change_Plot"].cat.categories)}
                
                for ent in sorted_entities:
                    mask_ent = (adata.obs["Change_Plot"] == ent).values
                    if np.sum(mask_ent) > 0:
                        change_ax.scatter(
                            adata.obsm["X_umap"][mask_ent, 0],
                            adata.obsm["X_umap"][mask_ent, 1],
                            c=color_map.get(ent, "#000000"),
                            s=15,
                            alpha=0.9,  # Higher opacity for highlighted points.
                            zorder=10,  # Keep highlighted points on top.
                            label=ent,
                        )

                change_ax.set_title(title4)
                change_ax.axis("off")

                # Draw trajectory arrows in HC -> MIA -> IA order without covering the points.
                if traj_centroids_df is not None and traj_centroids_df.shape[0] > 0:
                    _draw_feature_trajectories(
                        change_ax,
                        df_centroids=traj_centroids_df,
                        order=violin_order,  # Keep the HC -> MIA -> IA order.
                        color="black",
                        lw=1.0,
                        ms=8,
                        alpha=0.35,
                        shrink=10,
                    )

                # Manually build the legend with circular markers.
                from matplotlib.lines import Line2D
                legend_elements = []
                top_tbl = change_entity_info["top"]
                for ent in sorted_entities:
                    clean = str(ent).replace("_", " ").strip()
                    if ent in top_tbl.index:
                        v = float(top_tbl.loc[ent, "change"])
                        label = f"{clean} ({v:.2f}x)" if change_metric == "fold" else f"{clean} ({v:+.3f})"
                    else:
                        label = clean
                    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                                 markerfacecolor=color_map.get(ent, "#000000"), 
                                                 markersize=8, label=label))

                
                leg = change_ax.legend(
                    handles=legend_elements,
                    bbox_to_anchor=(1.0, 0.5),
                    loc="center left",
                    frameon=False,
                    title=f"Δ Pseudotime: {change_entity_info['end_group']}-{change_entity_info['start_group']}"
                        if change_metric != "fold"
                        else f"Change: {change_entity_info['start_group']}→{change_entity_info['end_group']} (fold)"
                )
                leg.get_title().set_fontweight("bold")
                leg.get_title().set_fontsize(9)

        plt.tight_layout()
        if savepath:
            plt.savefig(savepath)
        plt.show()

    print(f"\n{'='*60}")
    print("✓ Analysis completed.")
    print(f"{'='*60}\n")
    # Return pivot only when it exists; otherwise return None.
    return_pivot = pivot if 'pivot' in locals() else None
    return adata, return_pivot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

def plot_pseudotime_violin_and_ecdf(
    adata,
    group_col="group",
    pseudotime_key="dpt_pseudotime",
    group_order=("HC", "MIA", "IA"),
    group_colors=('#6B98C4', '#FFBC80', '#F5867F'),
    dpi=500,
    figsize=(12, 4),
    # ---- Match the reference style: opaque by default + full frame + framed legend + grid ----
    transparent=False,
    ecdf_linewidth=2.0,
    median_vline=True,
    median_vline_alpha=0.35,
    legend_loc="lower right",
    legend_fontsize=9,
    grid_alpha=0.2,
    savepath=None,
):
    """
    Left: violin plot (Scanpy). Right: ECDF (Matplotlib).
    Style aligned with the reference function:
      - Opaque background by default (transparent=False)
      - Keep all spines visible
      - Framed legend at lower right with fontsize=9
      - grid(alpha=0.2)
      - tight_layout + savefig
    """

    # --- Explicit color mapping driven only by group_order ---
    color_map = {g: c for g, c in zip(group_order, group_colors)}

    # --- 1) Explicitly fetch columns from obs ---
    if group_col not in adata.obs.columns:
        raise KeyError(f"Column '{group_col}' is missing from adata.obs.")
    if pseudotime_key not in adata.obs.columns:
        raise KeyError(f"Column '{pseudotime_key}' is missing from adata.obs.")

    dfp = adata.obs.loc[:, [group_col, pseudotime_key]].copy()
    dfp[group_col] = dfp[group_col].astype(str)

    # Clean pseudotime values.
    dfp[pseudotime_key] = pd.to_numeric(dfp[pseudotime_key], errors="coerce")
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna(subset=[group_col, pseudotime_key])

    # --- 2) Keep only groups that are present, in the exact group_order order ---
    present_groups = [g for g in group_order if g in set(dfp[group_col].values)]
    if len(present_groups) == 0:
        raise ValueError("None of the groups specified in group_order are present in dfp; plotting is not possible.")

    # --- 3) Build a temporary view for the Scanpy violin plot ---
    mask = adata.obs[group_col].astype(str).isin(present_groups)
    ad_view = adata[mask].copy()

    ad_view.obs[group_col] = pd.Categorical(
        ad_view.obs[group_col].astype(str),
        categories=present_groups,
        ordered=True
    )
    ad_view.uns[f"{group_col}_colors"] = [color_map[g] for g in present_groups]

    # --- 4) Plot ---
    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    # Reference style: opaque by default; only make it transparent when requested.
    if transparent:
        fig.patch.set_alpha(0.0)
        for ax in axs:
            ax.patch.set_alpha(0.0)

    # A) Violin
    sc.pl.violin(
        ad_view,
        keys=[pseudotime_key],
        groupby=group_col,
        order=present_groups,
        ax=axs[0],
        show=False
    )
    axs[0].set_title("Pseudotime distribution (violin)")
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Pseudotime")
    axs[0].grid(alpha=grid_alpha)

    # B) ECDF
    for g in present_groups:
        vals = dfp.loc[dfp[group_col] == g, pseudotime_key].to_numpy(dtype=float)
        vals = np.sort(vals)
        if vals.size == 0:
            continue

        y = np.arange(1, vals.size + 1) / vals.size
        axs[1].plot(
            vals, y,
            label=g,
            linewidth=ecdf_linewidth,
            color=color_map.get(g, "#808080")
        )

        if median_vline:
            med = float(np.median(vals))
            axs[1].axvline(
                med,
                linestyle="--",
                alpha=median_vline_alpha,
                linewidth=1.5,
                color=color_map.get(g, "#808080")
            )

    axs[1].set_title("Pseudotime distribution (ECDF)")
    axs[1].set_xlabel("Pseudotime")
    axs[1].set_ylabel("Cumulative proportion")
    axs[1].grid(alpha=grid_alpha)

    # Reference style: framed legend, lower right, fontsize=9.
    axs[1].legend(loc=legend_loc, fontsize=legend_fontsize,frameon=True)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def map_test_to_train_within_group_knn(
    adata_test,
    adata_train,
    group_col="group",
    rep="X_pca",
    k=15,
    sigma=None,
    map_umap=True,
    eps=1e-12,
    # ========== Additional plotting parameters ==========
    plot=True,
    plot_type="both",  # "umap" | "violin" | "both"
    violin_order=("HC", "MIA", "IA"),
    group_colors=("#6B98C4", "#FFBC80", "#F5867F"),
    cmap="Reds",
    figsize_umap=(24, 4),
    figsize_violin=(12, 4),
    dpi=500,
    transparent=True,
    savepath_umap=None,
    savepath_violin=None,
):
    """
    Within-group KNN mapping with automatic plotting, matching the training-set format.
    
    Features:
    1. Within-group KNN mapping: each test point searches neighbors only within
       the same group in the training set.
    2. Pseudotime mapping: weighted average of train dpt_pseudotime values.
    3. UMAP mapping: weighted average of train X_umap coordinates, preserving
       the training coordinate system.
    4. Automatic plotting: generates the same figure style as the training set
       (multi-panel UMAP + violin/ECDF).
    
    Example:
    ```python
    # Training set
    adata_train, _ = run_trajectory_analysis_ultimate(
        df_train, full_name_dic, 
        plot=True, 
        savepath="train_umap.png"
    )
    plot_pseudotime_violin_and_ecdf(adata_train, savepath="train_violin.png")
    
    # Test set: one call for mapping + plotting
    adata_test = map_test_to_train_within_group_knn(
        adata_test, adata_train,
        plot=True,
        savepath_umap="test_umap.png",
        savepath_violin="test_violin.png"
    )
    ```
    """
    
    # ========================================
    # Step 1: within-group KNN mapping (original logic preserved)
    # ========================================
    print("\n" + "="*60)
    print("Within-group KNN mapping")
    print("="*60)
    
    if "dpt_pseudotime" not in adata_train.obs:
        raise ValueError("adata_train is missing obs['dpt_pseudotime'].")
    if map_umap and ("X_umap" not in adata_train.obsm):
        raise ValueError("adata_train is missing obsm['X_umap'].")

    g_train = adata_train.obs[group_col].astype(str).values
    g_test  = adata_test.obs[group_col].astype(str).values

    if rep == "X_pca":
        if "X_pca" not in adata_train.obsm:
            raise ValueError("rep='X_pca' was requested, but adata_train.obsm['X_pca'] is missing.")
        if "X_pca" not in adata_test.obsm:
            raise ValueError("rep='X_pca' was requested, but adata_test.obsm['X_pca'] is missing.")
        Xtr_all = np.asarray(adata_train.obsm["X_pca"], dtype=float)
        Xte_all = np.asarray(adata_test.obsm["X_pca"], dtype=float)
    elif rep == "X":
        Xtr_all = np.asarray(adata_train.X, dtype=float)
        Xte_all = np.asarray(adata_test.X, dtype=float)
    else:
        raise ValueError("rep must be either 'X_pca' or 'X'.")

    pt_tr_all = np.asarray(adata_train.obs["dpt_pseudotime"].astype(float).values)
    if map_umap:
        um_tr_all = np.asarray(adata_train.obsm["X_umap"], dtype=float)

    pt_out = np.full(adata_test.n_obs, np.nan, dtype=float)
    um_out = np.full((adata_test.n_obs, 2), np.nan, dtype=float) if map_umap else None

    train_groups = np.unique(g_train)
    test_groups  = np.unique(g_test)
    missing_in_train = [g for g in test_groups if g not in set(train_groups)]
    if missing_in_train:
        raise ValueError(f"Test data contains groups that are absent from train data: {missing_in_train}")

    per_group_stats = {}

    for g in train_groups:
        te_idx = np.where(g_test == g)[0]
        if te_idx.size == 0:
            continue

        tr_idx = np.where(g_train == g)[0]
        if tr_idx.size == 0:
            raise ValueError(f"Training group {g} is empty.")

        Xtr = Xtr_all[tr_idx]
        Xte = Xte_all[te_idx]

        k_use = int(min(k, tr_idx.size))
        nn = NearestNeighbors(n_neighbors=k_use, metric="euclidean")
        nn.fit(Xtr)
        dists, nbr_local = nn.kneighbors(Xte, return_distance=True)

        if sigma is None:
            nonzero = dists[dists > 0]
            sig = float(np.median(nonzero)) if nonzero.size else 1.0
            if (not np.isfinite(sig)) or sig <= 0:
                sig = 1.0
        else:
            sig = float(sigma)

        w = np.exp(-(dists ** 2) / (2.0 * (sig ** 2)))
        w = w / (w.sum(axis=1, keepdims=True) + eps)

        nbr_global = tr_idx[nbr_local]

        pt_out[te_idx] = (w * pt_tr_all[nbr_global]).sum(axis=1)

        if map_umap:
            um_out[te_idx] = (w[:, :, None] * um_tr_all[nbr_global]).sum(axis=1)

        per_group_stats[g] = {
            "n_test": int(te_idx.size),
            "n_train": int(tr_idx.size),
            "k": int(k_use),
            "sigma": float(sig),
            "d1_median": float(np.median(dists[:, 0])) if dists.size else np.nan,
            "d1_p90": float(np.quantile(dists[:, 0], 0.90)) if dists.size else np.nan,
        }

    adata_test.obs["dpt_pseudotime"] = pt_out.astype(float)
    if map_umap:
        adata_test.obsm["X_umap"] = um_out.astype(np.float32)

    adata_test.uns["map_within_group_knn"] = {
        "rep": rep,
        "k": int(k),
        "sigma": None if sigma is None else float(sigma),
        "per_group": per_group_stats,
    }

    # Print per-group mapping statistics.
    print("\nWithin-group mapping statistics:")
    for g, info in per_group_stats.items():
        print(f"  {g}: test={info['n_test']}, train={info['n_train']}, "
              f"k={info['k']}, sigma={info['sigma']:.3f}, "
              f"d1_median={info['d1_median']:.3f}")

    # ========================================
    # Step 2: plotting (match the training-set format)
    # ========================================
    if plot:
        print("\n" + "="*60)
        print("Rendering test-set plots")
        print("="*60)
        
        # Keep group colors consistent with the training plots.
        default_map = {"HC": group_colors[0], "MIA": group_colors[1], "IA": group_colors[2]}
        adata_test.uns[f"{group_col}_colors"] = [default_map.get(g, "#808080") for g in violin_order]
        
        # --- Figure 1: multi-panel UMAP ---
        if plot_type in ["umap", "both"]:
            # Check required data.
            has_taxonomy = "Taxonomy_Plot" in adata_test.obs or "Taxonomy_Raw" in adata_test.obs
            
            # Decide the number of subplots using the same rule as the training plots.
            n_plots = 4 if has_taxonomy else 3
            
            fig, axs = plt.subplots(1, n_plots, figsize=figsize_umap, dpi=dpi)
            if n_plots == 1:
                axs = [axs]
            
            if transparent:
                fig.patch.set_alpha(0.0)
                for ax in axs:
                    ax.patch.set_alpha(0.0)
            
            # Panel 1: group labels
            sc.pl.umap(
                adata_test, color=group_col, ax=axs[0], show=False,
                alpha=0.5, legend_loc="on data", 
                title="Test Set - Trajectory Inference", 
                frameon=False
            )
            
            # Add group centroids using the same style as the training plots.
            group_to_xy = {}
            for g in violin_order:
                mask = (adata_test.obs[group_col].astype(str) == str(g)).values
                if np.sum(mask) > 0:
                    mu = np.mean(adata_test.obsm["X_umap"][mask], axis=0)
                    group_to_xy[g] = (float(mu[0]), float(mu[1]))
                    # Draw the centroid marker.
                    axs[0].scatter(
                        [mu[0]], [mu[1]],
                        s=150,
                        c=default_map.get(g, "k"),
                        edgecolors="white",
                        linewidths=1.2,
                        zorder=13,
                    )
            
            # Draw arrows in HC -> MIA -> IA order.
            from matplotlib.patches import FancyArrowPatch
            for a, b in zip(violin_order[:-1], violin_order[1:]):
                if a in group_to_xy and b in group_to_xy:
                    arrow = FancyArrowPatch(
                        posA=group_to_xy[a],
                        posB=group_to_xy[b],
                        arrowstyle="-|>",
                        mutation_scale=18,
                        lw=2.4,
                        color="black",
                        alpha=0.95,
                        shrinkA=18,
                        shrinkB=18,
                        zorder=14,
                    )
                    axs[0].add_patch(arrow)
            
            # Panel 2: pseudotime
            sc.pl.umap(
                adata_test, color="dpt_pseudotime", color_map=cmap, 
                ax=axs[1], show=False,
                title="Test Set - Pseudotime (Mapped)", 
                frameon=False
            )
            
            # Panel 3: taxonomy, when available
            if has_taxonomy:
                tax_col = "Taxonomy_Plot" if "Taxonomy_Plot" in adata_test.obs else "Taxonomy_Raw"
                sc.pl.umap(
                    adata_test, color=tax_col, ax=axs[2], show=False, 
                    alpha=0.5,
                    title="Test Set - Taxonomy", 
                    frameon=False
                )
            
            # Panel 4: validation view
            ax_idx = 3 if has_taxonomy else 2
            sc.pl.umap(
                adata_test, color="dpt_pseudotime", 
                ax=axs[ax_idx], show=False,
                title="Test Set - Validation", 
                frameon=False
            )
            
            plt.tight_layout()
            if savepath_umap:
                plt.savefig(savepath_umap, dpi=dpi, transparent=transparent)
                print(f"✓ UMAP figure saved to: {savepath_umap}")
            plt.show()
        
        # --- Figure 2: violin + ECDF ---
        if plot_type in ["violin", "both"]:
            _plot_pseudotime_violin_and_ecdf(
                adata_test,
                group_col=group_col,
                violin_order=violin_order,
                group_colors=group_colors,
                dpi=dpi,
                figsize=figsize_violin,
                transparent=transparent,
                savepath=savepath_violin,
            )

    print(f"\n{'='*60}")
    print("✓ Test-set mapping completed.")
    print(f"{'='*60}\n")

    return adata_test


# ==========================================
# Internal helper: violin + ECDF plotting
# ==========================================
def _plot_pseudotime_violin_and_ecdf(
    adata,
    group_col="group",
    pseudotime_key="dpt_pseudotime",
    violin_order=("HC", "MIA", "IA"),
    group_colors=('#6B98C4', '#FFBC80', '#F5867F'),
    dpi=500,
    figsize=(12, 4),
    transparent=True,
    ecdf_linewidth=2.0,
    median_vline=True,
    median_vline_alpha=0.35,
    savepath=None,
):
    """
    Violin + ECDF with styling aligned to the reference function:
      - Keep all spines visible
      - Framed legend at lower right with fontsize=9
      - grid(alpha=0.2)
      - tight_layout + savefig
      - Keep ordering/category/color logic unchanged
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scanpy as sc

    color_map = {g: c for g, c in zip(violin_order, group_colors)}

    # --- Prepare data ---
    if group_col not in adata.obs.columns:
        raise KeyError(f"Column '{group_col}' is missing from adata.obs.")
    if pseudotime_key not in adata.obs.columns:
        raise KeyError(f"Column '{pseudotime_key}' is missing from adata.obs.")

    dfp = adata.obs.loc[:, [group_col, pseudotime_key]].copy()
    dfp[group_col] = dfp[group_col].astype(str)
    dfp[pseudotime_key] = pd.to_numeric(dfp[pseudotime_key], errors="coerce")
    dfp = dfp.replace([np.inf, -np.inf], np.nan).dropna(subset=[group_col, pseudotime_key])

    present_groups = [g for g in violin_order if g in set(dfp[group_col].values)]
    if len(present_groups) == 0:
        raise ValueError("None of the groups specified in violin_order are present in dfp.")

    # --- Stable Scanpy view with explicit group ordering ---
    mask = adata.obs[group_col].astype(str).isin(present_groups)
    ad_view = adata[mask].copy()
    ad_view.obs[group_col] = pd.Categorical(
        ad_view.obs[group_col].astype(str),
        categories=present_groups,
        ordered=True
    )
    ad_view.uns[f"{group_col}_colors"] = [color_map[g] for g in present_groups]

    # --- Plot ---
    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    if transparent:
        fig.patch.set_alpha(0.0)
        for ax in axs:
            ax.patch.set_alpha(0.0)

    # A) Violin
    sc.pl.violin(
        ad_view,
        keys=[pseudotime_key],
        groupby=group_col,
        order=present_groups,
        ax=axs[0],
        show=False
    )
    axs[0].set_title("Pseudotime distribution (violin)")
    axs[0].set_xlabel("")
    axs[0].set_ylabel("Pseudotime")
    axs[0].grid(alpha=0.2)

    # B) ECDF
    for g in present_groups:
        vals = dfp.loc[dfp[group_col] == g, pseudotime_key].to_numpy(dtype=float)
        vals = np.sort(vals)
        if vals.size == 0:
            continue

        y = np.arange(1, vals.size + 1) / vals.size
        axs[1].plot(
            vals, y,
            label=g,
            linewidth=ecdf_linewidth,
            color=color_map.get(g, "#808080")
        )

        if median_vline:
            med = float(np.median(vals))
            axs[1].axvline(
                med,
                linestyle="--",
                alpha=median_vline_alpha,
                linewidth=1.5,
                color=color_map.get(g, "#808080")
            )

    axs[1].set_title("Pseudotime distribution (ECDF)")
    axs[1].set_xlabel("Pseudotime")
    axs[1].set_ylabel("Cumulative proportion")
    axs[1].legend(loc="lower right", fontsize=9,frameon=True)  # Keep the legend frame enabled.
    axs[1].grid(alpha=0.2)

    # Keep the full frame visible to match the reference style.
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, transparent=transparent)
        print(f"✓ Violin figure saved to: {savepath}")
    plt.show()
