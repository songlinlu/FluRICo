import numpy as np
import pandas as pd
from skbio.stats.distance import mantel
import warnings
warnings.filterwarnings('ignore')


def co_shift_bootstrap_distance(df_target, check_ko_list, corr_ref_full, n_boot=100, seed=42):
    rng = np.random.default_rng(seed)
    
    # 1. Apply CLR to the full dataset first
    #    (to ensure the geometric mean is computed based on all features)
    df_full = df_target.T  # convert to sample × full-feature matrix
    df_full_vals = df_full.values + 1e-12
    
    try:
        from skbio.stats.composition import clr
        full_clr_vals = clr(df_full_vals)
    except ImportError:
        gmean = np.exp(np.mean(np.log(df_full_vals), axis=1, keepdims=True))
        full_clr_vals = np.log(df_full_vals / gmean)
    
    df_full_clr = pd.DataFrame(full_clr_vals, 
                                index=df_full.index, 
                                columns=df_full.columns)
    
    # 2. Perform feature matching and subset extraction
    hit_ko_list = [ko for ko in check_ko_list if ko in df_full_clr.columns]
    
    if len(hit_ko_list) < 3:
        return np.nan, np.nan, np.nan, np.nan, len(hit_ko_list)
    
    df_check_clr = df_full_clr[hit_ko_list]
    
    # 3. Prepare the reference distance matrix
    corr_ref_sub = corr_ref_full.loc[hit_ko_list, hit_ko_list]
    dist_ref = 1 - corr_ref_sub
    
    # --- Part A: Observed value ---
    obs_corr = df_check_clr.corr(method='spearman')
    obs_dist = 1 - obs_corr
    
    obs_r, obs_p, _ = mantel(obs_dist, dist_ref, method='spearman', permutations=999)
    
    # --- Part B: Bootstrap ---
    boot_rs = []
    n_samples = df_check_clr.shape[0]
    
    for _ in range(n_boot):
        resampled_indices = rng.choice(n_samples, size=n_samples, replace=True)
        df_resampled = df_check_clr.iloc[resampled_indices]
        
        boot_corr = df_resampled.corr(method='spearman')
        boot_dist = 1 - boot_corr
        
        r_val, _, _ = mantel(boot_dist, dist_ref, method='spearman', permutations=0)
        boot_rs.append(r_val)
    
    boot_mean = np.mean(boot_rs)
    ci_low = np.percentile(boot_rs, 2.5)
    ci_high = np.percentile(boot_rs, 97.5)
    
    return obs_r, boot_mean, ci_low, ci_high, len(hit_ko_list)