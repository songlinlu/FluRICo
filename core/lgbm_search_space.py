from __future__ import annotations

from typing import Dict, List

# Only keep two profile names as requested.
# - lgbm_grid: LUAD default grid.
# - lgbm_grid_small_sample: LC/NSCLC default grid.
LGBM_PROFILE_GRIDS: Dict[str, Dict[str, List[float | int]]] = {
    "lgbm_grid": {
        "n_estimators": [200, 300, 500],
        "max_depth": [5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [15, 31, 50],
        "min_child_samples": [20, 30],
        "colsample_bytree": [1.0],
        "subsample": [1.0],
        "reg_alpha": [0.0],
        "reg_lambda": [0.0],
    },
    "lgbm_grid_small_sample": {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.02, 0.05, 0.1],
        "num_leaves": [4, 7],
        "min_child_samples": [3, 5, 10],
        "colsample_bytree": [1.0],
        "subsample": [1.0],
        "reg_alpha": [0.0],
        "reg_lambda": [0.0],
    },
}


DATASET_LGBM_PROFILE_CONFIG: Dict[str, Dict[str, str | List[str]]] = {
    "LUAD": {
        "default_profile": "lgbm_grid",
        "recommended_profile": "lgbm_grid",
        "available_profiles": ["lgbm_grid"],
    },
    "LC": {
        "default_profile": "lgbm_grid_small_sample",
        "recommended_profile": "lgbm_grid_small_sample",
        "available_profiles": ["lgbm_grid_small_sample"],
    },
    "NSCLC": {
        "default_profile": "lgbm_grid_small_sample",
        "recommended_profile": "lgbm_grid_small_sample",
        "available_profiles": ["lgbm_grid_small_sample"],
    },
}


def lgbm_grid_by_profile(profile: str) -> Dict[str, List[float | int]]:
    if profile not in LGBM_PROFILE_GRIDS:
        supported = ", ".join(sorted(LGBM_PROFILE_GRIDS))
        raise ValueError(f"Unsupported lgbm_profile='{profile}'. Supported: {supported}")
    return {key: list(values) for key, values in LGBM_PROFILE_GRIDS[profile].items()}


def default_lgbm_profile_for_dataset(dataset_key: str) -> str:
    token = str(dataset_key).strip().upper()
    if token not in DATASET_LGBM_PROFILE_CONFIG:
        raise ValueError(
            f"Unsupported dataset_key='{dataset_key}'. Supported keys: {sorted(DATASET_LGBM_PROFILE_CONFIG)}"
        )
    return str(DATASET_LGBM_PROFILE_CONFIG[token]["default_profile"])
