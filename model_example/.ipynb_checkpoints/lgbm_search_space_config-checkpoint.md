# LGBM Search Space Config

Search spaces are centralized in:

- `../core/lgbm_search_space.py`

Profiles (aligned with the historical utils profile runner):

- `lgbm_grid`
- `lgbm_grid_small_sample`

Dataset-level visible config:

- `LUAD`: default `lgbm_grid`
- `LC`: default `lgbm_grid_small_sample`
- `NSCLC`: default `lgbm_grid_small_sample`
