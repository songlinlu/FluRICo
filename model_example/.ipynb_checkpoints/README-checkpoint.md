# model training scripts

Step-wise training scripts:

1. `01_split_multiround.sh`
2. `02_build_task_views.sh`
3. `03_run_fastspar_all.sh`--require Fastspar, see details in https://github.com/scwatts/fastspar 
4. `04_run_flurico_all.sh`
5. `05_run_models_lgbm_all.sh`
6. `plot_bars_lgbm_valmax_testtieavg.sh`

Convenience wrappers:

- `run_stage1_prepare.sh` = step 01 + 02
- `run_stage2_after_fastspar.sh` = step 04 + 05


