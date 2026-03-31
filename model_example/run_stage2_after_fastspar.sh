#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/04_run_flurico_all.sh"
bash "$SCRIPT_DIR/05_run_models_lgbm_all.sh"

echo "[DONE] Stage-2 complete (flurico + lgbm modeling)."
