#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/01_split_multiround.sh"
bash "$SCRIPT_DIR/02_build_task_views.sh"

echo "[DONE] Stage-1 complete (split + task views)."
