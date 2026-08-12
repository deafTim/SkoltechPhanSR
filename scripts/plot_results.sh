#!/usr/bin/env bash
# Reads RUNS_DIR / PLOT_SAVE from scripts/bitrate_sr_config.sh — edit that file, then:
#   bash scripts/plot_results.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/bitrate_sr_config.sh"

PLOT_SHOW="${PLOT_SHOW:-0}"

ARGS=(
  --runs-dir "$RUNS_DIR"
  --save "$PLOT_SAVE"
)
if [[ "$PLOT_SHOW" == "1" ]]; then
  ARGS+=(--show)
fi

echo "RUNS_DIR=$RUNS_DIR"
echo "PLOT_SAVE=$PLOT_SAVE"
python -m bitrate_sr.plot "${ARGS[@]}"
