#!/usr/bin/env bash
# Reads RUNS_DIR / PLOT_SAVE from scripts/bitrate_sr_config.sh — edit that file, then:
#   bash scripts/plot_results.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "${BITRATE_SR_CONFIG:-$SCRIPT_DIR/bitrate_sr_config.sh}"

PLOT_SHOW="${PLOT_SHOW:-0}"
METHOD="${METHOD:-admm}"

# Avoid overwriting ADMM rd.png when plotting direct (and vice versa)
if [[ "$METHOD" == "direct" && "$PLOT_SAVE" == *"/rd.png" ]]; then
  PLOT_SAVE="${PLOT_SAVE%.png}_${METHOD}.png"
fi

ARGS=(
  --runs-dir "$RUNS_DIR"
  --save "$PLOT_SAVE"
  --method "$METHOD"
)
if [[ "$PLOT_SHOW" == "1" ]]; then
  ARGS+=(--show)
fi

echo "RUNS_DIR=$RUNS_DIR METHOD=$METHOD LORA_TARGET=${LORA_TARGET:-all}"
echo "PLOT_SAVE=$PLOT_SAVE"
python -m bitrate_sr.plot "${ARGS[@]}"
