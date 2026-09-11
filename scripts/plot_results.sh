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
# Tags: *_admm_wd (warm-start), *_admm_old (old-notebook ladder); PLOT_METHOD overrides
if [[ -n "${PLOT_METHOD:-}" ]]; then
  PLOT_FILTER="$PLOT_METHOD"
elif [[ "$METHOD" == "admm_oldlike" ]]; then
  PLOT_FILTER="admm_old"
elif [[ "$METHOD" == "admm" && "${ADMM_INIT:-pretrained}" == "direct" ]]; then
  PLOT_FILTER="admm_wd"
else
  PLOT_FILTER="$METHOD"
fi

# Avoid overwriting ADMM rd.png when plotting direct (and vice versa)
if [[ "$PLOT_SAVE" == *"/rd.png" ]]; then
  PLOT_SAVE="${PLOT_SAVE%.png}_${PLOT_FILTER}.png"
fi

ARGS=(
  --runs-dir "$RUNS_DIR"
  --save "$PLOT_SAVE"
  --method "$PLOT_FILTER"
  --group-by "${PLOT_GROUP_BY:-auto}"
)
if [[ "$PLOT_SHOW" == "1" ]]; then
  ARGS+=(--show)
fi

echo "RUNS_DIR=$RUNS_DIR METHOD=$METHOD LORA_TARGET=${LORA_TARGET:-all}"
echo "PLOT_SAVE=$PLOT_SAVE"
python -m bitrate_sr.plot "${ARGS[@]}"
