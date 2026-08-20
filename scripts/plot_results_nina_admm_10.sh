#!/usr/bin/env bash
# Plot results for nina_admm_10.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BITRATE_SR_CONFIG="$SCRIPT_DIR/bitrate_sr_config_nina_admm_10.sh"
exec bash "$SCRIPT_DIR/plot_results.sh"
