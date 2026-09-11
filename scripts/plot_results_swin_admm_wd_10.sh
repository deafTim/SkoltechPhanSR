#!/usr/bin/env bash
# Plot Swin ADMM warm-start (admm_wd) 10-image results.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BITRATE_SR_CONFIG="$SCRIPT_DIR/bitrate_sr_config_swin_admm_wd_10.sh"
exec bash "$SCRIPT_DIR/plot_results.sh"
