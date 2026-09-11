#!/usr/bin/env bash
# Plot Swin old-notebook ADMM (admm_old) 10-image results; curve vs target PSNR.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BITRATE_SR_CONFIG="$SCRIPT_DIR/bitrate_sr_config_swin_admm_old_10.sh"
exec bash "$SCRIPT_DIR/plot_results.sh"
