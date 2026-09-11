#!/usr/bin/env bash
# Sweep Swin old-notebook ADMM (target-PSNR ladder, 10 images).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BITRATE_SR_CONFIG="$SCRIPT_DIR/bitrate_sr_config_swin_admm_old_10.sh"
exec bash "$SCRIPT_DIR/run_sweep.sh"
