#!/usr/bin/env bash
# Sweep Swin ADMM warm-started from direct (10 images).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BITRATE_SR_CONFIG="$SCRIPT_DIR/bitrate_sr_config_swin_admm_wd_10.sh"
exec bash "$SCRIPT_DIR/run_sweep.sh"
