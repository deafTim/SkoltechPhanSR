#!/usr/bin/env bash
# Sweep swin_direct_10 (see bitrate_sr_config_swin_direct_10.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BITRATE_SR_CONFIG="$SCRIPT_DIR/bitrate_sr_config_swin_direct_10.sh"
exec bash "$SCRIPT_DIR/run_sweep.sh"
