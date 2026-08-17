#!/usr/bin/env bash
# Sweep Swin2SR + LoRA on all DIV2K train HR (see bitrate_sr_config_swin.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BITRATE_SR_CONFIG="$SCRIPT_DIR/bitrate_sr_config_swin.sh"
exec bash "$SCRIPT_DIR/run_sweep.sh"
