#!/usr/bin/env bash
# Sweep with attention-expand LoRA only (see bitrate_sr_config_attn_expand.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BITRATE_SR_CONFIG="$SCRIPT_DIR/bitrate_sr_config_attn_expand.sh"
exec bash "$SCRIPT_DIR/run_sweep.sh"
