#!/usr/bin/env bash
# =============================================================================
# Swin2SR + LoRA (Linear), 10-image pilot list.
# Smaller crop for GPU memory (e.g. Zhores 1080 Ti).
#
#   bash scripts/run_sweep_swin.sh
#   sbatch scripts/run_sweep_Zhores_swin.sbatch
#   bash scripts/plot_results_swin.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/bitrate_sr_config.sh"

BACKBONE="swin"
LORA_TARGET="all"

RUNS_DIR="${PROJECT_ROOT}/runs/bitrate_sr_swin"
PLOT_SAVE="${RUNS_DIR}/rd.png"
BPP_CSV="${PROJECT_ROOT}/runs/bitrate_sr/bpp_all_sorted.csv"

IMG_LIST="image_lists/bpp_diverse_10.txt"

METHOD="direct"
DIRECT_REF="gt"
STEPS=200
CHECKPOINT_EVERY=0

# HR crop / LR input (×2 SR → sizes must match)
CROP_SIZE=128
SCALED_SIZE=64
