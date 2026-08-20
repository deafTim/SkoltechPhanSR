#!/usr/bin/env bash
# =============================================================================
# Swin2SR + LoRA, DIRECT, 10 BPP-diverse images.
# Tags: imgXXXX_swin_psnr35_lam*_r4_direct
# Runs: ${PROJECT_ROOT}/runs/bitrate_sr_swin_direct_10
# Plot: rd_swin_direct_10.png
#
#   bash scripts/run_sweep_swin_direct_10.sh
#   sbatch scripts/run_sweep_Zhores_swin_direct_10.sbatch
#   bash scripts/plot_results_swin_direct_10.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/bitrate_sr_config.sh"

BACKBONE="swin"
LORA_TARGET="all"

RUNS_DIR="${PROJECT_ROOT}/runs/bitrate_sr_swin_direct_10"
PLOT_SAVE="${RUNS_DIR}/rd_swin_direct_10.png"
BPP_CSV="${PROJECT_ROOT}/runs/bitrate_sr/bpp_all_sorted.csv"

IMG_LIST="image_lists/bpp_diverse_10.txt"

LAMBDAS="0.1 0.2 0.3 0.4 0.5 0.6 0.8 1.0"

METHOD="direct"
DIRECT_REF="gt"
STEPS=200
CHECKPOINT_EVERY=0

CROP_SIZE=128
SCALED_SIZE=64
