#!/usr/bin/env bash
# =============================================================================
# NinaSR + LoRA, ADMM, 10 BPP-diverse images.
# Tags: imgXXXX_nina_psnr35_lam*_r4_admm
# Runs: ${PROJECT_ROOT}/runs/bitrate_sr_nina_admm_10
# Plot: rd_nina_admm_10.png
#
#   bash scripts/run_sweep_nina_admm_10.sh
#   sbatch scripts/run_sweep_Zhores_nina_admm_10.sbatch
#   bash scripts/plot_results_nina_admm_10.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/bitrate_sr_config.sh"

BACKBONE="nina"
LORA_TARGET="all"

RUNS_DIR="${PROJECT_ROOT}/runs/bitrate_sr_nina_admm_10"
PLOT_SAVE="${RUNS_DIR}/rd_nina_admm_10.png"
BPP_CSV="${PROJECT_ROOT}/runs/bitrate_sr/bpp_all_sorted.csv"

IMG_LIST="image_lists/bpp_diverse_10.txt"

LAMBDAS="0.1 0.2 0.3 0.4 0.5 0.6 0.8 1.0"

METHOD="admm"
CHECKPOINT_EVERY=0

INNERS=100
OUTERS=20

CROP_SIZE=512
SCALED_SIZE=256
