#!/usr/bin/env bash
# =============================================================================
# NinaSR + LoRA, ADMM (Z/T + projection), full DIV2K train HR (800 images).
# Tags: imgXXXX_nina_psnr35_lam*_r4_admm
# Runs:  ${PROJECT_ROOT}/runs/bitrate_sr_nina_admm
#
#   bash scripts/run_sweep_nina_admm.sh
#   sbatch scripts/run_sweep_Zhores_nina_admm.sbatch
#   bash scripts/plot_results_nina_admm.sh
#
# Does not touch runs/bitrate_sr_nina_full (direct).
# gpu_devel MaxWall=12h: resubmit — finished metrics.json are skipped.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/bitrate_sr_config.sh"

BACKBONE="nina"
LORA_TARGET="all"

RUNS_DIR="${PROJECT_ROOT}/runs/bitrate_sr_nina_admm"
PLOT_SAVE="${RUNS_DIR}/rd.png"
BPP_CSV="${PROJECT_ROOT}/runs/bitrate_sr/bpp_all_sorted.csv"

IMG_LIST="image_lists/div2k_train_hr_all.txt"

LAMBDAS="0.1 0.2 0.3 0.4 0.5 0.6 0.8 1.0"

METHOD="admm"
CHECKPOINT_EVERY=0

# OUTERS*INNERS = 2000 Adam steps (as in earlier ADMM curves)
INNERS=100
OUTERS=20

CROP_SIZE=512
SCALED_SIZE=256
