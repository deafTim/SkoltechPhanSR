#!/usr/bin/env bash
# =============================================================================
# NinaSR + LoRA, old-notebook ADMM loss (admm_oldlike), 10 BPP-diverse images.
# Same λ sweep as regular ADMM, but with the old loss:
#   |PSNR-target| (no 0.5 hinge), second term λ/10, lr=1e-4, Adam restarts.
# Tags: imgXXXX_nina_psnr35_lam*_r4_admm_old
# Runs: ${PROJECT_ROOT}/runs/bitrate_sr_nina_admm_old_10
# Plot: rd_nina_admm_old_10.png
#
#   bash scripts/run_sweep_nina_admm_old_10.sh
#   sbatch scripts/run_sweep_Zhores_nina_admm_old_10.sbatch
#   bash scripts/plot_results_nina_admm_old_10.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/bitrate_sr_config.sh"

BACKBONE="nina"
LORA_TARGET="all"

RUNS_DIR="${PROJECT_ROOT}/runs/bitrate_sr_nina_admm_old_10"
PLOT_SAVE="${RUNS_DIR}/rd_nina_admm_old_10.png"
BPP_CSV="${PROJECT_ROOT}/runs/bitrate_sr/bpp_all_sorted.csv"

IMG_LIST="image_lists/bpp_diverse_10.txt"

METHOD="admm_oldlike"
PLOT_METHOD="admm_old"
PLOT_GROUP_BY="lambda"

# λ sweep like regular ADMM; target PSNR fixed (no annealing ladder).
LAMBDAS="0.1 0.2 0.3 0.4 0.5 0.6 0.8 1.0"
TARGET_PSNR=35
ANNEAL_PSNRS=""

# Old-notebook loss / optimizer settings.
LR=1e-4
INNERS=100
OUTERS=20
REPEATS=1
CHECKPOINT_EVERY=0

CROP_SIZE=512
SCALED_SIZE=256
