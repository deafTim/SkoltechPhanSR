#!/usr/bin/env bash
# =============================================================================
# Swin2SR + LoRA, old-notebook ADMM (admm_oldlike), 10 BPP-diverse images.
# RD curve comes from the target-PSNR ladder at fixed lambda, as in
# train_superresolution_compression_admm_div2K_0025_old.ipynb
# Tags: imgXXXX_swin_psnr{36..32}_lam1_r4_admm_old
# Runs: ${PROJECT_ROOT}/runs/bitrate_sr_swin_admm_old_10
# Plot: rd_swin_admm_old_10.png
#
#   bash scripts/run_sweep_swin_admm_old_10.sh
#   sbatch scripts/run_sweep_Zhores_swin_admm_old_10.sbatch
#   bash scripts/plot_results_swin_admm_old_10.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/bitrate_sr_config.sh"

BACKBONE="swin"
LORA_TARGET="all"

RUNS_DIR="${PROJECT_ROOT}/runs/bitrate_sr_swin_admm_old_10"
PLOT_SAVE="${RUNS_DIR}/rd_swin_admm_old_10.png"
BPP_CSV="${PROJECT_ROOT}/runs/bitrate_sr/bpp_all_sorted.csv"

IMG_LIST="image_lists/bpp_diverse_10.txt"

METHOD="admm_oldlike"
PLOT_METHOD="admm_old"

# Old notebook: lambda fixed at 1, rate swept by lowering the target PSNR.
LAMBDAS="1.0"
ANNEAL_PSNRS="36 35 34 33 32"
TARGET_PSNR=36

# Old notebook: lr 1e-4, fresh Adam per repeat, plain |dPSNR| penalties.
LR=1e-4
INNERS=100
OUTERS=10
REPEATS=2
CHECKPOINT_EVERY=0

CROP_SIZE=128
SCALED_SIZE=64
