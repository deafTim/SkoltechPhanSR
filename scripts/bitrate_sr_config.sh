#!/usr/bin/env bash
# =============================================================================
# Bitrate-SR (ADMM + LoRA) config — edit here, then from repo root:
#   bash scripts/select_bpp_images.sh
#   bash scripts/run_sweep.sh
#   mkdir -p logs && sbatch scripts/run_sweep_Zhores.sbatch
#   bash scripts/plot_results.sh
# Do not pass flags on the command line.
# =============================================================================

# Paths
PROJECT_ROOT="/gpfs/gpfs0/timofey.glukhikh/Science_Phan"
DATA_DIR="${PROJECT_ROOT}/DIV2K/DIV2K_train_HR"
RUNS_DIR="${PROJECT_ROOT}/runs/bitrate_sr"
PLOT_SAVE="${RUNS_DIR}/rd.png"
BPP_CSV="${RUNS_DIR}/bpp_all_sorted.csv"
# Conda env (path form works when name is not registered in mambaforge)
CONDA_ENV="/home/timofey.glukhikh/.conda/science_phan"
# PLOT_SHOW=0

# Image list for training; also written by select_bpp_images.sh
IMG_LIST="image_lists/bpp_diverse_10.txt"
NUM_SELECTED=10
BPP_BATCH_SIZE=4

# Backbone: nina | swin  (also used for BPP ranking SR)
BACKBONE="nina"

# Nina LoRA scope: all | attention_expand  (body.*.body.2.body.3 only)
LORA_TARGET="all"

# Optimizer: admm | direct  (direct = one Adam loss, no Z/T)
METHOD="direct"
# Direct only: gt | sr_orig  (PSNR hinges vs ground truth or pretrained SR)
DIRECT_REF="gt"
STEPS=200
CHECKPOINT_EVERY=50

# ADMM / LoRA (OUTERS*INNERS = 1000 when METHOD=admm)
TARGET_PSNR=35
LAMBDAS="1.0 0.8 0.6 0.5 0.2"
LORA_R=4
LORA_ALPHA=8
LR=5e-4
INNERS=100
OUTERS=10

# Crop (for Swin use smaller, e.g. 128 / 64)
CROP_SIZE=512
SCALED_SIZE=256
CROP_LEFT=1328
CROP_TOP=826

# Compression
COMPRESSION_MODEL="cheng2020-attn"
COMPRESSION_QUALITY=6
