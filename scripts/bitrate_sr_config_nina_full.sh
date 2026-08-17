#!/usr/bin/env bash
# =============================================================================
# NinaSR + full LoRA (all layers), full DIV2K train HR (800 images).
# No intermediate step_*.pt; train.py skips runs that already have metrics.json.
#
#   bash scripts/run_sweep_nina_full.sh
#   sbatch scripts/run_sweep_Zhores_nina_full.sbatch
#   # after Swin array JOBID:
#   sbatch --dependency=afterok:JOBID scripts/run_sweep_Zhores_nina_full.sbatch
#   bash scripts/plot_results_nina_full.sh
#
# gpu_devel MaxWall=12h: one λ over 800 imgs may need several submits;
# resubmit the same sbatch — finished tags are skipped.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/bitrate_sr_config.sh"

BACKBONE="nina"
LORA_TARGET="all"

RUNS_DIR="${PROJECT_ROOT}/runs/bitrate_sr_nina_full"
PLOT_SAVE="${RUNS_DIR}/rd.png"
BPP_CSV="${PROJECT_ROOT}/runs/bitrate_sr/bpp_all_sorted.csv"

IMG_LIST="image_lists/div2k_train_hr_all.txt"

# All λ (ascending). Existing metrics.json tags are skipped.
LAMBDAS="0.1 0.2 0.3 0.4 0.5 0.6 0.8 1.0"

METHOD="direct"
DIRECT_REF="gt"
STEPS=200
CHECKPOINT_EVERY=0

# Full-res crop (×2 SR)
CROP_SIZE=512
SCALED_SIZE=256
