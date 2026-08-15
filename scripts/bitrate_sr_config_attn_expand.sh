#!/usr/bin/env bash
# =============================================================================
# Attention-expand LoRA only: body.{k}.body.2.body.3
# Full DIV2K train HR (800 images); no intermediate step_*.pt checkpoints.
#
#   bash scripts/run_sweep_attn_expand.sh
#   sbatch scripts/run_sweep_Zhores_attn_expand.sbatch
#   bash scripts/plot_results_attn_expand.sh
#
# Note: 800 images × 5 λ is heavy; each array task may need >8h — raise
# #SBATCH --time or split IMG_LIST if jobs hit the wall-clock limit.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/bitrate_sr_config.sh"

RUNS_DIR="${PROJECT_ROOT}/runs/bitrate_sr_attn_expand"
PLOT_SAVE="${RUNS_DIR}/rd.png"
BPP_CSV="${PROJECT_ROOT}/runs/bitrate_sr/bpp_all_sorted.csv"

LORA_TARGET="attention_expand"

# All DIV2K train HR images (0001.png … 0800.png)
IMG_LIST="image_lists/div2k_train_hr_all.txt"

# No intermediate LoRA checkpoints (metrics.json / SR.png / curves still saved)
CHECKPOINT_EVERY=0
