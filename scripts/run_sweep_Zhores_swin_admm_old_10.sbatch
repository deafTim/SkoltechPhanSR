#!/usr/bin/env bash
#SBATCH -J bsr_swin_aold
#SBATCH -p gpu_devel
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH -o logs/bitrate_sr_swin_admm_old_10_%A_%a.out
#SBATCH -e logs/bitrate_sr_swin_admm_old_10_%A_%a.err
#SBATCH --array=0-9
#
# swin_admm_old_10: old-notebook ADMM, one array task per image.
# Each task walks the whole target-PSNR ladder (ANNEAL_PSNRS) with lambda fixed,
# warm-starting LoRA/Z/T from the previous rung. Finished rungs are skipped on
# resubmit, so just resubmit after a 12h timeout.
# From repo root:
#   sbatch scripts/run_sweep_Zhores_swin_admm_old_10.sbatch

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/bitrate_sr_config_swin_admm_old_10.sh"

module unload python/3.12 2>/dev/null || true
if [[ -f /trinity/shared/opt/mambaforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /trinity/shared/opt/mambaforge3/etc/profile.d/conda.sh
  set +u
  conda activate "${CONDA_ENV}"
  set -u
fi

if [[ -z "${IMG_LIST}" || ! -f "$IMG_LIST" ]]; then
  echo "IMG_LIST must point to an existing file (got: '${IMG_LIST:-}')" >&2
  exit 1
fi

# Array index selects the image; lambda is fixed for this method.
N_IMGS="$(awk 'NF && $1 !~ /^#/' "$IMG_LIST" | wc -l | tr -d ' ')"
IMG_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IMG_INDEX >= N_IMGS )); then
  echo "IMG_INDEX=$IMG_INDEX >= images in $IMG_LIST ($N_IMGS); nothing to do"
  exit 0
fi

read -r -a LAMBDA_ARR <<< "$LAMBDAS"
L="${LAMBDA_ARR[0]}"

LOG_DIR="${RUNS_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/admm_old_swin_job${SLURM_ARRAY_JOB_ID:-local}_img${IMG_INDEX}.log"

echo "REPO_ROOT=$REPO_ROOT backbone=$BACKBONE method=$METHOD lambda=$L"
echo "ladder='$ANNEAL_PSNRS' repeats=$REPEATS outers=$OUTERS inners=$INNERS lr=$LR"
echo "img_index=$IMG_INDEX/$N_IMGS runs=$RUNS_DIR log=$LOG_FILE"

python -u -m bitrate_sr.train \
  --project-root "$PROJECT_ROOT" \
  --data-dir "$DATA_DIR" \
  --runs-dir "$RUNS_DIR" \
  --method "$METHOD" \
  --target-psnr "$TARGET_PSNR" \
  --anneal-psnrs "$ANNEAL_PSNRS" \
  --repeats "$REPEATS" \
  --checkpoint-every "${CHECKPOINT_EVERY:-0}" \
  --backbone "$BACKBONE" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-target "${LORA_TARGET:-all}" \
  --lr "$LR" \
  --inners "$INNERS" \
  --outers "$OUTERS" \
  --crop-size "$CROP_SIZE" \
  --scaled-size "$SCALED_SIZE" \
  --crop-left "$CROP_LEFT" \
  --crop-top "$CROP_TOP" \
  --compression-model "$COMPRESSION_MODEL" \
  --compression-quality "$COMPRESSION_QUALITY" \
  --img-list "$IMG_LIST" \
  --img-index "$IMG_INDEX" \
  --lambda "$L" 2>&1 | tee -a "$LOG_FILE"
