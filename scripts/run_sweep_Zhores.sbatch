#!/usr/bin/env bash
#SBATCH -J bitrate_sr
#SBATCH -p gpu_devel
#SBATCH --gpus=1
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH -o logs/bitrate_sr_%A_%a.out
#SBATCH -e logs/bitrate_sr_%A_%a.err
#SBATCH --array=0-4
#
# Same training as run_sweep.sh, but on Zhores (Slurm array over LAMBDAS).
# Submit from repo root (so SLURM_SUBMIT_DIR == repo root):
#   sbatch scripts/run_sweep_Zhores.sbatch
# Array size should match number of LAMBDAS in the config (default 5 → 0-4).
#
# Note: do NOT derive repo path from BASH_SOURCE — Slurm may copy this script
# into a spool dir where mkdir/logs is not writable.

set -euo pipefail


REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/bitrate_sr_config.sh"

module unload python/3.12 2>/dev/null || true
if [[ -f /trinity/shared/opt/mambaforge3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /trinity/shared/opt/mambaforge3/etc/profile.d/conda.sh
  # conda activate.d scripts reference unset vars; incompatible with set -u
  set +u
  conda activate "${CONDA_ENV}"
  set -u
fi

read -r -a LAMBDA_ARR <<< "$LAMBDAS"
L="${LAMBDA_ARR[$SLURM_ARRAY_TASK_ID]}"
if [[ -z "${L:-}" ]]; then
  echo "No lambda for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (LAMBDAS='$LAMBDAS')" >&2
  exit 1
fi

if [[ -z "${IMG_LIST}" || ! -f "$IMG_LIST" ]]; then
  echo "IMG_LIST must point to an existing file (got: '${IMG_LIST:-}')" >&2
  exit 1
fi

echo "REPO_ROOT=$REPO_ROOT CONDA_ENV=$CONDA_ENV lambda=$L"
python -m bitrate_sr.train \
  --project-root "$PROJECT_ROOT" \
  --data-dir "$DATA_DIR" \
  --runs-dir "$RUNS_DIR" \
  --target-psnr "$TARGET_PSNR" \
  --backbone "$BACKBONE" \
  --lora-r "$LORA_R" \
  --lora-alpha "$LORA_ALPHA" \
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
  --lambda "$L"
