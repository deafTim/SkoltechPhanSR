#!/usr/bin/env bash
# Reads settings from scripts/bitrate_sr_config.sh — edit that file, then:
#   bash scripts/run_sweep.sh
# On Zhores: sbatch scripts/run_sweep_Zhores.sbatch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/bitrate_sr_config.sh"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "BACKBONE=$BACKBONE TARGET_PSNR=$TARGET_PSNR"
echo "LAMBDAS=$LAMBDAS"
echo "IMG_LIST=$IMG_LIST"
echo "LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA LR=$LR INNERS=$INNERS OUTERS=$OUTERS"
echo "CROP_SIZE=$CROP_SIZE SCALED_SIZE=$SCALED_SIZE"

if [[ -z "${IMG_LIST}" || ! -f "$IMG_LIST" ]]; then
  echo "IMG_LIST must point to an existing file (got: '${IMG_LIST:-}')" >&2
  exit 1
fi

COMMON_ARGS=(
  --project-root "$PROJECT_ROOT"
  --data-dir "$DATA_DIR"
  --runs-dir "$RUNS_DIR"
  --target-psnr "$TARGET_PSNR"
  --backbone "$BACKBONE"
  --lora-r "$LORA_R"
  --lora-alpha "$LORA_ALPHA"
  --lr "$LR"
  --inners "$INNERS"
  --outers "$OUTERS"
  --crop-size "$CROP_SIZE"
  --scaled-size "$SCALED_SIZE"
  --crop-left "$CROP_LEFT"
  --crop-top "$CROP_TOP"
  --compression-model "$COMPRESSION_MODEL"
  --compression-quality "$COMPRESSION_QUALITY"
  --img-list "$IMG_LIST"
)

for L in $LAMBDAS; do
  echo "========== lambda=$L =========="
  python -m bitrate_sr.train "${COMMON_ARGS[@]}" --lambda "$L"
done

echo "Done. Results under: $RUNS_DIR"
