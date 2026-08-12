#!/usr/bin/env bash
# Reads settings from scripts/bitrate_sr_config.sh — edit that file, then:
#   bash scripts/select_bpp_images.sh
# Writes diverse selection into IMG_LIST (same path train uses).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/bitrate_sr_config.sh"

DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/DIV2K/DIV2K_train_HR}"
BPP_CSV="${BPP_CSV:-$RUNS_DIR/bpp_all_sorted.csv}"
NUM_SELECTED="${NUM_SELECTED:-10}"
BPP_BATCH_SIZE="${BPP_BATCH_SIZE:-4}"

echo "DATA_DIR=$DATA_DIR"
echo "IMG_LIST=$IMG_LIST"
echo "BPP_CSV=$BPP_CSV"
echo "BACKBONE=$BACKBONE NUM_SELECTED=$NUM_SELECTED BATCH=$BPP_BATCH_SIZE"

python -m bitrate_sr.select_bpp \
  --project-root "$PROJECT_ROOT" \
  --data-dir "$DATA_DIR" \
  --img-list-out "$IMG_LIST" \
  --bpp-csv "$BPP_CSV" \
  --num-selected "$NUM_SELECTED" \
  --backbone "$BACKBONE" \
  --crop-size "$CROP_SIZE" \
  --scaled-size "$SCALED_SIZE" \
  --batch-size "$BPP_BATCH_SIZE" \
  --compression-model "$COMPRESSION_MODEL" \
  --compression-quality "$COMPRESSION_QUALITY"

echo "Done. IMG_LIST ready for training: $IMG_LIST"
