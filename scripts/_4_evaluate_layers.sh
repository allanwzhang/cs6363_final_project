#!/bin/bash
# _4_evaluate_layers.sh
#
# Re-evaluate saved decoder checkpoints on the validation split.
# Run this anytime after _3_train_all.sh to reprint or update metrics.
#
# Evaluates TokenGrid decoders for layers 1, 4, 8, 12 by default.
# Saves outputs/full_run/eval_layer{N}.json for each.
#
# Usage:
#   bash scripts/_4_evaluate_layers.sh [dataset_root] [batch_size] [max_samples]
#
# Examples:
#   Full val:    bash scripts/_4_evaluate_layers.sh ./datasets 64 0
#   Quick check: bash scripts/_4_evaluate_layers.sh ./datasets 16 256

set -e
cd "$(dirname "$0")/.."

DATASET_ROOT=${1:-"./datasets"}
BATCH_SIZE=${2:-64}
MAX_SAMPLES=${3:-0}
DECODER_DIR="./outputs/full_run"
LAYERS=(1 4 8 12)

echo "Re-evaluating TokenGrid decoders from $DECODER_DIR"
echo ""

for LAYER in "${LAYERS[@]}"; do
    DECODER_PATH="$DECODER_DIR/TokenGrid_layer${LAYER}.pt"

    if [ ! -f "$DECODER_PATH" ]; then
        echo "Missing checkpoint: $DECODER_PATH — run _3_train_all.sh first"
        exit 1
    fi

    echo "--- Layer $LAYER ---"
    PYTHONPATH=. python testing/evaluate_reconstruction.py \
        --dataset_root  "$DATASET_ROOT" \
        --decoder_path  "$DECODER_PATH" \
        --layer         "$LAYER" \
        --output_size   64 \
        --batch_size    "$BATCH_SIZE" \
        --max_samples   "$MAX_SAMPLES" \
        --num_workers   2 \
        --output_dir    "$DECODER_DIR"
    echo ""
done
