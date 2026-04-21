#!/bin/bash
# _4_evaluate_layers.sh
# Evaluate MSE for each trained layer decoder on the validation split.
# Requires _3_train_layers.sh to have been run first.
#
# Usage:
#   bash scripts/_4_evaluate_layers.sh [dataset_root] [batch_size] [max_samples]
#
# Examples:
#   Full run:    bash scripts/_4_evaluate_layers.sh ./datasets 64 0
#   Quick test:  bash scripts/_4_evaluate_layers.sh ./datasets 16 256

set -e
cd "$(dirname "$0")/.."

DATASET_ROOT=${1:-"./datasets"}
BATCH_SIZE=${2:-64}
MAX_SAMPLES=${3:-0}
DECODER_DIR="./outputs/full_run"
LAYERS=(1 4 8 12)

echo "Decoder dir: $DECODER_DIR"
echo ""

for LAYER in "${LAYERS[@]}"; do
    DECODER_PATH="$DECODER_DIR/decoder_layer${LAYER}.pt"

    if [ ! -f "$DECODER_PATH" ]; then
        echo "Missing checkpoint: $DECODER_PATH — run _3_train_layers.sh first"
        exit 1
    fi

    echo "--- Layer $LAYER ---"
    PYTHONPATH=. python testing/evaluate_reconstruction.py \
        --dataset_root "$DATASET_ROOT" \
        --decoder_path "$DECODER_PATH" \
        --layer "$LAYER" \
        --output_size 64 \
        --batch_size "$BATCH_SIZE" \
        --max_samples "$MAX_SAMPLES" \
        --num_workers 2
    echo ""
done
