#!/bin/bash
# _3_train_layers.sh
# Train one TokenGridDecoder per CLIP layer (1, 4, 8, 12).
# Each decoder is saved to outputs/full_run/decoder_layer<N>.pt
#
# Usage:
#   bash scripts/_3_train_layers.sh [dataset_root] [epochs] [batch_size] [max_samples]
#
# Examples:
#   Full run:    bash scripts/_3_train_layers.sh ./datasets 5 64 0
#   Quick test:  bash scripts/_3_train_layers.sh ./datasets 1 16 512

set -e
cd "$(dirname "$0")/.."

DATASET_ROOT=${1:-"./datasets"}
EPOCHS=${2:-5}
BATCH_SIZE=${3:-64}
MAX_SAMPLES=${4:-0}   # 0 = use full dataset
OUTPUT_DIR="./outputs/full_run"
LAYERS=(1 4 8 12)

echo "Dataset:     $DATASET_ROOT"
echo "Epochs:      $EPOCHS"
echo "Batch size:  $BATCH_SIZE"
echo "Max samples: $MAX_SAMPLES (0 = all)"
echo "Output dir:  $OUTPUT_DIR"
echo ""

for LAYER in "${LAYERS[@]}"; do
    echo "==============================="
    echo " Training decoder — layer $LAYER"
    echo "==============================="
    PYTHONPATH=. python training/train_decoder.py \
        --dataset_root "$DATASET_ROOT" \
        --layer "$LAYER" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --output_size 64 \
        --max_samples "$MAX_SAMPLES" \
        --num_workers 2 \
        --output_dir "$OUTPUT_DIR"
    echo ""
done

echo "All layer decoders saved to $OUTPUT_DIR"
