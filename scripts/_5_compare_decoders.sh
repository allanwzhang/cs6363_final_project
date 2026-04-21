#!/bin/bash
# _5_compare_decoders.sh
# Train all three decoder architectures (TokenGrid, MLP, Pooled) on layers
# 1, 4, 8, and 12. Reports an MSE table and saves a comparison grid image.
#
# Outputs:
#   outputs/decoder_comparison/results.txt
#   outputs/decoder_comparison/viz_compare.png
#
# Usage:
#   bash scripts/_5_compare_decoders.sh [dataset_root] [train_samples] [val_samples] [epochs] [batch_size]
#
# Examples:
#   Full run:    bash scripts/_5_compare_decoders.sh ./datasets 20000 4000 5 64
#   Quick test:  bash scripts/_5_compare_decoders.sh ./datasets 512 256 2 16

set -e
cd "$(dirname "$0")/.."

DATASET_ROOT=${1:-"./datasets"}
TRAIN_SAMPLES=${2:-20000}
VAL_SAMPLES=${3:-4000}
EPOCHS=${4:-5}
BATCH_SIZE=${5:-64}
OUTPUT_DIR="./outputs/decoder_comparison"

echo "Dataset:       $DATASET_ROOT"
echo "Train samples: $TRAIN_SAMPLES"
echo "Val samples:   $VAL_SAMPLES"
echo "Epochs:        $EPOCHS"
echo "Batch size:    $BATCH_SIZE"
echo "Output dir:    $OUTPUT_DIR"
echo ""

PYTHONPATH=. python testing/compare_decoders.py \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --layers 1 4 8 12 \
    --train_samples "$TRAIN_SAMPLES" \
    --val_samples "$VAL_SAMPLES" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "Results table:"
cat "$OUTPUT_DIR/results.txt"
echo ""
echo "Visualization: $OUTPUT_DIR/viz_compare.png"
