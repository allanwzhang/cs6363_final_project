#!/bin/bash
# _3_train_all.sh
#
# THE main training script. Trains all 3 decoder architectures (TokenGrid, MLP,
# Pooled) on CLIP layers 1, 4, 8, 12 in one go. Evaluates each one immediately
# after training and saves everything to outputs/full_run/:
#
#   {arch}_layer{N}.pt      checkpoint for each decoder × layer combo (12 total)
#   results.csv             MSE / PSNR / SSIM for every combination
#   results.txt             same table, human-readable
#   viz_compare.png         image grid  (rows = decoder, cols = layer)
#   bar_mse.png             grouped bar chart — MSE
#   bar_psnr.png            grouped bar chart — PSNR
#   bar_ssim.png            grouped bar chart — SSIM
#
# Usage:
#   bash scripts/_3_train_all.sh [dataset_root] [train_samples] [val_samples] [epochs] [batch_size]
#
# Examples:
#   Full run (20k imgs, 5 epochs):  bash scripts/_3_train_all.sh ./datasets 20000 4000 5 64
#   Quick test (512 imgs, 2 epochs): bash scripts/_3_train_all.sh ./datasets 512  256  2 16

set -e
cd "$(dirname "$0")/.."

DATASET_ROOT=${1:-"./datasets"}
TRAIN_SAMPLES=${2:-20000}
VAL_SAMPLES=${3:-4000}
EPOCHS=${4:-5}
BATCH_SIZE=${5:-64}
OUTPUT_DIR="./outputs/full_run"

echo "========================================"
echo " Training all decoders on all layers"
echo "========================================"
echo "Dataset:       $DATASET_ROOT"
echo "Train samples: $TRAIN_SAMPLES"
echo "Val samples:   $VAL_SAMPLES"
echo "Epochs:        $EPOCHS"
echo "Batch size:    $BATCH_SIZE"
echo "Output dir:    $OUTPUT_DIR"
echo ""

PYTHONPATH=. python testing/compare_decoders.py \
    --dataset_root  "$DATASET_ROOT" \
    --output_dir    "$OUTPUT_DIR" \
    --layers 1 4 8 12 \
    --train_samples "$TRAIN_SAMPLES" \
    --val_samples   "$VAL_SAMPLES" \
    --epochs        "$EPOCHS" \
    --batch_size    "$BATCH_SIZE"

echo ""
echo "All done. Outputs in $OUTPUT_DIR"
