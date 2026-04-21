#!/bin/bash
# _2_smoketest.sh
# Verify that CLIP loads, the dataset is found, and tensor shapes are correct.
# Expected output:
#   Input:  torch.Size([4, 3, 224, 224])
#   Layer 4: torch.Size([4, 50, 768])
#   Layer 12: torch.Size([4, 50, 768])
#   Pooler: torch.Size([4, 768])
#   Grid size: 7

set -e
cd "$(dirname "$0")/.."

DATASET_ROOT=${1:-"./datasets"}

PYTHONPATH=. python testing/clip_smoketest.py
