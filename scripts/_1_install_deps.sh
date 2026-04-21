#!/bin/bash
# _1_install_deps.sh
# Install all required Python dependencies.
# Run this first, or skip if your environment already has them.

set -e

pip install transformers torchvision tqdm matplotlib pillow
