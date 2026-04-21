#!/bin/bash
# _6_visualize_layers.sh
#
# Re-generate original vs reconstruction images for layers 1, 4, 8, 12.
# Uses TokenGrid checkpoints saved by _3_train_all.sh.
# Run this anytime after _3_train_all.sh.
#
# Outputs:
#   outputs/full_run/viz_layer{N}.png  for each layer
#
# Usage:
#   bash scripts/_6_visualize_layers.sh [dataset_root]

set -e
cd "$(dirname "$0")/.."

DATASET_ROOT=${1:-"./datasets"}
DECODER_DIR="./outputs/full_run"
LAYERS=(1 4 8 12)

for LAYER in "${LAYERS[@]}"; do
    DECODER_PATH="$DECODER_DIR/TokenGrid_layer${LAYER}.pt"

    if [ ! -f "$DECODER_PATH" ]; then
        echo "Missing checkpoint: $DECODER_PATH — run _3_train_all.sh first"
        exit 1
    fi

    echo "Generating viz for layer $LAYER..."

    PYTHONPATH=. python - <<PY
import os, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F

from data.celeba_dataset import build_celeba_dataloader
from data.transforms import get_basic_image_transform
from model.clip_wrapper import CLIPVisionWrapper
from model.decoders import TokenGridDecoder

layer = $LAYER
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, loader = build_celeba_dataloader(
    root="$DATASET_ROOT", split="valid",
    transform=get_basic_image_transform(224),
    batch_size=6, shuffle=False, num_workers=0,
)

batch = next(iter(loader))
images = batch["image"]

clip_model = CLIPVisionWrapper(device=device, freeze=True)
decoder = TokenGridDecoder(output_size=64).to(device)
decoder.load_state_dict(torch.load("$DECODER_PATH", map_location=device))
decoder.eval()

with torch.no_grad():
    feats = clip_model.extract_features(images=images.to(device), layers=[layer])
    recon = decoder(feats.hidden_states[layer]).cpu()

target = F.interpolate(images, size=(64, 64), mode="bilinear", align_corners=False)

n = 4
fig = plt.figure(figsize=(10, 4.5))
fig.suptitle(f"Layer {layer}: Original (top) vs Reconstruction (bottom)", fontsize=12)
for i in range(n):
    plt.subplot(2, n, i + 1)
    plt.imshow(target[i].permute(1, 2, 0).clamp(0, 1).numpy())
    plt.axis("off")
    plt.subplot(2, n, n + i + 1)
    plt.imshow(recon[i].permute(1, 2, 0).clamp(0, 1).numpy())
    plt.axis("off")

plt.tight_layout()
save_path = os.path.join("$DECODER_DIR", f"viz_layer{layer}.png")
plt.savefig(save_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {save_path}")
PY
done
