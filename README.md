# Interpreting Transformer Blocks in CLIP Through Celebrity Face Reconstruction

This repository contains code for our CS6363 final project investigating how visual information evolves across layers of the CLIP vision transformer. We analyze intermediate representations by training reconstruction decoders on hidden states extracted from different transformer depths.

Our goal is to better understand what information is preserved or discarded as CLIP moves from early spatial features toward higher-level semantic representations.

Project code accompanies the paper:  
**"Interpreting Transformer Blocks in CLIP Through Celebrity Face Reconstruction"**

---

## Overview

Transformer-based vision models such as CLIP are often treated as black boxes. In this project, we probe intermediate hidden states of the CLIP vision encoder by attempting to reconstruct input face images from representations at different layers.

Our pipeline:

```
CelebA images
→ CLIP vision encoder (frozen)
→ extract hidden states (layers 4, 12)
→ train reconstruction decoder
→ compare reconstruction quality
```

By measuring reconstruction fidelity across layers, we study how spatial detail changes with transformer depth.

---

## Repository Structure

```
clip_face_project/
│
├── data/        # Dataset loading and transforms
├── model/       # CLIP wrapper, decoders, sparse autoencoder
├── training/    # Training scripts
├── testing/     # Evaluation and visualization scripts
├── utils/       # Metrics and plotting helpers
├── outputs/     # Saved checkpoints and reconstructions
```

---

## Installation

Clone the repository:

```
git clone https://github.com/allanwzhang/cs6363_final_project.git
cd cs6363_final_project
```

Install dependencies:

```
pip install -r requirements.txt
```

We recommend running experiments with a GPU-enabled environment (e.g., Google Colab or a local CUDA setup).

---

## Dataset Setup

We use the CelebA dataset via torchvision.

Update the dataset path if needed (default assumes):

```
./datasets/
```

Example usage:

```
python data/inspect_data.py
```

This verifies dataset loading and preprocessing.

---

## Quick Start

Run a CLIP feature extraction smoke test:

```
python testing/clip_smoketest.py
```

Train a reconstruction decoder from layer 4:

```
python training/train_decoder.py --layer 4
```

Evaluate reconstruction quality:

```
python testing/evaluate_reconstruction.py     --decoder_path outputs/decoder_layer4.pt
```

Visualize reconstruction examples:

```
python testing/visualize_recons.py     --decoder_path outputs/decoder_layer4.pt
```

---

## Model Details

We use:

- CLIP vision encoder: clip-vit-base-patch32
- Frozen encoder weights
- Hidden state extraction from intermediate layers
- Token-grid reconstruction decoder

Primary experiments reconstruct images from:

- layer 4 representations
- layer 12 representations

This allows comparison between early spatial features and later semantic embeddings.

---

## Outputs

Trained models and reconstructions are saved to:

```
outputs/
```

Typical contents include:

```
decoder_layer4.pt
decoder_layer12.pt
reconstruction grids
evaluation metrics
```

---

## Sparse Autoencoder (Optional Extension)

The repository also includes a sparse autoencoder module for probing interpretability of latent features:

```
python training/train_sae.py
```

This component is exploratory and not required for the main reconstruction experiments.

---

## Reproducibility

All experiments were implemented in PyTorch using the HuggingFace CLIP vision encoder.

The codebase is structured so that reconstruction experiments can be reproduced with:

```
python training/train_decoder.py
```

and evaluated using scripts in:

```
testing/
```

---

## Authors

Allan Zhang  
Rohan Rashingkar  
Franklin Hu  

CS6363 Final Project
