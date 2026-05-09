# sDiMA — Sub-optimized DiMA for Protein Sequence Generation

A lightweight, resource-constrained implementation of the DiMA (Diffusion on Language Model Encodings) framework for protein sequence generation, trained on a temporally split PDB dataset.

## Overview
sDiMA adapts the original DiMA pipeline to run on a single consumer GPU (RTX 4070 Ti, 16GB) in 6-7 hours instead of DiMA's 10 days on 4x A100s. The core architecture is preserved — frozen ESM-2 35M encoder, Tan-10 diffusion denoiser with self-conditioning, and a fine-tuned decoder — with the denoiser halved from 12 to 6 transformer layers.

A key original finding of this work is a clear inverse relationship between generated sequence length and pLDDT score, with sequences in the 80-130 residue range achieving the highest structural confidence.

## Pipeline
1. Dataset: PDB sequences, temporally split (2010-2015 train, 2016 test)
2. Generate 10,000 raw sequences via diffusion
3. Score with Bio Score (30%) + ESM-2 PLL proxy (70%)
4. Dimensionality reduction: PCA -> UMAP
5. K-means clustering (k=20), select top sequences per cluster
6. Fold top 2,000 via ESMFold
7. Filter by pLDDT > 30, evaluate metrics

## Results (Final Pipeline)
- Mean pLDDT: 37.23
- FD-seq vs Test: 0.800
- CD@0.95: 0.987 (near-zero duplication)
- Novelty vs Test: 74.5%
- Mean Levenshtein Match: 25.48%

## Dataset
https://github.com/GDSC-IIIT-V/protein-classification-ml-model

## Reference
Meshchaninov et al. (2025). Diffusion on Language Model Encodings for Protein Sequence Generation. ICML 2025.
https://arxiv.org/abs/2403.03726
