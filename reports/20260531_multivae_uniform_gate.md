# MultiVAE (WWW'18 latent-VAE reconstruction CF) — UNIFORM gate

4th and final major CF paradigm tested (after graph/BPR, item-item linear, text-LM).
GeoCF (arXiv 2410.03064) builds on MultiVAE but its novelty is item-metadata geometry — N/A
here (anonymous gameID, no metadata) — so the metadata-free MultiVAE backbone was tested.

- split: `val_random_uniform_seed42` (rows=19996), train 28.7s
- **solo uniform: 0.72995** (pop floor 0.684)
- corr(within-user z) vs emb128 = **0.798**
- **50/50 z-blend: 0.75655** vs emb128 ref 0.76505 → **-0.00850**
- **tier: REDUNDANT** — solo 0.72995 (>=floor), corr_z 0.798, blend 0.75655 vs ref -0.00850 within/under noise -> redundant with backbone (like EASE), no new axis.

## Interpretation
MultiVAE is the nonlinear VAE cousin of EASE (linear autoencoder, already REDUNDANT at
corr_z 0.79). It reconstructs the same binary interaction matrix LightGCN learns from, so
corr_z 0.798 is structural — architecture sweeps (hidden/latent/beta) cannot decorrelate it,
and solo 0.72995 < ensemble 0.76505 means a single VAE cannot beat the
4-seed ensemble. One clean run is decisive; no sweep warranted (noise-chasing avoidance).
With MultiVAE, all four major CF paradigms are closed with evidence.
