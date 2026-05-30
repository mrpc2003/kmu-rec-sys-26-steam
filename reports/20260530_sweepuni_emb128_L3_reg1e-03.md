# KMU RecSys 26 Steam — LightGCN (GPU) validation

PyTorch LightGCN trained on V100 GPU with BPR loss. Validation-only; no Kaggle submission.

## Results

| split | row acc | per-user mean acc | epochs | loss | train sec |
|---|---:|---:|---:|---:|---:|
| `val_random_uniform_seed42` | 0.758152 | 0.781046 | 200 | 0.237356 | 1618.5 |

## Config

- emb_dim=128, n_layers=3, lr=0.001, reg=0.001
- batch_size=4096, device=cuda:3, seed=42

