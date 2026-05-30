# KMU RecSys 26 Steam — LightGCN (GPU) validation

PyTorch LightGCN trained on V100 GPU with BPR loss. Validation-only; no Kaggle submission.

## Results

| split | row acc | per-user mean acc | epochs | loss | train sec |
|---|---:|---:|---:|---:|---:|
| `val_random_uniform_seed42` | 0.757752 | 0.782616 | 200 | 0.148728 | 1657.3 |

## Config

- emb_dim=128, n_layers=4, lr=0.001, reg=0.0001
- batch_size=4096, device=cuda:2, seed=42

