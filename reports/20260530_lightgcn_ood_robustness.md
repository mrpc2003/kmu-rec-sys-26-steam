# KMU RecSys 26 Steam — LightGCN (GPU) validation

PyTorch LightGCN trained on V100 GPU with BPR loss. Validation-only; no Kaggle submission.

## Results

| split | row acc | per-user mean acc | epochs | loss | train sec |
|---|---:|---:|---:|---:|---:|
| `val_random_uniform_seed42` | 0.754451 | 0.777635 | 200 | 0.171112 | 1774.7 |
| `val_random_communitypop_seed42` | 0.572314 | 0.591446 | 200 | 0.171112 | 1773.1 |
| `val_recent_communitypop_seed42` | 0.555511 | 0.573836 | 200 | 0.172888 | 1619.6 |

## Config

- emb_dim=64, n_layers=3, lr=0.001, reg=0.0001
- batch_size=4096, device=cuda:3, seed=42

