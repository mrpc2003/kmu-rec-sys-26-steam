# KMU RecSys 26 Steam — LightGCN (GPU) validation

PyTorch LightGCN trained on V100 GPU with BPR loss. Validation-only; no Kaggle submission.

## Results

| split | row acc | per-user mean acc | epochs | loss | train sec |
|---|---:|---:|---:|---:|---:|
| `val_random_uniform_seed42` | 0.757752 | 0.780993 | 200 | 0.172145 | 1637.7 |

## Config

- emb_dim=64, n_layers=3, lr=0.001, reg=0.0001
- batch_size=4096, device=cuda:1, seed=2024

