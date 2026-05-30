# KMU RecSys 26 Steam — LightGCN (GPU) validation

PyTorch LightGCN trained on V100 GPU with BPR loss. Validation-only; no Kaggle submission.

## Results

| split | row acc | per-user mean acc | epochs | loss | train sec |
|---|---:|---:|---:|---:|---:|
| `val_random_uniform_seed42` | 0.758052 | 0.780674 | 200 | 0.170616 | 1716.2 |

## Config

- emb_dim=64, n_layers=3, lr=0.001, reg=0.0001
- batch_size=4096, device=cuda:2, seed=7

