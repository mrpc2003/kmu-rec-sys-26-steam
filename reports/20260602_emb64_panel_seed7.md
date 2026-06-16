# KMU RecSys 26 Steam — LightGCN (GPU) validation

PyTorch LightGCN trained on V100 GPU with BPR loss. Validation-only; no Kaggle submission.

## Results

| split | row acc | per-user mean acc | epochs | loss | train sec |
|---|---:|---:|---:|---:|---:|
| `val_random_uniform_seed7` | 0.749650 | 0.771328 | 200 | 0.171669 | 1796.4 |
| `val_random_uniform_seed123` | 0.749350 | 0.770178 | 200 | 0.173301 | 1782.2 |

## Config

- emb_dim=64, n_layers=3, lr=0.001, reg=0.0001
- batch_size=4096, device=cuda:3, seed=7
