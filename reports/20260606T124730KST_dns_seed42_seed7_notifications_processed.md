# 2026-06-06 12:47 KST — DNS seed42/seed7 late notifications processed

## Trigger

Received late completion notifications for original split `val_random_uniform_seed42`:

- `proc_eabe514f7b69` — DNS pool=1 seed7, complete
- `proc_1b64d33ed426` — DNS pool=1 seed42, complete

These were already reflected in the four-seed aggregate.

## Confirmed original split results

| split | model seed | row accuracy | vs emb128 4-seed ref 0.76505 | status |
|---|---:|---:|---:|---|
| `val_random_uniform_seed42` | 42 | 0.76565 | +0.00060 | weak positive, not enough alone |
| `val_random_uniform_seed42` | 123 | 0.76265 | -0.00240 | weak |
| `val_random_uniform_seed42` | 2024 | 0.76395 | -0.00110 | weak |
| `val_random_uniform_seed42` | 7 | 0.76305 | -0.00200 | weak |

Four-seed aggregate remains:

- `dns_pool1_mean_z`: 0.76605, vs ref +0.00100
- `dns_pool1_mean_raw`: 0.76595, vs ref +0.00090
- `dns_pool1_neg_mean_rank`: 0.76535, vs ref +0.00030

Verdict: one-split weak positive; expand, do not submit.

## Current active expansion

All four jobs for independent split `val_random_uniform_seed123` are now running:

| split | model seed | GPU | process session | status |
|---|---:|---:|---|---|
| `val_random_uniform_seed123` | 42 | 0 | `proc_299db1d67b1b` | running |
| `val_random_uniform_seed123` | 123 | 1 | `proc_bcfc9b96abf7` | running |
| `val_random_uniform_seed123` | 2024 | 2 | `proc_e98b6356cfd6` | running |
| `val_random_uniform_seed123` | 7 | 3 | `proc_f32d0d53c3fd` | running |

GPU3 has a stale `[Not Found]` allocation around 4.3GB, but no active process was visible and memory headroom is large; seed7 uses this GPU for a validation-only low-memory job.

## Safety

- Kaggle submit executed: false
- Submission CSV created: false
- Hidden labels/external scraping: false
- Aggressive runner guard/quarantine unchanged.
