# 2026-06-06 12:46 KST — DNS pool=1 seed2024 processed and aggregate escalated

## Trigger

Background process `proc_7fac8fe737cb` completed: DNS pool=1, model seed 2024, split `val_random_uniform_seed42`.

## Single-seed DNS pool=1 panel on `val_random_uniform_seed42`

| model seed | row accuracy | vs emb128 4-seed ref 0.76505 | status |
|---:|---:|---:|---|
| 42 | 0.76565 | +0.00060 | weak positive, within/noise-adjacent |
| 123 | 0.76265 | -0.00240 | weak |
| 2024 | 0.76395 | -0.00110 | weak |
| 7 | 0.76305 | -0.00200 | weak |

## Four-seed aggregate on `val_random_uniform_seed42`

Aggregated score files under:

`artifacts/dns_pool1_seed_panel/aggregate_val_random_uniform_seed42/`

| aggregate score | row accuracy | vs emb128 4-seed ref | note |
|---|---:|---:|---|
| `dns_pool1_mean_z` | 0.76605 | +0.00100 | best one-split result |
| `dns_pool1_mean_raw` | 0.76595 | +0.00090 | also positive |
| `dns_pool1_neg_mean_rank` | 0.76535 | +0.00030 | not enough |

Interpretation: this is a weak one-split positive above the ±0.0007 noise band, but not submission-worthy. It must be tested on independent uniform splits before candidate promotion.

## Escalation launched — no submit

Started three background jobs on independent split `val_random_uniform_seed123`:

| split | model seed | GPU | process session | out dir |
|---|---:|---:|---|---|
| `val_random_uniform_seed123` | 42 | 0 | `proc_299db1d67b1b` | `artifacts/dns_pool1_multisplit/val_random_uniform_seed123/seed42` |
| `val_random_uniform_seed123` | 123 | 1 | `proc_bcfc9b96abf7` | `artifacts/dns_pool1_multisplit/val_random_uniform_seed123/seed123` |
| `val_random_uniform_seed123` | 2024 | 2 | `proc_e98b6356cfd6` | `artifacts/dns_pool1_multisplit/val_random_uniform_seed123/seed2024` |

Remaining to launch after one GPU frees:

- `val_random_uniform_seed123`, model seed 7
- then `val_random_uniform_seed7`, model seeds 42/123/2024/7

## Safety

- Kaggle submit executed: false
- Submission CSV created: false
- Hidden labels/external scraping: false
- Aggressive runner guard/quarantine unchanged.
