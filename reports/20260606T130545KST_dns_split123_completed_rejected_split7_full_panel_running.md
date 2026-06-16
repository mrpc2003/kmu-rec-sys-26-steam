# 2026-06-06 13:05 KST — DNS split123 completed and rejected; split7 full panel running

## Trigger

Received completion notifications for remaining `val_random_uniform_seed123` DNS pool=1 jobs:

- `proc_e98b6356cfd6` — seed2024: complete
- `proc_299db1d67b1b` — seed42: complete
- `proc_f32d0d53c3fd` — seed7: complete

Together with seed123, this completes the second independent uniform split panel.

## `val_random_uniform_seed123` single-seed results

| model seed | row accuracy | vs emb128 4-seed ref 0.76505 | status |
|---:|---:|---:|---|
| 42 | 0.75955 | -0.00550 | weak |
| 123 | 0.76135 | -0.00370 | weak |
| 2024 | 0.75865 | -0.00640 | weak |
| 7 | 0.75975 | -0.00530 | weak |

## `val_random_uniform_seed123` four-seed aggregate

Artifacts:

`artifacts/dns_pool1_multisplit/val_random_uniform_seed123/aggregate/aggregate_summary.json`

| aggregate score | row accuracy | vs emb128 4-seed ref | verdict |
|---|---:|---:|---|
| `dns_pool1_neg_mean_rank` | 0.76075 | -0.00430 | reject |
| `dns_pool1_mean_z` | 0.76065 | -0.00440 | reject |
| `dns_pool1_mean_raw` | 0.75925 | -0.00580 | reject |

Interpretation: the weak positive on `val_random_uniform_seed42` did not transfer to the second uniform split. DNS pool=1 is now very likely split-specific noise, not a candidate. Keep no-submit stance.

## Continuation launched

The third independent uniform split `val_random_uniform_seed7` now has all four seeds running:

| split | model seed | GPU | process session | status |
|---|---:|---:|---|---|
| `val_random_uniform_seed7` | 42 | 0 | `proc_31fb7df5f047` | running |
| `val_random_uniform_seed7` | 123 | 1 | `proc_73617e53a7c7` | running |
| `val_random_uniform_seed7` | 2024 | 2 | `proc_17e267f8fc15` | running |
| `val_random_uniform_seed7` | 7 | 3 | `proc_6fb208e8f301` | running |

If split7 also fails, close DNS pool=1 formally and move to the next independent no-submit axis.

## Safety

- Kaggle submit executed: false
- Submission CSV created: false
- Hidden labels/external scraping: false
- Aggressive runner guard/quarantine unchanged.
