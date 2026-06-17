# Semantic Residual Probe — Qwen/Qwen3-Embedding-0.6B

Validation-only probe using fold-train in-bundle review text. No Kaggle submit and no candidate/submission CSV were created.

## Safety flags

- validation_only: true
- kaggle_submit_executed: false
- candidate_csv_written: false
- hidden_label_access: false
- external_steam_scraping: false
- text source: `data/raw/public/data/train.json` joined only by each fold's `train_interactions.csv.row_idx`

## Coverage

| split | base acc | items w/text | emb dim | bin covered rows | missing user | missing item | reused |
|---|---:|---:|---:|---:|---:|---:|---:|
| val_random_uniform_seed42 | 0.76505 | 2437 | 1024 | 19996 | 0 | 0 | True |

## Top variants by mean delta

| variant | mean Δ | pos splits | fixes | breaks | Fisher p | strict pass |
|---|---:|---:|---:|---:|---:|---:|
| `base_plus_am0p020_sem_bin_resid` | +0.000200 | 1/3 | 51 | 47 | 0.762 | False |
| `base_plus_am0p020_sem_bin` | +0.000200 | 1/3 | 49 | 45 | 0.7572 | False |
| `base_plus_ap0p010_sem_htr` | +0.000200 | 1/3 | 28 | 24 | 0.6778 | False |
| `base_plus_ap0p010_sem_bin_resid` | +0.000200 | 1/3 | 26 | 22 | 0.6655 | False |
| `base_plus_ap0p010_sem_bin` | +0.000200 | 1/3 | 24 | 20 | 0.6516 | False |
| `base_plus_ap0p050_sem_bin_resid` | +0.000100 | 1/3 | 127 | 125 | 0.9498 | False |
| `base_plus_ap0p010_sem_htr_resid` | +0.000100 | 1/3 | 28 | 26 | 0.8919 | False |
| `base_plus_am0p010_sem_bin_resid` | +0.000100 | 1/3 | 26 | 24 | 0.8877 | False |
| `base_plus_am0p010_sem_bin` | +0.000100 | 1/3 | 25 | 23 | 0.8854 | False |
| `base_plus_am0p010_sem_htr_resid` | +0.000100 | 1/3 | 25 | 23 | 0.8854 | False |
| `base_plus_am0p010_sem_htr` | +0.000000 | 0/3 | 25 | 25 | 1 | False |
| `base_plus_ap0p005_sem_bin` | +0.000000 | 0/3 | 9 | 9 | 1 | False |
| `base_plus_ap0p005_sem_htr` | +0.000000 | 0/3 | 9 | 9 | 1 | False |
| `base_plus_ap0p005_sem_htr_resid` | +0.000000 | 0/3 | 9 | 9 | 1 | False |
| `base_plus_ap0p100_sem_htr` | -0.000100 | 0/3 | 219 | 221 | 0.962 | False |

## Verdict

- verdict: `NO_STRICT_PASS`
- best variant: `base_plus_am0p020_sem_bin_resid`
- best mean Δ: +0.000200 versus MDE +0.00355
- best positive splits: 1/3
- best flips: fixes=51, breaks=47

## Notes

- `sem_*_resid` variants remove linear association with within-user base score and log-popularity before blending.
- The grid is exploratory/predeclared for validation triage only; below-MDE positives are manual-risk/no-submit signals, not candidates.
- Output score artifacts are validation-labeled diagnostics under `artifacts/semantic_residual_probe/`, not submission-like `ID,Label` files.
