# Semantic Residual Probe — tfidf-svd-smoke

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
| val_random_uniform_seed42 | 0.76505 | 2437 | 128 | 19996 | 0 | 0 | True |

## Top variants by mean delta

| variant | mean Δ | pos splits | fixes | breaks | Fisher p | strict pass |
|---|---:|---:|---:|---:|---:|---:|
| `base_plus_am0p010_sem_htr` | +0.000700 | 1/3 | 30 | 16 | 0.05408 | False |
| `base_plus_am0p010_sem_bin` | +0.000600 | 1/3 | 33 | 21 | 0.1337 | False |
| `base_plus_am0p010_sem_bin_resid` | +0.000500 | 1/3 | 33 | 23 | 0.2288 | False |
| `base_plus_am0p010_sem_htr_resid` | +0.000500 | 1/3 | 30 | 20 | 0.2026 | False |
| `base_plus_ap0p010_sem_bin_resid` | +0.000200 | 1/3 | 23 | 19 | 0.644 | False |
| `base_plus_ap0p010_sem_bin` | +0.000200 | 1/3 | 18 | 14 | 0.5966 | False |
| `base_plus_ap0p005_sem_htr` | +0.000200 | 1/3 | 10 | 6 | 0.4545 | False |
| `base_plus_ap0p005_sem_bin_resid` | +0.000200 | 1/3 | 10 | 6 | 0.4545 | False |
| `base_plus_ap0p005_sem_htr_resid` | +0.000200 | 1/3 | 10 | 6 | 0.4545 | False |
| `base_plus_am0p020_sem_htr_resid` | +0.000100 | 1/3 | 59 | 57 | 0.9261 | False |
| `base_plus_am0p020_sem_htr` | +0.000100 | 1/3 | 57 | 55 | 0.9248 | False |
| `base_plus_ap0p005_sem_bin` | +0.000100 | 1/3 | 11 | 9 | 0.8238 | False |
| `base_plus_am0p020_sem_bin` | +0.000000 | 0/3 | 58 | 58 | 1 | False |
| `base_plus_am0p020_sem_bin_resid` | +0.000000 | 0/3 | 58 | 58 | 1 | False |
| `base_plus_ap0p010_sem_htr` | -0.000100 | 0/3 | 18 | 20 | 0.8714 | False |

## Verdict

- verdict: `NO_STRICT_PASS`
- best variant: `base_plus_am0p010_sem_htr`
- best mean Δ: +0.000700 versus MDE +0.00355
- best positive splits: 1/3
- best flips: fixes=30, breaks=16

## Notes

- `sem_*_resid` variants remove linear association with within-user base score and log-popularity before blending.
- The grid is exploratory/predeclared for validation triage only; below-MDE positives are manual-risk/no-submit signals, not candidates.
- Output score artifacts are validation-labeled diagnostics under `artifacts/semantic_residual_probe/`, not submission-like `ID,Label` files.
