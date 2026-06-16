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
| val_random_uniform_seed42 | 0.76505 | 2437 | 128 | 19996 | 0 | 0 | False |

## Top variants by mean delta

| variant | mean Δ | pos splits | fixes | breaks | Fisher p | strict pass |
|---|---:|---:|---:|---:|---:|---:|
| `base_plus_ap0p050_sem_bin` | +0.000500 | 1/3 | 115 | 105 | 0.5441 | False |
| `base_plus_am0p010_sem_htr` | +0.000400 | 1/3 | 22 | 14 | 0.243 | False |
| `base_plus_am0p010_sem_htr_resid` | +0.000400 | 1/3 | 22 | 14 | 0.243 | False |
| `base_plus_ap0p050_sem_bin_resid` | +0.000300 | 1/3 | 114 | 108 | 0.7373 | False |
| `base_plus_ap0p050_sem_htr` | +0.000300 | 1/3 | 113 | 107 | 0.7361 | False |
| `base_plus_ap0p050_sem_htr_resid` | +0.000300 | 1/3 | 113 | 107 | 0.7361 | False |
| `base_plus_ap0p010_sem_htr_resid` | +0.000300 | 1/3 | 28 | 22 | 0.4799 | False |
| `base_plus_ap0p020_sem_bin_resid` | +0.000200 | 1/3 | 50 | 46 | 0.7596 | False |
| `base_plus_ap0p010_sem_bin_resid` | +0.000200 | 1/3 | 26 | 22 | 0.6655 | False |
| `base_plus_am0p010_sem_bin` | +0.000200 | 1/3 | 18 | 14 | 0.5966 | False |
| `base_plus_am0p010_sem_bin_resid` | +0.000200 | 1/3 | 18 | 14 | 0.5966 | False |
| `base_plus_ap0p020_sem_htr_resid` | +0.000100 | 1/3 | 52 | 50 | 0.9212 | False |
| `base_plus_ap0p020_sem_bin` | +0.000100 | 1/3 | 48 | 46 | 0.9179 | False |
| `base_plus_am0p020_sem_htr` | +0.000100 | 1/3 | 41 | 39 | 0.9111 | False |
| `base_plus_ap0p010_sem_htr` | +0.000100 | 1/3 | 26 | 24 | 0.8877 | False |

## Verdict

- verdict: `NO_STRICT_PASS`
- best variant: `base_plus_ap0p050_sem_bin`
- best mean Δ: +0.000500 versus MDE +0.00355
- best positive splits: 1/3
- best flips: fixes=115, breaks=105

## Notes

- `sem_*_resid` variants remove linear association with within-user base score and log-popularity before blending.
- The grid is exploratory/predeclared for validation triage only; below-MDE positives are manual-risk/no-submit signals, not candidates.
- Output score artifacts are validation-labeled diagnostics under `artifacts/semantic_residual_probe/`, not submission-like `ID,Label` files.
