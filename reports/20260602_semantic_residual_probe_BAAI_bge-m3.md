# Semantic Residual Probe — BAAI/bge-m3

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
| val_random_uniform_seed42 | 0.76505 | 2437 | 1024 | 19996 | 0 | 0 | False |
| val_random_uniform_seed7 | 0.76095 | 2437 | 1024 | 19996 | 0 | 0 | False |
| val_random_uniform_seed123 | 0.75995 | 2437 | 1024 | 19996 | 0 | 0 | False |

## Top variants by mean delta

| variant | mean Δ | pos splits | fixes | breaks | Fisher p | strict pass |
|---|---:|---:|---:|---:|---:|---:|
| `base_plus_ap0p010_sem_htr_resid` | +0.000267 | 2/3 | 75 | 59 | 0.1019 | False |
| `base_plus_ap0p010_sem_bin` | +0.000233 | 2/3 | 75 | 61 | 0.07496 | False |
| `base_plus_ap0p010_sem_bin_resid` | +0.000233 | 2/3 | 75 | 61 | 0.06561 | False |
| `base_plus_ap0p020_sem_htr_resid` | +0.000233 | 1/3 | 135 | 121 | 0.04195 | False |
| `base_plus_ap0p020_sem_bin_resid` | +0.000233 | 1/3 | 138 | 124 | 0.03933 | False |
| `base_plus_ap0p010_sem_htr` | +0.000200 | 1/3 | 74 | 62 | 0.1006 | False |
| `base_plus_ap0p020_sem_htr` | +0.000167 | 1/3 | 130 | 120 | 0.02672 | False |
| `base_plus_ap0p005_sem_htr` | +0.000133 | 2/3 | 35 | 27 | 0.866 | False |
| `base_plus_ap0p005_sem_htr_resid` | +0.000100 | 2/3 | 35 | 29 | 0.9503 | False |
| `base_plus_am0p010_sem_bin` | +0.000067 | 2/3 | 78 | 74 | 0.9709 | False |
| `base_plus_ap0p020_sem_bin` | +0.000067 | 1/3 | 132 | 128 | 0.1111 | False |
| `base_plus_ap0p005_sem_bin` | +0.000033 | 2/3 | 33 | 31 | 0.8044 | False |
| `base_plus_ap0p005_sem_bin_resid` | +0.000033 | 2/3 | 33 | 31 | 0.8044 | False |
| `base_plus_am0p010_sem_bin_resid` | +0.000033 | 2/3 | 76 | 74 | 0.9941 | False |
| `base_plus_am0p020_sem_bin_resid` | +0.000033 | 1/3 | 156 | 154 | 0.9133 | False |

## Verdict

- verdict: `NO_STRICT_PASS`
- best variant: `base_plus_ap0p010_sem_htr_resid`
- best mean Δ: +0.000267 versus MDE +0.00355
- best positive splits: 2/3
- best flips: fixes=75, breaks=59

## Notes

- `sem_*_resid` variants remove linear association with within-user base score and log-popularity before blending.
- The grid is exploratory/predeclared for validation triage only; below-MDE positives are manual-risk/no-submit signals, not candidates.
- Output score artifacts are validation-labeled diagnostics under `artifacts/semantic_residual_probe/`, not submission-like `ID,Label` files.
