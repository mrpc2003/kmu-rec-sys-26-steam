# Semantic Residual Probe — nomic-ai/modernbert-embed-base

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
| val_random_uniform_seed42 | 0.76505 | 2437 | 768 | 19996 | 0 | 0 | False |
| val_random_uniform_seed7 | 0.76095 | 2437 | 768 | 19996 | 0 | 0 | False |
| val_random_uniform_seed123 | 0.75995 | 2437 | 768 | 19996 | 0 | 0 | False |

## Top variants by mean delta

| variant | mean Δ | pos splits | fixes | breaks | Fisher p | strict pass |
|---|---:|---:|---:|---:|---:|---:|
| `base_plus_ap0p050_sem_htr_resid` | +0.000433 | 2/3 | 349 | 323 | 0.5044 | False |
| `base_plus_am0p020_sem_htr` | +0.000367 | 3/3 | 164 | 142 | 0.6702 | False |
| `base_plus_ap0p050_sem_htr` | +0.000300 | 2/3 | 340 | 322 | 0.3314 | False |
| `base_plus_am0p010_sem_htr` | +0.000267 | 3/3 | 77 | 61 | 0.6597 | False |
| `base_plus_am0p010_sem_htr_resid` | +0.000233 | 2/3 | 76 | 62 | 0.6747 | False |
| `base_plus_ap0p005_sem_bin_resid` | +0.000233 | 2/3 | 36 | 22 | 0.1179 | False |
| `base_plus_am0p020_sem_htr_resid` | +0.000200 | 2/3 | 161 | 149 | 0.724 | False |
| `base_plus_am0p020_sem_bin` | +0.000200 | 2/3 | 151 | 139 | 0.7097 | False |
| `base_plus_am0p010_sem_bin_resid` | +0.000200 | 2/3 | 75 | 63 | 0.6509 | False |
| `base_plus_am0p010_sem_bin` | +0.000200 | 2/3 | 73 | 61 | 0.6352 | False |
| `base_plus_ap0p005_sem_bin` | +0.000200 | 2/3 | 33 | 21 | 0.2282 | False |
| `base_plus_am0p020_sem_bin_resid` | +0.000167 | 2/3 | 155 | 145 | 0.803 | False |
| `base_plus_ap0p010_sem_htr` | +0.000167 | 2/3 | 64 | 54 | 0.3715 | False |
| `base_plus_ap0p005_sem_htr` | +0.000167 | 2/3 | 33 | 23 | 0.205 | False |
| `base_plus_ap0p100_sem_bin` | +0.000167 | 1/3 | 687 | 677 | 0.6711 | False |

## Verdict

- verdict: `NO_STRICT_PASS`
- best variant: `base_plus_ap0p050_sem_htr_resid`
- best mean Δ: +0.000433 versus MDE +0.00355
- best positive splits: 2/3
- best flips: fixes=349, breaks=323

## Notes

- `sem_*_resid` variants remove linear association with within-user base score and log-popularity before blending.
- The grid is exploratory/predeclared for validation triage only; below-MDE positives are manual-risk/no-submit signals, not candidates.
- Output score artifacts are validation-labeled diagnostics under `artifacts/semantic_residual_probe/`, not submission-like `ID,Label` files.
