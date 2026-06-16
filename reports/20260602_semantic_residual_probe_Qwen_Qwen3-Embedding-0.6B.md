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
| val_random_uniform_seed7 | 0.76095 | 2437 | 1024 | 19996 | 0 | 0 | True |
| val_random_uniform_seed123 | 0.75995 | 2437 | 1024 | 19996 | 0 | 0 | True |

## Top variants by mean delta

| variant | mean Δ | pos splits | fixes | breaks | Fisher p | strict pass |
|---|---:|---:|---:|---:|---:|---:|
| `base_plus_ap0p010_sem_htr_resid` | +0.000367 | 3/3 | 77 | 55 | 0.2128 | False |
| `base_plus_ap0p010_sem_htr` | +0.000367 | 3/3 | 72 | 50 | 0.2264 | False |
| `base_plus_ap0p010_sem_bin` | +0.000300 | 3/3 | 67 | 49 | 0.4761 | False |
| `base_plus_ap0p010_sem_bin_resid` | +0.000267 | 3/3 | 69 | 53 | 0.606 | False |
| `base_plus_ap0p005_sem_htr` | +0.000233 | 2/3 | 38 | 24 | 0.3562 | False |
| `base_plus_am0p020_sem_bin_resid` | +0.000167 | 2/3 | 164 | 154 | 0.9624 | False |
| `base_plus_ap0p020_sem_htr_resid` | +0.000167 | 2/3 | 137 | 127 | 0.4946 | False |
| `base_plus_ap0p005_sem_htr_resid` | +0.000167 | 2/3 | 34 | 24 | 0.6812 | False |
| `base_plus_ap0p005_sem_bin` | +0.000133 | 2/3 | 34 | 26 | 0.8463 | False |
| `base_plus_ap0p020_sem_htr` | +0.000133 | 2/3 | 133 | 125 | 0.2905 | False |
| `base_plus_am0p010_sem_htr_resid` | +0.000100 | 2/3 | 85 | 79 | 0.9871 | False |
| `base_plus_ap0p005_sem_bin_resid` | +0.000100 | 2/3 | 33 | 27 | 0.8041 | False |
| `base_plus_am0p020_sem_bin` | +0.000067 | 2/3 | 156 | 152 | 0.8783 | False |
| `base_plus_am0p010_sem_htr` | +0.000067 | 1/3 | 82 | 78 | 0.9935 | False |
| `base_plus_ap0p020_sem_bin` | +0.000067 | 2/3 | 126 | 122 | 0.4858 | False |

## Verdict

- verdict: `NO_STRICT_PASS`
- best variant: `base_plus_ap0p010_sem_htr_resid`
- best mean Δ: +0.000367 versus MDE +0.00355
- best positive splits: 3/3
- best flips: fixes=77, breaks=55

## Notes

- `sem_*_resid` variants remove linear association with within-user base score and log-popularity before blending.
- The grid is exploratory/predeclared for validation triage only; below-MDE positives are manual-risk/no-submit signals, not candidates.
- Output score artifacts are validation-labeled diagnostics under `artifacts/semantic_residual_probe/`, not submission-like `ID,Label` files.
