# Last-slot validation-only probe: robust rank aggregation (seed42 uniform)
**Safety:** validation_only=true · candidate_csv_written=false · kaggle_submit_executed=false
Rows=19996, users=4736, split=val_random_uniform_seed42. Baseline is emb128 4-seed raw mean.
## Top results
| score | acc | delta_vs_base | flips | fixes | breaks | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| vote_128_then_base | 0.76525 | +0.00020 | 58 | 31 | 27 | 0.6936 |
| boundary_rule_score | 0.76515 | +0.00010 | 28 | 15 | 13 | 0.8501 |
| base_emb128_raw_mean | 0.76505 | +0.00000 | 0 | 0 | 0 | 1.0000 |
| uncert_128_64_lamm0p5 | 0.76485 | -0.00020 | 252 | 124 | 128 | 0.8501 |
| rankagg_128_mean | 0.76475 | -0.00030 | 196 | 95 | 101 | 0.7210 |
| rankagg_128_median | 0.76475 | -0.00030 | 218 | 106 | 112 | 0.7349 |
| rrf_128_k60 | 0.76475 | -0.00030 | 196 | 95 | 101 | 0.7210 |
| uncert_128_lam1p0 | 0.76465 | -0.00040 | 182 | 87 | 95 | 0.6038 |
| rrf_128_k5 | 0.76465 | -0.00040 | 198 | 95 | 103 | 0.6189 |
| rrf_128_k10 | 0.76465 | -0.00040 | 208 | 100 | 108 | 0.6274 |
| uncert_128_lamm0p5 | 0.76455 | -0.00050 | 116 | 53 | 63 | 0.4034 |
| uncert_128_lamm1p0 | 0.76445 | -0.00060 | 198 | 93 | 105 | 0.4344 |

## Gate verdict
**REJECT** — best non-baseline does not exceed +0.00355 MDE and/or McNemar p<0.05 gate. Best non-baseline: `vote_128_then_base` delta=+0.00020, p=0.6936.
