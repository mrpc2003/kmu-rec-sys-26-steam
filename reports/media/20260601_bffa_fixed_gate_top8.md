| 순위 | fixed variant | mean Δ | min~max Δ | pos splits | fixes/breaks | p | gate |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | zr_feat_item_funny_rate / lam0.05 | +0.00083 | -0.00020~+0.00220 | 2/3 | 366/316 | 0.06053 | fail |
| 2 | zr_feat_cooc_norm_max / lam0.10 | +0.00077 | +0.00050~+0.00130 | 3/3 | 693/647 | 0.21894 | fail |
| 3 | zr_feat_date_compat / lam0.05 | +0.00070 | +0.00030~+0.00110 | 3/3 | 356/314 | 0.11313 | fail |
| 4 | zr_feat_cooc_norm_max / lam0.05 | +0.00050 | +0.00010~+0.00120 | 3/3 | 342/312 | 0.25678 | fail |
| 5 | zr_feat_item_early_access_rate / lam0.05 | +0.00047 | +0.00000~+0.00120 | 2/3 | 238/210 | 0.20203 | fail |
| 6 | zr_feat_cooc_norm_sum / lam0.05 | +0.00043 | -0.00020~+0.00130 | 2/3 | 361/335 | 0.34333 | fail |
| 7 | zr_feat_cooc_norm_mean / lam0.05 | +0.00043 | -0.00020~+0.00130 | 2/3 | 361/335 | 0.34333 | fail |
| 8 | zr_feat_item_hours_std / lam0.05 | +0.00043 | -0.00070~+0.00190 | 2/3 | 351/325 | 0.33629 | fail |
| gate | required | ≥ +0.00355 | all robust | ≥2/3 | fixes>breaks | p<0.05 | strict |
