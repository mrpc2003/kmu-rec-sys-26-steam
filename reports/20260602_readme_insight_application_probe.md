# README insight application probe

- verdict: **README_MANUAL_RISK_ONLY**
- source README SHA256: `039a986734b47097be4cf0eea03ad3a8ce2adc2eaa56c920cbd3016c52f36576`
- base: `emb128 L4 reg1e-3 4-seed LightGCN`
- splits: `val_random_uniform_seed42, val_random_uniform_seed7, val_random_uniform_seed123`
- variants: `219`
- strict pass count: `0` (requires all 3 splits positive, mean Δ ≥ 0.00355, fixes>breaks, p<0.05)
- manual-risk signal count: `2`
- safety: validation-only; no hidden/test labels; no candidate CSV; no Kaggle submit.

## README hints applied / reconciled

| README hint | Applied here | Outcome |
|---|---|---|
| Per-user positive:negative = 1:1 | Base and all variants decode with per-user top-half on each validation user | hard constraint preserved |
| Popularity + CF hybrid / alpha tuning | Exact README-style BPR/ALS + positive popularity alphas evaluated, then re-added as z/rank/raw residuals on the final LightGCN base | see table below |
| hours_transformed confidence | `bpr_htr` / `als_htr` axes included in the residual grid | see table below |
| LightGCN | current strongest LightGCN ensemble used as the base to test whether README residuals still add signal | base verified on all splits |
| Ensemble | evaluated fixed weighted z/rank and boundary-only ensembles against the base; no submission artifact written | see verdict |

## Base verification

| split | base row acc | expected |
|---|---:|---:|
| `val_random_uniform_seed42` | 0.765053 | 0.765053 |
| `val_random_uniform_seed7` | 0.760952 | 0.760952 |
| `val_random_uniform_seed123` | 0.759952 | 0.759952 |

## Standalone README-style BPR/ALS/pop floors

| split | best standalone README axis | row acc | Δ vs base |
|---|---|---:|---:|
| `val_random_uniform_seed42` | `score_als_f32_it30_alpha20_popa4` | 0.736847 | -0.028206 |
| `val_random_uniform_seed7` | `score_als_f32_it30_alpha20_popa8` | 0.734547 | -0.026405 |
| `val_random_uniform_seed123` | `score_als_f32_it30_alpha20_popa4` | 0.733747 | -0.026205 |

## Top README-derived residual variants

| rank | variant | family | axis | mean Δ | min~max Δ | splits+ | fixes | breaks | p | strict |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `z_base_plus_score_als_htr_f32_it30_alpha20_popa8_w0.05` | readme_weighted_z | `score_als_htr_f32_it30_alpha20_popa8` | +0.000834 | +0.000200~+0.001200 | 3/3 | 173 | 123 | 0.00432 | False |
| 2 | `z_base_plus_score_als_htr_f32_it30_alpha20_popa4_w0.05` | readme_weighted_z | `score_als_htr_f32_it30_alpha20_popa4` | +0.000800 | +0.000400~+0.001000 | 3/3 | 186 | 138 | 0.008922 | False |
| 3 | `z_base_plus_score_als_f32_it30_alpha20_popa4_w0.2` | readme_weighted_z | `score_als_f32_it30_alpha20_popa4` | +0.000900 | +0.000000~+0.002000 | 2/3 | 567 | 513 | 0.1068 | False |
| 4 | `z_base_plus_score_als_f32_it30_alpha20_popa4_w0.1` | readme_weighted_z | `score_als_f32_it30_alpha20_popa4` | +0.000800 | +0.000400~+0.001200 | 3/3 | 326 | 278 | 0.05573 | False |
| 5 | `z_base_plus_score_als_f32_it30_alpha20_w0.1` | readme_weighted_z | `score_als_f32_it30_alpha20` | +0.000567 | -0.000100~+0.000900 | 2/3 | 519 | 485 | 0.2977 | False |
| 6 | `z_base_plus_score_als_f32_it30_alpha20_popa8_w0.1` | readme_weighted_z | `score_als_f32_it30_alpha20_popa8` | +0.000567 | -0.000300~+0.001400 | 2/3 | 293 | 259 | 0.1601 | False |
| 7 | `z_base_plus_score_als_f32_it30_alpha20_popa4_w0.05` | readme_weighted_z | `score_als_f32_it30_alpha20_popa4` | +0.000567 | +0.000300~+0.001000 | 3/3 | 172 | 138 | 0.06072 | False |
| 8 | `z_base_plus_score_als_f32_it30_alpha20_popa8_w0.05` | readme_weighted_z | `score_als_f32_it30_alpha20_popa8` | +0.000567 | +0.000100~+0.001200 | 3/3 | 165 | 131 | 0.05492 | False |
| 9 | `z_base_plus_score_als_f32_it30_alpha20_popa8_w0.2` | readme_weighted_z | `score_als_f32_it30_alpha20_popa8` | +0.000533 | -0.000200~+0.001500 | 2/3 | 524 | 492 | 0.3308 | False |
| 10 | `z_base_plus_score_bpr_f32_it100_popa4_w0.05` | readme_weighted_z | `score_bpr_f32_it100_popa4` | +0.000533 | +0.000100~+0.000900 | 3/3 | 170 | 138 | 0.07716 | False |
| 11 | `z_base_plus_score_pop_log_w0.05` | readme_weighted_z | `score_pop_log` | +0.000500 | +0.000300~+0.000700 | 3/3 | 190 | 160 | 0.121 | False |
| 12 | `z_base_plus_score_als_htr_f32_it30_alpha20_popa4_w0.1` | readme_weighted_z | `score_als_htr_f32_it30_alpha20_popa4` | +0.000467 | -0.000100~+0.000800 | 2/3 | 324 | 296 | 0.2782 | False |
| 13 | `z_base_plus_score_pop_sqrt_w0.05` | readme_weighted_z | `score_pop_sqrt` | +0.000467 | +0.000300~+0.000700 | 3/3 | 175 | 147 | 0.1323 | False |
| 14 | `z_base_plus_score_als_f32_it30_alpha20_w0.05` | readme_weighted_z | `score_als_f32_it30_alpha20` | +0.000433 | -0.000200~+0.001000 | 2/3 | 297 | 271 | 0.2942 | False |
| 15 | `z_base_plus_score_bpr_htr_f32_it100_popa4_w0.05` | readme_weighted_z | `score_bpr_htr_f32_it100_popa4` | +0.000433 | +0.000100~+0.000800 | 3/3 | 164 | 138 | 0.1501 | False |
| 16 | `z_base_plus_score_als_htr_f32_it30_alpha20_popa8_w0.2` | readme_weighted_z | `score_als_htr_f32_it30_alpha20_popa8` | +0.000400 | -0.000600~+0.001300 | 2/3 | 561 | 537 | 0.4876 | False |
| 17 | `z_base_plus_score_pop_norm_w0.05` | readme_weighted_z | `score_pop_norm` | +0.000400 | +0.000000~+0.000800 | 2/3 | 157 | 133 | 0.1767 | False |
| 18 | `z_base_plus_score_als_f32_it30_alpha20_w0.025` | readme_weighted_z | `score_als_f32_it30_alpha20` | +0.000333 | -0.000500~+0.000900 | 2/3 | 144 | 124 | 0.2458 | False |
| 19 | `z_base_plus_score_bpr_f32_it100_w0.05` | readme_weighted_z | `score_bpr_f32_it100` | +0.000300 | -0.000300~+0.001400 | 1/3 | 343 | 325 | 0.5107 | False |
| 20 | `z_base_plus_score_als_htr_f32_it30_alpha20_popa4_w0.025` | readme_weighted_z | `score_als_htr_f32_it30_alpha20_popa4` | +0.000300 | +0.000200~+0.000400 | 3/3 | 100 | 82 | 0.2075 | False |
| 21 | `z_base_plus_score_bpr_f32_it100_popa8_w0.05` | readme_weighted_z | `score_bpr_f32_it100_popa8` | +0.000300 | -0.000100~+0.000900 | 2/3 | 158 | 140 | 0.3247 | False |
| 22 | `z_base_plus_score_als_htr_f32_it30_alpha20_popa8_w0.025` | readme_weighted_z | `score_als_htr_f32_it30_alpha20_popa8` | +0.000300 | +0.000000~+0.000500 | 2/3 | 85 | 67 | 0.1677 | False |
| 23 | `z_base_plus_score_als_f32_it30_alpha20_popa4_w0.025` | readme_weighted_z | `score_als_f32_it30_alpha20_popa4` | +0.000267 | -0.000100~+0.000900 | 1/3 | 91 | 75 | 0.2442 | False |
| 24 | `raw_base_plus_score_pop_sqrt_w0.02` | readme_raw_pop_prior | `score_pop_sqrt` | +0.000267 | +0.000000~+0.000700 | 2/3 | 61 | 45 | 0.1448 | False |
| 25 | `raw_base_plus_score_pop_log_w0.02` | readme_raw_pop_prior | `score_pop_log` | +0.000233 | +0.000100~+0.000300 | 3/3 | 75 | 61 | 0.2649 | False |

## Interpretation

README의 구조 힌트(유저별 top-half, LightGCN, popularity+CF, hours confidence)는 모두 실제 파이프라인에 적용되었고, 이 프로브는 특히 final backbone 위에 남은 README residual을 다시 얹어본 검증이다.
Mean Δ가 MDE 0.00355를 넘는 strict pass가 없으면, README에서 얻은 신호는 이미 LightGCN backbone에 흡수됐거나 popularity sampler artifact로 남은 것으로 본다.
