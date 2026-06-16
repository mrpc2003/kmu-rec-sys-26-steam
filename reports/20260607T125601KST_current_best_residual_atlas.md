# Current-best residual atlas — 20260607T125604KST

Validation-only diagnostic around `rank_blend_emb128_emb192` current-best style. No Kaggle submission, no full-test candidate CSV.

## Verdict

- verdict: `STRICT_PASS_DIAGNOSTIC_NEEDS_INDEPENDENT_CONFIRMATION`
- strict_diagnostic_pass_count: `5`
- tested_variants: `896`

## Base rankblend validation

- val_random_uniform_seed42: acc=`0.764653`, errors=`4706`, FP/FN=`2353/2353`
- val_random_uniform_seed7: acc=`0.761152`, errors=`4776`, FP/FN=`2388/2388`
- val_random_uniform_seed123: acc=`0.759052`, errors=`4818`, FP/FN=`2409/2409`

## Top diagnostic residual variants

| rank | variant | mean Δ | min Δ | pos | fixes/breaks | p | strict diag |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `zrankblend_plus_score_als_f32_it30_alpha20_popa4_w+0.30_band2` | +0.001734 | +0.001100 | 3/3 | 532/428 | 0.0008772 | True |
| 2 | `zrankblend_plus_score_als_f32_it30_alpha20_popa4_w+0.30_band1` | +0.001667 | +0.001000 | 3/3 | 518/418 | 0.001201 | True |
| 3 | `zrankblend_plus_score_als_f32_it30_alpha20_popa4_w+0.30_all` | +0.001634 | +0.000900 | 3/3 | 531/433 | 0.001769 | True |
| 4 | `zrankblend_plus_score_als_f32_it30_alpha20_popa4_w+0.30_band3` | +0.001634 | +0.000900 | 3/3 | 531/433 | 0.001769 | True |
| 5 | `zrankblend_plus_score_als_f32_it30_alpha20_popa4_w+0.20_band1` | +0.001534 | +0.001000 | 3/3 | 469/377 | 0.00174 | True |
| 6 | `zrankblend_plus_score_als_htr_f32_it30_alpha20_popa8_w+0.30_band2` | +0.001467 | +0.001100 | 3/3 | 516/428 | 0.004606 | False |
| 7 | `zrankblend_plus_score_als_htr_f32_it30_alpha20_popa4_w+0.30_band2` | +0.001467 | +0.001400 | 3/3 | 532/444 | 0.005329 | False |
| 8 | `zrankblend_plus_compat_item_date_w+0.10_all` | +0.001434 | +0.001100 | 3/3 | 458/372 | 0.00315 | False |
| 9 | `zrankblend_plus_compat_item_date_w+0.10_band3` | +0.001434 | +0.001100 | 3/3 | 458/372 | 0.00315 | False |
| 10 | `zrankblend_plus_score_als_htr_f32_it30_alpha20_popa4_w+0.20_all` | +0.001434 | +0.001000 | 3/3 | 457/371 | 0.003114 | False |
| 11 | `zrankblend_plus_score_als_htr_f32_it30_alpha20_popa4_w+0.20_band2` | +0.001434 | +0.001000 | 3/3 | 457/371 | 0.003114 | False |
| 12 | `zrankblend_plus_score_als_htr_f32_it30_alpha20_popa4_w+0.20_band3` | +0.001434 | +0.001000 | 3/3 | 457/371 | 0.003114 | False |
| 13 | `zrankblend_plus_compat_item_date_w+0.10_band1` | +0.001434 | +0.000900 | 3/3 | 452/366 | 0.002936 | False |
| 14 | `zrankblend_plus_score_als_f32_it30_alpha20_popa4_w+0.20_band2` | +0.001434 | +0.000800 | 3/3 | 469/383 | 0.003566 | False |
| 15 | `zrankblend_plus_compat_item_date_w+0.10_band2` | +0.001400 | +0.001100 | 3/3 | 456/372 | 0.003895 | False |
| 16 | `zrankblend_plus_score_als_htr_f32_it30_alpha20_popa8_w+0.30_all` | +0.001400 | +0.001100 | 3/3 | 516/432 | 0.006992 | False |
| 17 | `zrankblend_plus_score_als_htr_f32_it30_alpha20_popa8_w+0.30_band3` | +0.001400 | +0.001100 | 3/3 | 516/432 | 0.006992 | False |
| 18 | `zrankblend_plus_score_als_htr_f32_it30_alpha20_popa4_w+0.30_all` | +0.001400 | +0.001300 | 3/3 | 532/448 | 0.007985 | False |
| 19 | `zrankblend_plus_score_als_htr_f32_it30_alpha20_popa4_w+0.30_band3` | +0.001400 | +0.001300 | 3/3 | 532/448 | 0.007985 | False |
| 20 | `zrankblend_plus_score_als_f32_it30_alpha20_popa4_w+0.20_all` | +0.001400 | +0.000700 | 3/3 | 469/385 | 0.004481 | False |

## Stable false-negative vs false-positive feature gaps

Positive `FN-FP` means missed positives had larger feature values than false positives. This is diagnostic only, not a submission rule.

| rank | feature | mean FN-FP | + splits | - splits |
|---:|---|---:|---:|---:|
| 1 | `rel_item_text_vs_user` | +40.691792 | 3 | 0 |
| 2 | `item_text_mean` | +40.691792 | 3 | 0 |
| 3 | `rel_item_date_vs_user` | +27.491532 | 3 | 0 |
| 4 | `item_date_mean` | +27.491532 | 3 | 0 |
| 5 | `compat_item_text` | -27.460299 | 0 | 3 |
| 6 | `item_hours_mean` | -18.257764 | 0 | 3 |
| 7 | `compat_item_date` | -5.851672 | 0 | 3 |
| 8 | `score_als_htr_f32_it30_alpha20_popa8` | -0.743035 | 0 | 3 |
| 9 | `log_item_pop` | -0.728138 | 0 | 3 |
| 10 | `rel_log_item_pop_vs_user_hist` | -0.728138 | 0 | 3 |
| 11 | `score_als_f32_it30_alpha20_popa8` | -0.686082 | 0 | 3 |
| 12 | `score_bpr_f32_it100_popa8` | -0.575964 | 0 | 3 |
| 13 | `rel_item_htr_vs_user` | -0.479683 | 0 | 3 |
| 14 | `item_htr_mean` | -0.479683 | 0 | 3 |
| 15 | `score_als_htr_f32_it30_alpha20_popa4` | -0.460631 | 0 | 3 |
| 16 | `score_als_f32_it30_alpha20_popa4` | -0.403678 | 0 | 3 |
| 17 | `compat_log_item_pop` | -0.333507 | 0 | 3 |
| 18 | `score_bpr_f32_it100_popa4` | -0.293560 | 0 | 3 |
| 19 | `compat_item_htr` | -0.103073 | 0 | 3 |
| 20 | `rank_disagreement_abs` | +0.046527 | 3 | 0 |

## Highest-error buckets per split

### val_random_uniform_seed42
| bucket | rows | base acc | error | FP | FN |
|---|---:|---:|---:|---:|---:|
| `log_user_deg_q5` | 4032 | 0.7133 | 0.2867 | 578 | 578 |
| `log_item_pop_q3` | 4132 | 0.7144 | 0.2856 | 566 | 614 |
| `log_item_pop_q4` | 4029 | 0.7173 | 0.2827 | 759 | 380 |
| `boundary_margin_le_1` | 9472 | 0.7197 | 0.2803 | 1317 | 1338 |
| `compat_log_item_pop_q4` | 3999 | 0.7289 | 0.2711 | 657 | 427 |
| `compat_log_item_pop_q5` | 4000 | 0.7348 | 0.2652 | 685 | 376 |
| `log_item_pop_q2` | 3925 | 0.7419 | 0.2581 | 337 | 676 |
| `item_htr_mean_q3` | 3998 | 0.7451 | 0.2549 | 518 | 501 |
| `boundary_margin_le_2` | 14664 | 0.7466 | 0.2534 | 1851 | 1865 |
| `compat_item_htr_q5` | 4000 | 0.7472 | 0.2527 | 568 | 443 |

### val_random_uniform_seed7
| bucket | rows | base acc | error | FP | FN |
|---|---:|---:|---:|---:|---:|
| `log_item_pop_q3` | 4068 | 0.6890 | 0.3110 | 625 | 640 |
| `log_user_deg_q5` | 4032 | 0.7093 | 0.2907 | 586 | 586 |
| `boundary_margin_le_1` | 9472 | 0.7184 | 0.2816 | 1335 | 1332 |
| `log_item_pop_q4` | 4025 | 0.7205 | 0.2795 | 739 | 386 |
| `compat_log_item_pop_q5` | 4000 | 0.7235 | 0.2765 | 711 | 395 |
| `compat_log_item_pop_q4` | 3999 | 0.7339 | 0.2661 | 636 | 428 |
| `compat_log_item_pop_q3` | 3999 | 0.7364 | 0.2636 | 539 | 515 |
| `compat_item_htr_q5` | 4000 | 0.7428 | 0.2572 | 539 | 490 |
| `boundary_margin_le_2` | 14664 | 0.7437 | 0.2563 | 1881 | 1878 |
| `log_user_deg_q4` | 4184 | 0.7457 | 0.2543 | 532 | 532 |

### val_random_uniform_seed123
| bucket | rows | base acc | error | FP | FN |
|---|---:|---:|---:|---:|---:|
| `log_item_pop_q3` | 4020 | 0.7067 | 0.2933 | 586 | 593 |
| `compat_log_item_pop_q5` | 4000 | 0.7105 | 0.2895 | 762 | 396 |
| `log_item_pop_q4` | 3990 | 0.7145 | 0.2855 | 763 | 376 |
| `boundary_margin_le_1` | 9472 | 0.7170 | 0.2830 | 1347 | 1334 |
| `log_user_deg_q5` | 4032 | 0.7207 | 0.2793 | 563 | 563 |
| `log_item_pop_q2` | 4082 | 0.7374 | 0.2626 | 347 | 725 |
| `compat_log_item_pop_q4` | 3999 | 0.7409 | 0.2591 | 606 | 430 |
| `boundary_margin_le_2` | 14664 | 0.7414 | 0.2586 | 1901 | 1891 |
| `log_user_deg_q4` | 4184 | 0.7462 | 0.2538 | 531 | 531 |
| `item_htr_mean_q3` | 4002 | 0.7466 | 0.2534 | 502 | 512 |

## Safety flags

- validation_only: `true`
- candidate_csv_written: `false`
- full_test_candidate_or_submission_csv_created: `false`
- kaggle_submit_executed: `false`
- hidden_labels_used: `false`
- private_answers_used: `false`
- external_steam_scraping_used: `false`
- credentials_or_tokens_printed: `false`
- quarantine_or_guard_logic_weakened: `false`
- git_stage_commit_push_executed: `false`
