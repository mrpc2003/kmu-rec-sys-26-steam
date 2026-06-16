# competition public/private guard v1

- validation_only: true
- no_kaggle_submit: true
- candidate_csv_written: false
- full_test_candidate_materialized: false
- public_lb_feedback_used: true, only as negative-control calibration from already submitted historical candidates

## verdict

`PASS_NEGATIVE_CONTROLS_SEPARATED`

This is a guard harness, not a model. It makes the competition-writeup lesson operational: do not trust small normal-validation deltas when they resemble already public-failed row families.

## negative-control separation

- controls checked: 11
- validation/public pairs with normal validation positive but public negative: 10
- those rejected by guard-adjusted delta: 10
- separated families: als_residual_rankblend, boundary_scoreblend, frontier_z_boundary, tagcf_boundary
- validation/public delta correlation on usable controls: 0.626794

| candidate | family | val delta | public delta | diff rows | other-failed overlap | boundary≤3 | guard delta | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `emb128_emb64_zblend` | capacity_blend |  | -0.000100 | 596 | 0.924 | 0.998 |  | `FLAG_HIGH_FALSE_POSITIVE_RISK` |
| `als_htr_popa4_w0.025` | als_residual_rankblend | 0.001300 | -0.000200 | 278 | 1.000 | 1.000 | -0.000300 | `REJECT_NEGATIVE_CONTROL` |
| `als_htr_popa4_w0.05` | als_residual_rankblend | 0.001300 | -0.000200 | 278 | 1.000 | 1.000 | -0.000300 | `REJECT_NEGATIVE_CONTROL` |
| `als_htr_popa4_w0.1` | als_residual_rankblend | 0.001300 | -0.000200 | 278 | 1.000 | 1.000 | -0.000300 | `REJECT_NEGATIVE_CONTROL` |
| `als_htr_popa4_w0.2` | als_residual_rankblend | 0.001434 | -0.000400 | 300 | 0.983 | 1.000 | -0.000166 | `REJECT_NEGATIVE_CONTROL` |
| `als_popa4_w0.2` | als_residual_rankblend | 0.001400 | -0.000300 | 276 | 0.964 | 1.000 | -0.000200 | `REJECT_NEGATIVE_CONTROL` |
| `boundary_scoreblend_z128_z192_z64_w-0.75` | boundary_scoreblend | 0.000633 | -0.000700 | 582 | 0.742 | 0.997 | -0.002017 | `REJECT_NEGATIVE_CONTROL` |
| `boundary_scoreblend_z128_z192_z64_w2` | boundary_scoreblend | 0.000433 | -0.002500 | 596 | 0.914 | 1.000 | -0.002567 | `REJECT_NEGATIVE_CONTROL` |
| `frontier_z_w1920_w64-0.25` | frontier_z_boundary | 0.000267 | -0.001100 | 556 | 0.941 | 0.996 | -0.002233 | `REJECT_NEGATIVE_CONTROL` |
| `otto_coplay_top5_reverse_recent_forced` | otto_covisitation | 0.000667 | -0.000100 | 508 | 0.909 | 1.000 | -0.000833 | `REJECT_NEGATIVE_CONTROL` |
| `tagcf_seed2024_sym_a0.1_raw_zblend_bw0.5` | tagcf_boundary | 0.000767 | -0.002100 | 646 | 0.805 | 0.995 | -0.002733 | `REJECT_NEGATIVE_CONTROL` |

## bucket-level false-positive audit

Top buckets below are historical danger zones. They are descriptive guard features, not public-LB tuning thresholds.

| dimension | bucket | changed rows | mismatch frac | public-negative frac | mean public delta | mean validation delta | note |
|---|---|---:|---:|---:|---:|---:|---|
| candidate_count_bucket | 03-04 | 1280 | 0.900 | 1.000 | -0.000862 | 0.000826 | false_positive_bucket |
| item_degree_decile | 4 | 869 | 0.898 | 1.000 | -0.000872 | 0.000828 | false_positive_bucket |
| user_degree_decile | 0 | 546 | 0.892 | 1.000 | -0.000878 | 0.000852 | false_positive_bucket |
| user_degree_decile | 5 | 396 | 0.889 | 1.000 | -0.000823 | 0.000857 | false_positive_bucket |
| item_degree_decile | 6 | 587 | 0.888 | 1.000 | -0.000844 | 0.000827 | false_positive_bucket |
| user_degree_decile | 2 | 444 | 0.887 | 1.000 | -0.000885 | 0.000825 | false_positive_bucket |
| item_degree_decile | 1 | 442 | 0.887 | 1.000 | -0.000867 | 0.000810 | false_positive_bucket |
| user_degree_decile | 6 | 548 | 0.887 | 1.000 | -0.000903 | 0.000807 | false_positive_bucket |
| item_degree_decile | 5 | 766 | 0.886 | 1.000 | -0.000897 | 0.000819 | false_positive_bucket |
| item_degree_decile | 9 | 78 | 0.885 | 1.000 | -0.000886 | 0.000851 | false_positive_bucket |
| user_degree_decile | 1 | 537 | 0.883 | 1.000 | -0.000848 | 0.000888 | false_positive_bucket |
| boundary_distance_bucket | boundary_le1 | 4597 | 0.880 | 1.000 | -0.000875 | 0.000826 | false_positive_bucket |
| candidate_count_bucket | 07-10 | 1160 | 0.878 | 1.000 | -0.000886 | 0.000820 | false_positive_bucket |
| item_degree_decile | 7 | 400 | 0.877 | 1.000 | -0.000847 | 0.000846 | false_positive_bucket |
| user_degree_decile | 8 | 500 | 0.876 | 1.000 | -0.000927 | 0.000761 | false_positive_bucket |
| item_degree_decile | 8 | 241 | 0.876 | 1.000 | -0.000881 | 0.000834 | false_positive_bucket |

## how to use for the next smoke

1. Keep the normal seed42 smoke gate: solo, fixed z-blend, fixes/breaks, corr_z, and no head-only lift.
2. Add this guard report before any escalation. A small positive validation delta is not enough if the row/bucket profile matches `false_positive_bucket` rows.
3. Do not create a full-test candidate only to measure overlap. Full-test overlap is allowed here only because these are already-existing historical negative controls.
4. If a legacy candidate file already exists, compare its changed-row mask to `reports/competition_failed_family_overlap_matrix.csv`; high overlap with failed families is a reject/flag, not a retune prompt.
5. For validation-only new smoke, compare changed-row bucket profile against `reports/competition_false_positive_bucket_audit.csv` and require guard-adjusted evidence above the existing `+0.0007` noise band.

## outputs

- `reports/competition_public_private_guard_v1.md`
- `reports/competition_public_private_guard_v1.json`
- `reports/competition_failed_family_overlap_matrix.csv`
- `reports/competition_false_positive_bucket_audit.csv`

COMPETITION_PUBLIC_PRIVATE_GUARD_V1_DONE
