# Current-best ALS independent confirmation reconciliation

- Timestamp: `20260607T144257KST`
- Source report: `reports/20260607T130533KST_current_best_als_independent_confirmation.json`
- Verdict: `INDEPENDENT_WEAK_POSITIVE_STRICT_FAIL`
- Ledger action: `appended` in `reports/failed_axes.json`

## Strict pre-registered row

- Variant: `pre_registered_atlas_top_als_f32_popa4_w0.30_band2`
- Mean Δ vs current-best rankblend: `+0.0008001600`
- Min split Δ: `-0.0003000600`
- Positive splits: `2/3`
- Fixes/breaks: `504/456`
- Pooled exact p: `0.12924401684163647`
- Split deltas: `{'val_random_uniform_seed314': 0.0008001600320064473, 'val_random_uniform_seed2025': 0.001900380076015229, 'val_random_uniform_seed2718': -0.0003000600120024455}`
- Strict gate pass: `false`

## Diagnostic best row

- Variant: `atlas_als_f32_popa4_w0.20_band1`
- Mean Δ: `+0.0011335600`
- Min Δ: `+0.0004000800`
- Positive splits: `3/3`
- Fixes/breaks: `462/394`
- Pooled exact p: `0.021965674090633346`
- Strict gate pass: `false`

## Decision

The ALS residual axis is a real weak-positive research signal but **not candidate/submission eligible**. The pre-registered row fails effect size, min-split, positive-split, and p-value gates. The best diagnostic row is 3/3 positive and p<0.05, but its mean Δ is only `+0.0011335600`, below the `+0.0015` escalation threshold.

## Safety

- validation_only: `true`
- candidate_csv_written: `false`
- full_test_candidate_or_submission_csv_created: `false`
- kaggle_submit_executed: `false`
- hidden_labels_used: `false`
- external_steam_scraping_used: `false`
- git_stage_commit_push_executed: `false`
