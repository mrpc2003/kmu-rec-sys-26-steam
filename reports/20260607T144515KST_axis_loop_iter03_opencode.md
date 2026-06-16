# OpenCode axis loop iteration 03 — 20260607T144515KST

## Decision

Verdict: **NO_SAFE_AXIS**

I did not launch a new probe. After reviewing the two prior loop reports for this same run, the manual no-safe-axis stop summary, current-best ALS independent confirmation, the controller gate logic, and the runnable script inventory, I found no materially fresh bounded validation-only axis that avoids the closed/stalled/quarantined families.

## Best remaining metric and why it is rejected

- Variant: `diagnostic_atlas_als_f32_popa4_w0.20_band1_from_20260607T130533KST_current_best_als_independent_confirmation`
- Mean delta vs rankblend: `+0.0011335600453423744`
- Min delta vs rankblend: `+0.0004000800160031126`
- Positive splits: `3/3`
- Fixes/breaks: `462/394`
- Pooled exact p: `0.021965674090633346`
- Quarantine/same-family conflict: `true`

This row is not a candidate because it fails the strict `mean_delta >= +0.0015` threshold, is the same ALS/rankblend residual diagnostic already rejected, and was not the pre-registered independent row. The pre-registered ALS row failed with mean `+0.0008001600320064103`, min `-0.0003000600120024455`, `2/3` positive splits, and p `0.12924401684163647`.

## Search performed

- Read `reports/20260607T144515KST_axis_loop_iter01_opencode.json`.
- Read `reports/20260607T144515KST_axis_loop_iter02_opencode.json`.
- Read `reports/20260607T145142KST_after_als_manual_no_safe_axis_stop_summary.json`.
- Read `reports/20260607T130533KST_current_best_als_independent_confirmation.json`.
- Inspected `scripts/current_best_residual_atlas.py` and `scripts/opencode_hermes_axis_rejection_loop.py`.
- Ran direct `rg` searches over scripts/reports/state for validation, probe, quarantine, candidate/full-test, and submit markers.
- Used ast-grep to inventory Python `__main__` entrypoints and `to_csv` write sites.

## Ranked next-axis review

1. **Exact-K/subset-loss or candidate-count-aware LightGCN variants**: not launched; exact-K is explicitly closed/stalled in the prompt.
2. **Hyperbolic/SGL/XSimGCL/DirectAU/SASRec/DIN/MultiVAE/AlphaRec/TurboCF families**: not launched; prior reports classify them as redundant, weak, below floor, or too large for a credible bounded probe.
3. **Boundary/frontier/aggressive gated residuals or pairwise factories**: not launched; prompt and state mark these as closed/quarantined/public-negative or near-duplicate risk.
4. **UserKNN cleanup or jackknife expansion**: not launched; both are stalled/incomplete and not a materially new independent axis.
5. **Semantic/text/README/LM residuals**: not launched; prompt classifies these as weak or redundant.

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
- recursive_cron_scheduled: `false`

## Report paths

- JSON: `reports/20260607T144515KST_axis_loop_iter03_opencode.json`
- Markdown: `reports/20260607T144515KST_axis_loop_iter03_opencode.md`
