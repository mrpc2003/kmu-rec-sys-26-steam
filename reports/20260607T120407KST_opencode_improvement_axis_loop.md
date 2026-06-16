# OpenCode improvement-axis loop — 20260607T120407KST

## Verdict

`NO_SAFE_AXIS`

No fresh bounded validation-only improvement axis was launched. The only recent axis with apparent signal, OTTO/source-separated co-visitation, failed independent strict confirmation and then failed public transfer versus the current best `candidate_rank_blend_emb128_emb192.csv` public `0.77825`.

## Safety

- validation_only: `true`
- candidate_csv_written: `false`
- full_test_candidate_or_submission_csv_created: `false`
- kaggle_submit_executed: `false`
- hidden/private labels used: `false`
- external Steam scraping used: `false`
- credentials/tokens printed: `false`
- quarantine/guard logic weakened: `false`
- git stage/commit/push executed: `false`
- recursive cron scheduled: `false`

## Top inspected metric

Best actionable recent strict-confirmation row inspected:

- Variant: `pre_registered_old_panel_top_coplay_top5_reverse_recent`
- mean Δ vs base: `+0.0006668000`
- min Δ vs base: `-0.0006001200`
- positive splits: `2/3`
- fixes / breaks: `424 / 384`
- pooled exact p: `0.1700198675`
- quarantine/near-duplicate conflict: `true`, because the same OTTO family later produced public `0.77815`, below current best `0.77825`
- strict gate pass: `false`

## Closed or avoided axes

- **OTTO/source-separated co-visitation and reverse-recent residuals**: `closed_or_avoided` — Initial old-panel smoke looked promising, but independent strict confirmation failed: mean Δ +0.0006668000, min Δ -0.0006001200, 2/3 positive splits, fixes/breaks 424/384, pooled p 0.1700199. Forced public probe scored 0.77815, below current best 0.77825.
- **DNS pool=1 / dynamic negative sampling**: `closed_or_avoided` — Prompt and reports classify as split-specific noise with negative three-uniform-panel mean and only 1/3 positive splits.
- **UserKNN gated residual broad/fine grids**: `closed_or_avoided` — Fine-grid expected report is missing after stalled invalid-divide warnings; prior UserKNN surfaces are not a safe fresh axis and broad relaunch is explicitly forbidden.
- **Jackknife uncertainty boundary**: `weak_signal_avoided` — Smoke top row vote_consensus__high_capacity_gap__B1__w0.1: mean Δ +0.0003667400, min Δ -0.0012002400, 2/3 positive, fixes/breaks 252/230, p 0.338815; expanded report absent.
- **Boundary/frontier/capacity/rankblend residual/TAG-CF families**: `quarantine_or_public_negative_avoided` — aggressive_quota_runner_state records public-negative transfer and quarantines for ALS rankblend residuals, boundary scoreblends, frontier capacity variants, and TAG-CF full-test family.
- **Hours-confidence, exact-K subset objective, temporal compatibility, boundary covariate residual, SL@K-lite, last-slot sparse agreement**: `closed_or_avoided` — Prompt lists these as no-gain, regression, weak/pop-trap, all-splits-negative, or rejected; no strict-gate-supporting fresh independent artifact was found.
- **Raw semantic/README/LM text residuals and similar train-review semantic probes**: `closed_or_avoided` — Prior reports and prompt classify as weak or redundant; no bounded validation-only script surface appears likely to produce 3/3 positive support beyond current public best.
- **Full-test materialization scripts and autonomous submit runner**: `forbidden_this_tick` — Code-surface search found materialize/submission paths, including aggressive_quota_runner and materializers; none were run because the hard contract forbids Kaggle submit and full-test/uploadable candidate CSV creation.

## Inspected artifacts

- `README.md`
- `reports/failed_axes.json`
- `state/aggressive_quota_runner_state.json`
- `reports/20260607T120407KST_opencode_axis_loop_prompt.md`
- `reports/20260607T120218KST_opencode_hermes_axis_rejection_loop_summary.json`
- `reports/20260607T120245KST_axis_loop_iter01_opencode.json`
- `reports/20260607T090941KST_opencode_improvement_axis_loop.json`
- `reports/20260607T095549KST_otto_independent_uniform_confirmation.json`
- `reports/20260607T095549KST_otto_independent_uniform_confirmation.md`
- `reports/20260607T113300KST_otto_independent_uniform_reconciliation.md`
- `reports/20260607T114059KST_otto_forced_post_submission_analysis.json`
- `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json`
- `reports/20260603T174950KST_userknn_residual_probe.json`
- `reports/20260603T180707KST_userknn_gated_residual_probe.json`
- `reports/20260604T115818KST_last_slot_sparse_agreement_probe.json`
- `scripts/otto_source_covisit_smoke.py`
- `scripts/otto_source_covisit_followup_grid.py`
- `scripts/userknn_gated_residual_probe.py`
- `scripts/jackknife_uncertainty_boundary_probe.py`
- `scripts/aggressive_quota_runner.py`
- `scripts/materialize_otto_forced_candidate.py`
- `scripts/materialize_rank_blend_emb128_emb192.py`

## Next-axis notes

1. A genuinely independent validation-label-free base-model family could be considered only if it can be evaluated against the current live-best/rankblend anchor on multiple validation splits without full-test materialization.
2. UserKNN cleanup is lower priority because it is a stalled/closed family and a broad relaunch is explicitly disallowed.
3. Boundary/frontier/capacity/rankblend/jackknife-style diagnostics should remain closed unless a truly independent validation design passes the strict gate.
