# OpenCode improvement-axis loop — 20260607T063604KST

## Verdict

`NO_SAFE_AXIS`

I did not launch a new probe. Current evidence does not justify a fresh, independent, bounded validation-only run that could plausibly clear the strict gate without repeating a stalled/weak/closed/quarantined family or creating a disallowed one-split blip.

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

## Active process and resource observations

- `ps -eo pid,ppid,stat,comm,args | rg -i 'kaggle|submit|aggressive_quota|userknn|jackknife|python|cuda|train|probe'` matched only:
  - Hermes gateway process.
  - The Hermes wrapper/current `opencode run` for this exact prompt.
  - The transient process-search shell/`rg` command itself.
- No active `kaggle competitions submit` process was observed.
- No active `scripts/aggressive_quota_runner.py` process was observed in this run.
- No active UserKNN fine-grid or jackknife expanded probe process was observed.
- GPU snapshot: V100 GPU0 `0 MiB/0%`, GPU1 `0 MiB/1%`, GPU2 `0 MiB/0%`, GPU3 `4320 MiB/2%`.

## Completed and incomplete probe classification

| Axis / artifact | Classification | Evidence |
|---|---|---|
| UserKNN gated residual fine grid | `STALLED_INCOMPLETE` | `reports/20260606T132450KST_userknn_gated_residual_fine.{json,md}` are still missing. `logs/userknn_gated_residual_fine_20260606T132450KST.log` contains 40,121 lines dominated by repeated `RuntimeWarning: invalid value encountered in divide` from `scripts/userknn_residual_probe.py:114`, with no metric report. Relaunching the broad grid is explicitly disallowed. |
| Jackknife uncertainty boundary expanded | `FAILED_INCOMPLETE_NO_METRIC_REPORT` | `reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.{json,md}` are still missing. PID file contains `28646`; current process scan did not find it live. Log has 12 lines and stops mid `val_random_uniform_seed123`. |
| Jackknife uncertainty boundary smoke | `WEAK_SIGNAL_STRICT_GATE_FAIL` | `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json`: best `vote_consensus__high_capacity_gap__B1__w0.1`, mean Δ `+0.0003667400`, min Δ `-0.0012002400`, positive splits `2/3`, fixes/breaks `252/230`, pooled exact p `0.338815`. Fails mean, min, 3/3 positive, and p gates. |
| UserKNN gated residual smoke | `WEAK_SIGNAL_STRICT_GATE_FAIL` | `reports/20260603T180707KST_userknn_gated_residual_probe.json`: top mean Δ `+0.0009001800`, min Δ `+0.0002000400`, positive splits `3/3`, fixes/breaks `406/352`, pooled exact p `0.0541518`. Mean is below +0.0015 and p is not `< 0.05`; lower-mean p-passing variants remain below the strict mean threshold. |
| DNS pool=1 panel | `CLOSED_NO_CANDIDATE` | `reports/20260606T125011KST_dns_pool1_panel_aggregate.json` plus prompt state classify it as split-specific noise with negative mean panel deltas and only `1/3` positive splits. |
| Last-slot sparse agreement | `REJECT` | `reports/20260604T115818KST_last_slot_sparse_agreement_probe.json` is a closed negative/weak axis. |
| Rankblend/ALS residual, boundary scoreblend, frontier/capacity, TAG-CF full-test families | `CLOSED_OR_BLOCKED_BY_QUARANTINE` | `state/aggressive_quota_runner_state.json` records public-negative/non-improving transfer: rankblend ALS residuals, boundary scoreblend, frontier/capacity, and TAG-CF variants all scored below current public best `0.77825` and are quarantined or skipped for near-family reruns. |

## Strict gate check

Required gate:

- mean Δ `>= +0.0015`
- min Δ `>= 0`
- `3/3` positive uniform splits
- fixes `>` breaks
- pooled exact/McNemar p `< 0.05` when available
- `validation_only=true`
- `candidate_csv_written=false`
- `kaggle_submit_executed=false`
- no near-duplicate or quarantine conflict

No inspected completed axis satisfies all of these. The closest completed UserKNN smoke is stable-looking but below the required mean and p thresholds. The jackknife smoke is both weaker and split-negative. The larger expansions are not results because they lack JSON/MD metric reports.

## New probe decision

No new probe was launched.

Reason: a bounded validation-only run must be fresh and independent. At this point the available candidates would either repeat UserKNN, jackknife/boundary, rankblend/residual, DNS/capacity, TAG-CF, semantic/text, temporal, exact-K, SL@K-lite, or last-slot axes already weak/negative/closed, or they would be underpowered diagnostics with no credible route to strict pass. The explicit instruction also forbids relaunching the broad UserKNN fine-grid and the same jackknife expanded grid.

## Ranked next-axis hypotheses

1. **Tiny UserKNN finite-value/complexity diagnostic after isolating the invalid-divide path**
   - Not launched now because it is implementation diagnostics, not a credible improvement axis. It should only be considered if it is capped and proves finite features before any three-split probe.
2. **New non-leaky out-of-fold residual calibration using only existing validation artifacts**
   - Not launched now because nearby residual/rankblend/boundary families show negative public transfer and quarantine conflict.
3. **Genuinely new backbone/loss family beyond the LightGCN capacity frontier**
   - Not launched now because SASRec, TAG-CF, SGL/XSimGCL, Hyperbolic, MultiVAE, AlphaRec, semantic/text, temporal, exact-K, and hours-confidence directions are already weak/negative/closed or too large for a bounded validation-only run.
4. **DNS/capacity micro-ensemble refinement**
   - Not launched now because existing DNS/capacity/frontier evidence is marginal, split-specific, public-noise-prone, and below the strict gate.

## Inspected evidence

- Reports:
  - `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json`
  - `reports/20260603T180707KST_userknn_gated_residual_probe.json`
  - `reports/20260607T052646KST_opencode_improvement_axis_loop.json`
  - `reports/20260607T041626KST_opencode_improvement_axis_loop.json`
  - `reports/20260607T053207KST_improvement_axis_cron_status.json`
  - `reports/20260606T202326KST_repeat_until_candidate_status.md`
  - `reports/20260606T224445KST_repeat_until_candidate_status.json`
- Logs:
  - `logs/userknn_gated_residual_fine_20260606T132450KST.log`
  - `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log`
  - `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.pid`
- State/policy:
  - `state/aggressive_quota_runner_state.json`
  - `state/autonomous_submission_policy.json`
- Trusted script surfaces searched/seen:
  - `scripts/userknn_gated_residual_probe.py`
  - `scripts/userknn_residual_probe.py`
  - `scripts/jackknife_uncertainty_boundary_probe.py`
  - `scripts/aggressive_quota_runner.py`
  - `scripts/rank_blend_mcnemar.py`
- Direct searches used `rg` for strict gate, quarantine, closed-axis, and safety evidence, plus `ast-grep` over Python subprocess call sites.

## Artifacts produced by this run

- `reports/20260607T063604KST_opencode_improvement_axis_loop.md`
- `reports/20260607T063604KST_opencode_improvement_axis_loop.json`

No logs, validation artifacts, candidate CSVs, submission CSVs, Kaggle submissions, staged files, commits, or pushes were produced.
