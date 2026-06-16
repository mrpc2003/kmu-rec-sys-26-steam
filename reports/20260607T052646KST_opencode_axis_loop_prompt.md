# KMURecSys26 no-submit improvement-axis loop prompt — 20260607T052646KST

CRITICAL EXECUTION RULES: Answer and act entirely yourself in this single OpenCode run. Do NOT delegate to sub-agents. Do NOT wait for parallel agents. Do NOT say you are waiting. Stay within this repository only: `/opt/data/kaggle/kmu-rec-sys-26-steam`.

## Hard safety contract

Forbidden actions, with no exceptions in this run:
- Do NOT run `kaggle competitions submit` or any command that submits to Kaggle.
- Do NOT create any full-test candidate/submission CSV or uploadable artifact; do not write under `submissions/`.
- Do NOT use hidden labels, private answers, leaderboard probing, or external Steam scraping/data collection.
- Do NOT print credentials/tokens or read raw credential files.
- Do NOT weaken quarantine, duplicate, or guard logic.
- Do NOT git stage, commit, push, or change remotes.
- Do NOT recursively schedule cron jobs.

Allowed actions:
- Inspect existing repo files/reports/logs/state.
- Write validation-only reports under `reports/` and logs under `logs/`.
- If and only if you find a genuinely fresh, independent, bounded validation-only axis, you may implement or run a tiny validation-only diagnostic/probe using repo-trusted scripts. It must be bounded to a short foreground run, must not create full-test candidate CSVs, and must set/report `validation_only=true`, `candidate_csv_written=false`, `kaggle_submit_executed=false`.
- Prefer no probe over a weak one-split blip or repeating a closed/stalled/quarantined axis.

## Current objective and strict gate

Find a real improvement axis beyond current public best behavior.
- Known public best: `candidate_rank_blend_emb128_emb192.csv`, public `0.77825`.
- Main internal uniform reference: emb128 4-seed ref `0.76505`.
- Weak one-split blips around `±0.0007` are not candidates; they require expansion only if an axis is not closed/stalled and has a credible independent rationale.

STRICT_PASS requires all of:
- mean Δ `>= +0.0015`
- min Δ `>= 0`
- 3/3 positive splits
- fixes > breaks
- pooled exact/McNemar p `< 0.05` when available
- `validation_only=true`
- `candidate_csv_written=false`
- `kaggle_submit_executed=false`
- no near-duplicate/quarantine conflict

Otherwise classify as `WEAK_SIGNAL`, `REJECT`, `STALLED_INCOMPLETE`, `FAILED_INCOMPLETE_NO_METRIC_REPORT`, or `NO_SAFE_AXIS`.

## Evidence you must inspect before deciding

1. Active/current status from Hermes this tick:
   - No Hermes background process is active.
   - OS process scan found only a pre-existing `scripts/aggressive_quota_runner.py` pair, not started by this cron. Latest watchdog `logs/aggressive_quota_runner_watchdog_20260607T052059KST.log` says quota remaining 5 but no validation-positive unsubmitted variant; it is idle/waiting.
   - GPU snapshot: V100 GPUs mostly idle; GPU3 has 4320 MiB used by an unrelated/pre-existing process. CPU/memory are available.
2. Required recent reports/logs:
   - UserKNN expected report `reports/20260606T132450KST_userknn_gated_residual_fine.json` is missing; only `logs/userknn_gated_residual_fine_20260606T132450KST.log` exists. Previous cron status classified it as `STALLED_INCOMPLETE` after ~14h51m with repeated invalid-divide warnings and no metric report/artifacts. Do not relaunch the same broad fine-grid.
   - Jackknife expanded expected reports `reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.{json,md}` are missing; `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log` is only 12 lines and stopped mid `val_random_uniform_seed123`; PID file value 28646 is not live. Classify as `FAILED_INCOMPLETE_NO_METRIC_REPORT`, not a result.
   - Jackknife smoke report `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json` exists and is `WEAK_SIGNAL`: top `vote_consensus__high_capacity_gap__B1__w0.1`, mean Δ `+0.0003667400`, min Δ `-0.0012002400`, positive splits `2/3`, fixes/breaks `252/230`, p `0.3388`.
   - Previous OpenCode loop `reports/20260607T041626KST_opencode_improvement_axis_loop.{md,json}` returned `NO_SAFE_AXIS`; do not merely repeat its text—either find a materially new axis or confirm no safe axis with updated evidence.
   - Previous overall status `reports/20260607T042202KST_improvement_axis_cron_status.{md,json}` recorded the same closures and no new probe.
3. Runner/quarantine context:
   - `state/aggressive_quota_runner_state.json` lists quarantined/public-negative families including rankblend ALS/BPR residuals, boundary scoreblend, frontier/capacity, and TAG-CF full-test.
   - `state/autonomous_submission_policy.json` exists, but THIS cron is hard no-submit/no-candidate regardless of that older policy.
4. Known closed axes to avoid repeating:
   - DNS pool=1 rejected as split-specific noise (`three_uniform_panel` mean deltas negative, positive 1/3).
   - hours-confidence edge weighting no gain.
   - exact-K subset objective no gain.
   - temporal compatibility large regression.
   - boundary covariate residual weak/pop-trap.
   - SL@K-lite all splits negative.
   - last-slot sparse agreement reject.
   - raw semantic/README/LM text probes weak or redundant.
   - capacity frontier/emb192 marginal public noise.
   - public-tested rankblend variants did not safely beat current best.

## Your task

Inspect the evidence and current trusted scripts. Then choose exactly one of:

A. `NO_SAFE_AXIS`: write the required Markdown/JSON reports explaining why no fresh independent bounded validation-only axis is credible now. Do not launch a probe.

B. `NEXT_PROBE_COMPLETED_OR_RUNNING`: only if you find a materially fresh and bounded validation-only axis that does not repeat the closed/stalled/quarantined list, implement/run at most one tiny bounded validation-only probe. It must produce a report with strict classification. It must not create candidate/submission CSVs or use Kaggle submit. If a probe is still running when your OpenCode run ends, record PID/log/path and why it is bounded; avoid duplicate runners.

A tiny UserKNN NaN/complexity diagnostic is allowed only if it is materially different from the stalled broad fine-grid, has hard row/split/candidate caps, proves finite intermediate values, and writes only reports/logs. If it is merely a rerun of the fine-grid, choose NO_SAFE_AXIS.

## Required output files

Write:
- `reports/20260607T052646KST_opencode_improvement_axis_loop.md`
- `reports/20260607T052646KST_opencode_improvement_axis_loop.json`

The JSON must include:
- `verdict`
- `safety_flags` with the booleans above
- `active_processes_observed`
- `completed_probe_classification`
- `new_probe` object with `launched`, `command`, `pid_file`, `log`, `report_json`, `report_md`, `artifact_dir`
- `ranked_next_axis_hypotheses`
- `artifacts_reports_produced`

End your final OpenCode text with exactly one sentinel line, one of:
- `OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS`
- `OPENCODE_AXIS_LOOP_DONE_NEXT_PROBE_RUNNING`
- `OPENCODE_AXIS_LOOP_DONE_IMPROVEMENT_AXIS_FOUND`
- `OPENCODE_AXIS_LOOP_DONE_WEAK_OR_REJECTED`

Remember: no Kaggle submit, no full-test candidate/submission CSV, no external Steam scraping, no credentials, no git staging/commit/push.
