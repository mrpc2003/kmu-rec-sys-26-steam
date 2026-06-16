# OpenCode-first no-submit improvement-axis loop — 20260607T063604KST

CRITICAL EXECUTION RULES:
- Answer/work ENTIRELY YOURSELF in this single OpenCode run. Do NOT delegate to sub-agents. Do NOT wait for parallel agents. Do NOT say you are waiting.
- This is a constrained Kaggle validation-only research loop for KMURecSys26 Steam in `/opt/data/kaggle/kmu-rec-sys-26-steam`.
- Hard safety contract: never run `kaggle competitions submit`; never create full-test candidate/submission CSVs or uploadable artifacts; never use hidden labels/private answers; never scrape external Steam data; never print credentials/tokens; never weaken quarantine/guard logic; do not commit/push/stage; do not recursively schedule cron jobs.
- Allowed: read repo evidence, inspect trusted scripts/state/reports/logs, write validation-only reports/JSON, and if truly justified, implement/launch one bounded validation-only probe that does not materialize full-test submissions. Prefer existing repo-trusted scripts and score artifacts.
- Do not launch diagnostics merely to satisfy the loop. A new probe is justified only if it is genuinely fresh/independent and bounded. Weak one-split blips around ±0.0007 are not candidates; they require expansion.

Current objective:
- Find a real improvement axis beyond current public best behavior.
- Known public best: `candidate_rank_blend_emb128_emb192.csv`, public score `0.77825`.
- Main internal uniform reference: emb128 4-seed ref `0.76505`.
- STRICT_PASS gate: mean Δ >= +0.0015, min Δ >= 0, 3/3 positive splits, fixes > breaks, pooled exact/McNemar p < 0.05 when available, `validation_only=true`, `candidate_csv_written=false`, `kaggle_submit_executed=false`, no near-duplicate/quarantine conflict.

Known closed/rejected axes to avoid repeating:
- DNS pool=1 rejected as split-specific noise: three_uniform_panel mean deltas negative, positive 1/3.
- hours-confidence edge weighting no gain.
- exact-K subset objective no gain.
- temporal compatibility large regression.
- boundary covariate residual weak/pop-trap.
- SL@K-lite all splits negative.
- last-slot sparse agreement reject.
- raw semantic/README/LM text probes weak or redundant.
- capacity frontier/emb192 marginal public noise.
- public-tested rankblend variants did not safely beat current best.
- UserKNN broad fine-grid stalled/incomplete; do not relaunch it broadly.
- Jackknife uncertainty boundary smoke weak and expanded run failed/incomplete; do not relaunch same expanded grid.

State just observed by Hermes before this OpenCode run:
- Hermes background process list was empty.
- GPUs: V100 GPU0 0 MiB/0%, GPU1 0 MiB/1%, GPU2 0 MiB/0%, GPU3 4320 MiB/1%.
- Required report check:
  - `reports/20260606T132450KST_userknn_gated_residual_fine.json` missing.
  - `reports/20260606T132450KST_userknn_gated_residual_fine.md` missing.
  - `logs/userknn_gated_residual_fine_20260606T132450KST.log` exists but is 40,121 lines of repeated `RuntimeWarning: invalid value encountered in divide`; no metric report.
  - `reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.json` missing.
  - `reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.md` missing.
  - `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.pid` contains 28646, not live.
  - `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log` has 12 lines, stops mid `val_random_uniform_seed123`.
- Previous completed classifications:
  - `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json`: top `vote_consensus__high_capacity_gap__B1__w0.1`, mean Δ +0.0003667400, min Δ -0.0012002400, positive splits 2/3, fixes/breaks 252/230, p=0.338815 => WEAK_SIGNAL_STRICT_GATE_FAIL.
  - `reports/20260603T180707KST_userknn_gated_residual_probe.json`: best mean Δ +0.0009001800, min Δ +0.0002000400, positive 3/3, fixes/breaks 406/352, p=0.05415 => WEAK_SIGNAL_STRICT_GATE_FAIL.
- A pre-existing `scripts/aggressive_quota_runner.py` was observed and stopped by Hermes because this cron's current hard contract is no-submit/no-candidate. Do not restart it. No active `kaggle competitions submit` process remained after stop.

Required output artifacts for this OpenCode run:
1. Write Markdown report: `reports/20260607T063604KST_opencode_improvement_axis_loop.md`.
2. Write JSON report: `reports/20260607T063604KST_opencode_improvement_axis_loop.json`.
3. JSON must include: `verdict` (`STRICT_PASS`, `WEAK_SIGNAL`, `REJECT`, or `NO_SAFE_AXIS`), `safety_flags`, `active_processes_observed`, `completed_probe_classification`, `new_probe` object with `launched`/paths/command, `ranked_next_axis_hypotheses`, `strict_gate`, `inspected_evidence`, and `artifacts_reports_produced`.
4. If no credible fresh axis exists, do not launch any probe; set verdict `NO_SAFE_AXIS` and explain precisely why.
5. If you launch a probe, it must be bounded validation-only, write logs/reports under `logs/` and `reports/`, avoid full-test submission artifacts, and explicitly record `validation_only=true`, `candidate_csv_written=false`, `kaggle_submit_executed=false`.
6. End your final text with exactly one sentinel line, one of:
   - `OPENCODE_AXIS_LOOP_DONE_IMPROVEMENT_AXIS_FOUND`
   - `OPENCODE_AXIS_LOOP_DONE_NEXT_PROBE_RUNNING`
   - `OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS`
   - `OPENCODE_AXIS_LOOP_DONE_BLOCKED`

Do not output secrets. Do not submit to Kaggle. Do not create submission/candidate CSVs. Do not commit/stage/push.
