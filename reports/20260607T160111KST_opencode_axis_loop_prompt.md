# OpenCode-first no-submit improvement-axis discovery tick — 20260607T160111KST

CRITICAL EXECUTION RULES:
- Answer/act entirely yourself in this single OpenCode run. Do NOT delegate to sub-agents or parallel agents. Do NOT say you are waiting on agents.
- You are a constrained validation-only worker inside `/opt/data/kaggle/kmu-rec-sys-26-steam`.
- If you cannot identify a safe fresh axis, write the required JSON/MD reports and end with `OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS`.

## Hard safety contract (non-negotiable)
Forbidden actions:
- NEVER run `kaggle competitions submit` or any Kaggle API submit call.
- NEVER create full-test candidate/submission/uploadable CSVs, especially under `submissions/` or root `candidate*.csv` / `submission*.csv`.
- NEVER use hidden labels, private answers, leaked data, or scrape external Steam data.
- NEVER print credentials/tokens/secrets.
- NEVER weaken quarantine/guard/similarity logic.
- NEVER stage, commit, push, or schedule cron jobs.

Allowed actions:
- Inspect repo-trusted scripts, reports, logs, state, and validation artifacts.
- Write validation-only scripts/artifacts/logs/reports under `scripts/`, `logs/`, `reports/`, or non-uploadable `artifacts/` only.
- If you implement or launch a probe, it must be bounded, validation-only, no-submit, no-full-test-candidate, and must write a JSON and Markdown report. Prefer CPU/trusted scripts. Use GPU only if truly needed. If GPU is needed, use the V100-compatible stack:
  `uv run --python 3.13 --extra-index-url https://download.pytorch.org/whl/cu128 --with 'torch==2.10.0+cu128' ...`

## Current objective
Find a real independent improvement axis beyond current public-best behavior.
- Known public best: `submissions/candidate_rank_blend_emb128_emb192.csv`, public score `0.77825`.
- Main internal uniform reference: `emb128 4-seed`, reference `0.76505`.
- Weak one-split or ±0.0007 blips are NOT candidates; they require expansion.
- Candidate-like escalation requires multi-split support, fixes > breaks / paired significance when available, no near-duplicate/quarantine conflict, and strict gate below.

## Strict gate to apply to any completed probe
Only `STRICT_PASS` if ALL are true:
- `validation_only=true`
- `candidate_csv_written=false`
- `kaggle_submit_executed=false`
- mean Δ >= `+0.0015`
- min Δ >= `0`
- `3/3` positive splits
- fixes > breaks
- pooled exact / paired p < `0.05` when available
- no near-duplicate/quarantine conflict
Otherwise classify as `WEAK_SIGNAL`, `REJECT`, `STALLED_INCOMPLETE`, or `NO_SAFE_AXIS`.

## Required evidence to inspect before proposing anything
1. Required stale/weak reports:
   - `reports/20260606T132450KST_userknn_gated_residual_fine.json` (expected absent/stalled; check if anything new exists)
   - `reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.json` and `.md` (expected absent/incomplete; check log)
   - `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log`
2. Latest loop/reconciliation:
   - `reports/20260607T145505KST_improvement_axis_cron_status.json`
   - `reports/20260607T145142KST_after_als_manual_no_safe_axis_stop_summary.json`
   - `reports/20260607T145927KST_after_als_exit13_iter03_reconciliation.json`
   - `reports/20260607T144515KST_axis_loop_iter03_opencode.json` and `.md`
3. Quarantine/closure ledgers:
   - `reports/failed_axes.json`
   - `state/aggressive_quota_runner_state.json`
4. Recent validation-only evidence, if relevant:
   - `reports/20260607T125601KST_current_best_residual_atlas.json`
   - `reports/20260607T130533KST_current_best_als_independent_confirmation.json`
   - OTTO/source-covisit reports from 20260607 morning, only as closed/negative context.

## Known closed axes to avoid repeating
Do NOT relaunch or retune these unless you have a materially new independent validation design that is not a near-duplicate:
- DNS pool=1 rejected as split-specific noise: three_uniform_panel mean deltas negative, positive 1/3.
- Hours-confidence edge weighting: no gain.
- Exact-K subset objective: no gain.
- Temporal compatibility: large regression.
- Boundary covariate residual: weak/pop-trap.
- SL@K-lite: all splits negative.
- Last-slot sparse agreement: reject.
- Raw semantic/README/LM text probes: weak or redundant.
- Capacity frontier/emb192 marginal public noise.
- Public-tested rankblend variants did not safely beat current best.
- OTTO/source-separated co-visitation: independent confirmation failed strict gate; forced public 0.77815 < current 0.77825.
- Current-best ALS/rankblend residual diagnostics: pre-registered row failed strict gate; diagnostic row 3/3 and p<0.05 but mean only +0.00113356 and same-family/quarantine-conflicted.
- UserKNN gated residual fine-grid and jackknife uncertainty boundary: currently stalled/weak/incomplete, not candidate axes.

## Your task
1. Inspect the evidence above and current repo state.
2. If there is a credible fresh independent validation-only axis, either:
   - implement/launch ONE bounded validation-only probe, or
   - write a precise implementation/launch command that Hermes can run, if you decide launching inside this run is unsafe.
   The probe must not write uploadable full-test CSVs and must have report JSON/MD paths.
3. If no credible fresh axis exists, do NOT launch busywork. Write a `NO_SAFE_AXIS` report.
4. In all cases, write exactly these two reports:
   - JSON: `reports/20260607T160111KST_opencode_improvement_axis_loop.json`
   - Markdown: `reports/20260607T160111KST_opencode_improvement_axis_loop.md`
5. The JSON must include at least:
```json
{
  "timestamp_kst": "20260607T160111KST",
  "verdict": "STRICT_PASS|WEAK_SIGNAL|REJECT|STALLED_INCOMPLETE|NO_SAFE_AXIS|NEXT_PROBE_RUNNING|FAILED_BLOCKER",
  "safety_flags": {
    "validation_only": true,
    "candidate_csv_written": false,
    "full_test_candidate_or_submission_csv_created": false,
    "kaggle_submit_executed": false,
    "hidden_labels_used": false,
    "private_answers_used": false,
    "external_steam_scraping_used": false,
    "credentials_or_tokens_printed": false,
    "quarantine_or_guard_logic_weakened": false,
    "git_stage_commit_push_executed": false,
    "recursive_cron_scheduled": false
  },
  "evidence_reviewed": [],
  "active_or_completed_processes": [],
  "completed_probe_classification": null,
  "new_probe": {"launched": false, "command": null, "pid": null, "log": null, "report_json": null, "report_md": null},
  "best_or_top_metrics": null,
  "ranked_next_axis_hypotheses": [],
  "why_not_candidate": null,
  "sentinel": "OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS"
}
```
6. End your final assistant text with exactly one of:
   - `OPENCODE_AXIS_LOOP_DONE_IMPROVEMENT_AXIS_FOUND`
   - `OPENCODE_AXIS_LOOP_DONE_NEXT_PROBE_RUNNING`
   - `OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS`
   - `OPENCODE_AXIS_LOOP_DONE_WEAK_SIGNAL`
   - `OPENCODE_AXIS_LOOP_DONE_FAILED_BLOCKER`

Remember: this cron tick must not submit to Kaggle and must not materialize any full-test candidate/submission CSV.
