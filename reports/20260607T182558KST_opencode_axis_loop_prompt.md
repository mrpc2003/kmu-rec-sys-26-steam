# OpenCode-first no-submit improvement-axis loop — 20260607T182558KST

CRITICAL EXECUTION RULES:
- Answer/act entirely yourself in this single OpenCode run. Do NOT delegate to sub-agents. Do NOT wait for parallel agents. Do NOT say you are waiting on other agents.
- Work only inside repo `/opt/data/kaggle/kmu-rec-sys-26-steam`.
- This is a constrained validation-only Kaggle RecSys research tick. Hermes will independently verify your outputs.

## Hard safety contract
Forbidden actions:
1. Do not run `kaggle competitions submit` or any Kaggle submission/API submit path.
2. Do not create full-test candidate/submission CSVs, uploadable prediction files, or files under `submissions/`.
3. Do not use hidden labels, private answers, leaked data, or test labels.
4. Do not scrape or collect external Steam data.
5. Do not print credentials/tokens/API keys or read raw auth files.
6. Do not weaken quarantine/guard logic, failed-axis ledgers, or autonomous safety policy.
7. Do not git stage/commit/push.
8. Do not create or modify cron jobs; no recursive scheduling.
9. Do not launch untracked long-lived background processes. If a validation probe cannot finish within this bounded run, write a probe design/report only.

Allowed actions:
- Inspect existing repo-trusted scripts, reports, state, artifacts, and logs.
- Write ONLY the required report JSON/MD listed below, plus compact validation-only diagnostic artifacts under `reports/`, `logs/`, or `artifacts/` if you actually run a bounded validation-only probe.
- You may run a small bounded validation-only probe only if it is a fresh independent axis and does not write candidate/full-test CSVs.

## Current objective and strict gate
Find a real improvement axis beyond current public-best behavior.
Known public best: `candidate_rank_blend_emb128_emb192.csv`, public score `0.77825`.
Main internal uniform reference: `emb128` 4-seed reference `0.76505`.
Weak one-split blips around ±0.0007 are not candidates; they require expansion. Candidate-like escalation requires all of:
- `validation_only=true`
- `candidate_csv_written=false`
- `kaggle_submit_executed=false`
- mean Δ >= +0.0015
- min Δ >= 0
- 3/3 positive splits
- fixes > breaks
- pooled exact paired p < 0.05 when available
- no near-duplicate, failed-axis, or quarantine conflict

## Current reconstructed state from Hermes at 2026-06-07 18:25 KST
- Hermes tracked background processes: none.
- No live relevant validation process was found except current checker wrappers.
- Actual Kaggle submit-like processes: none.
- GPU summary: GPU0/1/2 idle 0 MiB; GPU3 shows 4320 MiB but `nvidia-smi pmon` shows no owning process and no repo process matched. Treat this as stale/orphan display; do not launch a GPU job just to use it.
- Pre-run `submissions/*.csv` count: 23. Do not change it.
- Tracked dirty files pre-exist: `reports/failed_axes.json`, `scripts/aggressive_quota_runner.py`, `state/aggressive_quota_runner_state.json`. Do not stage/commit/push.

## Required specific checks to include
1. `reports/20260606T132450KST_userknn_gated_residual_fine.json` and `.md` are absent. Log `logs/userknn_gated_residual_fine_20260606T132450KST.log` exists but prior ticks classified it as warning-dominated `STALLED_INCOMPLETE`; do not relaunch the broad fine-grid.
2. `reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.json` and `.md` are absent. Log `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log` is 12 lines, stops after `val_random_uniform_seed123` invalid-value warning, and has no final report.
3. `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json` is `WEAK_SIGNAL`: top `vote_consensus__high_capacity_gap__B1__w0.1`, mean Δ `+0.0003667400146696309`, min Δ `-0.0012002400480095599`, positive `2/3`, fixes/breaks `252/230`, p `0.33881500709211204`.
4. Latest prior no-safe tick: `reports/20260607T172126KST_improvement_axis_cron_status.json` and `reports/20260607T171551KST_opencode_improvement_axis_loop.json` returned `NO_SAFE_AXIS`, launched no probe, and had clean safety flags.
5. Read and respect `reports/failed_axes.json`, `state/aggressive_quota_runner_state.json`, `reports/20260607T125601KST_current_best_residual_atlas.json`, and `reports/20260607T130533KST_current_best_als_independent_confirmation.json` if present.

## Known closed axes to avoid repeating
- DNS pool=1 rejected as split-specific noise: three_uniform_panel mean deltas negative, positive 1/3.
- hours-confidence edge weighting: no gain.
- exact-K subset objective: no gain.
- temporal compatibility: large regression.
- boundary covariate residual: weak/pop-trap.
- SL@K-lite: all splits negative.
- last-slot sparse agreement: reject.
- raw semantic/README/LM text probes: weak or redundant.
- capacity frontier/emb192: marginal public noise.
- public-tested rankblend variants: did not safely beat current best.
- OTTO/source-separated co-visitation: independent validation failed strict gate and forced public score `0.77815` did not beat `0.77825`.
- ALS/rankblend/current-best residual diagnostics: strongest diagnostic row was below +0.0015 mean gate and quarantine/same-family conflicted; pre-registered independent row had negative min split and non-significant p.
- UserKNN gated fine-grid and jackknife expanded: stalled/incomplete; do not relaunch broad duplicates.

## What to do in this run
A. Inspect current evidence and script surfaces compactly.
B. If you find a genuinely fresh independent validation-only axis with enough rationale, either:
   - run one bounded validation-only probe that finishes in this OpenCode run and writes only reports/logs/artifacts, OR
   - if it is too large, write a precise probe design and classify as `WEAK_SIGNAL`/`NO_SAFE_AXIS` rather than launching.
C. If no credible fresh axis exists, do not force a probe. Return `NO_SAFE_AXIS` with evidence.
D. Never create candidate/full-test CSVs; never submit; never schedule; never stage/commit.

## Required output files
Write both files:
- `reports/20260607T182558KST_opencode_improvement_axis_loop.json`
- `reports/20260607T182558KST_opencode_improvement_axis_loop.md`

The JSON must include at least:
```json
{
  "timestamp_kst": "20260607T182558KST",
  "verdict": "STRICT_PASS|WEAK_SIGNAL|REJECT|NO_SAFE_AXIS|NEXT_PROBE_RUNNING|FAILED_BLOCKER",
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
  "best_or_top_metrics": {},
  "ranked_next_axis_hypotheses": [],
  "why_not_candidate": "...",
  "sentinel": "OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS"
}
```

Use exactly one final sentinel line in the Markdown and in your final assistant text, chosen from:
- `OPENCODE_AXIS_LOOP_DONE_IMPROVEMENT_AXIS_FOUND`
- `OPENCODE_AXIS_LOOP_DONE_NEXT_PROBE_RUNNING`
- `OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS`
- `OPENCODE_AXIS_LOOP_DONE_WEAK_SIGNAL`
- `OPENCODE_AXIS_LOOP_DONE_FAILED_BLOCKER`
