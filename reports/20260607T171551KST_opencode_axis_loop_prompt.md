# KMURecSys26 Steam OpenCode-first no-submit improvement-axis loop — 20260607T171551KST

CRITICAL EXECUTION RULES:
- Answer/work entirely yourself in this single OpenCode run. Do NOT delegate to sub-agents. Do NOT wait for parallel agents. Do NOT say you are waiting.
- Scope: repository `/opt/data/kaggle/kmu-rec-sys-26-steam` only.
- You are a constrained adviser/worker. Hermes will verify and classify your output; do not assume your own verdict is final.
- Bounded run: if you implement/run a probe, keep it validation-only and bounded within this OpenCode run (rough target <= 8 minutes). Do not leave a long-running/background process unless absolutely necessary; if you do, write pid/log/report paths and mark it as running. Prefer a concrete design/report over launching a broad expensive grid.

HARD SAFETY CONTRACT — DO NOT VIOLATE:
- Never run `kaggle competitions submit` or any Kaggle API/CLI submission equivalent.
- Never create full-test candidate/submission CSVs, uploadable submission files, or files under `submissions/`.
- Never use hidden labels/private answers.
- Never scrape external Steam data or collect non-approved external data.
- Never print credentials/tokens/secrets.
- Never weaken quarantine/guard logic or bypass failed-axis ledgers.
- Do not stage, commit, push, or schedule cron jobs.
- Allowed: validation-only scripts, validation-only artifacts/logs/reports under `reports/`, `logs/`, `artifacts/validation_only_*`, and read-only inspection of existing project files.

CURRENT OBJECTIVE:
Find a real improvement axis beyond current public best behavior for KMURecSys26 Steam.
- Known public best: `candidate_rank_blend_emb128_emb192.csv`, public score `0.77825`.
- Main internal uniform reference: emb128 4-seed ref `0.76505`.
- Weak one-split blips around +/-0.0007 are NOT candidates; they require expansion.
- Candidate-like escalation requires: multi-split support, fixes > breaks / paired significance when available, no near-duplicate/quarantine conflict.
- STRICT_PASS gate for a completed probe: mean delta >= +0.0015, min delta >= 0, 3/3 positive splits, fixes > breaks, pooled exact p < 0.05, validation_only=true, candidate_csv_written=false, kaggle_submit_executed=false.

RECENT REQUIRED CHECKS ALREADY OBSERVED BY HERMES BEFORE THIS RUN:
- No Hermes background processes currently tracked.
- No relevant repo validation process was live in `/proc` at 2026-06-07 17:14 KST, aside from the checker itself.
- No Kaggle submit process was live.
- GPU: V100s mostly idle; GPU3 displayed 4320 MiB but `nvidia-smi` reported no running process and ps matched no repo process.
- OpenCode CLI: `/opt/data/home/.local/bin/opencode`, version 1.15.11; providers list shows configured credentials (do not print tokens).

SPECIFIC ARTIFACTS TO INSPECT/RESPECT:
1. `reports/20260606T132450KST_userknn_gated_residual_fine.json`
   - Expected final JSON is absent. Prior status classified UserKNN gated residual fine-grid as `STALLED_INCOMPLETE` due warning-dominated log/no metric report. Do not simply relaunch UserKNN fine-grid.
2. `reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.json` and `.md`
   - Expected expanded files are absent. `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log` is only 12 lines and stops after seed123 invalid-value warning. No live pid/process found. Treat expanded run as incomplete/stalled unless you find new evidence.
3. `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json`
   - Smoke/probe top row `vote_consensus__high_capacity_gap__B1__w0.1`: mean delta +0.0003667400146696309, min -0.0012002400480095599, positive 2/3, fixes/breaks 252/230, p=0.33881500709211204. Classification: WEAK_SIGNAL_STRICT_GATE_FAIL.
4. Latest OpenCode/cron evidence:
   - `reports/20260607T160717KST_improvement_axis_cron_status.json` / `.md`: fresh OpenCode run at 16:01 KST returned `NO_SAFE_AXIS`, no probe launched, safety flags clean.
   - `reports/20260607T160111KST_opencode_improvement_axis_loop.json` / `.md` / raw text / log: prior OpenCode verdict `NO_SAFE_AXIS`.
   - `reports/20260607T145142KST_after_als_manual_no_safe_axis_stop_summary.json` and `reports/20260607T145927KST_after_als_exit13_iter03_reconciliation.json`: after-ALS loop ended with no safe axis and no forbidden output.
5. Failed/quarantined state:
   - `reports/failed_axes.json` includes stacker public regression, OTTO forced public negative vs current best, and current-best ALS independent strict failure.
   - `state/aggressive_quota_runner_state.json` closes public-tested rankblend/ALS boundary/frontier/tagcf variants.

KNOWN CLOSED AXES — AVOID REPEATING WITHOUT A MATERIALLY NEW INDEPENDENT DESIGN:
- DNS pool=1 rejected as split-specific noise (three_uniform_panel mean deltas negative, positive 1/3).
- Hours-confidence edge weighting no gain.
- Exact-K subset objective no gain.
- Temporal compatibility large regression.
- Boundary covariate residual weak/pop-trap.
- SL@K-lite all splits negative.
- Last-slot sparse agreement reject.
- Raw semantic/README/LM text probes weak or redundant.
- Capacity frontier/emb192 marginal public noise.
- Public-tested rankblend variants did not safely beat current best.
- OTTO/source-separated co-visitation: strict failure and forced public 0.77815 < current best 0.77825.
- Current-best ALS/rankblend residual diagnostic: strongest diagnostic row mean +0.00113356, min +0.00040008, 3/3, p=0.02197, but below +0.0015 mean gate and same-family/quarantine conflict; pre-registered independent row mean +0.00080016 with min negative.
- UserKNN fine-grid and jackknife boundary are stalled/weak, not fresh axes.

TASK FOR THIS RUN:
1. Inspect current evidence, scripts, logs, ledgers, and recent reports.
2. Decide whether there is a genuinely fresh independent validation-only axis that is not closed/quarantined/near-duplicate and can be tested safely.
3. If and only if there is a credible bounded validation-only probe, implement/launch it without creating full-test/uploadable CSVs. Use repo-trusted scripts where possible. If GPU is needed, use V100-compatible torch only: `uv run --python 3.13 --extra-index-url https://download.pytorch.org/whl/cu128 --with 'torch==2.10.0+cu128' ...`.
4. If no safe fresh axis exists, do not launch busywork. Produce a clear `NO_SAFE_AXIS` report.

REQUIRED OUTPUT FILES:
- Write JSON report to `reports/20260607T171551KST_opencode_improvement_axis_loop.json`.
- Write Markdown report to `reports/20260607T171551KST_opencode_improvement_axis_loop.md`.
- If you run a validation probe, also write its bounded log/report paths and classify it.

REQUIRED JSON SCHEMA (include all fields; add extra fields only if useful):
```json
{
  "timestamp_kst": "20260607T171551KST",
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
  "new_probe": {
    "launched": false,
    "command": null,
    "pid": null,
    "log": null,
    "report_json": null,
    "report_md": null
  },
  "best_or_top_metrics": null,
  "ranked_next_axis_hypotheses": [],
  "why_not_candidate": "",
  "sentinel": "OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS"
}
```

FINAL SENTINEL:
End your final assistant text with exactly one of these lines, matching the JSON verdict/new_probe status:
- `OPENCODE_AXIS_LOOP_DONE_IMPROVEMENT_AXIS_FOUND`
- `OPENCODE_AXIS_LOOP_DONE_NEXT_PROBE_RUNNING`
- `OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS`
- `OPENCODE_AXIS_LOOP_DONE_WEAK_SIGNAL`
- `OPENCODE_AXIS_LOOP_DONE_FAILED_BLOCKER`

Remember: no Kaggle submission, no full-test/uploadable CSV, no external Steam scraping, no git stage/commit/push, no recursive cron.
