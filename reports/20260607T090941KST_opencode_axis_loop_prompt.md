# KMURecSys26 Steam no-submit improvement-axis discovery — 20260607T090941KST

CRITICAL EXECUTION RULES:
- Answer/work entirely yourself in this single OpenCode run. Do NOT delegate to sub-agents, do NOT wait for parallel agents, and do NOT say you are waiting for another agent.
- You may inspect and edit files in this one repo only: `/opt/data/kaggle/kmu-rec-sys-26-steam`.
- Hard safety contract: never run `kaggle competitions submit`; never create full-test candidate/submission CSVs or uploadable Kaggle artifacts; never write under `submissions/`; never use hidden labels/private answers; never scrape external Steam data; never print credentials/tokens; never weaken quarantine/guard logic; do not `git add`, commit, push, or schedule cron jobs.
- Allowed outputs: validation-only scripts, validation artifacts under `artifacts/`, logs under `logs/`, reports under `reports/`, and compact JSON summaries. Test-side/full-test prediction materialization is forbidden in this cron job.
- If you launch any validation probe, it must be bounded and no-submit/no-candidate by construction. Prefer `timeout` around long validation commands. Record pid/log/report paths if it continues after your run.

Current objective:
Find a real improvement axis beyond current public best behavior, not a weak one-split blip. Current public best candidate is `candidate_rank_blend_emb128_emb192.csv` public `0.77825`; main internal uniform reference is emb128 4-seed ref `0.76505`. Weak ±0.0007 signals are not candidates; they require expansion.

Strict escalation gate used by Hermes after you finish:
- `validation_only=true`
- `candidate_csv_written=false`
- `full_test_candidate_or_submission_csv_created=false`
- `kaggle_submit_executed=false`
- mean Δ vs base >= `+0.0015`
- min split Δ >= `0`
- 3/3 positive splits
- fixes > breaks
- pooled exact/McNemar p < `0.05` when available
- no near-duplicate/quarantine/public-negative conflict

Known required readback/classification before new work:
- `reports/20260606T132450KST_userknn_gated_residual_fine.json` / `.md`: expected reports are missing. The previous broad fine-grid stalled; do not relaunch it.
- `logs/userknn_gated_residual_fine_20260606T132450KST.log`: warning-dominated stalled log.
- `reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.json` / `.md`: expected reports are missing.
- `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log`: only 12 lines, stopped mid split.
- `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json` and `.md`: completed smoke is WEAK_SIGNAL; top mean Δ `+0.0003667400`, min Δ `-0.0012002400`, positive `2/3`, fixes/breaks `252/230`, p `0.338815`.
- Latest rejection loop: `reports/20260607T074546KST_opencode_hermes_axis_rejection_loop_summary.json` / `.md` reached `MAX_ITERS_REACHED_NO_STRICT_PASS` after 6 OpenCode iterations; strongest boundary/frontier metric row is blocked by quarantine/public-negative family conflict.
- Latest external methodology scan: `reports/20260607T082037KST_external_dacon_kaggle_methodology_scan.md` suggests one under-tested high-priority transferable idea: OTTO-style multi-source co-visitation/transition scoring using existing train data only. But inspect existing `scripts/paper_guided_next_steps.py`, `reports/20260530_paper_guided_next_steps.*`, and `reports/20260530_stage3_blend.*` first: time-decay ItemKNN/BM25 and Stage3 blends already exist, so do NOT relaunch a near-duplicate plain ItemKNN/time-BM25 probe.

Closed axes to avoid repeating unless you can prove a material, fresh, bounded distinction:
- DNS pool=1 rejected as split-specific noise; hours-confidence edge weighting no gain; exact-K subset objective no gain; temporal compatibility large regression; boundary covariate residual weak/pop-trap; SL@K-lite all splits negative; last-slot sparse agreement reject; raw semantic/README/LM text probes weak or redundant; capacity frontier/emb192 marginal public noise; public-tested rankblend variants did not safely beat current best.
- UserKNN broad fine-grid is stalled/closed; jackknife expanded is failed/incomplete; boundary/frontier/rankblend/TAG-CF families are public-negative/quarantined in `state/aggressive_quota_runner_state.json`.

Your task in this one bounded OpenCode run:
1. Inspect current evidence, scripts, reports, and state enough to avoid duplicate/closed axes.
2. Decide whether a truly fresh, independent validation-only axis is available now. The best candidate clue is OTTO-style source-separated co-visitation/transition scoring beyond the already-tested plain/time-decay ItemKNN/BM25: e.g. separate source features for co-play count, temporal order, last-K history, hours/log-hours weighting, and user-history max/mean/sum, evaluated only on validation splits against the current LightGCN/rankblend reference.
3. If and only if this is not a near-duplicate and can be bounded, implement or launch a **validation-only smoke**. Prefer 3 uniform splits if cheap; otherwise make a one-split smoke explicitly labeled `ONE_SPLIT_DIAGNOSTIC_NOT_CANDIDATE` and prescribe expansion. No test/full-test outputs. Write artifacts under `artifacts/opencode_axis_loop_20260607T090941KST/` and reports under `reports/`.
4. If no safe bounded axis is credible, do not launch a probe. Explain why with evidence.
5. Always write BOTH:
   - `reports/20260607T090941KST_opencode_improvement_axis_loop.json`
   - `reports/20260607T090941KST_opencode_improvement_axis_loop.md`
6. The JSON must be parseable and include at least:
```json
{
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
  "axis_decision": "...",
  "new_probe": {
    "launched": false,
    "status": "not_launched|running|completed|failed",
    "command": null,
    "pid_file": null,
    "log": null,
    "report_json": null,
    "report_md": null,
    "artifact_dir": "artifacts/opencode_axis_loop_20260607T090941KST"
  },
  "best_or_top_metrics": {},
  "closed_or_rejected_axes_checked": [],
  "ranked_next_axis_hypotheses": [],
  "artifacts_reports_produced": {},
  "verdict": "NO_SAFE_AXIS|NEXT_PROBE_RUNNING|WEAK_SIGNAL|STRICT_PASS|REJECT|FAILED_BLOCKER"
}
```
7. End your final text with exactly one sentinel line, one of:
   - `OPENCODE_AXIS_LOOP_DONE_NEXT_PROBE_RUNNING`
   - `OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS`
   - `OPENCODE_AXIS_LOOP_DONE_WEAK_SIGNAL`
   - `OPENCODE_AXIS_LOOP_DONE_STRICT_PASS`
   - `OPENCODE_AXIS_LOOP_DONE_FAILED_BLOCKER`

Remember: no Kaggle submit, no full-test candidate/submission CSV, no external Steam scraping, no credentials, no git staging/commit/push, no recursive cron.
