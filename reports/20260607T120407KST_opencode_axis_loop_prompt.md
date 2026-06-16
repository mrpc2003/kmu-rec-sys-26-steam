# KMURecSys26 Steam OpenCode-first improvement-axis loop — 20260607T120407KST

CRITICAL EXECUTION RULES:
- Answer and act ENTIRELY YOURSELF in this single OpenCode run. Do NOT delegate to sub-agents. Do NOT wait for parallel agents. Do NOT say you are waiting.
- This is a constrained no-submit validation-only Kaggle/RecSys research tick in repo `/opt/data/kaggle/kmu-rec-sys-26-steam`.
- Before proposing or launching any new improvement axis, inspect current evidence/artifacts in this repo and avoid closed/rejected/quarantined families.
- Keep the run bounded. If a probe cannot complete quickly and safely, write a report that recommends the next bounded command but do not start unbounded/orphaned background processes.

Hard safety contract (non-negotiable):
- NEVER run `kaggle competitions submit` or any Kaggle submit API.
- NEVER create a full-test candidate/submission CSV or uploadable artifact. Do not write under `submissions/`; do not materialize files with `candidate`, `submission`, or full-test prediction semantics. Validation-only score/artifact files are allowed if clearly validation-fold-only.
- NEVER use hidden labels, private answers, leaked public/private LB data, or test labels.
- NEVER scrape or download external Steam data. No external data collection.
- NEVER print credentials, tokens, auth files, `.netrc`, Kaggle JSON, W&B/HF/OpenAI tokens, or secret env vars.
- NEVER weaken quarantine/guard logic.
- Do not stage/commit/push. Do not schedule cron jobs. Do not modify unrelated files.

Current objective:
Find a real improvement axis beyond current public best behavior. Known public best: `candidate_rank_blend_emb128_emb192.csv` public `0.77825`. Main internal uniform reference: emb128 4-seed ref `0.76505`. Weak one-split blips around ±0.0007 are NOT candidates; they require expansion. Escalation requires multi-split support, fixes > breaks / paired significance when available, and no near-duplicate/quarantine conflict.

Strict classification gate for any completed probe:
- `validation_only=true`
- `candidate_csv_written=false`
- `kaggle_submit_executed=false`
- mean Δ >= `+0.0015`
- min Δ >= `0`
- 3/3 positive splits
- fixes > breaks
- pooled exact p < `0.05` when available
- no quarantine / near-duplicate conflict
Otherwise classify `WEAK_SIGNAL`, `REJECT`, `STALLED_INCOMPLETE`, or `NO_SAFE_AXIS`.

Known closed axes / do not repeat as-is:
- DNS pool=1 rejected as split-specific noise: three_uniform_panel mean deltas negative, positive 1/3.
- UserKNN gated residual fine-grid launched at `20260606T132450KST` stalled with repeated invalid divide warnings; expected `reports/20260606T132450KST_userknn_gated_residual_fine.json` is missing. Do not relaunch broad fine-grid as-is.
- Jackknife uncertainty boundary smoke: `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json` is `WEAK_SIGNAL`: top `vote_consensus__high_capacity_gap__B1__w0.1`, mean Δ `+0.0003667400`, min Δ `-0.0012002400`, positive splits `2/3`, fixes/breaks `252/230`, p `0.338815`.
- Jackknife expanded expected files `reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.json/.md` and log `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log` were not found by Hermes in this tick.
- Hours-confidence edge weighting no gain; exact-K subset objective no gain; temporal compatibility large regression; boundary covariate residual weak/pop-trap; SL@K-lite all splits negative; last-slot sparse agreement reject; raw semantic/README/LM text probes weak/redundant; capacity frontier/emb192 marginal public noise; public-tested rankblend variants did not safely beat current best.
- OTTO/copurchase source axis: independent uniform confirmation `reports/20260607T095549KST_otto_independent_uniform_confirmation.json` failed strict gate: registered row mean Δ `+0.0006668000`, min Δ `-0.0006001200`, positive splits `2/3`, p `0.1700199`; a later forced full-test/public probe scored `0.77815`, below current best `0.77825`. Treat as closed/no escalation unless a genuinely independent validation design passes strict gate versus current best.
- Failed/quarantined public-transfer families are recorded in `reports/failed_axes.json` and `state/aggressive_quota_runner_state.json`; inspect before proposing similar axes.

Fresh Hermes tick observations before this OpenCode run:
- OpenCode version available: `1.15.11`.
- GPU state: 4x V100 visible; essentially idle. `nvidia-smi` showed no running processes, despite GPU3 memory display. CPU/memory healthy.
- No active live UserKNN/Jackknife/OpenCode/parallel validation probe was found in process inspection, only Hermes itself.
- A concurrent/previous dry-run artifact at `reports/20260607T120218KST_opencode_hermes_axis_rejection_loop_summary.json` did not call OpenCode and is not sufficient for this tick.
- Existing `state/autonomous_submission_policy.json` mentions older autonomous submit policy, but THIS cron's hard contract overrides it: do not submit and do not create full-test candidate CSVs.

Allowed outputs for this OpenCode run:
1. Write a concise JSON report to `reports/20260607T120407KST_opencode_improvement_axis_loop.json`.
2. Write a concise Markdown report to `reports/20260607T120407KST_opencode_improvement_axis_loop.md`.
3. Optionally patch/create validation-only scripts or run a tiny/bounded validation-only probe ONLY if it cannot write full-test/uploadable outputs and finishes within this bounded run. Prefer repo-trusted scripts and artifacts. If GPU is required, use V100-compatible command pattern only: `uv run --python 3.13 --extra-index-url https://download.pytorch.org/whl/cu128 --with 'torch==2.10.0+cu128' ...`.
4. Do not write under `submissions/`. Do not write files whose names look uploadable candidate/submission CSVs.

Required JSON schema (include all fields):
```json
{
  "timestamp_kst": "20260607T120407KST",
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
  "inspected_artifacts": [],
  "closed_or_avoided_axes": [],
  "axis_decision": "STRICT_PASS|WEAK_SIGNAL|REJECT|NO_SAFE_AXIS|NEXT_PROBE_RUNNING|FAILED_BLOCKER",
  "best_or_top_metrics": {
    "variant": null,
    "mean_delta_vs_base": null,
    "min_delta_vs_base": null,
    "positive_splits": null,
    "fixes": null,
    "breaks": null,
    "pooled_p_exact": null,
    "quarantine_conflict": false,
    "strict_gate_pass": false
  },
  "new_probe": {
    "launched": false,
    "status": "not_launched|completed|running|failed|blocked",
    "command": null,
    "pid_file": null,
    "log": null,
    "report_json": null,
    "report_md": null,
    "artifact_dir": null,
    "bounded_seconds": null
  },
  "ranked_next_axis_hypotheses": [],
  "verdict_reason": ""
}
```

If you identify a genuinely fresh axis, prefer a validation-only probe that evaluates multiple existing validation splits and row-level fixes/breaks rather than any full-test materialization. If no safe high-signal axis exists, say so clearly and do not manufacture a probe.

Final sentinel requirement: end your final assistant text with exactly one of these lines:
- `OPENCODE_AXIS_LOOP_DONE_IMPROVEMENT_AXIS_FOUND`
- `OPENCODE_AXIS_LOOP_DONE_NEXT_PROBE_RUNNING`
- `OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS`
- `OPENCODE_AXIS_LOOP_DONE_WEAK_SIGNAL`
- `OPENCODE_AXIS_LOOP_DONE_REJECT`
- `OPENCODE_AXIS_LOOP_DONE_FAILED_BLOCKER`
