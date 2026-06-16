You are OpenCode running one bounded validation-only improvement-axis iteration in `/opt/data/kaggle/kmu-rec-sys-26-steam`.

Goal: find a fresh independent validation-only improvement axis, or launch exactly one bounded validation-only probe, while avoiding closed, stalled, weak, quarantined, or near-duplicate axes.

Hard safety forbids:
- Do not run `kaggle competitions submit`.
- Do not create full-test candidate CSVs, submission CSVs, or any CSV under `submissions/`.
- Do not use hidden labels, private labels, external Steam scraping, credentials, tokens, or secret material.
- Do not stage, commit, amend, or push Git changes.
- Do not schedule cron, edit cron, or launch recursive OpenCode/controller loops.
- Do not weaken quarantine/guard logic.

Allowed scope for this OpenCode iteration:
- Read code/reports/logs/state.
- Write concise validation-only reports/logs/artifacts under `reports/`, `logs/`, and `artifacts/opencode_hermes_axis_loop_20260606T224148Z/iter01/`.
- If launching a probe, launch exactly one bounded validation-only probe and record its PID/log/report paths.

Required output files:
- Markdown report: `reports/20260606T224148Z_axis_loop_iter01_opencode.md`
- JSON report: `reports/20260606T224148Z_axis_loop_iter01_opencode.json`

The JSON report must include at minimum:
```json
{
  "verdict": "IMPROVEMENT_AXIS_FOUND | NEXT_PROBE_RUNNING | NO_SAFE_AXIS | BLOCKED",
  "safety_flags": {
    "validation_only": true,
    "candidate_csv_written": false,
    "full_test_candidate_or_submission_csv_created": false,
    "kaggle_submit_executed": false,
    "hidden_labels_used": false,
    "private_answers_used": false,
    "external_steam_scraping_used": false,
    "credentials_or_tokens_printed": false,
    "git_stage_commit_push_executed": false,
    "recursive_cron_scheduled": false
  },
  "best_candidate_metrics": {
    "mean_delta": null,
    "min_delta": null,
    "positive_splits": null,
    "fixes": null,
    "breaks": null,
    "pooled_p": null,
    "mcnemar_p": null,
    "quarantine_or_near_duplicate_conflict": false
  },
  "new_probe": {
    "launched": false,
    "still_running": false,
    "pid": null,
    "pid_file": null,
    "log": null,
    "report_json": null,
    "report_md": null,
    "validation_only": true
  },
  "rejection_reasons": []
}
```

Hermes strict pass criteria for any improvement axis:
- mean_delta >= 0.0015
- min_delta >= 0
- positive_splits >= 3, or the literal string "3/3"
- fixes > breaks
- pooled or McNemar p < 0.05 when present
- no quarantine or near-duplicate conflict
- all safety flags above remain safe

## Concise current state

Timestamp UTC: `20260606T224148Z`

Recent reports:
- `reports/20260607T073730KST_opencode_controller_impl_prompt.md` size=4324 mtime=1780785450
- `reports/20260607T064127KST_improvement_axis_cron_status.md` size=3861 mtime=1780782212
- `reports/20260607T064127KST_improvement_axis_cron_status.json` size=9111 mtime=1780782189
- `reports/20260607T063604KST_opencode_improvement_axis_loop_raw_text.md` size=1382 mtime=1780782071
- `reports/20260607T063604KST_opencode_improvement_axis_loop.json` size=11842 mtime=1780782022
- `reports/20260607T063604KST_opencode_improvement_axis_loop.md` size=7548 mtime=1780782022
- `reports/20260607T063604KST_opencode_axis_loop_prompt.md` size=5445 mtime=1780781801
- `reports/20260607T053207KST_improvement_axis_cron_status.md` size=3348 mtime=1780777997
- `reports/20260607T053207KST_improvement_axis_cron_status.json` size=8292 mtime=1780777976
- `reports/20260607T052646KST_opencode_improvement_axis_loop_raw_text.md` size=1228 mtime=1780777878
- `reports/20260607T052646KST_opencode_improvement_axis_loop.json` size=10823 mtime=1780777832
- `reports/20260607T052646KST_opencode_improvement_axis_loop.md` size=5370 mtime=1780777832
- `reports/20260607T052646KST_opencode_axis_loop_prompt.md` size=7178 mtime=1780777651
- `reports/20260607T042202KST_improvement_axis_cron_status.md` size=3668 mtime=1780773792
- `reports/20260607T042202KST_improvement_axis_cron_status.json` size=7885 mtime=1780773771
- `reports/20260607T042200KST_userknn_stalled_opencode_no_safe_axis_reconciled.md` size=3354 mtime=1780773720

Recent logs:
- `logs/20260607T073730KST_opencode_controller_impl.jsonl` size=329170 mtime=1780785706
- `logs/aggressive_quota_runner_watchdog_20260607T064435KST.log` size=1795 mtime=1780785290
- `logs/opencode_improvement_axis_loop_20260607T063604KST.jsonl` size=683374 mtime=1780782060
- `logs/aggressive_quota_runner_watchdog_20260607T052059KST.log` size=2361 mtime=1780781481
- `logs/opencode_improvement_axis_loop_20260607T052646KST.jsonl` size=578352 mtime=1780777867
- `logs/aggressive_quota_runner_watchdog_20260606T113031KST.log` size=30095 mtime=1780776272
- `logs/opencode_improvement_axis_loop_20260607T041626KST.jsonl` size=447643 mtime=1780773648
- `logs/userknn_gated_residual_fine_20260606T132450KST.log` size=3711020 mtime=1780773396
- `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log` size=872 mtime=1780756647
- `logs/opencode_improvement_axis_loop_20260606T220406KST.jsonl` size=821159 mtime=1780753656
- `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.pid` size=6 mtime=1780753321
- `logs/20260606T220406KST_jackknife_uncertainty_boundary_probe.log` size=4479 mtime=1780753294
- `logs/dns_pool1_val_random_uniform_seed7_seed7_20260606T130538KST.log` size=748 mtime=1780719782
- `logs/dns_pool1_val_random_uniform_seed7_seed42_20260606T130345KST.log` size=614 mtime=1780719662
- `logs/dns_pool1_val_random_uniform_seed7_seed2024_20260606T130319KST.log` size=624 mtime=1780719626
- `logs/dns_pool1_val_random_uniform_seed7_seed123_20260606T130318KST.log` size=619 mtime=1780719626

Recent state files:
- `state/aggressive_quota_runner_state.json` size=20809 mtime=1780476153
- `state/autonomous_submission_policy.json` size=2218 mtime=1780402967

Interesting active processes (redacted):
-     170     153 Ssl  hermes          /opt/hermes/.venv/bin/python3 /opt/hermes/.venv/bin/hermes gateway run
-    4064       1 Ssl  uv              uv run --with pandas --with numpy --with scipy --with wandb python /opt/data/kaggle/kmu-rec-sys-26-steam/scripts/aggressive_quota_runner.py --sleep-no-quota 300 --sleep-no-candidate 600 --sleep-after-submit 21600
-    4080    4064 Sl   python          /opt/data/home/.cache/uv/builds-v0/.tmpBjMbQu/bin/python /opt/data/kaggle/kmu-rec-sys-26-steam/scripts/aggressive_quota_runner.py --sleep-no-quota 300 --sleep-no-candidate 600 --sleep-after-submit 21600
-    5354       1 Sl   opencode        opencode run You are OpenCode implementing a bounded controller script in the repo /opt/data/kaggle/kmu-rec-sys-26-steam.  CRITICAL EXECUTION RULES: - Answer/work entirely yourself in this single OpenCode run. Do NOT delegate to sub-agents. Do NOT wait for parallel agents. - Your task is to implement the controller script only. Do NOT run Kaggle, do NOT submit, do NOT create full-test candidate/submission CSVs, and do NOT write under submissions/. - Do NOT sc

Active Kaggle-submit process count from controller check: `0`


End your final text with exactly one sentinel line from this set:
OPENCODE_AXIS_LOOP_DONE_IMPROVEMENT_AXIS_FOUND
OPENCODE_AXIS_LOOP_DONE_NEXT_PROBE_RUNNING
OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS
OPENCODE_AXIS_LOOP_DONE_BLOCKED
