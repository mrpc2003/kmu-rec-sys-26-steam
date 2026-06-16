# 2026-06-07 04:22 KST — UserKNN exit 143 and OpenCode no-safe-axis reconciliation

## Trigger

Received notification for background process `proc_d98ef5d36b4a`:

- script command: `scripts/userknn_gated_residual_probe.py`
- intended outputs:
  - `reports/20260606T132450KST_userknn_gated_residual_fine.json`
  - `reports/20260606T132450KST_userknn_gated_residual_fine.md`
- exit code: `143`

## Reconciliation

This was not a successful probe completion. It is classified as:

`STALLED_INCOMPLETE`

Evidence:

- expected JSON report: missing
- expected Markdown report: missing
- log exists: `logs/userknn_gated_residual_fine_20260606T132450KST.log`
- log size observed: `3711020` bytes
- log line count observed by OpenCode: `40121`
- tail consists of repeated `RuntimeWarning: invalid value encountered in divide` from `scripts/userknn_residual_probe.py:114`
- no final `done UserKNN gated fine-grid follow-up` line and no metric report was produced

Therefore this broad fine-grid is **not a candidate** and should not be relaunched as-is.

## OpenCode-first loop follow-up

The updated recurring OpenCode-first loop already detected the stall and launched a bounded OpenCode run:

- prompt: `reports/20260607T041626KST_opencode_axis_loop_prompt.md`
- log: `logs/opencode_improvement_axis_loop_20260607T041626KST.jsonl`
- result report md: `reports/20260607T041626KST_opencode_improvement_axis_loop.md`
- result report json: `reports/20260607T041626KST_opencode_improvement_axis_loop.json`

OpenCode result:

`OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS`

OpenCode classification summary:

- UserKNN broad fine-grid: `STALLED_INCOMPLETE`
- Jackknife expanded: `FAILED_INCOMPLETE_NO_METRIC_REPORT`
- Jackknife smoke: `WEAK_SIGNAL_STRICT_GATE_FAIL`
  - mean Δ `+0.0003667400`
  - min Δ `-0.0012002400`
  - positive splits `2/3`
  - p `0.338815`
- DNS pool=1 panel: `CLOSED_NO_CANDIDATE`
- UserKNN gated smoke: `WEAK_SIGNAL_STRICT_GATE_FAIL`
  - top mean Δ `+0.0009001800`
  - min Δ `+0.0002000400`
  - positive splits `3/3`
  - fixes/breaks `406/352`
  - p `0.0541518`
  - fails strict gate because mean Δ < `+0.0015` and p is not < `0.05`

OpenCode did not launch a new probe because it judged that every credible next move would either repeat stalled/closed/quarantined families or be too weak to meet the strict gate.

## Hermes verification

- Parsed `reports/20260607T041626KST_opencode_improvement_axis_loop.json` successfully with `python3 -m json.tool`.
- Checked for newer CSVs under `submissions/` after the OpenCode report timestamp: count `0`.
- Current related active processes: only pre-existing `aggressive_quota_runner.py`; no live OpenCode/UserKNN/Jackknife probe process remains.

Note: simple string scan finds `kaggle competitions submit` and `submissions/` inside the OpenCode report because those strings are quoted as **forbidden actions / no-submission assertions**, not as executed commands or output paths.

## Safety flags

- validation_only: true
- candidate_csv_written: false
- kaggle_submit_executed: false
- hidden_labels_used: false
- external_steam_scraping_used: false
- git_stage_commit_push_executed: false

## Loop status

The hourly OpenCode-first recurring loop remains scheduled as job `4d627b59804f`; next iterations will continue searching, but this tick did not produce a strict or weak-launchable new axis.
