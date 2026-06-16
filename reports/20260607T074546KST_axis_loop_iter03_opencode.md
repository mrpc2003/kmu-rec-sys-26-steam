# OpenCode axis-finding iteration 03 — 20260607T074546KST

## Verdict

`NO_SAFE_AXIS` — I did not launch a new probe. Iteration 03 found no fresh independent bounded validation-only axis that could plausibly pass Hermes' strict gate without repeating a closed/stalled family or conflicting with quarantine/public-negative evidence.

## Safety flags

- validation_only: true
- candidate_csv_written: false
- full_test_candidate_or_submission_csv_created: false
- kaggle_submit_executed: false
- hidden_labels_used / private_answers_used: false
- external_steam_scraping_used: false
- credentials_or_tokens_printed: false
- quarantine_or_guard_logic_weakened: false
- git_stage_commit_push_executed: false
- recursive_cron_scheduled: false

## Evidence checked

- Iteration state: `reports/20260607T074546KST_axis_loop_iter02_opencode.{json,md}`, `reports/20260607T074528KST_opencode_hermes_axis_rejection_loop_summary.json`, and recent improvement-loop reports.
- Safety/quarantine state: `state/aggressive_quota_runner_state.json`, `state/autonomous_submission_policy.json`.
- Metric surfaces: `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json`, `reports/20260603T180707KST_userknn_gated_residual_probe.json`, `reports/20260603T164750KST_boundary_factory_frontier.json`, `reports/manual_qa_boundary_pairwise_factory.json`, plus `reports/` and `artifacts/` metric verdict searches.
- Direct searches: `rg` over reports/scripts/state/artifacts for strict gates, deltas, p-values, quarantine and closed-axis terms; `rg` over scripts for axis/probe surfaces; `ast-grep` for Python `subprocess.run` and pandas `to_csv` call sites; process sweep for active Kaggle/probe/controller processes.

## Axis review

| Axis | Decision reason |
|---|---|
| UserKNN gated residual fine-grid / smoke | Fine-grid reports absent with repeated invalid-divide warnings; smoke mean Δ `+0.000900` and p `0.05415` fail strict mean/p gates. |
| Jackknife uncertainty boundary | Smoke weak: mean Δ `+0.0003667`, min Δ `-0.0012002`, `2/3` positive, p `0.3388`; expanded run remains report-missing/incomplete. |
| Boundary/frontier/capacity scoreblend | Some internal rows clear mean/p gates, but the family is public-negative and quarantined (`boundary_scoreblend` and `frontier_z` public deltas below current best). |
| Rankblend ALS/BPR residual | Public-negative/non-improving despite validation-positive signals; same-family variants are quarantined/skipped. |
| Layer mixture, SL@K-lite, multi-interest SVD, SASRec/xSIMGCL/semantic/text/README/LM, temporal/hours/exact-K/last-slot/DNS | Already weak, split-inconsistent, below floor, redundant, public-noise risk, or explicitly closed. |

## New probe

None launched. The only apparent strict metric-bearing rows were not valid fresh axes because they fall inside public-negative/quarantined boundary/frontier families. Launching another bounded probe would either duplicate a prohibited family or create an underpowered diagnostic that cannot satisfy the stated strict gate.

## Output files

- JSON: `reports/20260607T074546KST_axis_loop_iter03_opencode.json`
- Markdown: `reports/20260607T074546KST_axis_loop_iter03_opencode.md`
- Artifact directory reserved: `artifacts/opencode_hermes_axis_loop_20260607T074546KST`
