# OpenCode axis-finding iteration 05 — 20260607T074546KST

## Verdict

**NO_SAFE_AXIS.** I did not launch a probe. The remaining plausible surfaces either repeat closed/stalled families, conflict with quarantine/public-negative evidence, or would be weak one-split diagnostics that cannot satisfy Hermes' strict gate.

## Safety flags

- validation_only: true
- candidate_csv_written: false
- full_test_candidate_or_submission_csv_created: false
- kaggle_submit_executed: false
- hidden_labels_used: false
- private_answers_used: false
- external_steam_scraping_used: false
- credentials_or_tokens_printed: false
- quarantine_or_guard_logic_weakened: false
- git_stage_commit_push_executed: false
- recursive_cron_scheduled: false

## Probe decision

No new probe was launched. `scripts/turbocf_uniform_gate.py` was the only locally implemented surface that looked not-yet-mentioned in the iteration prompt, but prior advisory explicitly closes EASE/ALS/MultiVAE/TurboCF/itemKNN as redundant. Launching it would repeat a closed family. Boundary/frontier/rankblend/TAG-CF rows remain blocked by public-negative/quarantine evidence, and UserKNN/jackknife remain stalled or weak.

## Best apparent metric row

The strongest apparent metric-bearing row remains the blocked boundary/frontier internal validation family from `reports/20260603T164750KST_boundary_factory_frontier.json`: mean Δ `+0.0017003400680135987`, 3 positive splits, p `0.0005242948680109718`. It is not actionable because `min_delta` and fixes/breaks are unavailable and the family conflicts with quarantine/public-negative transfer recorded in `state/aggressive_quota_runner_state.json`.

## Evidence checked

- Read iteration-04 report, recent improvement-axis reports, multi-interest SVD smoke, and quota runner state.
- Ran broad `rg` searches across `reports/`, `scripts/`, `state/`, and README for axis/probe/quarantine/safety/family evidence.
- Ran `ast-grep` for `subprocess.run(...)` and `to_csv(...)` script surfaces.
- Compiled relevant controller/probe scripts with `py_compile`.
- Scanned processes for Kaggle submit and active probe/controller lines; only the expected Hermes/OpenCode loop and search command were observed.

## Required output files

- `reports/20260607T074546KST_axis_loop_iter05_opencode.json`
- `reports/20260607T074546KST_axis_loop_iter05_opencode.md`
- artifact directory created: `artifacts/opencode_hermes_axis_loop_20260607T074546KST/`
