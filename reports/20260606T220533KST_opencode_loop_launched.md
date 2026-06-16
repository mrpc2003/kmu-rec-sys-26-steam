# 2026-06-06 22:05 KST — OpenCode improvement-axis loop launched

## User request

우현 requested using OpenCode to run the loop, with the explicit goal of finding an improvement axis.

## Readiness

OpenCode was verified before launch:

- binary: `/opt/data/tools/node-v24.11.1-linux-x64/bin/opencode`
- version: `1.15.12`
- smoke test sentinel: `OPENCODE_LOOP_SMOKE_OK`

## Running OpenCode loop

- process session: `proc_3d61e70ffa23`
- prompt: `reports/20260606T220406KST_opencode_axis_loop_prompt.md`
- JSONL log: `logs/opencode_improvement_axis_loop_20260606T220406KST.jsonl`
- extracted raw text path: `reports/20260606T220406KST_opencode_improvement_axis_loop_raw_text.md`
- status file: `reports/20260606T220406KST_opencode_improvement_axis_loop_status.md`
- expected OpenCode report: `reports/20260606T220406KST_opencode_improvement_axis_loop.md`
- expected OpenCode JSON: `reports/20260606T220406KST_opencode_improvement_axis_loop.json`

## OpenCode safety contract embedded in prompt

OpenCode was explicitly forbidden to:

- call `kaggle competitions submit`
- create submission/full-test candidate CSVs
- read hidden labels or external/private answer sources
- scrape external Steam reviews
- print credentials/tokens
- weaken guard/quarantine logic
- commit/push/stage changes

Allowed outputs were limited to validation-only scripts/artifacts/logs/reports.

## Current concurrent work

- Existing UserKNN gated residual fine-grid remains running and was included in the OpenCode prompt as an active process to avoid duplicate launch.
- Aggressive runner remains active; its guard/quarantine was not modified.

## Current status

OpenCode loop is running in the background with completion notification enabled. Hermes will reconcile its report/logs when it exits.
