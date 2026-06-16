# OpenCode↔Hermes rejection loop summary — 20260607T074546KST

- final_verdict: `MAX_ITERS_REACHED_NO_STRICT_PASS`
- dry_run: `false`
- iterations: `6`

## Safety contract

- no Kaggle submit
- no full-test candidate/submission CSV creation
- no hidden labels/private answers/external Steam scraping
- no git stage/commit/push
- no recursive cron scheduling

## Iterations

| iter | sentinel | Hermes verdict | strict | rejection reasons | report |
|---:|---|---|---:|---|---|
| 1 | `OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS` | `REJECTED_CONTINUE` | `false` | best_metrics_failed:mean_delta,min_delta,positive_splits,fixes_gt_breaks | `reports/20260607T074546KST_axis_loop_iter01_opencode.json` |
| 2 | `OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS` | `REJECTED_CONTINUE` | `false` | best_metrics_failed:mean_delta,min_delta,positive_splits,fixes_gt_breaks | `reports/20260607T074546KST_axis_loop_iter02_opencode.json` |
| 3 | `OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS` | `REJECTED_CONTINUE` | `false` | best_metrics_failed:min_delta,fixes_gt_breaks,quarantine | `reports/20260607T074546KST_axis_loop_iter03_opencode.json` |
| 4 | `OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS` | `REJECTED_CONTINUE` | `false` | best_metrics_failed:min_delta,fixes_gt_breaks,quarantine | `reports/20260607T074546KST_axis_loop_iter04_opencode.json` |
| 5 | `OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS` | `REJECTED_CONTINUE` | `false` | best_metrics_failed:min_delta,fixes_gt_breaks,quarantine | `reports/20260607T074546KST_axis_loop_iter05_opencode.json` |
| 6 | `OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS` | `REJECTED_CONTINUE` | `false` | best_metrics_failed:min_delta,fixes_gt_breaks,quarantine | `reports/20260607T074546KST_axis_loop_iter06_opencode.json` |

## Artifacts

- JSON: `reports/20260607T074546KST_opencode_hermes_axis_rejection_loop_summary.json`
- Markdown: `reports/20260607T074546KST_opencode_hermes_axis_rejection_loop_summary.md`
