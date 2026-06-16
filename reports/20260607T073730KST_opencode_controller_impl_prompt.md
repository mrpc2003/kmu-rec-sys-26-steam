You are OpenCode implementing a bounded controller script in the repo /opt/data/kaggle/kmu-rec-sys-26-steam.

CRITICAL EXECUTION RULES:
- Answer/work entirely yourself in this single OpenCode run. Do NOT delegate to sub-agents. Do NOT wait for parallel agents.
- Your task is to implement the controller script only. Do NOT run Kaggle, do NOT submit, do NOT create full-test candidate/submission CSVs, and do NOT write under submissions/.
- Do NOT schedule cron jobs, do NOT modify existing cron, do NOT git add/commit/push.
- Allowed writes: scripts/opencode_hermes_axis_rejection_loop.py and optionally reports/logs for compile/smoke notes.
- The script itself will orchestrate future OpenCode axis-finding iterations, but this implementation run must not start that loop.
- Keep the script dependency-light: Python stdlib only.
- The script must be executable as:
  python3 scripts/opencode_hermes_axis_rejection_loop.py --max-iters 6 --opencode-timeout-sec 900 --probe-wait-sec 3600

Controller behavior required:
1. Workdir fixed/default: /opt/data/kaggle/kmu-rec-sys-26-steam.
2. Create a run directory under artifacts/opencode_hermes_axis_loop_<TS>/ and report/log files under reports/ and logs/.
3. Each iteration:
   - Collect concise state from recent reports/logs/processes without printing secrets.
   - Write reports/<TS>_axis_loop_iterNN_prompt.md for an OpenCode run.
   - Prompt OpenCode to find a fresh independent validation-only improvement axis, or launch exactly one bounded validation-only probe, avoiding closed/stalled axes.
   - Hard forbid `kaggle competitions submit`, full-test candidate/submission CSVs, hidden labels, external Steam scraping, tokens, git stage/commit/push, recursive cron.
   - Require OpenCode to write reports/<TS>_axis_loop_iterNN_opencode.{json,md}, and to end final text with one of:
     OPENCODE_AXIS_LOOP_DONE_IMPROVEMENT_AXIS_FOUND
     OPENCODE_AXIS_LOOP_DONE_NEXT_PROBE_RUNNING
     OPENCODE_AXIS_LOOP_DONE_NO_SAFE_AXIS
     OPENCODE_AXIS_LOOP_DONE_BLOCKED
   - Run `opencode run <prompt> --format json </dev/null` via subprocess with timeout, capture JSONL to logs/<TS>_axis_loop_iterNN_opencode.jsonl.
   - Extract JSONL text parts into reports/<TS>_axis_loop_iterNN_opencode_raw_text.md.
4. Hermes review step implemented in the script:
   - Parse OpenCode JSON report when present.
   - Verify safety flags: validation_only true; candidate_csv/full_test false; kaggle_submit false; hidden/private labels false; external scraping false; git stage/commit/push false.
   - Check no new .csv under submissions/ since iteration start.
   - Check no active `kaggle competitions submit` process, excluding the checker itself by avoiding that exact full phrase in ps grep.
   - If report has metrics for top/best candidate, classify strict pass only if mean_delta >= 0.0015, min_delta >= 0, positive_splits >= 3 or "3/3", fixes > breaks, pooled/mcnemar p < 0.05 if present, and no quarantine/near-duplicate conflict flag.
   - If OpenCode launched a probe and it is still running, wait/poll up to --probe-wait-sec for its report_json/report_md, then review. Do not start a new OpenCode iteration while a launched probe is still legitimately running.
   - If rejected/no_safe/blocker, append a compact rejection record and continue to next iteration until max_iters.
   - Stop early if strict pass or improvement-axis-found passes Hermes strict review.
5. Produce final controller summary JSON/MD:
   reports/<TS>_opencode_hermes_axis_rejection_loop_summary.{json,md}
   Include safety status, per-iteration OpenCode sentinel, Hermes verdict, rejection reasons, any running probe, and artifact paths.
6. Include argparse options: --max-iters, --opencode-timeout-sec, --probe-wait-sec, --sleep-between-iters-sec, --dry-run, --opencode-bin, --workdir.
7. In --dry-run, do not call OpenCode; write a one-iteration fake prompt and summary with dry_run=true.
8. Print only concise progress lines, never secrets.
9. Add clear docstring and robust error handling.
10. After writing the script, run Python compile check and a dry-run smoke command only:
    python3 -m py_compile scripts/opencode_hermes_axis_rejection_loop.py
    python3 scripts/opencode_hermes_axis_rejection_loop.py --dry-run --max-iters 1

End your final response with exactly:
OPENCODE_CONTROLLER_IMPL_DONE
