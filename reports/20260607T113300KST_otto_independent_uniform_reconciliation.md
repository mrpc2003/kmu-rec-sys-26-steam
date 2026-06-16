# OTTO independent uniform confirmation — final reconciliation

- Timestamp: `20260607T1133xxKST`
- Source result JSON: `reports/20260607T095549KST_otto_independent_uniform_confirmation.json`
- Source result Markdown: `reports/20260607T095549KST_otto_independent_uniform_confirmation.md`
- Background session: `proc_9c3f4cce0a62`
- Process exit: `0`

## Verdict

`INDEPENDENT_DIAGNOSTIC_ONLY_POSITIVE_STRICT_FAIL`

The pre-registered confirmation row did **not** pass the independent strict gate, so candidate/submission escalation is **not allowed**.

## Pre-registered strict row

- Variant: `pre_registered_old_panel_top_coplay_top5_reverse_recent`
- Terms: `base + 0.090 * score_coplay_top5_mean + 0.040 * score_reverse_recent`
- mean Δ vs base: `+0.0006668000`
- min split Δ: `-0.0006001200`
- positive splits: `2/3`
- fixes/breaks: `424/384`
- pooled exact p: `0.1700198675`

Split deltas:

- `val_random_uniform_seed314`: `+0.0010002000`
- `val_random_uniform_seed2025`: `+0.0016003201`
- `val_random_uniform_seed2718`: `-0.0006001200`

Strict gate failures:

- mean Δ `< +0.0015`
- min Δ `< 0`
- positive splits `2/3`, not `3/3`
- p `>= 0.05`

## Diagnostic rows

No diagnostic row passed the strict gate. The best diagnostic row remained positive on average but failed because `seed2718` was negative:

- `looso_seed42_choice_coplay_top5_w0.100_reverse_recent_w0.070`
  - mean Δ `+0.0013669401`
  - min Δ `-0.0006001200`
  - positive splits `2/3`
  - fixes/breaks `548/466`
  - p `0.0109325640`

## Safety verification

Verified after completion:

- `python3 -m py_compile`: PASS for changed OTTO scripts
- result JSON parse: PASS
- active relevant processes: none
- `submissions/` new files after launch: none
- uploadable `ID,Label` CSV under run root: none
- run-root CSVs are validation artifacts with columns such as `ID,userID,gameID,Label,...`
- Kaggle submit process: none
- hidden/private labels: not used
- external Steam scraping: not used
- git add/commit/push: not executed
- secret-ish scan: only policy/variable-name false positives (`credentials_or_tokens_printed`, local variable `token`)

## Runner state

- no-submit OpenCode discovery cron `4d627b59804f`: resumed after the confirmation run
- submit-capable watchdog `272808a2bcca`: remains paused because no strict candidate exists

## Interpretation

The OTTO co-visitation residual is a real-looking weak positive axis, but the fresh independent panel shows it is not stable enough for submission work. Same-panel/LOOSO positives were likely over-tuned or panel-specific. Keep it as research evidence, not as a candidate.
