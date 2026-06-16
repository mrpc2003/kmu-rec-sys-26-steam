# Improvement-axis cron status — 2026-06-07 09:18:46 KST

## Verdict

- **New OpenCode/probe launched:** yes — bounded OpenCode run `20260607T090941KST` completed.
- **New axis found:** OTTO-style source-separated co-visitation residual smoke.
- **Classification:** `WEAK_SIGNAL_STRICT_GATE_FAIL`.
- **Strict-pass candidate-like axis:** none.
- **Kaggle submit / candidate CSV:** not executed / not created.

## New probe result

OpenCode implemented and ran `scripts/otto_source_covisit_smoke.py` with the bounded command:

```text
timeout 600 env -u VIRTUAL_ENV UV_NO_ACTIVE_VENV=1 uv run --with numpy --with pandas --with scipy python scripts/otto_source_covisit_smoke.py
```

Top aggregate row:

| metric | value |
|---|---:|
| variant | `base_plus_score_coplay_top5_mean_w0.2` |
| mean Δ vs emb128 4-seed base | `+0.0012335800` |
| min split Δ | `+0.0005001000` |
| positive splits | `3/3` |
| fixes / breaks | `699 / 625` |
| pooled exact p | `0.0447918733` |

Split deltas:

- `val_random_uniform_seed42`: `+0.0005001000`
- `val_random_uniform_seed7`: `+0.0013002601`
- `val_random_uniform_seed123`: `+0.0019003801`

This is a real weak multi-split signal, but it fails the predeclared strict gate because mean Δ is below `+0.0015`. Therefore it is **not** a candidate and must not be submitted/materialized as a full-test CSV.

## Required prior probe checks

| Axis | Status | Classification | Evidence |
|---|---:|---|---|
| UserKNN gated residual fine-grid | not running | `STALLED_INCOMPLETE_PREVIOUSLY_CLOSED` | `reports/20260606T132450KST_userknn_gated_residual_fine.{json,md}` still missing; broad fine-grid not relaunched. |
| Jackknife uncertainty boundary expanded | not running | `FAILED_INCOMPLETE_NO_METRIC_REPORT` | expected expanded `{json,md}` missing; log has only 12 lines and stops mid `val_random_uniform_seed123`. |
| Jackknife boundary smoke | completed | `WEAK_SIGNAL_STRICT_GATE_FAIL` | top mean Δ `+0.0003667`, min Δ `-0.0012002`, positives `2/3`, fixes/breaks `252/230`, p `0.338815`. |

## Safety status

- `validation_only=true`
- active `kaggle competitions submit` processes after tick: `0`
- full-test candidate/submission CSVs created by this tick: `false`
- validation artifact CSV headers: non-uploadable; each split file has `19,996` rows and includes validation labels/features, not submission schema.
- hidden/private labels: not used
- external Steam scraping: not used
- quarantine/guard weakening: not done
- git stage/commit/push: not done
- recursive cron scheduling: not done
- secret scan over selected new files: no credential hits

## Resources after tick

- Hermes background processes: none
- project probe processes matching OTTO/UserKNN/jackknife/aggressive runner: none
- GPU0: `0/32768 MiB`, util `0%`
- GPU1: `0/32768 MiB`, util `0%`
- GPU2: `0/32768 MiB`, util `1%`
- GPU3: `4320/32768 MiB`, util `2%`

## Artifacts

- Status JSON: `reports/20260607T091846KST_improvement_axis_cron_status.json`
- Status MD: `reports/20260607T091846KST_improvement_axis_cron_status.md`
- OpenCode prompt: `reports/20260607T090941KST_opencode_axis_loop_prompt.md`
- OpenCode JSONL: `logs/opencode_improvement_axis_loop_20260607T090941KST.jsonl`
- OpenCode raw text: `reports/20260607T090941KST_opencode_improvement_axis_loop_raw_text.md`
- OpenCode report: `reports/20260607T090941KST_opencode_improvement_axis_loop.{json,md}`
- Probe script: `scripts/otto_source_covisit_smoke.py`
- Probe log: `logs/20260607T090941KST_otto_source_covisit_smoke.log`
- Validation artifacts: `artifacts/opencode_axis_loop_20260607T090941KST/otto_source_covisit/`

## Verification

- OpenCode version: `1.15.11`; smoke sentinel passed.
- OpenCode final extracted sentinel/report verdict: `OPENCODE_AXIS_LOOP_DONE_WEAK_SIGNAL` / `WEAK_SIGNAL`.
- `python3 -m py_compile scripts/otto_source_covisit_smoke.py`: OK.
- JSON parse: OK.
- selected `git diff --check`: OK.
- selected secret scan: OK.

## Exact next action

Do **not** submit and do **not** create a full-test candidate. Treat OTTO source-separated co-visitation as a weak, non-candidate signal. A future tick can run a bounded validation-only expansion around this source family only if it stays OpenCode-first, avoids test/submission materialization, and predeclares a stricter multi-split/noise gate; otherwise continue searching for a stronger independent axis.
