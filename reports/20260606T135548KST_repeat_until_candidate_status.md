# 2026-06-06 13:55 KST — repeat-until-candidate status

## Safety / scope

- Kaggle submit executed: **false** (`kaggle competitions submit` not run)
- Submission CSV created/sent: **false**
- Hidden labels / external Steam scraping: **false**
- Aggressive runner guard and quarantine: unchanged; existing `aggressive_quota_runner.py` process is still alive.

## Active processes checked

- `aggressive_quota_runner.py`: alive (`uv run` PID 7613, Python PID 7630), elapsed about 2h23m at check time.
- `lightgcn_exactk_subset_loss.py`: no active process.
- `hours_confidence_lightgcn_gate.py`: no active process.
- Active KMURecSys validation job: `scripts/userknn_gated_residual_probe.py` fine-grid follow-up
  - launcher PID 18483, `uv run` PID 18804, Python PID 18812
  - elapsed about 29m at check time, ~100% CPU
  - log: `logs/userknn_gated_residual_fine_20260606T132450KST.log`
  - intended reports: `reports/20260606T132450KST_userknn_gated_residual_fine.{md,json}` (not written yet; job still running)
  - artifact dir: `artifacts/userknn_gated_residual_fine_20260606T132450KST`

## GPU checked

- GPU0/1/2: 0 MiB used and effectively free.
- GPU3: 4320 MiB shown with stale/orphan allocation; compute-app query reports PID `595850` as `[Not Found]`, so GPU3 remains avoided.
- No GPU job launched this tick because a bounded validation-only KMURecSys probe is already running.

## Required artifact/state readback

- `state/aggressive_quota_runner_state.json`: public best still `0.77825`; prior failed/quarantined families include rankblend/ALS, boundary scoreblend/frontier, and TAG-CF variants. Quarantine/duplicate guards remain aggressive.
- `logs/latest_exactk_subset_outdir.txt`: points to `artifacts/exactk_subset_20260606T104621KST`, but the completed full summary is present at `artifacts/exactk_subset/val_random_uniform_seed42/summary.json`.
- Exact-K subset full summary: `SUBSET_NO_GAIN_NOISE`; subset vs BPR fine-tune isolated delta `+0.00000`, fixes=73 breaks=73, p=0.93404. Treat exact-K as closed unless a deliberate new hyperparameter reason is introduced.
- Exact-K smoke summary: `SUBSET_NO_GAIN_NOISE`; subset vs BPR fine-tune delta `-0.00010`.
- Hours-confidence summaries:
  - `user_quantile`: `CONF_PLATEAU_NO_GAIN`, delta `-0.00010` vs binary ref.
  - `item_quantile`: `CONF_PLATEAU_NO_GAIN`, delta `+0.00060` vs binary ref, still inside noise band and not `CONF_GAIN_CHECK_ENSEMBLE`.
  - `balanced`: `CONF_PLATEAU_NO_GAIN`, delta `+0.00020` vs binary ref.
- `artifacts/temporal_compat/val_random_uniform_seed42/summary.json`: all tested temporal reranks regress (`T_only`, `rank_sum`, `rank_sum_resid`, `boundary_swap`). Do not repeat.
- `artifacts/boundary_covariate/val_random_uniform_seed42/summary.json`: popularity-residualized signals weak/ambiguous; verdict soft no-go below escalation bar. Do not repeat.
- DNS pool=1 multisplit was already closed in `artifacts/dns_pool1_multisplit/three_uniform_panel/three_uniform_panel_summary.json` / `reports/20260606T132607KST_dns_split7_late_notifications_reconciled.md`: best aggregate mean vs ref `-0.002698`, only 1/3 positive splits; final verdict `DNS_POOL1_REJECT_SPLIT_SPECIFIC_NOISE`.

## Candidate status

- Strict candidate found this tick: **none**.
- Weak candidate axis requiring expansion: **none**.
- Tiny one-split/single-seed blips remain rejected as traps; no Kaggle preflight/submission materialization was attempted.

## Next action

Continue monitoring the active UserKNN gated residual fine-grid job. When it writes `reports/20260606T132450KST_userknn_gated_residual_fine.{md,json}`, evaluate it only against the strict gate (3 uniform splits, mean delta >= +0.00355, >=2/3 positive, fixes > breaks, paired p < 0.05, row-diff distinct). If it fails, close the UserKNN fine-grid axis and move to a genuinely orthogonal bounded no-submit probe instead of repeating near-neighbor variants.
