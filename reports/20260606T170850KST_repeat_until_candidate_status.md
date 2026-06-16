# 2026-06-06 17:08 KST — repeat-until-candidate status

## Safety / scope

- Kaggle submit executed: **false** (`kaggle competitions submit` process count checked as 0; no submit command run this tick)
- Submission/candidate CSV created or sent: **false**
- Hidden labels / external Steam scraping: **false**
- Existing aggressive runner guards/quarantines: **unchanged**

## Required process / GPU check

- `aggressive_quota_runner.py`: alive (`uv run` PID 7613, Python PID 7630), elapsed about 5h36m at 17:06 KST.
- `lightgcn_exactk_subset_loss.py`: no active process found.
- `hours_confidence_lightgcn_gate.py`: no active process found.
- Active KMURecSys validation job: `scripts/userknn_gated_residual_probe.py` fine-grid follow-up
  - launcher PID 18483, `uv run` PID 18804, Python PID 18812
  - elapsed about 3h44m at 17:08 KST; Python child still running at ~100% CPU.
  - liveness: CPU TIME advanced from 03:43:56 to 03:44:01 over a 5s sample; log mtime current.
  - log: `logs/userknn_gated_residual_fine_20260606T132450KST.log`
  - log snapshot: ~9,967 lines / ~922 KB at this tick; output remains dominated by NumPy divide warnings from `scripts/userknn_residual_probe.py:114`.
  - expected reports still pending: `reports/20260606T132450KST_userknn_gated_residual_fine.{md,json}`
  - expected artifact dir: `artifacts/userknn_gated_residual_fine_20260606T132450KST` (no files yet)
- GPU state from `nvidia-smi` at 17:06 KST:
  - GPU0: 0 MiB, 0% util
  - GPU1: 0 MiB, 0% util
  - GPU2: 0 MiB, ~1% util, no process shown
  - GPU3: 4320 MiB, ~2% util, no process shown by `pmon`; treated as stale/orphan allocation and avoided.
- Hermes-tracked background process list: empty; live OS processes above are the source of truth.

## Required artifact/state readback

- `state/aggressive_quota_runner_state.json`: public best remains `0.77825`; submitted/quarantined families include rankblend/ALS residuals, boundary scoreblend/frontier, and TAG-CF. Guard policy still blocks rapid quota burns, exact/near duplicates, and same-family reruns after non-improving public transfer.
- `logs/latest_exactk_subset_outdir.txt`: `artifacts/exactk_subset_20260606T104621KST`; that timestamped outdir has no readable summary at the expected path, so canonical completed summary was used.
- Exact-K subset full: `artifacts/exactk_subset/val_random_uniform_seed42/summary.json` tier `SUBSET_NO_GAIN_NOISE`; isolated subset-vs-BPR delta `+0.00000`, fixes=73 breaks=73, p=0.93404. Axis remains closed without a deliberate new hyperparameter reason.
- Exact-K smoke: `artifacts/exactk_subset_smoke/val_random_uniform_seed42/summary.json` tier `SUBSET_NO_GAIN_NOISE`; subset-vs-BPR delta `-0.00010`.
- Hours-confidence summaries:
  - `user_quantile`: `CONF_PLATEAU_NO_GAIN`, delta `-0.00010` vs binary ref.
  - `item_quantile`: `CONF_PLATEAU_NO_GAIN`, delta `+0.00060` vs binary ref; within the 0.0007 noise band and **not** `CONF_GAIN_CHECK_ENSEMBLE`.
  - `balanced`: `CONF_PLATEAU_NO_GAIN`, delta `+0.00020`.
- Temporal compatibility: `artifacts/temporal_compat/val_random_uniform_seed42/summary.json` — all tested temporal reranks regress; do not repeat.
- Boundary covariate: `artifacts/boundary_covariate/val_random_uniform_seed42/summary.json` — residualized covariates weak/ambiguous; soft no-go below escalation bar; do not repeat.

## Candidate status

- Strict candidate found this tick: **none**.
- Weak candidate axis requiring 3-split/4-seed expansion: **none**.
- Tiny one-split/single-seed blips remain rejected as traps. No Kaggle preflight/submission materialization was attempted.

## Action taken / next action

- No new experiment launched in this tick because the bounded validation-only `userknn_gated_residual_probe.py` fine-grid job is still actively running and has not written its report yet.
- Next check should parse `reports/20260606T132450KST_userknn_gated_residual_fine.{md,json}` once written. If it does not meet the strict 3-uniform-split gate, close the UserKNN fine-grid axis and pivot to a genuinely orthogonal no-submit smoke rather than repeating nearby UserKNN/rankblend/boundary/TAG-CF/semantic/temporal/exact-K/pop-bias variants.
