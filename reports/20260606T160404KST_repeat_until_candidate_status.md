# 2026-06-06 16:04 KST — repeat-until-candidate status

## Safety / scope

- Kaggle submit executed: **false** (`kaggle competitions submit` not run)
- Submission/candidate CSV created or sent: **false**
- Hidden labels / external Steam scraping: **false**
- Existing aggressive runner guards/quarantines: **unchanged**

## Required process / GPU check

- `aggressive_quota_runner.py`: alive (`uv run` PID 7613, Python PID 7630), elapsed about 4h32m at 16:03 KST.
- `lightgcn_exactk_subset_loss.py`: no active process found.
- `hours_confidence_lightgcn_gate.py`: no active process found.
- Active KMURecSys validation job: `scripts/userknn_gated_residual_probe.py` fine-grid follow-up
  - launcher PID 18483, `uv run` PID 18804, Python PID 18812
  - elapsed about 2h39m at 16:03 KST; Python child still running at ~100% CPU.
  - log: `logs/userknn_gated_residual_fine_20260606T132450KST.log`
  - log mtime was current at the check; line count around 7103 and current output is dominated by NumPy divide warnings from `scripts/userknn_residual_probe.py:114`.
  - expected reports still pending: `reports/20260606T132450KST_userknn_gated_residual_fine.{md,json}`
  - expected artifact dir: `artifacts/userknn_gated_residual_fine_20260606T132450KST` (no files yet)
- GPU state from `nvidia-smi`:
  - GPU0/1/2: 0 MiB used, effectively free.
  - GPU3: 4320 MiB stale/orphan allocation; compute-app query reports PID `595850` as `[Not Found]`, so GPU3 remains avoided.

## Required artifact/state readback

- `state/aggressive_quota_runner_state.json`: public best still `0.77825`; submitted/quarantined families include rankblend/ALS residuals, boundary scoreblend/frontier, and TAG-CF. Guards remain aggressive.
- `logs/latest_exactk_subset_outdir.txt`: points to `artifacts/exactk_subset_20260606T104621KST`; canonical completed summaries were reconciled instead.
- Exact-K subset full: `artifacts/exactk_subset/val_random_uniform_seed42/summary.json` tier `SUBSET_NO_GAIN_NOISE`; isolated subset-vs-BPR delta `+0.00000`, fixes=73 breaks=73, p=0.93404. Axis remains closed unless there is a deliberate new hyperparameter reason.
- Exact-K smoke: `artifacts/exactk_subset_smoke/val_random_uniform_seed42/summary.json` tier `SUBSET_NO_GAIN_NOISE`; subset-vs-BPR delta `-0.00010`.
- Hours-confidence summaries:
  - `user_quantile`: `CONF_PLATEAU_NO_GAIN`, delta `-0.00010` vs binary ref.
  - `item_quantile`: `CONF_PLATEAU_NO_GAIN`, delta `+0.00060` vs binary ref; inside noise band and **not** `CONF_GAIN_CHECK_ENSEMBLE`.
  - `balanced`: `CONF_PLATEAU_NO_GAIN`, delta `+0.00020`.
- Temporal compatibility: `artifacts/temporal_compat/val_random_uniform_seed42/summary.json` — all tested temporal reranks regress (`T_only`, `rank_sum`, `rank_sum_resid`, `boundary_swap`); do not repeat.
- Boundary covariate: `artifacts/boundary_covariate/val_random_uniform_seed42/summary.json` — residualized covariates weak/ambiguous; soft no-go below escalation bar; do not repeat.
- DNS pool=1 multisplit remains rejected from earlier report: best aggregate mean vs ref `-0.002698`, only 1/3 positive splits, final verdict `DNS_POOL1_REJECT_SPLIT_SPECIFIC_NOISE`.
- Multi-interest SVD64 smoke remains rejected: best 50/50 z-blend delta `-0.07141`, solo below floor.

## Candidate status

- Strict candidate found this tick: **none**.
- Weak candidate axis requiring 3-split/4-seed expansion: **none**.
- Tiny one-split/single-seed blips remain rejected as traps. No Kaggle preflight/submission materialization was attempted.

## Action taken / next action

- No new experiment launched in this tick because one bounded validation-only KMURecSys probe is already actively running and has not yet written its report.
- Next check should parse `reports/20260606T132450KST_userknn_gated_residual_fine.{md,json}` once written. If it does not meet the strict 3-uniform-split candidate gate, close the UserKNN fine-grid axis and move to a genuinely orthogonal no-submit smoke rather than repeating nearby UserKNN/rankblend/boundary/TAG-CF/semantic/temporal/exact-K/pop-bias variants.
