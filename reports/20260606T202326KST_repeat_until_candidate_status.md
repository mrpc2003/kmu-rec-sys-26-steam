# 2026-06-06 20:23 KST — repeat-until-candidate status

## Safety / scope

- Kaggle submit executed this tick: **false** (`kaggle competitions submit` was not run by this tick; process scan did not show a submit process).
- New submission/candidate CSV created or sent: **false** (observed `submissions/*.csv` count remains 22; newest files are from 2026-06-03 KST/UTC mtime area; no candidate materialization attempted).
- Hidden labels / external Steam scraping: **false**.
- Existing aggressive runner guards/quarantines: **unchanged**; no guard weakening and no quarantine edits.

## Required process / GPU check

- `aggressive_quota_runner.py`: alive (`uv run` PID 7613, Python PID 7630), elapsed ~8h50m at 20:20–20:23 KST. Latest watchdog lines at 20:22 KST: quota `0/5`, public best `0.77825` (`candidate_rank_blend_emb128_emb192.csv`), “No validation-positive unsubmitted variant available; waiting for new experiment axes.”
- `lightgcn_exactk_subset_loss.py`: no active process found.
- `hours_confidence_lightgcn_gate.py`: no active process found.
- Active KMURecSys validation job: `scripts/userknn_gated_residual_probe.py` fine-grid follow-up
  - launcher PID 18483, `uv run` PID 18804, Python PID 18812.
  - elapsed ~6h56m at 20:21 KST; Python child at ~100% CPU.
  - liveness proof: CPU TIME advanced from `06:56:54` to `06:57:02` over an 8s sample; log mtime remained current at ~20:23 KST.
  - log: `logs/userknn_gated_residual_fine_20260606T132450KST.log` (~18,825 lines / ~1.74 MB at this tick; dominated by NumPy divide warnings from `scripts/userknn_residual_probe.py:114`).
  - expected reports still pending: `reports/20260606T132450KST_userknn_gated_residual_fine.{md,json}`.
  - expected artifact dir: `artifacts/userknn_gated_residual_fine_20260606T132450KST` (no completed result files yet).
- Hermes-tracked background process list: empty; OS `ps` is the source of truth.
- GPU state at 20:20 KST:
  - GPU0: 0 MiB, 0% util
  - GPU1: 0 MiB, 0% util
  - GPU2: 0 MiB, ~1% util
  - GPU3: 4320 MiB, ~1% util, no process in `pmon`; continue to treat as stale/orphan allocation and avoid.

## Required artifact/state readback

- `state/aggressive_quota_runner_state.json`: public best remains `0.77825`; submitted/quarantined families include rankblend/ALS residuals, boundary scoreblend/frontier, and TAG-CF. Guard policy still blocks rapid quota burns, exact/near duplicates, and same-family reruns after non-improving public transfer.
- `logs/latest_exactk_subset_outdir.txt`: points at `artifacts/exactk_subset_20260606T104621KST`; that directory has no completed summary files, so canonical completed exact-K summaries were used for the decision.
- Exact-K subset full: `artifacts/exactk_subset/val_random_uniform_seed42/summary.json` tier `SUBSET_NO_GAIN_NOISE`; isolated subset-vs-BPR delta `+0.00000`, fixes=73 breaks=73, p=0.93404. Axis remains closed unless there is a deliberate new hyperparameter reason.
- Exact-K smoke: `artifacts/exactk_subset_smoke/val_random_uniform_seed42/summary.json` tier `SUBSET_NO_GAIN_NOISE`; subset-vs-BPR delta `-0.00010`.
- Hours-confidence summaries:
  - `user_quantile`: `CONF_PLATEAU_NO_GAIN`, delta `-0.00010` vs binary ref.
  - `item_quantile`: `CONF_PLATEAU_NO_GAIN`, delta `+0.00060` vs binary ref; inside the script’s no-gain/noise tier and **not** the ensemble-escalation tier.
  - `balanced`: `CONF_PLATEAU_NO_GAIN`, delta `+0.00020`.
  - `binary_control`: `CONF_PLATEAU_NO_GAIN`, delta `+0.00000`.
- Temporal compatibility: `artifacts/temporal_compat/val_random_uniform_seed42/summary.json` — all tested temporal reranks regress (`REGRESS`); do not repeat.
- Boundary covariate: `artifacts/boundary_covariate/val_random_uniform_seed42/summary.json` — residualized covariates weak/ambiguous; verdict is soft no-go below escalation bar; do not repeat.
- DNS pool=1 three-uniform panel: `artifacts/dns_pool1_multisplit/three_uniform_panel/three_uniform_panel_summary.json` verdict `DNS_POOL1_REJECT_SPLIT_SPECIFIC_NOISE`; best aggregate still negative on mean vs reference.

## Candidate status

- Strict candidate found this tick: **none**.
- Weak candidate axis requiring 3-split/4-seed expansion: **none**.
- Tiny one-split/single-seed blips remain rejected as traps. No Kaggle preflight/submission materialization was attempted.

## Action taken / next action

- No new experiment launched in this tick because the bounded validation-only `userknn_gated_residual_probe.py` fine-grid job is still actively running and has not written its report yet.
- Next tick should parse `reports/20260606T132450KST_userknn_gated_residual_fine.{md,json}` once present. If it does not meet the strict 3-uniform-split gate, close the UserKNN fine-grid axis and pivot to one genuinely orthogonal validation-only smoke, avoiding rankblend/ALS, boundary scoreblend, TAG-CF, semantic residual, UserKNN, temporal, exact-K, SL@K-lite, and pop-bias near-duplicates.
