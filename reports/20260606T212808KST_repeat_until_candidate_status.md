# KMURecSys26 no-submit candidate discovery status — 2026-06-06 21:28:08 KST

## Safety
- Mode: validation-only monitoring tick; no Kaggle submission command run by this tick.
- No new submission/candidate CSV was created by this tick.
- Existing aggressive runner guards/quarantines were only read, not modified.
- Existing `submissions/*.csv` count observed: 22; latest mtimes remain from 2026-06-03, not this tick.
- Report scan observed 0 positive submit/candidate-write markers in JSON/JSONL reports.

## Live processes
- Aggressive runner is still present:
  - PID 7613 `uv run ... scripts/aggressive_quota_runner.py --sleep-no-quota 300 --sleep-no-candidate 600 --sleep-after-submit 21600`
  - Child PID 7630 Python runner, CPU ~0%, elapsed ~09:57.
- Active validation-only CPU-bound probe is still running and live:
  - Wrapper PID 18483, `userknn_gated_residual_fine_20260606T132450KST`.
  - Python child PID 18812, elapsed ~08:03, CPU time advanced from 08:03:29 to 08:03:32 over a 3s sample, ~100% CPU.
  - Log mtime was current at 2026-06-06 21:27:36 KST despite repeated NumPy divide warnings.
  - Pending outputs:
    - `reports/20260606T132450KST_userknn_gated_residual_fine.json`
    - `reports/20260606T132450KST_userknn_gated_residual_fine.md`
    - `artifacts/userknn_gated_residual_fine_20260606T132450KST/`
    - `logs/userknn_gated_residual_fine_20260606T132450KST.log`

## GPU state
- GPU0: V100 32GB, 0 MiB used, 0% util.
- GPU1: V100 32GB, 0 MiB used, 0% util.
- GPU2: V100 32GB, 0 MiB used, ~1% util.
- GPU3: V100 32GB, 4320 MiB used, ~2% util, no owning process shown by `nvidia-smi pmon`; treat as stale/orphan allocation and do not refill there.

## Required artifact readback
- `state/aggressive_quota_runner_state.json`
  - mtime: 2026-06-03 17:42:33 KST.
  - last public best seen: 0.77825.
  - operating policy still records exact-SHA duplicate and near-duplicate guards plus 21600s post-submit sleep.
  - quarantined family map present; skipped variants count observed: 111.
- `logs/latest_exactk_subset_outdir.txt`
  - points to `artifacts/exactk_subset_20260606T104621KST`, but the canonical completed full summary remains under `artifacts/exactk_subset/val_random_uniform_seed42/summary.json`.
- Exact-K subset loss:
  - `artifacts/exactk_subset/val_random_uniform_seed42/summary.json`: subset vs BPR delta +0.00000, vs pretrained -0.00180, McNemar fixes=73 breaks=73 p=0.93404; tier is the no-gain/noise label, so exact-K remains closed absent a deliberate new hyperparameter reason.
  - Smoke rerun `artifacts/exactk_subset_smoke/val_random_uniform_seed42/summary.json`: subset vs BPR -0.00010, vs pretrained -0.00030; same no-gain/noise tier.
- Hours-confidence LightGCN:
  - `balanced`: acc 0.76225 vs ref 0.76205, delta +0.00020; plateau/no-gain tier.
  - `user_quantile`: acc 0.76195, delta -0.00010; plateau/no-gain tier.
  - `item_quantile`: acc 0.76265, delta +0.00060; plateau/no-gain tier.
  - No mode returned the ensemble-expansion tier, so no 3-split/4-seed expansion was launched.
- Temporal compatibility:
  - `artifacts/temporal_compat/val_random_uniform_seed42/summary.json`: all tested rerank variants regressed heavily vs base (`T_only`, `rank_sum`, `rank_sum_resid`, `boundary_swap`); do not repeat.
- Boundary covariate:
  - `artifacts/boundary_covariate/val_random_uniform_seed42/summary.json`: popularity-residualized novel covariates are weak/ambiguous and below escalation bar; soft no-go.

## Candidate status
- Strict candidate: none found. Current artifacts do not approach the 3-split mean-delta, 2/3-positive, fixes>breaks, paired-significance, and row-diff-distinct gate.
- Weak candidate axis: none found. Hours-confidence modes were all plateau/no-gain; the active UserKNN fine-grid job has not produced its final report yet.
- Action this tick: no new experiment launched because a bounded validation-only probe is live and making CPU progress. Next tick should parse the pending UserKNN fine-grid JSON/MD paths above before considering any orthogonal smoke.
