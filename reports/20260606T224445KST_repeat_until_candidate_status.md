# KMURecSys26 Steam no-submit candidate-discovery tick — 2026-06-06 22:44:45 KST

## Safety

- validation_only: true
- kaggle_submit_executed_by_this_tick: false
- actual Kaggle CLI submit processes: 0
- candidate/submission CSV created by this tick: false
- hidden labels / external Steam scraping: not used
- aggressive runner guard/quarantine files were not weakened or edited by this tick

## Active processes

- `aggressive_quota_runner.py`: live (`uv` PID 7613, child PID 7630), idle/no GPU; existing guard state preserved.
- `userknn_gated_residual_probe.py`: live PID 18812, elapsed about 9h19m, CPU time advancing; output pending.
  - log: `logs/userknn_gated_residual_fine_20260606T132450KST.log`
  - expected reports: `reports/20260606T132450KST_userknn_gated_residual_fine.{md,json}`
  - artifact dir: `artifacts/userknn_gated_residual_fine_20260606T132450KST`
- OpenCode improvement-axis loop: live PID 26417, launched earlier with explicit no-submit/no-candidate safety prompt.
  - status: `reports/20260606T220406KST_opencode_improvement_axis_loop_status.md`
  - log: `logs/opencode_improvement_axis_loop_20260606T220406KST.jsonl`
  - expected final report: `reports/20260606T220406KST_opencode_improvement_axis_loop.{md,json}`
- New validation-only jackknife uncertainty boundary expanded probe: live PID 28974 via timeout wrapper PID 28965.
  - command family: `scripts/jackknife_uncertainty_boundary_probe.py --weights 0.025,0.05,0.075,0.1,0.15,0.2,0.3 --bands 1,2,3,4`
  - log: `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log`
  - expected reports: `reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.{md,json}`
  - artifact dir: `artifacts/opencode_axis_loop_20260606T220406KST/jackknife_uncertainty_boundary_expanded`

## GPU snapshot

- GPU0: 0 / 32768 MiB, 0% util
- GPU1: 0 / 32768 MiB, 0% util
- GPU2: 0 / 32768 MiB, ~1% util
- GPU3: 4320 / 32768 MiB, ~2% util, no owning process in `nvidia-smi pmon`; treat as stale/orphan and avoid scheduling there.

No new GPU job was launched because useful CPU-bound validation-only work is live and the current queue items are already closed or running.

## Required artifact/state readback

- `state/aggressive_quota_runner_state.json`: public best remains 0.77825; quarantines include failed rankblend/ALS residual families, boundary scoreblend/frontier variants, and TAG-CF public-transfer misses. State still has autonomous runner enabled; this tick did not modify it.
- `logs/latest_exactk_subset_outdir.txt`: points to `artifacts/exactk_subset_20260606T104621KST`, but that outdir has no summary files. Canonical completed summary is `artifacts/exactk_subset/val_random_uniform_seed42/summary.json`.
- Exact-K subset loss: canonical tier is `SUBSET_NO_GAIN_NOISE`; subset fine-tune = BPR fine-tune (0.76025), both below pretrained 0.76205; closed unless a deliberate new hyperparameter reason appears.
- Hours-confidence modes: all present and plateau/no-gain tier.
  - user_quantile: acc 0.76195, delta -0.00010
  - item_quantile: acc 0.76265, delta +0.00060, still below noise/escalation bar
  - balanced: acc 0.76225, delta +0.00020
- Temporal compatibility: present and rejected; best reported combiner `rank_sum_resid` 0.67243 vs base 0.76505, large regression.
- Boundary covariate expansion: present and soft no-go; residualized signals are weak/ambiguous below escalation bar.

## New result during this tick

OpenCode's smoke jackknife-uncertainty boundary probe completed validation-only:

- report: `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.md`
- JSON: `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json`
- verdict: `WEAK_SIGNAL`, strict pass count 0
- best smoke variant: `vote_consensus__high_capacity_gap__B1__w0.1`
  - 3-split mean delta: +0.000367
  - positives: 2/3 splits
  - min split delta: -0.001200
  - fixes/breaks: 252 / 230
  - paired p: 0.3388

This is not a strict candidate: gain is far below +0.00355, p is not significant, and one split is negative. It is only being expanded as a bounded validation-only probe to see whether the same uncertainty axis has any stable setting; no Kaggle submission or full-test candidate CSV is allowed.

## Candidate status

No strict candidate and no script-prescribed weak-candidate escalation tier found. Current action is monitor the live expanded jackknife probe plus the long UserKNN fine-grid, then parse their reports when complete.

## Next tick checklist

1. Check whether `reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.{md,json}` exists and classify it against strict gates.
2. Check whether `reports/20260606T132450KST_userknn_gated_residual_fine.{md,json}` exists and classify strict/weak/reject.
3. Reconstruct OS process state because the expanded probe was launched by OpenCode as an orphaned timeout wrapper, not as a Hermes-tracked process.
4. Do not launch hours/exact-K/temporal/boundary repeats unless a genuinely new deliberate axis is introduced.
