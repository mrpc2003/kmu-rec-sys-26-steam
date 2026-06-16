# OpenCode improvement-axis loop — 20260606T220406KST

## Safety flags
- validation_only: `true`
- candidate_csv_written: `false`
- kaggle_submit_executed: `false`
- hidden_labels_used: `false`
- external_scraping_used: `false`

## Active/completed process summary
- UserKNN gated residual fine-grid: **RUNNING** (`scripts/userknn_gated_residual_probe.py`, pids 18483/18804/18812). I did not duplicate this grid.
- Jackknife uncertainty boundary smoke: **COMPLETED**, `WEAK_SIGNAL`, strict pass count `0`. Best variant `vote_consensus__high_capacity_gap__B1__w0.1`: mean Δ `+0.0003667`, min Δ `-0.0012002`, positive splits `2/3`, fixes/breaks `252/230`, p `0.3388`. Reject for escalation.
- Expanded jackknife uncertainty boundary probe: **RUNNING** under `timeout 3600`, pids 28965/28966/28974, log `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log`.
- Aggressive quota runner remains pre-existing/running; no submit command was issued by this loop.

## Closed-axis summary since this run
- **UserKNN gated residual fine-grid**: Still running; no result to classify; no duplicate grid launched.
- **Jackknife uncertainty boundary smoke**: WEAK_SIGNAL only: best mean Δ +0.0003667, min Δ -0.0012002, 2/3 positive, fixes/breaks 252/230, p=0.3388. Not a strict improvement axis.
- **Initial broad jackknife grid attempt**: Killed by 600s timeout before completion; script was optimized and rerun as bounded smoke.

## Ranked next-axis hypotheses
1. **Expanded jackknife seed/capacity uncertainty boundary rerank** (`RUNNING`)
   - independence: Uses only model-family seed uncertainty, capacity disagreement, and boundary-local calibration from emb128/emb192 validation scores; no UserKNN, DNS, hours confidence, exact-K, temporal features, text semantics, or test candidate marginal.
   - why credible: Smoke found a small positive mean in vote-consensus capacity-gap gates on 2/3 splits and writes row-level score artifacts; expansion checks whether wider bands/weights repair split42 without over-claiming.
2. **Active UserKNN gated residual fine-grid** (`RUNNING_PREEXISTING`)
   - independence: User-neighborhood residual from train interactions, distinct from seed uncertainty; already running from prior process.
   - why credible: Prior 3-split weak signals around +0.0008~+0.0009; fine-grid may determine whether this is robust or noise.
3. **Per-user calibration of existing rankblend by unsupervised margin strata only** (`NOT_LAUNCHED`)
   - independence: Would avoid new model families and use only rankblend margin/error-prone strata; not launched because the jackknife expansion is already testing a stronger boundary-local uncertainty version.
   - why credible: Could identify low-margin regimes where public-best rankblend is unstable.

## Commands run or launched
- `python -m py_compile scripts/jackknife_uncertainty_boundary_probe.py`
- `UV_LINK_MODE=copy uv run --with numpy --with pandas --with scipy python scripts/jackknife_uncertainty_boundary_probe.py --weights 0.05,0.1 --bands 1 --json reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json --md reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.md --out-dir artifacts/opencode_axis_loop_20260606T220406KST/jackknife_uncertainty_boundary`
- `timeout 3600 env UV_LINK_MODE=copy uv run --with numpy --with pandas --with scipy python scripts/jackknife_uncertainty_boundary_probe.py --weights 0.025,0.05,0.075,0.1,0.15,0.2,0.3 --bands 1,2,3,4 --json reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.json --md reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.md --out-dir artifacts/opencode_axis_loop_20260606T220406KST/jackknife_uncertainty_boundary_expanded > logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log 2>&1 &`

## Artifacts/reports produced
- scripts:
  - `scripts/jackknife_uncertainty_boundary_probe.py`
- reports:
  - `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.md`
  - `reports/20260606T220406KST_jackknife_uncertainty_boundary_probe.json`
  - `reports/20260606T220406KST_opencode_improvement_axis_loop.md`
  - `reports/20260606T220406KST_opencode_improvement_axis_loop.json`
- logs:
  - `logs/20260606T220406KST_jackknife_uncertainty_boundary_probe.log`
  - `logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log`
- validation_artifacts:
  - `artifacts/opencode_axis_loop_20260606T220406KST/jackknife_uncertainty_boundary/split_metrics.csv`
  - `artifacts/opencode_axis_loop_20260606T220406KST/jackknife_uncertainty_boundary/aggregate.csv`
  - `artifacts/opencode_axis_loop_20260606T220406KST/jackknife_uncertainty_boundary/val_random_uniform_seed42__top_variant_scores.csv`
  - `artifacts/opencode_axis_loop_20260606T220406KST/jackknife_uncertainty_boundary/val_random_uniform_seed7__top_variant_scores.csv`
  - `artifacts/opencode_axis_loop_20260606T220406KST/jackknife_uncertainty_boundary/val_random_uniform_seed123__top_variant_scores.csv`
  - `artifacts/opencode_axis_loop_20260606T220406KST/jackknife_uncertainty_boundary_expanded/`

## Verdict
`NEXT_PROBE_RUNNING`

## Next Hermes action
Monitor logs/20260606T220406KST_jackknife_uncertainty_boundary_expanded.log and reports/20260606T220406KST_jackknife_uncertainty_boundary_expanded.json. Escalate only if strict_pass_count > 0 with mean Δ >= +0.0015, min Δ >= 0, 3/3 positive splits, fixes > breaks, pooled exact p < 0.05, and no quarantine/near-duplicate conflict. Also continue monitoring reports/20260606T132450KST_userknn_gated_residual_fine.json for the active UserKNN fine-grid.
