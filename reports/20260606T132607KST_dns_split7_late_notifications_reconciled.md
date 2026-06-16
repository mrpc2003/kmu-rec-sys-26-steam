# 2026-06-06 13:26 KST — DNS split7 late notifications reconciled

## Trigger

Received late completion notifications for the remaining DNS pool=1 jobs on `val_random_uniform_seed7`:

- `proc_17e267f8fc15` — seed2024, row accuracy `0.76095`, vs ref `-0.00410`
- `proc_31fb7df5f047` — seed42, row accuracy `0.76045`, vs ref `-0.00460`
- `proc_6fb208e8f301` — seed7, row accuracy `0.75835`, vs ref `-0.00670`

These were already included in the three-uniform-split DNS panel closure.

## Final DNS pool=1 panel

Artifact:

`artifacts/dns_pool1_multisplit/three_uniform_panel/three_uniform_panel_summary.json`

| aggregate score | mean accuracy | mean vs ref | min vs ref | positive splits | verdict |
|---|---:|---:|---:|---:|---|
| `dns_pool1_neg_mean_rank` | 0.762352 | -0.002698 | -0.004298 | 1/3 | reject |
| `dns_pool1_mean_z` | 0.762252 | -0.002798 | -0.004998 | 1/3 | reject |
| `dns_pool1_mean_raw` | 0.761852 | -0.003198 | -0.005798 | 1/3 | reject |

Final verdict remains: **DNS_POOL1_REJECT_SPLIT_SPECIFIC_NOISE**.

## Current active next axis

Validation-only UserKNN gated residual fine-grid is running:

- process: `proc_d98ef5d36b4a`
- report md: `reports/20260606T132450KST_userknn_gated_residual_fine.md`
- report json: `reports/20260606T132450KST_userknn_gated_residual_fine.json`
- artifact dir: `artifacts/userknn_gated_residual_fine_20260606T132450KST`

Runtime warnings observed from `z_within_user` divide with zero std are expected and guarded by `np.where(std > 1e-12, ..., 0.0)`; continue unless it exits nonzero.

## Safety

- Kaggle submit executed: false
- Submission CSV created: false
- Hidden labels/external scraping: false
- Aggressive runner guard/quarantine unchanged.
