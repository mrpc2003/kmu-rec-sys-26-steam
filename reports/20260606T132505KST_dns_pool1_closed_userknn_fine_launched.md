# 2026-06-06 13:25 KST — DNS pool=1 closed; UserKNN fine-grid follow-up launched

## Trigger

Received completion notification for `val_random_uniform_seed7` DNS pool=1 seed123:

- process: `proc_73617e53a7c7`
- split: `val_random_uniform_seed7`
- seed: 123
- row accuracy: `0.75865`
- vs emb128 4-seed reference `0.76505`: `-0.00640`

## Additional reconciliation

The remaining DNS split7 jobs were checked and completed/aggregated:

| split | model seed | row accuracy | vs ref |
|---|---:|---:|---:|
| `val_random_uniform_seed7` | 42 | 0.76045 | -0.00460 |
| `val_random_uniform_seed7` | 123 | 0.75865 | -0.00640 |
| `val_random_uniform_seed7` | 2024 | 0.76095 | -0.00410 |
| `val_random_uniform_seed7` | 7 | 0.75835 | -0.00670 |

Split7 aggregate:

| aggregate score | row accuracy | vs ref |
|---|---:|---:|
| `dns_pool1_neg_mean_rank` | 0.76095 | -0.00410 |
| `dns_pool1_mean_raw` | 0.76035 | -0.00470 |
| `dns_pool1_mean_z` | 0.76005 | -0.00500 |

## Three-uniform-split panel

Artifact:

`artifacts/dns_pool1_multisplit/three_uniform_panel/three_uniform_panel_summary.json`

| aggregate score | mean vs ref | min vs ref | positive splits | verdict |
|---|---:|---:|---:|---|
| `dns_pool1_neg_mean_rank` | -0.00270 | -0.00430 | 1/3 | reject |
| `dns_pool1_mean_z` | -0.00280 | -0.00500 | 1/3 | reject |
| `dns_pool1_mean_raw` | -0.00320 | -0.00580 | 1/3 | reject |

Final DNS pool=1 verdict: **REJECT — original split positive was split-specific noise**.

## Next no-submit axis launched

Because the user requested repeat-until-candidate behavior, a follow-up validation-only UserKNN gated residual fine-grid was launched immediately instead of idling.

- process: `proc_d98ef5d36b4a`
- command family: `scripts/userknn_gated_residual_probe.py`
- weights: `0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3`
- bands: `1,2,3,4,5`
- report md: `reports/20260606T132450KST_userknn_gated_residual_fine.md`
- report json: `reports/20260606T132450KST_userknn_gated_residual_fine.json`
- artifact dir: `artifacts/userknn_gated_residual_fine_20260606T132450KST`

Rationale: previous user-gated UserKNN probe had weak 3/3 positive variants around mean delta +0.0008~+0.0009 but no strict pass; this fine-grid tests whether a nearby bounded setting clears the stricter gate. No candidate/submission CSV is written.

## Safety

- Kaggle submit executed: false
- Submission CSV created: false
- Hidden labels/external scraping: false
- Aggressive runner guard/quarantine unchanged.
