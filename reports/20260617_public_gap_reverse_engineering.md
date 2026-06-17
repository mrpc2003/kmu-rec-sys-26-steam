# Validation ↔ Public Gap Reverse Engineering

Safety: validation-only analysis. No Kaggle submit, no hidden labels, no external Steam scraping, no candidate CSV.

## Target

- emb128 public: `0.77745`
- rankblend public: `0.77825`
- public delta(rankblend-emb128): `+0.00080`

## Fitted public-mimic weighting

- splits used: `20` uniform panel splits
- fitted beta: `-0.150`
- beta meaning: negative rows are weighted by item popularity after z-normalization; beta < 0 means tail/low-pop negatives matter more.

| metric | unweighted uniform | public-mimic weighted | public target |
|---|---:|---:|---:|
| emb128 mean acc | 0.760717 | 0.775605 | 0.77745 |
| rankblend mean acc | 0.760972 | 0.775844 | 0.77825 |
| rankblend Δ vs emb128 | +0.000255 | +0.000239 | +0.00080 |

## Top beta rows

| beta | emb128 mean | rankblend mean | rankblend Δ | abs target err | Δ err |
|---:|---:|---:|---:|---:|---:|
| -0.150 | 0.775605 | 0.775844 | +0.000239 | 0.001845 | 0.000561 |
| -0.200 | 0.780071 | 0.780302 | +0.000231 | 0.002621 | 0.000569 |
| -0.100 | 0.770894 | 0.771140 | +0.000246 | 0.006556 | 0.000554 |
| -0.250 | 0.784300 | 0.784521 | +0.000221 | 0.006850 | 0.000579 |
| -0.300 | 0.788300 | 0.788510 | +0.000210 | 0.010850 | 0.000590 |
| -0.050 | 0.765933 | 0.766185 | +0.000252 | 0.011517 | 0.000548 |
| -0.350 | 0.792079 | 0.792277 | +0.000198 | 0.014629 | 0.000602 |
| +0.000 | 0.760717 | 0.760972 | +0.000255 | 0.016733 | 0.000545 |
| -0.400 | 0.795646 | 0.795831 | +0.000185 | 0.018196 | 0.000615 |
| -0.450 | 0.799010 | 0.799182 | +0.000172 | 0.021560 | 0.000628 |

## Feature shift at fitted beta

- negative_item_pop_mean_unweighted: `65.9574`
- negative_item_pop_mean_weighted: `56.2421`
- negative_log_item_pop_mean_unweighted: `3.7733`
- negative_log_item_pop_mean_weighted: `3.6538`

## Interpretation

- Public absolute score is matched only after making uniform validation negatives easier/tail-heavier than the raw uniform panel.
- This supports the hypothesis that the old uniform split got the direction roughly right but overstated hard/head-item negatives.
- Even after calibration, rankblend's public-sized edge remains tiny. Treat this surrogate as a ranking/triage tool, not as evidence for blind submissions.
