I detect advisory-only analysis with strict no-write/no-submit constraints, so I’ll inspect only existing local reports/docs and return one concrete validation axis without modifying artifacts.
The latest reports show the low-cost CF/spectral branch is genuinely closed; I’m checking the prior axis ledger to avoid recommending a quarantined duplicate.
## Short verdict

`NO_SAFE_AXIS` locally, unless Steam item metadata is professor-approved and supplied/allowed. The best single next axis is therefore a **metadata side-information validation probe**, but it is `DIAGNOSTIC_ONLY` until approval.

## One next axis

**Professor-approved Steam item metadata side-information content model**: genres/tags/categories/release date, joined only to known `gameID`s, used to build user content profiles and validation-only scores.

## Why independent from closed axes

This is not another CF/spectral/LightGCN/loss/seed/blend retune. The closed axes all reuse interaction graph structure or LightGCN score residuals; metadata adds exogenous item attributes that could explain the remaining `0.78795` gap if the leader used allowed side info.

## Minimal commands or implementation sketch

No command should run until approval/metadata exists.

Small script design: `scripts/score_metadata_sideinfo_validation.py`

- Inputs: `--split-dir artifacts/validation_uniform_panel20_.../<split>` and `--metadata approved_metadata.csv`
- Build item feature vectors from tags/genres/categories/release buckets.
- Build each user profile by averaging metadata vectors of training positives only.
- Score validation candidate pairs by cosine/user-profile similarity.
- Evaluate solo and rank/z blends with existing emb128/emb192 validation scores.
- Run on all 20 uniform splits, not full test, no candidate CSV.

Example validation-only shape:

```bash
uv run python scripts/score_metadata_sideinfo_validation.py \
  --split-root artifacts/validation_uniform_panel20_20260612T214626KST \
  --metadata data/approved_steam_metadata.csv \
  --ref-score-root artifacts/scores/...emb128_validation_scores... \
  --out-root artifacts/scores/metadata_sideinfo_validation_only \
  --no-test --no-candidate-csv
```

## Gate criteria

Promote only if all hold across the 20 clean uniform splits:

- mean Δ vs `candidate_rank_blend_emb128_emb192` validation surrogate ≥ `+0.0020`
- min split Δ ≥ `0`
- at least 16/20 splits positive
- paired test significant, target `p < 0.05`
- metadata solo is not trivially weak: should approach at least high-0.75 uniform accuracy or provide low-correlation blend gain
- no use of unapproved scraping or hidden labels

## Risks / stop conditions

Stop immediately if metadata is not explicitly allowed. Without approved metadata, the repo evidence says no safe local axis remains: EASE/ItemKNN, GF-CF, ALS/WMF, UserKNN, boundary/frontier, TAG-CF, hours gates, seed/capacity expansion, and forced blends are already weak, redundant, or public-negative. In that case, professor-approved Steam metadata is the only remaining high-value path I would not classify as repeating a closed family.

## Safety flags

- validation_only: true
- candidate_csv_written: false
- kaggle_submit_executed: false
- hidden_labels_used: false
- external_scraping_used: false

OPENCODE_FOLLOWUP_AXIS_ADVISORY_DONE