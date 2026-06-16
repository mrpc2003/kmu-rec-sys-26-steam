# OTTO-style co-visitation validation run — Hermes summary

- Timestamp: 20260607T093800KST
- Project: KMURecSys26 Steam
- Scope: DACON/Kaggle methodology scan에서 도출한 OTTO식 source-separated co-visitation/transition residual axis를 validation-only로 실행·검증.

## Safety

- Kaggle submit: not executed
- `submissions/` write: none
- full-test candidate/submission CSV: none
- hidden/private labels: not used
- external Steam scraping: not used
- git add/commit/push: not executed
- submit-capable watchdog `272808a2bcca`: remains paused
- no-submit OpenCode loop `4d627b59804f`: remains enabled/scheduled

Secret scan only matched literal safety flag names such as `credentials_or_tokens_printed`; no actual credential/token values were found.

## Artifacts

Primary OpenCode smoke:

- Script: `scripts/otto_source_covisit_smoke.py`
- Report JSON: `reports/20260607T090941KST_opencode_improvement_axis_loop.json`
- Report MD: `reports/20260607T090941KST_opencode_improvement_axis_loop.md`
- Validation artifacts: `artifacts/opencode_axis_loop_20260607T090941KST/otto_source_covisit/`

Hermes follow-up grid:

- Script: `scripts/otto_source_covisit_followup_grid.py`
- Report JSON: `reports/20260607T092546KST_otto_source_covisit_followup_grid.json`
- Report MD: `reports/20260607T092546KST_otto_source_covisit_followup_grid.md`

Targeted diagnostic fine grid:

- Report JSON: `reports/20260607T093002KST_otto_source_covisit_targeted_fine_grid.json`
- Report MD: `reports/20260607T093002KST_otto_source_covisit_targeted_fine_grid.md`

LOOSO confirmation:

- Report JSON: `reports/20260607T093521KST_otto_source_covisit_looso_confirmation.json`
- Report MD: `reports/20260607T093521KST_otto_source_covisit_looso_confirmation.md`

## Results

### Initial 3-split smoke

Top variant:

- `base_plus_score_coplay_top5_mean_w0.2`
- mean Δ vs base: `+0.0012335800`
- min split Δ: `+0.0005001000`
- positive splits: `3/3`
- fixes/breaks: `699/625`
- pooled exact p: `0.0447918733`
- verdict: `WEAK_SIGNAL`

Reason: it passes sign/fixes/p-value, but misses strict mean Δ `+0.0015`.

### Follow-up grid

Top variant:

- `base_plus_score_coplay_top5_mean_w0.12_plus_score_last5_forward_w0.03`
- mean Δ vs base: `+0.0014002801`
- min split Δ: `+0.0005001000`
- positive splits: `3/3`
- fixes/breaks: `509/425`
- pooled exact p: `0.0065798594`
- verdict: `WEAK_SIGNAL`

Reason: still below mean Δ `+0.0015`.

### Targeted fine grid

Top variant:

- `base_plus_score_coplay_top5_mean_w0.090_plus_score_reverse_recent_w0.040`
- mean Δ vs base: `+0.0017670201`
- min split Δ: `+0.0016003201`
- positive splits: `3/3`
- fixes/breaks: `450/344`
- pooled exact p: `0.0001904323`
- targeted strict rows: `36`
- verdict: `STRICT_PASS_DIAGNOSTIC_NEEDS_INDEPENDENT_CONFIRMATION`

Interpretation: strict criteria pass on the same three splits after targeted weight search. Because the weights were selected on the same validation panel, this is a diagnostic strict signal, not a submission candidate.

### Leave-one-split-out confirmation

Protocol: choose weights on 2 splits and evaluate on the held-out split.

- holdout mean Δ: `+0.0012669201`
- holdout min Δ: `+0.0005001000`
- holdout positive splits: `3/3`
- verdict: `LOOSO_WEAK_POSITIVE`

Holdout details:

- holdout `val_random_uniform_seed42`: chosen `base_plus_score_coplay_top5_mean_w0.100_plus_score_reverse_recent_w0.070`, holdout Δ `+0.0005001000`
- holdout `val_random_uniform_seed7`: chosen `base_plus_score_coplay_top5_mean_w0.095_plus_score_reverse_recent_w0.035`, holdout Δ `+0.0015003001`
- holdout `val_random_uniform_seed123`: chosen `base_plus_score_coplay_top5_mean_w0.090_plus_score_reverse_recent_w0.040`, holdout Δ `+0.0018003601`

Interpretation: signal remains sign-stable under LOOSO but drops below strict mean Δ. This supports the axis as real/positive, but not yet safe enough for candidate materialization.

## Decision

- Do not create a submission/candidate CSV.
- Do not resume submit-capable watchdog.
- Treat OTTO source co-visitation as the first genuinely positive external-methodology axis in this round, but keep it in `needs independent confirmation` state.

Recommended next step:

1. Freeze the interpretable family `coplay_top5_mean + reverse_recent`.
2. Confirm on a fresh independent uniform split if base LightGCN scores are available or can be generated validation-only.
3. If fresh-split mean Δ remains `>= +0.0015` with positive min split, only then consider preflight for candidate materialization and ask user approval.
