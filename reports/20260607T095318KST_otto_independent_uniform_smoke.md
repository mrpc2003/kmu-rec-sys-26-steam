# OTTO independent uniform confirmation

- Timestamp: 20260607T095318KST
- Safety: validation-only; no full-test pairs; no candidate/submission CSV; no Kaggle submit; no hidden/private labels; no external scraping.
- Fresh uniform split seeds: `[314]`
- Model seeds per split: `[42, 123, 2024, 7]`
- Verdict: `INDEPENDENT_STRICT_CONFIRMATION_PASS`

## Strict pre-registered row

- Variant: `pre_registered_old_panel_top_coplay_top5_reverse_recent`
- terms: `[('score_coplay_top5_mean', 0.09), ('score_reverse_recent', 0.04)]`
- mean Δ vs base: +0.0030006001
- min split Δ: +0.0030006001
- positive splits: 1/1
- fixes/breaks: 173/113
- pooled exact p: 0.0004647758366388276
- split deltas: `{'val_random_uniform_seed314': 0.0030006001200240107}`

## Diagnostic rows

1. `primary_smoke_coplay_top5_w0.200` meanΔ=+0.0046009202, minΔ=+0.0046009202, pos=1/1, fixes/breaks=272/180, p=1.7536175117908018e-05, deltas={'val_random_uniform_seed314': 0.004600920184036794}
2. `looso_seed42_choice_coplay_top5_w0.100_reverse_recent_w0.070` meanΔ=+0.0036007201, minΔ=+0.0036007201, pos=1/1, fixes/breaks=221/149, p=0.00021411670700935133, deltas={'val_random_uniform_seed314': 0.0036007201440287906}
3. `followup_top_coplay_top5_w0.120_last5_forward_w0.030` meanΔ=+0.0034006801, minΔ=+0.0034006801, pos=1/1, fixes/breaks=189/121, p=0.00013379189924597775, deltas={'val_random_uniform_seed314': 0.0034006801360272343}
4. `looso_seed7_choice_coplay_top5_w0.095_reverse_recent_w0.035` meanΔ=+0.0032006401, minΔ=+0.0032006401, pos=1/1, fixes/breaks=174/110, p=0.00017483265988526935, deltas={'val_random_uniform_seed314': 0.003200640128025678}

## Artifacts

- run_root: `artifacts/otto_independent_uniform_smoke_20260607T095318KST`
- validation_root: `artifacts/otto_independent_uniform_smoke_20260607T095318KST/validation`
- base_root: `artifacts/otto_independent_uniform_smoke_20260607T095318KST/lightgcn_emb128`
- otto_root: `artifacts/otto_independent_uniform_smoke_20260607T095318KST/otto_source_covisit`
- log_root: `logs/otto_independent_uniform_20260607T095318KST`
- out_json: `reports/20260607T095318KST_otto_independent_uniform_smoke.json`
- out_md: `reports/20260607T095318KST_otto_independent_uniform_smoke.md`
