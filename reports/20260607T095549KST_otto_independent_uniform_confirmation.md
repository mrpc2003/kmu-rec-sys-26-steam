# OTTO independent uniform confirmation

- Timestamp: 20260607T095549KST
- Safety: validation-only; no full-test pairs; no candidate/submission CSV; no Kaggle submit; no hidden/private labels; no external scraping.
- Fresh uniform split seeds: `[314, 2025, 2718]`
- Model seeds per split: `[42, 123, 2024, 7]`
- Verdict: `INDEPENDENT_DIAGNOSTIC_ONLY_POSITIVE_STRICT_FAIL`

## Strict pre-registered row

- Variant: `pre_registered_old_panel_top_coplay_top5_reverse_recent`
- terms: `[('score_coplay_top5_mean', 0.09), ('score_reverse_recent', 0.04)]`
- mean Δ vs base: +0.0006668000
- min split Δ: -0.0006001200
- positive splits: 2/3
- fixes/breaks: 424/384
- pooled exact p: 0.1700198674835568
- split deltas: `{'val_random_uniform_seed314': 0.0010002000400080036, 'val_random_uniform_seed2025': 0.0016003200640127835, 'val_random_uniform_seed2718': -0.000600120024004891}`

## Diagnostic rows

1. `looso_seed42_choice_coplay_top5_w0.100_reverse_recent_w0.070` meanΔ=+0.0013669401, minΔ=-0.0006001200, pos=2/3, fixes/breaks=548/466, p=0.010932564012406056, deltas={'val_random_uniform_seed314': 0.001900380076015229, 'val_random_uniform_seed2025': 0.0028005601120223433, 'val_random_uniform_seed2718': -0.000600120024004891}
2. `primary_smoke_coplay_top5_w0.200` meanΔ=+0.0012669201, minΔ=-0.0002000400, pos=2/3, fixes/breaks=707/631, p=0.040287976192964085, deltas={'val_random_uniform_seed314': 0.0011002200440087817, 'val_random_uniform_seed2025': 0.0029005801160232325, 'val_random_uniform_seed2718': -0.00020004000800166732}
3. `followup_top_coplay_top5_w0.120_last5_forward_w0.030` meanΔ=+0.0012002400, minΔ=-0.0004000800, pos=2/3, fixes/breaks=489/417, p=0.018284334740730986, deltas={'val_random_uniform_seed314': 0.0017003400680136727, 'val_random_uniform_seed2025': 0.0023004600920183416, 'val_random_uniform_seed2718': -0.0004000800160032236}
4. `looso_seed7_choice_coplay_top5_w0.095_reverse_recent_w0.035` meanΔ=+0.0009668600, minΔ=-0.0003000600, pos=2/3, fixes/breaks=438/380, p=0.046199890172053525, deltas={'val_random_uniform_seed314': 0.0011002200440087817, 'val_random_uniform_seed2025': 0.0021004200840167853, 'val_random_uniform_seed2718': -0.0003000600120024455}

## Artifacts

- run_root: `artifacts/otto_independent_uniform_20260607T095549KST`
- validation_root: `artifacts/otto_independent_uniform_20260607T095549KST/validation`
- base_root: `artifacts/otto_independent_uniform_20260607T095549KST/lightgcn_emb128`
- otto_root: `artifacts/otto_independent_uniform_20260607T095549KST/otto_source_covisit`
- log_root: `logs/otto_independent_uniform_20260607T095549KST`
- out_json: `reports/20260607T095549KST_otto_independent_uniform_confirmation.json`
- out_md: `reports/20260607T095549KST_otto_independent_uniform_confirmation.md`
