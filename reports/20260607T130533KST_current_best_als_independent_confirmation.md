# Current-best ALS residual independent confirmation

- Timestamp: `20260607T130533KST`
- Safety: validation-only; no full-test candidate CSV; no Kaggle submit; no hidden/private labels; no external scraping.
- Verdict: `INDEPENDENT_WEAK_POSITIVE_STRICT_FAIL`

## Strict pre-registered atlas row

- Variant: `pre_registered_atlas_top_als_f32_popa4_w0.30_band2`
- feature/weight/band: `score_als_f32_it30_alpha20_popa4` / `0.3` / `2`
- mean Δ vs current-best rankblend: +0.0008001600
- min split Δ: -0.0003000600
- positive splits: 2/3
- fixes/breaks: 504/456
- pooled exact p: 0.12924401684163647
- split deltas: `{'val_random_uniform_seed314': 0.0008001600320064473, 'val_random_uniform_seed2025': 0.001900380076015229, 'val_random_uniform_seed2718': -0.0003000600120024455}`

## Diagnostic variants

1. `atlas_als_f32_popa4_w0.20_band1` meanΔ=+0.0011335600, minΔ=+0.0004000800, pos=3/3, fixes/breaks=462/394, p=0.021965674090633346, deltas={'val_random_uniform_seed314': 0.0009001800360072254, 'val_random_uniform_seed2025': 0.0021004200840167853, 'val_random_uniform_seed2718': 0.0004000800160031126}
2. `pre_registered_atlas_top_als_f32_popa4_w0.30_band2` meanΔ=+0.0008001600, minΔ=-0.0003000600, pos=2/3, fixes/breaks=504/456, p=0.12924401684163647, deltas={'val_random_uniform_seed314': 0.0008001600320064473, 'val_random_uniform_seed2025': 0.001900380076015229, 'val_random_uniform_seed2718': -0.0003000600120024455}
3. `atlas_als_f32_popa4_w0.30_band1` meanΔ=+0.0008001600, minΔ=-0.0001000200, pos=2/3, fixes/breaks=504/456, p=0.12924401684163647, deltas={'val_random_uniform_seed314': 0.000600120024004891, 'val_random_uniform_seed2025': 0.001900380076015229, 'val_random_uniform_seed2718': -0.00010002000400088917}
4. `atlas_als_f32_popa4_w0.30_all` meanΔ=+0.0008001600, minΔ=-0.0003000600, pos=2/3, fixes/breaks=507/459, p=0.13044025564744519, deltas={'val_random_uniform_seed314': 0.0008001600320064473, 'val_random_uniform_seed2025': 0.001900380076015229, 'val_random_uniform_seed2718': -0.0003000600120024455}
5. `atlas_als_f32_popa4_w0.30_band3` meanΔ=+0.0008001600, minΔ=-0.0003000600, pos=2/3, fixes/breaks=506/458, p=0.13004180257850934, deltas={'val_random_uniform_seed314': 0.0008001600320064473, 'val_random_uniform_seed2025': 0.001900380076015229, 'val_random_uniform_seed2718': -0.0003000600120024455}

## Artifacts

- run_root: `artifacts/current_best_als_independent_20260607T130533KST`
- emb192_root: `artifacts/current_best_als_independent_20260607T130533KST/lightgcn_emb192`
- als_root: `artifacts/current_best_als_independent_20260607T130533KST/als_scores`
- log_root: `logs/current_best_als_independent_20260607T130533KST`
- out_json: `reports/20260607T130533KST_current_best_als_independent_confirmation.json`
- out_md: `reports/20260607T130533KST_current_best_als_independent_confirmation.md`

## Safety flags

- validation_only: `true`
- candidate_csv_written: `false`
- full_test_candidate_or_submission_csv_created: `false`
- kaggle_submit_executed: `false`
- hidden_labels_used: `false`
- private_answers_used: `false`
- external_steam_scraping_used: `false`
- credentials_or_tokens_printed: `false`
- git_stage_commit_push_executed: `false`
- recursive_cron_scheduled: `false`
