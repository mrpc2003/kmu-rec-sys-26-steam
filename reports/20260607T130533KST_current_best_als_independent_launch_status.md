# Current-best ALS independent confirmation launch status

- Timestamp: `20260607T130533KST`
- Background session: `proc_57b038d788c6`
- Purpose: confirm the same-panel residual-atlas top row on a fresh independent uniform panel before any candidate escalation.
- Strict pre-registered row: `zrankblend_plus_score_als_f32_it30_alpha20_popa4_w+0.30_band2`
- Source fresh split panel: `artifacts/otto_independent_uniform_20260607T095549KST/validation/`
- Reused base: existing independent emb128 4-seed scores from `artifacts/otto_independent_uniform_20260607T095549KST/lightgcn_emb128/`
- New work in progress: emb192 L4 reg1e-3 4-seed scores for seeds 42/123/2024/7 on split seeds 314/2025/2718, plus validation-only ALS score generation.
- Driver log: `logs/20260607T130533KST_current_best_als_independent_driver.log`
- Expected final JSON: `reports/20260607T130533KST_current_best_als_independent_confirmation.json`
- Expected final Markdown: `reports/20260607T130533KST_current_best_als_independent_confirmation.md`

## Safety

- validation_only: `true`
- candidate_csv_written: `false`
- full_test_candidate_or_submission_csv_created: `false`
- kaggle_submit_executed: `false`
- hidden_labels_used: `false`
- private_answers_used: `false`
- external_steam_scraping_used: `false`
- git_stage_commit_push_executed: `false`
- no-submit OpenCode cron `4d627b59804f` is temporarily paused to avoid overlap.
- submit-capable watchdog `272808a2bcca` remains paused.

## Current observed progress

- 4 GPU/worker jobs started for `val_random_uniform_seed314` emb192 seeds 42/123/2024/7.
- Worker logs reached epoch 20/200, so the run is actively training rather than stalled.
