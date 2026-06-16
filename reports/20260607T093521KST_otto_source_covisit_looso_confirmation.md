# OTTO source co-visitation LOOSO confirmation

- Timestamp: 20260607T093521KST
- Safety: validation-only; no candidate/submission CSV; no Kaggle submit.
- Protocol: choose weights on 2 splits, evaluate on held-out split.
- Verdict: `LOOSO_WEAK_POSITIVE`
- Holdout mean Δ: +0.0012669201
- Holdout min Δ: +0.0005001000
- Holdout positive splits: 3/3

## Holdout rows

- holdout `val_random_uniform_seed42`: chosen `base_plus_score_coplay_top5_mean_w0.100_plus_score_reverse_recent_w0.070`, train_meanΔ=+0.0020004001, train_minΔ=+0.0018003601, holdoutΔ=+0.0005001000, chosen_all_deltas={'val_random_uniform_seed42': 0.0005001000200040018, 'val_random_uniform_seed7': 0.0018003600720144508, 'val_random_uniform_seed123': 0.0022004400880176744}
- holdout `val_random_uniform_seed7`: chosen `base_plus_score_coplay_top5_mean_w0.095_plus_score_reverse_recent_w0.035`, train_meanΔ=+0.0017503501, train_minΔ=+0.0016003201, holdoutΔ=+0.0015003001, chosen_all_deltas={'val_random_uniform_seed42': 0.0016003200640128945, 'val_random_uniform_seed7': 0.0015003000600120053, 'val_random_uniform_seed123': 0.001900380076015229}
- holdout `val_random_uniform_seed123`: chosen `base_plus_score_coplay_top5_mean_w0.090_plus_score_reverse_recent_w0.040`, train_meanΔ=+0.0017503501, train_minΔ=+0.0016003201, holdoutΔ=+0.0018003601, chosen_all_deltas={'val_random_uniform_seed42': 0.0016003200640128945, 'val_random_uniform_seed7': 0.001900380076015229, 'val_random_uniform_seed123': 0.0018003600720144508}
