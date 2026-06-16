# KMURecSys26 Steam no-submit improvement-axis loop

- Timestamp: 20260607T090941KST
- Safety: validation-only; no Kaggle submit; no candidate/submission CSV; no hidden/private labels; no external Steam scraping.
- Verdict: `WEAK_SIGNAL`

## Axis decision

launched_fresh_bounded_validation_only_otto_source_separated_covisit_smoke

The launched smoke is a bounded OTTO-style source-separated co-visitation residual gate. It differs from prior plain/time-decay ItemKNN/BM25 by keeping separate co-play, ordered transition, last-K, and hours-weighted source scores, then testing small residual weights against the emb128 4-seed LightGCN reference across three uniform validation splits.

## Top aggregate metric

- Variant: `base_plus_score_coplay_top5_mean_w0.2`
- mean Δ vs base: +0.0012335800
- min split Δ: +0.0005001000
- positive splits: 3/3
- fixes/breaks: 699/625
- pooled exact p: 0.044791873285533086
- split deltas: `{'val_random_uniform_seed42': 0.0005001000200040018, 'val_random_uniform_seed7': 0.001300260052010449, 'val_random_uniform_seed123': 0.001900380076015229}`

## Closed/rejected axes checked

- UserKNN gated residual fine-grid: missing expected reports; stalled warning-dominated log; not relaunched
- jackknife uncertainty boundary expanded: missing expected reports; 12-line log stopped mid split
- jackknife uncertainty boundary smoke: WEAK_SIGNAL mean +0.00036674, min -0.00120024, 2/3 positive, p 0.338815
- boundary/frontier/rankblend/TAG-CF: public-negative/quarantined family conflicts in aggressive_quota_runner_state.json
- plain/time-decay ItemKNN/BM25: already tested in paper_guided_next_steps and Stage3 blend; this probe keeps separate OTTO-style source features and residual-gates only

## Produced artifacts

- report_json: `reports/20260607T090941KST_opencode_improvement_axis_loop.json`
- report_md: `reports/20260607T090941KST_opencode_improvement_axis_loop.md`
- artifact_dir: `artifacts/opencode_axis_loop_20260607T090941KST`
- split_score_files: `['/opt/data/kaggle/kmu-rec-sys-26-steam/artifacts/opencode_axis_loop_20260607T090941KST/otto_source_covisit/val_random_uniform_seed42/validation_otto_source_scores.csv', '/opt/data/kaggle/kmu-rec-sys-26-steam/artifacts/opencode_axis_loop_20260607T090941KST/otto_source_covisit/val_random_uniform_seed7/validation_otto_source_scores.csv', '/opt/data/kaggle/kmu-rec-sys-26-steam/artifacts/opencode_axis_loop_20260607T090941KST/otto_source_covisit/val_random_uniform_seed123/validation_otto_source_scores.csv']`

## Strict gate status

Strict pass count: `0`
