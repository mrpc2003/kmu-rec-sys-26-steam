# Boundary pairwise candidate factory

- safety: no Kaggle submit; no hidden labels; no Steam scraping; public test score files only for materialization.
- base: `rank_blend_emb128_emb192_public_best_style`

## Top validation variants

| rank | variant | mean Δ vs rankblend | pos splits | fixes/breaks | pooled p | changed | manual-risk |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `boundary_scoreblend_z128_z192_z64_w-0.695` | +0.001034 | 2/3 | 782/720 | 0.1155 | 1502 | True |
| 2 | `boundary_scoreblend_z128_z192_z64_w-0.69` | +0.001000 | 3/3 | 776/716 | 0.1266 | 1492 | True |
| 3 | `boundary_scoreblend_z128_z192_z64_w-0.685` | +0.000967 | 2/3 | 772/714 | 0.1392 | 1486 | True |
| 4 | `boundary_scoreblend_z128_z192_z64_w-0.7` | +0.000867 | 2/3 | 788/736 | 0.1914 | 1524 | True |
| 5 | `boundary_scoreblend_z128_z192_z64_w-0.75` | +0.000633 | 2/3 | 841/803 | 0.3615 | 1644 | True |
| 6 | `boundary_scoreblend_z128_z192_z64_w2` | +0.000433 | 2/3 | 849/823 | 0.541 | 1672 | True |
| 7 | `boundary_scoreblend_z128_z192_z64_w3` | +0.000033 | 2/3 | 985/983 | 0.982 | 1968 | True |
| 8 | `boundary_scoreblend_z128_z192_z64_w-0.6` | +0.001367 | 3/3 | 707/625 | 0.02642 | 1332 | True |
| 9 | `boundary_scoreblend_z128_z192_z64_w0.2` | +0.001367 | 3/3 | 489/407 | 0.006777 | 896 | True |
| 10 | `boundary_scoreblend_z128_z192_z64_w-0.55` | +0.001334 | 3/3 | 680/600 | 0.0272 | 1280 | True |

## Materialized

- file: `submissions/candidate_autorun_boundary_factory_fine_latest.csv`
- sha256: `5df98f28490873916f373bf504b773abfdb19d5db98828b4d27c92c897bfdea3`
- preflight: `{'rows': 19998, 'expected_rows': 19998, 'columns': ['ID', 'Played'], 'id_unique': True, 'id_contiguous': True, 'labels_binary': True, 'label_1': 9999, 'label_0': 9999, 'bad_users_tophalf': 0}`
