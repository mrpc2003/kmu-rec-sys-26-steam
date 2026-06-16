# Boundary pairwise candidate factory

- safety: no Kaggle submit; no hidden labels; no Steam scraping; public test score files only for materialization.
- base: `rank_blend_emb128_emb192_public_best_style`

## Top validation variants

| rank | variant | mean Δ vs rankblend | pos splits | fixes/breaks | pooled p | changed | manual-risk |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `frontier_z_w1920_w64-0.25` | +0.000267 | 2/3 | 770/754 | 0.7008 | 1524 | True |
| 2 | `frontier_z_w1921.25_w640.25` | +0.001700 | 3/3 | 476/374 | 0.0005243 | 850 | True |
| 3 | `frontier_z_w1921.5_w640.25` | +0.001567 | 3/3 | 506/412 | 0.002128 | 918 | True |
| 4 | `boundary_scoreblend_z128_z192_z64_w-0.6` | +0.001367 | 3/3 | 707/625 | 0.02642 | 1332 | True |
| 5 | `boundary_scoreblend_z128_z192_z64_w0.2` | +0.001367 | 3/3 | 489/407 | 0.006777 | 896 | True |
| 6 | `boundary_scoreblend_z128_z192_z64_w-0.55` | +0.001334 | 3/3 | 680/600 | 0.0272 | 1280 | True |
| 7 | `boundary_scoreblend_z128_z192_z64_w0.1` | +0.001334 | 2/3 | 472/392 | 0.007161 | 864 | True |
| 8 | `boundary_scoreblend_z128_z192_z64_w-0.625` | +0.001300 | 3/3 | 725/647 | 0.0376 | 1372 | True |
| 9 | `boundary_scoreblend_z128_z192_z64_w0.3` | +0.001300 | 2/3 | 501/423 | 0.01127 | 924 | True |
| 10 | `boundary_scoreblend_z128_z192_z64_w-0.65` | +0.001234 | 3/3 | 749/675 | 0.05301 | 1424 | True |

## Materialized

- file: `submissions/candidate_autorun_boundary_factory_frontier_latest.csv`
- sha256: `e2872806738cb511d6c45cd828508cdb2f31f9040a463a8091235271adc26b1d`
- preflight: `{'rows': 19998, 'expected_rows': 19998, 'columns': ['ID', 'Played'], 'id_unique': True, 'id_contiguous': True, 'labels_binary': True, 'label_1': 9999, 'label_0': 9999, 'bad_users_tophalf': 0}`
