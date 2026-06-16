# Boundary pairwise candidate factory

- safety: no Kaggle submit; no hidden labels; no Steam scraping; public test score files only for materialization.
- base: `rank_blend_emb128_emb192_public_best_style`

## Top validation variants

| rank | variant | mean Δ vs rankblend | pos splits | fixes/breaks | pooled p | changed | manual-risk |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `boundary_scoreblend_z128_z192_z64_w-0.75` | +0.000633 | 2/3 | 841/803 | 0.3615 | 1644 | True |
| 2 | `boundary_scoreblend_z128_z192_z64_w2` | +0.000433 | 2/3 | 849/823 | 0.541 | 1672 | True |
| 3 | `boundary_scoreblend_z128_z192_z64_w3` | +0.000033 | 2/3 | 985/983 | 0.982 | 1968 | True |
| 4 | `boundary_scoreblend_z128_z192_z64_w0.2` | +0.001367 | 3/3 | 489/407 | 0.006777 | 896 | True |
| 5 | `boundary_scoreblend_z128_z192_z64_w0.1` | +0.001334 | 2/3 | 472/392 | 0.007161 | 864 | True |
| 6 | `boundary_scoreblend_z128_z192_z64_w0.3` | +0.001300 | 2/3 | 501/423 | 0.01127 | 924 | True |
| 7 | `boundary_scoreblend_z128_z192_z64_w-0.5` | +0.000934 | 2/3 | 632/576 | 0.1135 | 1208 | True |
| 8 | `boundary_pairwise_vote2_B4_tau0_guard0_cap1` | +0.000834 | 3/3 | 484/434 | 0.1058 | 918 | True |
| 9 | `boundary_pairwise_vote2_B8_tau0_guard0_cap1` | +0.000834 | 3/3 | 484/434 | 0.1058 | 918 | True |
| 10 | `boundary_pairwise_vote2_B16_tau0_guard0_cap1` | +0.000834 | 3/3 | 484/434 | 0.1058 | 918 | True |

## Materialized

- file: `submissions/candidate_autorun_boundary_factory_latest.csv`
- sha256: `8966b80f4a5c807f5747d9766e9b004624c9671a8adb2e03e30f38d7733a9822`
- preflight: `{'rows': 19998, 'expected_rows': 19998, 'columns': ['ID', 'Played'], 'id_unique': True, 'id_contiguous': True, 'labels_binary': True, 'label_1': 9999, 'label_0': 9999, 'bad_users_tophalf': 0}`
