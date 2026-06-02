# README rank-blend residual materialization

- base: `rank_blend_emb128_emb192_public_best_style`

## Top validation variants

| rank | variant | mean Δ vs rankblend | pos splits | fixes/breaks | pooled p | manual-risk |
|---:|---|---:|---:|---:|---:|---|
| 1 | `rankblend_z_plus_score_als_htr_f32_it30_alpha20_popa4_w0.2` | +0.001434 | 3/3 | 457/371 | 0.003114 | True |
| 2 | `rankblend_z_plus_score_als_f32_it30_alpha20_popa4_w0.2` | +0.001400 | 3/3 | 469/385 | 0.004481 | True |
| 3 | `rankblend_z_plus_score_als_htr_f32_it30_alpha20_popa4_w0.025` | +0.001300 | 3/3 | 419/341 | 0.005187 | True |
| 4 | `rankblend_z_plus_score_als_htr_f32_it30_alpha20_popa4_w0.05` | +0.001300 | 3/3 | 419/341 | 0.005187 | True |
| 5 | `rankblend_z_plus_score_als_htr_f32_it30_alpha20_popa4_w0.1` | +0.001300 | 3/3 | 422/344 | 0.005366 | True |
| 6 | `rankblend_z_plus_score_als_htr_f32_it30_alpha20_popa8_w0.025` | +0.001267 | 3/3 | 425/349 | 0.006983 | True |
| 7 | `rankblend_z_plus_score_als_htr_f32_it30_alpha20_popa8_w0.05` | +0.001267 | 3/3 | 425/349 | 0.006983 | True |
| 8 | `rankblend_z_plus_score_als_htr_f32_it30_alpha20_popa8_w0.1` | +0.001267 | 3/3 | 428/352 | 0.007205 | True |
| 9 | `rankblend_z_plus_score_als_f32_it30_alpha20_popa4_w0.025` | +0.001267 | 3/3 | 433/357 | 0.007583 | True |
| 10 | `rankblend_z_plus_score_als_f32_it30_alpha20_popa4_w0.05` | +0.001267 | 3/3 | 433/357 | 0.007583 | True |

## Materialized

- file: `/opt/data/kaggle/kmu-rec-sys-26-steam/submissions/candidate_autorun_rankblend_z_plus_score_als_htr_f32_it30_alpha20_popa4_w0.2.csv`
- sha256: `23f01b4e8354e0b4ee49f2cd266f10b558e1da9f85c1d6b7d357c0488a9cdd22`
- preflight: `{'rows': 19998, 'expected_rows': 19998, 'columns': ['ID', 'Played'], 'id_unique': True, 'id_contiguous': True, 'labels_binary': True, 'label_1': 9999, 'label_0': 9999, 'bad_users_tophalf': 0}`
