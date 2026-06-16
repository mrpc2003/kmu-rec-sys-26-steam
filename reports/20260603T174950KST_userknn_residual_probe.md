# UserKNN residual probe

No Kaggle submission is performed by this report.

## Base rankblend

- val_random_uniform_seed42: acc=0.766753
- val_random_uniform_seed7: acc=0.761252
- val_random_uniform_seed123: acc=0.760452

## Top aggregate variants

| rank | variant | mean Δ | min Δ | pos | fixes | breaks | p |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `zbase_plus_score_userknn_max_w0.25_band2` | +0.000533 | -0.000600 | 2/3 | 568 | 536 | 0.3508 |
| 2 | `zbase_plus_score_userknn_max_w0.25` | +0.000433 | -0.000600 | 2/3 | 568 | 542 | 0.453 |
| 3 | `zbase_plus_score_userknn_max_w0.25_band1` | +0.000200 | -0.001500 | 2/3 | 535 | 523 | 0.7352 |
| 4 | `zbase_plus_score_userknn_popnorm_w0.25_band2` | +0.000067 | -0.001500 | 2/3 | 474 | 470 | 0.9222 |
| 5 | `zbase_plus_score_userknn_popnorm_w0.25` | +0.000067 | -0.001500 | 2/3 | 475 | 471 | 0.9223 |
| 6 | `zbase_plus_score_userknn_top10_w0.25_band2` | -0.000033 | -0.002000 | 2/3 | 471 | 473 | 0.974 |
| 7 | `zbase_plus_score_userknn_top10_w0.25_band1` | -0.000033 | -0.002200 | 2/3 | 463 | 465 | 0.9738 |
| 8 | `zbase_plus_score_userknn_popnorm_w0.25_band1` | -0.000067 | -0.002000 | 2/3 | 467 | 471 | 0.922 |
| 9 | `zbase_plus_score_userknn_top10_w0.25` | -0.000100 | -0.002000 | 2/3 | 470 | 476 | 0.8709 |
| 10 | `zbase_plus_score_userknn_sum_w0.25_band2` | -0.000267 | -0.001200 | 1/3 | 457 | 473 | 0.6228 |
| 11 | `zbase_plus_score_userknn_sum_w0.25` | -0.000300 | -0.001200 | 1/3 | 456 | 474 | 0.5772 |
| 12 | `zbase_plus_score_userknn_sum_w0.25_band1` | -0.000333 | -0.001600 | 2/3 | 454 | 474 | 0.5328 |
| 13 | `zbase_plus_score_userknn_top10_w0.5_band1` | -0.000800 | -0.002000 | 1/3 | 732 | 780 | 0.2268 |
| 14 | `zbase_plus_score_userknn_popnorm_w0.5_band1` | -0.000834 | -0.002701 | 1/3 | 648 | 698 | 0.1817 |
| 15 | `zbase_plus_score_userknn_popnorm_w0.5_band2` | -0.000867 | -0.002501 | 1/3 | 677 | 729 | 0.1738 |
| 16 | `zbase_plus_score_userknn_popnorm_w0.5` | -0.000900 | -0.002501 | 1/3 | 687 | 741 | 0.1607 |
| 17 | `zbase_plus_score_userknn_top10_w0.5_band2` | -0.001000 | -0.003001 | 1/3 | 765 | 825 | 0.1389 |
| 18 | `zbase_plus_score_userknn_top10_w0.5` | -0.001034 | -0.003001 | 1/3 | 775 | 837 | 0.1287 |
| 19 | `zbase_plus_score_userknn_sum_w0.5_band1` | -0.001400 | -0.002801 | 0/3 | 645 | 729 | 0.02511 |
| 20 | `zbase_plus_score_userknn_sum_w0.5_band2` | -0.001600 | -0.002901 | 0/3 | 668 | 764 | 0.01203 |
| 21 | `zbase_plus_score_userknn_sum_w-0.25_band1` | -0.001767 | -0.002000 | 0/3 | 411 | 517 | 0.0005602 |
| 22 | `zbase_plus_score_userknn_sum_w0.5` | -0.001867 | -0.003001 | 0/3 | 676 | 788 | 0.003705 |
| 23 | `zbase_plus_score_userknn_sum_w-0.25_band2` | -0.002100 | -0.002300 | 0/3 | 417 | 543 | 5.351e-05 |
| 24 | `zbase_plus_score_userknn_popnorm_w-0.25_band1` | -0.002100 | -0.002801 | 0/3 | 384 | 510 | 2.826e-05 |
| 25 | `zbase_plus_score_userknn_max_w0.5_band1` | -0.002134 | -0.003801 | 0/3 | 1065 | 1193 | 0.007512 |
| 26 | `zbase_plus_score_userknn_sum_w-0.25` | -0.002167 | -0.002400 | 0/3 | 419 | 549 | 3.295e-05 |
| 27 | `zbase_plus_score_userknn_popnorm_w-0.25_band2` | -0.002234 | -0.002701 | 0/3 | 394 | 528 | 1.147e-05 |
| 28 | `zbase_plus_score_userknn_popnorm_w-0.25` | -0.002234 | -0.002601 | 0/3 | 400 | 534 | 1.307e-05 |
| 29 | `zbase_plus_score_userknn_max_w0.5_band2` | -0.002267 | -0.004101 | 0/3 | 1206 | 1342 | 0.007473 |
| 30 | `zbase_plus_score_userknn_max_w0.5` | -0.002300 | -0.004201 | 0/3 | 1238 | 1376 | 0.00736 |