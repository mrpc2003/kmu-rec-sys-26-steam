# Last-slot sparse agreement probe

No Kaggle submission is performed. No candidate CSV is written.

- verdict: **REJECT**
- strict pass count: `0`
- manual-risk count: `0`

Strict threshold: mean Δ >= +0.0020, min Δ >= +0.0008, 3/3 positive, pooled p < 0.01, fixes-breaks >= 120.
Manual-risk threshold: mean Δ >= +0.0015, min Δ >= +0.0005, 3/3 positive, pooled p < 0.03.

| rank | variant | mean Δ | min Δ | pos | fixes | breaks | p | gate users |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `semantic_resid3__vote_zmean__mv3__B1__all_users__w0.05` | -0.000033 | -0.001400 | 2/3 | 528 | 530 | 0.9755 | 1958 |
| 2 | `semantic_resid3__vote__mv3__B1__all_users__w0.05` | -0.000033 | -0.001500 | 2/3 | 518 | 520 | 0.9752 | 1958 |
| 3 | `semantic_resid3__vote_zmean__mv3__B1__train_deg_high_q0.4__w0.05` | -0.000100 | -0.001100 | 1/3 | 436 | 442 | 0.866 | 1270 |
| 4 | `semantic_resid3__vote__mv3__B1__train_deg_high_q0.4__w0.05` | -0.000100 | -0.001200 | 1/3 | 426 | 432 | 0.8645 | 1270 |
| 5 | `semantic_resid3__vote_zmean__mv3__B1__train_deg_high_q0.2__w0.05` | -0.000100 | -0.001200 | 1/3 | 482 | 488 | 0.8725 | 1617 |
| 6 | `semantic_resid3__vote__mv3__B1__train_deg_high_q0.2__w0.05` | -0.000100 | -0.001300 | 1/3 | 472 | 478 | 0.8711 | 1617 |
| 7 | `semantic_resid3__vote_zmean__mv3__B1__n_high_q0.8__w0.05` | -0.000167 | -0.000900 | 1/3 | 361 | 371 | 0.7394 | 675 |
| 8 | `semantic_resid3__vote_zmean__mv2__B1__train_deg_high_q0.4__w0.05` | -0.000167 | -0.000900 | 1/3 | 517 | 527 | 0.7806 | 1742 |
| 9 | `semantic_resid3__vote__mv3__B1__n_high_q0.8__w0.05` | -0.000167 | -0.001000 | 1/3 | 351 | 361 | 0.7359 | 675 |
| 10 | `semraw2_readme2__vote__mv2__B2__n_high_q0.8__w0.05` | -0.000167 | -0.001700 | 2/3 | 528 | 538 | 0.7828 | 1217 |
| 11 | `semraw2_readme2__vote__mv2__B2__train_deg_high_q0.2__w0.05` | -0.000167 | -0.001800 | 2/3 | 742 | 752 | 0.8159 | 2860 |
| 12 | `semantic_resid3__vote__mv2__B1__train_deg_high_q0.4__w0.05` | -0.000200 | -0.001000 | 1/3 | 504 | 516 | 0.7305 | 1742 |
| 13 | `semantic_resid3__vote_zmean__mv2__B1__train_deg_high_q0.2__w0.05` | -0.000233 | -0.001100 | 1/3 | 573 | 587 | 0.7027 | 2242 |
| 14 | `semantic_resid3__vote__mv3__B2__all_users__w0.05` | -0.000233 | -0.001600 | 1/3 | 520 | 534 | 0.6889 | 2507 |
| 15 | `semraw2_readme2__vote__mv2__B1__train_deg_high_q0.2__w0.05` | -0.000233 | -0.002100 | 1/3 | 733 | 747 | 0.7354 | 2623 |
| 16 | `semraw2_readme2__vote__mv2__B1__n_high_q0.8__w0.05` | -0.000267 | -0.002000 | 2/3 | 517 | 533 | 0.6434 | 1068 |
| 17 | `semantic_resid3__vote__mv2__B1__train_deg_high_q0.2__w0.05` | -0.000267 | -0.001200 | 1/3 | 560 | 576 | 0.6563 | 2242 |
| 18 | `semantic_resid3__vote_zmean__mv3__B2__all_users__w0.05` | -0.000267 | -0.001500 | 1/3 | 532 | 548 | 0.6481 | 2507 |
| 19 | `semraw2_readme2__vote__mv2__B2__train_deg_high_q0.4__w0.05` | -0.000267 | -0.001800 | 1/3 | 662 | 678 | 0.682 | 2237 |
| 20 | `semantic_resid3__vote_zmean__mv2__B1__n_high_q0.8__w0.05` | -0.000300 | -0.000900 | 1/3 | 428 | 446 | 0.5653 | 895 |
| 21 | `semantic_resid3__vote__mv3__B2__train_deg_high_q0.4__w0.05` | -0.000300 | -0.001300 | 1/3 | 427 | 445 | 0.5648 | 1665 |
| 22 | `semantic_resid3__vote__mv3__B2__train_deg_high_q0.2__w0.05` | -0.000300 | -0.001400 | 1/3 | 473 | 491 | 0.584 | 2090 |
| 23 | `semantic_resid3__vote__mv2__B1__n_high_q0.8__w0.05` | -0.000333 | -0.001000 | 1/3 | 414 | 434 | 0.5141 | 895 |
| 24 | `semraw2_readme2__vote__mv3__B1__bd_low_q0.4__w0.05` | -0.000333 | -0.001100 | 1/3 | 202 | 222 | 0.3562 | 175 |
| 25 | `semraw2_readme2__vote_zmean__mv3__B1__bd_low_q0.4__w0.05` | -0.000333 | -0.001100 | 1/3 | 202 | 222 | 0.3562 | 175 |
| 26 | `semraw2_readme2__vote__mv3__B1__bd_low_q0.4__w0.1` | -0.000333 | -0.001100 | 1/3 | 202 | 222 | 0.3562 | 175 |
| 27 | `semraw2_readme2__vote_zmean__mv3__B1__bd_low_q0.4__w0.1` | -0.000333 | -0.001100 | 1/3 | 202 | 222 | 0.3562 | 175 |
| 28 | `semraw2_readme2__vote__mv3__B2__bd_low_q0.4__w0.05` | -0.000333 | -0.001100 | 1/3 | 202 | 222 | 0.3562 | 175 |
| 29 | `semraw2_readme2__vote_zmean__mv3__B2__bd_low_q0.4__w0.05` | -0.000333 | -0.001100 | 1/3 | 202 | 222 | 0.3562 | 175 |
| 30 | `semraw2_readme2__vote__mv3__B2__bd_low_q0.4__w0.1` | -0.000333 | -0.001100 | 1/3 | 202 | 222 | 0.3562 | 175 |