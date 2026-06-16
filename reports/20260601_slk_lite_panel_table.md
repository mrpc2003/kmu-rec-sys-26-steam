| split/epoch | control acc | variant acc | Δ | fixes | breaks | exact p | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| aggregate e1 | 0.761052 | 0.752017 | -0.009035 | 1603 | 2145 | 8.59e-19 | REJECT |
| aggregate e2 | 0.760919 | 0.739248 | -0.021671 | 2288 | 3588 | 6.24e-65 | REJECT |
| val_random_uniform_seed42 e1 | 0.762953 | 0.753851 | -0.009102 | 519 | 701 | 2.2e-07 | REJECT |
| val_random_uniform_seed7 e1 | 0.759652 | 0.748350 | -0.011302 | 512 | 738 | 1.97e-10 | REJECT |
| val_random_uniform_seed123 e1 | 0.760552 | 0.753851 | -0.006701 | 572 | 706 | 0.000199 | REJECT |
