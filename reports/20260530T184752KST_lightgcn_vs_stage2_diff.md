# Post-submission diff: LightGCN vs Stage2 (full test pairs)

- Rows: 19,998
- Disagreement: 3,098 (15.49%)
- LightGCN promoted (s2=0→lg=1): 1,549
- LightGCN demoted  (s2=1→lg=0): 1,549

## Disagreement by user candidate-count

| bucket | users | total candidates | disagree | frac |
|---|---:|---:|---:|---:|
| 2 | 2141 | 4282 | 604 | 0.1411 |
| 3-4 | 1307 | 5228 | 812 | 0.1553 |
| 5-6 | 665 | 3990 | 648 | 0.1624 |
| 7-10 | 454 | 3948 | 602 | 0.1525 |
| 11+ | 170 | 2550 | 432 | 0.1694 |

## Disagreement by item popularity (quintile)

| pop_bin | mean_pop | n | disagree | frac |
|---:|---:|---:|---:|---:|
| 0 | 19 | 4188 | 541 | 0.1292 |
| 1 | 34 | 3880 | 707 | 0.1822 |
| 2 | 65 | 3961 | 946 | 0.2388 |
| 3 | 139 | 3983 | 656 | 0.1647 |
| 4 | 393 | 3986 | 248 | 0.0622 |

## Item popularity at flip points

| direction | n | mean item_pop | median item_pop |
|---|---:|---:|---:|
| promoted (s2=0→lg=1) | 1549 | 69.3 | 53 |
| demoted  (s2=1→lg=0) | 1549 | 99.2 | 60 |
| all rows | 19998 | 129.5 | 62 |