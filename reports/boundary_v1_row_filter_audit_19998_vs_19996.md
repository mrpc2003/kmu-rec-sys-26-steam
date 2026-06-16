# boundary v1 row-filter audit — 19,998 vs 19,996

- validation_only: true
- no_kaggle_submit: true
- candidate_csv_written: false
- public_lb_feedback_used: false

## 판정

19,998 rows / 4,737 users와 19,996 rows / 4,736 users 차이는 artifact mismatch가 아니라, panel20 split builder의 의도된 사용자 필터링으로 설명된다.
문제 사용자는 `u57101927` 한 명이고, raw `pairs.csv`에는 2개 row가 있다. 해당 사용자는 train interaction이 1개라 holdout을 만들면 fold-train에 남길 interaction이 없어져 panel에서 제외된다.

## 숫자 확인

| 항목 | rows | users | 비고 |
|---|---:|---:|---|
| raw pairs.csv | 19998 | 4737 | all candidate counts even = True |
| panel20 candidates | [19996] | [4736] | heldout positives = [9998] |

## 제외된 사용자

| userID | raw pair rows | raw pair IDs | raw games |
|---|---:|---|---|
| `u57101927` | 2 | [2658, 11536] | ['g94132851', 'g91440237'] |

## phase-1 주의점

scored boundary evaluation에서는 이 panel20 필터를 그대로 쓰거나, 다른 필터를 쓰는 경우 denominator와 user boundary를 별도로 기록해야 한다.
diff-band precision curve를 public rows로 환산할 때도 panel 기준 denominator와 raw pairs 기준 denominator를 섞지 않는다.
