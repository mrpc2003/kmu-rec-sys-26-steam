# boundary v1 phase-1 gate policy

- validation_only: true
- no_kaggle_submit: true
- public_lb_feedback_used: true
- candidate_csv_written: false

## 보정 이유

phase-0에서 기존 boundary 계열은 validation flip precision 0.505~0.512였지만 public implied precision은 0.46~0.49로 내려갔다.
그래서 phase-1 제출 검토 gate는 이론상 필요한 public precision보다 높게 잡는다.

## 운영 gate

| total diff | public changed est. | top2 이론 | top2 validation gate | top1 이론 | top1 validation gate |
|---:|---:|---:|---:|---:|---:|
| 50 | 25.0 | 불가능 | 불가능 | 불가능 | 불가능 |
| 100 | 50.0 | 0.770 | 0.820 | 불가능 | 불가능 |
| 150 | 75.0 | 0.680 | 0.720 | 불가능 | 불가능 |
| 300 | 150.0 | 0.590 | 0.650 | 0.823 | 0.850 |
| 500 | 250.0 | 0.554 | 0.604 | 0.694 | 0.740 |
| 850 | 425.0 | 0.532 | 0.582 | 0.614 | 0.740 |

## 제출 전 고정 조건

- scored boundary cross-fit 평가 전까지 candidate CSV 생성 금지.
- 300 diff band에서 2~3등권 net gain이 보이지 않으면 full-test candidate 생성 금지.
- 1등 도전은 300 diff band 0.85 근처 또는 500+ diff band 0.73~0.75 유지가 없으면 제출 목표로 보지 않는다.
- 기존 public-failed boundary rows와 overlap이 높으면 precision이 좋아 보여도 candidate로 보지 않는다.
