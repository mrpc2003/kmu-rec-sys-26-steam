# boundary_specialist_v1_rowflip_constrained — phase-0 no-submit artifacts

- validation_only: true
- no_kaggle_submit: true
- public_lb_feedback_used: true
- candidate_csv_written: false
- new_full_test_scoring_performed: false
- existing_full_test_score_artifacts_read: true
- external_metadata_used: false

## 생성 파일

```text
reports/boundary_public_failure_calibration.csv
reports/boundary_public_like_split_panel.json
reports/boundary_v1_diffband_precision_curve.csv
```

## 핵심 해석

이번 산출물은 boundary specialist 제출 후보가 아니라, 기존 boundary 계열의 public negative-transfer를 calibration set으로 고정하는 phase-0 기록이다.
기존 boundary/frontier/TAGCF 계열은 validation에서 양수였지만 public에서 current best 0.77825를 넘지 못했다.
따라서 v1은 후보 CSV 생성이 아니라 row-flip precision을 예측할 수 있는지부터 검증해야 한다.

## 기존 boundary 계열 public 결과

| candidate | public | public Δ | row diff | validation precision | overlap with failed boundary union |
|---|---:|---:|---:|---:|---:|
| `boundary_scoreblend_z128_z192_z64_w-0.75` | 0.77755 | -0.00070 | 582 | 0.512 | 1.000 |
| `boundary_scoreblend_z128_z192_z64_w2` | 0.77575 | -0.00250 | 596 | 0.508 | 1.000 |
| `frontier_z_w1920_w64-0.25` | 0.77715 | -0.00110 | 556 | 0.505 | 1.000 |
| `tagcf_seed2024_sym_a0.1_raw_zblend_bw0.5` | 0.77615 | -0.00210 | 646 |  | 1.000 |

## split panel 상태

- existing split panel: 20 splits
- status: score coverage를 30~50 split으로 확장한 상태는 아직 아님. 현재 파일은 risk measurement panel metadata다.

## diff-band 요구 precision

| total diff band | public changed est. | top2 precision req. | top1 precision req. |
|---:|---:|---:|---:|
| 50 | 25.0 | 불가능 | 불가능 |
| 100 | 50.0 | 0.770 | 불가능 |
| 150 | 75.0 | 0.680 | 불가능 |
| 300 | 150.0 | 0.590 | 0.823 |
| 500 | 250.0 | 0.554 | 0.694 |
| 850 | 425.0 | 0.532 | 0.614 |

## 다음 단계

1. 이 calibration을 기준으로 기존 boundary artifact bucket을 피할 수 있는 feature만 남긴다.
2. ridge logistic / pairwise logistic cross-fit으로 boundary-only flip proposal을 평가한다.
3. 300 diff band에서 2~3등권 net gain이 보이지 않으면 full-test candidate를 만들지 않는다.
