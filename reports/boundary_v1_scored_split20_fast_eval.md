# boundary v1 scored split20 fast ridge eval

- validation_only: true
- no_kaggle_submit: true
- candidate_csv_written: false
- full_test_candidate_written: false

## coverage

- complete scored splits: 20
- mean anchor accuracy: 0.760702
- mean boundary band rows: 17242.0

## aggregate diff-band result

| model | band | mean precision | mean net rows | positive split ratio | worst split | top2 pass | top1 pass |
|---|---:|---:|---:|---:|---:|---|---|
| ridge_fast | 100 | 0.545 | 6.30 | 0.70 | -14 | False | False |
| ridge_fast | 150 | 0.545 | 6.30 | 0.70 | -14 | False | False |
| ridge_fast | 300 | 0.545 | 6.30 | 0.70 | -14 | False | False |
| ridge_fast | 500 | 0.545 | 6.30 | 0.70 | -14 | False | False |
| ridge_fast | 850 | 0.545 | 6.30 | 0.70 | -14 | False | False |
| ridge_fast | 50 | 0.555 | 5.50 | 0.75 | -14 | False | False |

## submission readiness

`FAIL__NO_SCORED_GATE_PASS`

top2/top1 pass가 없으면 full-test candidate를 만들지 않는다.
