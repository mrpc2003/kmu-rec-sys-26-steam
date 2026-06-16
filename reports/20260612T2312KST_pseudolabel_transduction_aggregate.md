# Pseudo-label transduction probe aggregate

- run root: `artifacts/pseudolabel_transduction_20260612T2312KST`
- validation only: true
- Kaggle submit: false
- candidate CSV written: false
- external metadata: false

## Overall

| top_n | margin | runs | splits | mean student | mean teacher | mean Δ | min Δ | max Δ | +runs | pseudo precision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 12 | 3 | 0.761927 | 0.761986 | -0.000058 | -0.001500 | +0.001900 | 6 | 0.8442 |

## Gate

- pass: `False`
- required: mean Δ >= +0.0050, min Δ >= -0.0015, splits >= 3

## Best row

```json
{
  "top_n": 1.0,
  "min_margin": 0.0,
  "n": 12.0,
  "splits": 3.0,
  "mean_student_acc": 0.7619273854770955,
  "mean_teacher_acc": 0.7619857304794292,
  "mean_delta": -5.834500233376875e-05,
  "min_delta": -0.0015003000600120053,
  "max_delta": 0.001900380076015229,
  "positive_runs": 6.0,
  "mean_pseudo_precision": 0.8442426801801801
}
```
