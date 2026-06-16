# Pseudo-label transduction probe aggregate

- run root: `artifacts/pseudolabel_margin_transduction_20260613T0246KST`
- validation only: true
- Kaggle submit: false
- candidate CSV written: false
- external metadata: false

## Overall

| top_n | margin | runs | splits | mean student | mean teacher | mean Δ | min Δ | max Δ | +runs | pseudo precision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.5 | 12 | 3 | 0.761236 | 0.761986 | -0.000750 | -0.002801 | +0.001700 | 4 | 0.9109 |
| 1 | 2.5 | 12 | 3 | 0.760852 | 0.761986 | -0.001134 | -0.004701 | +0.001000 | 1 | 0.9444 |

## Gate

- pass: `False`
- required: mean Δ >= +0.0050, min Δ >= -0.0015, splits >= 3

## Best row

```json
{
  "top_n": 1.0,
  "min_margin": 1.5,
  "n": 12.0,
  "splits": 3.0,
  "mean_student_acc": 0.7612355804494233,
  "mean_teacher_acc": 0.7619857304794292,
  "mean_delta": -0.0007501500300059564,
  "min_delta": -0.0028005601120223433,
  "max_delta": 0.0017003400680136727,
  "positive_runs": 4.0,
  "mean_pseudo_precision": 0.91091019467421
}
```
