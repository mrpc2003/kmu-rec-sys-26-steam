# LightGCN checkpoint prediction averaging aggregate

- run root: `artifacts/lightgcn_checkpoint_avg_smoke_20260613T0106KST`
- validation only: true
- Kaggle submit: false
- candidate CSV written: false
- external metadata: false

## Overall

| variant | splits | mean acc | baseline | mean Δ | min Δ | max Δ | +splits |
|---|---:|---:|---:|---:|---:|---:|---:|
| `score_ep2` | 1 | 0.722044 | 0.765053 | -0.043009 | -0.043009 | -0.043009 | 0 |
| `score_avg_last2_1_2` | 1 | 0.721944 | 0.765053 | -0.043109 | -0.043109 | -0.043109 | 0 |
| `score_ep1` | 1 | 0.721244 | 0.765053 | -0.043809 | -0.043809 | -0.043809 | 0 |

## Gate

- pass: `False`
- required: mean Δ >= +0.0015, min Δ >= +0.0000, splits >= 3
- full-test materialization remains blocked until explicit later approval.

## Best row

```json
{
  "variant": "score_ep2",
  "splits": 1,
  "mean_acc": 0.7220444088817763,
  "mean_baseline": 0.7650530106021204,
  "mean_delta": -0.04300860172034404,
  "min_delta": -0.04300860172034404,
  "max_delta": -0.04300860172034404,
  "positive_splits": 0
}
```
