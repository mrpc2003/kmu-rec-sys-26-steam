# LightGCN checkpoint prediction averaging aggregate

- run root: `artifacts/lightgcn_checkpoint_avg_20260613T0106KST`
- validation only: true
- Kaggle submit: false
- candidate CSV written: false
- external metadata: false

## Overall

| variant | splits | mean acc | baseline | mean Δ | min Δ | max Δ | +splits |
|---|---:|---:|---:|---:|---:|---:|---:|
| `score_avg_last3_160_200` | 3 | 0.762052 | 0.761986 | +0.000067 | -0.000300 | +0.000500 | 1 |
| `score_ep200` | 3 | 0.761986 | 0.761986 | +0.000000 | +0.000000 | +0.000000 | 0 |
| `score_avg_last2_180_200` | 3 | 0.761852 | 0.761986 | -0.000133 | -0.000300 | +0.000000 | 0 |
| `score_ep180` | 3 | 0.761719 | 0.761986 | -0.000267 | -0.000300 | -0.000200 | 0 |
| `score_ep160` | 3 | 0.760452 | 0.761986 | -0.001534 | -0.002200 | -0.001100 | 0 |
| `score_avg_all_120_200` | 3 | 0.759985 | 0.761986 | -0.002000 | -0.002200 | -0.001900 | 0 |
| `score_ep140` | 3 | 0.757818 | 0.761986 | -0.004168 | -0.005501 | -0.002701 | 0 |
| `score_ep120` | 3 | 0.756185 | 0.761986 | -0.005801 | -0.006901 | -0.004401 | 0 |

## Gate

- pass: `False`
- required: mean Δ >= +0.0015, min Δ >= +0.0000, splits >= 3
- full-test materialization remains blocked until explicit later approval.

## Best row

```json
{
  "variant": "score_avg_last3_160_200",
  "splits": 3,
  "mean_acc": 0.7620524104820964,
  "mean_baseline": 0.7619857304794292,
  "mean_delta": 6.668000266722245e-05,
  "min_delta": -0.00030006001200233445,
  "max_delta": 0.0005001000200040018,
  "positive_splits": 1
}
```
