# TAG-CF 3-split panel aggregate — 2026-06-01

**Safety:** validation_only=true · candidate_csv_written=false · kaggle_submit_executed=false

## Split-level best results

| split | base acc | best non-base | delta | fixes | breaks | McNemar p | gate |
|---|---:|---|---:|---:|---:|---:|---|
| seed42 | 0.762052 | score_tag_sym_a0p25_raw | +0.003201 | 323 | 259 | 0.0090 | WEAK_SIGNAL_PANEL_ONLY |
| seed7 | 0.759652 | score_tag_sym_a0p1_raw | +0.000200 | 95 | 91 | 0.8259 | REJECT |
| seed123 | 0.760052 | score_blend_sym_a0p1_raw | +0.001300 | 70 | 44 | 0.0192 | WEAK_SIGNAL_PANEL_ONLY |

## Fixed-variant panel check

The best fixed variant by mean delta was `score_blend_sym_a0p1_raw`:

- deltas: `+0.001200, -0.000200, +0.001300`
- mean delta: `+0.000767`
- positive splits: `2/3`
- pooled fixes/breaks: `168/122`
- pooled McNemar p: `0.00823`

The seed42-winning fixed variant `score_tag_sym_a0p25_raw` did not transfer:

- deltas: `+0.003201, -0.002100, -0.001000`
- mean delta: `+0.000033`
- positive splits: `1/3`
- pooled fixes/breaks: `843/841`
- pooled McNemar p: `0.9806`

## Verdict

**REJECT for submission / no full-test candidate.**

Rationale:

1. No fixed variant reaches the predeclared MDE. Best fixed mean delta is only `+0.000767`, far below `+0.00355`.
2. The seed42-winning variant regresses on both additional splits, indicating split-specific noise.
3. Best-per-split selection gives mean `+0.001567`, but this is selection-biased and still below MDE.
4. `score_blend_sym_a0p1_raw` has pooled McNemar p=0.008, but one split is negative and the absolute effect is too small for a last-slot Kaggle submission.

Next recommended cheap probe per AI-Q: **LightGCN++-style layer-mixture rescoring** over `h0..hK`, validation-only.
