# Review axis inside the stacker — honest GroupKFold OOF

- LightGCN: 0.63883
- stacker base (LightGCN+Stage2): 0.64843
- stacker +review: 0.6483
- **review increment: -0.00013** (no help)

| variant | sqrtpop | recent | popbin | mean |
|---|---:|---:|---:|---:|
| base | 0.67934 | 0.64853 | 0.61742 | 0.64843 |
| plus_review | 0.67914 | 0.65213 | 0.61362 | 0.6483 |

## Decision

- Review features add only -0.00013 → NOT worth the extra test-score generation / complexity. Keep the LightGCN+Stage2 stacker.