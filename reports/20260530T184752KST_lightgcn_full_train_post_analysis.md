# Post-Analysis: LightGCN full-train submission

- Submitted: 2026-05-30 09:48:32 UTC (KST 18:48)
- File: artifacts/lightgcn_20260530/test_full_train/candidate_lightgcn_full_train.csv
- SHA256: a3dbe043f0f8b781d8c35aea88b7a1f561fa7b705b34edf6c7b7d0451eceb2a6
- Status: SubmissionStatus.COMPLETE
- Public score: 0.76245
- Previous best: 0.74594 (Stage2 mean-z blend)
- Delta: +0.01651

## Validation vs public

| split | LightGCN | Stage2 | Δ |
|---|---:|---:|---:|
| random_sqrtpop | 0.6748 | 0.6597 | +0.0151 |
| recent_sqrtpop | 0.6396 | 0.6260 | +0.0136 |
| random_popbin | 0.6020 | 0.5908 | +0.0112 |
| **mean** | **0.63880** | **0.62550** | **+0.01330** |

- Public delta: +0.01651
- Local mean gain: +0.01330
- Transfer ratio (public_delta / local_mean_gain): 1.2414
- Verdict: SUCCESS — local validation predicted direction and magnitude correctly. Transfer ratio is in a healthy range.

## Direction registry

Registering as a confirmed-good direction:
- Family: graph collaborative filtering, BPR-trained
- Architecture: LightGCN (no feature transform, no nonlinearity)
- Hyperparams: emb_dim=64, n_layers=3, lr=1e-3, reg=1e-4, batch_size=4096, epochs=200, seed=42
- Train data: full pos-only matrix from train.json (175k interactions, 6710 users, 2437 items)
- Decoding: per-user top-half on raw inner-product score

## Quota

- Used today (UTC 2026-05-30): 2/5 (the prior Stage2 + this LightGCN). 3 remaining.
- Per protocol: stop. No chain-submit. New best becomes the anchor; any next file requires fresh exact-file approval.
