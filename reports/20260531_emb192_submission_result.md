# emb192 제출 결과 — 실제 public LB 확정 (2026-05-31 KST)

emb192 capacity-frontier 후보를 우현 승인 하에 실제 Kaggle public LB에 제출한 결과.

## 결과

| 후보 | public | uniform surrogate | SHA256 |
|---|---:|---:|---|
| emb128 4-seed (final-2 #1) | **0.77745** | 0.76505 | `7e3191de…` |
| **emb192 4-seed** (이번 제출) | **0.77715** | 0.76615 | `1b3a6056…` |
| Δ (emb192 − emb128) | **−0.00030** | +0.00110 | — |

`SubmissionStatus.COMPLETE`, 제출 시각 2026-05-31 09:43 UTC.

## 해석 — paired McNemar가 옳았다

- surrogate(uniform)에서 emb192 4-seed 앙상블이 +0.0011 높았지만, 이는 noise band(0.0007)의
  1.6배에 불과한 marginal 신호였고 **paired McNemar p=0.4137(통계적 동전던지기)** 로 비유의 판정.
- 실제 public LB에서 emb192는 **0.77715 = emb128 대비 −0.0003**. 즉 surrogate의 +0.0011은
  진짜 개선이 아니라 noise였고, paired 검정이 제출 전에 정확히 경고한 그대로다.
- seed42 단독 uniform +0.0046이 high outlier였다는 진단도 확증: 4-seed 앙상블·실제 LB 모두에서
  capacity 증가의 실이득은 0 또는 음(-).

## 결론

- **capacity frontier가 실제 LB에서도 종결.** emb128이 backbone capacity sweet spot이며, emb192/256/320
  모두 실질 개선 없음. 백본 차원 확장 축은 닫힘.
- **final-2 변동 없음:** #1 emb128 4-seed (public **0.77745**), #2 emb64 4-seed (0.77125).
  emb192(0.77715)는 #1보다 낮아 교체 근거 없음.
- validation-first 원칙의 모범 사례: surrogate marginal gain을 paired 검정으로 de-noise → 비유의 →
  실측이 확인. 향후 noise-level surrogate 신호는 paired 검정으로 게이트하면 헛제출을 방지할 수 있다.
