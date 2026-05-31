# 새 방향 탐색 종합 — CF 축 포화 진단 (KMURecSys26 Steam, 2026-05-31 KST)

목표: 현재 best(emb128_L4_reg1e-3 4-seed 앙상블, public **0.77745**)에서 **큰 폭 개선**이
가능한 새 방향 탐색. 결론을 먼저: **순수 CF 축은 사실상 포화**됐고, 큰 폭 개선은 고전 CF
모델 추가가 아니라 강하면서 직교적인 새 모델(예: 그래프 contrastive)에서만 가능하다.

## 헤드룸 (uniform = public surrogate 기준)

- emb64 4-seed 앙상블: 0.76145, emb128 4-seed 앙상블: 0.76505
- corr(z64, z128) = **0.9747**, 50/50 z-blend = 0.76335 (**−0.0017, 손해**) → BPR-LightGCN 패밀리 포화
- best-of-2 oracle(emb64,emb128) = **0.78636** (emb128 대비 **+0.021뿐**)
- "neither correct" 행 = **21.4%** (두 모델 모두 틀림 = 본질적으로 어려운 행)

## 검증한 직교 후보 축 (parameter-free 기준: solo 또는 동등가중 blend가 emb128 초과해야 채택)

| 축 | 핵심 | solo (uniform) | corr_z vs emb128 | eq-blend(50/50) | 판정 |
|---|---|---:|---:|---:|---|
| ALS (hours-confidence WMF, Hu 2008) | 미사용 플레이타임 신호 | 0.714 (popa2) / 0.624 (pure) | 0.73 / 0.43 | 손해 | ❌ redundant |
| EASE λ1000 (closed-form item-item) | 다른 기하(선형 AE) | 0.727 | 0.79 | −0.0033 | ❌ redundant |
| ItemKNN htr | item-item 유사도 | 0.734 | 0.83 | −0.0050 | ❌ redundant |
| **DirectAU** (align+uniform, KDD'22) | loss 교체, uniform 평가와 정합 | 0.547~0.597 | **0.16~0.34** | 손해 | ❌ 직교하나 약함 |

## 핵심 교훈

1. **직교성만으로는 부족하다 — 강하면서 직교적이어야 한다.** DirectAU는 corr 0.16으로 가장 직교적이었지만
   단독 0.55로 너무 약해(popularity 0.684 미만) blend에 해롭다. 작은 그래프(6710u×2437i)에서 uniformity 항이
   유저별 소수 후보(median 4)의 미세 ranking을 파괴. gamma↑일수록 더 직교·더 약함.
2. **고전 CF 패밀리(ALS/EASE/ItemKNN)는 BPR-LightGCN과 직교적이지 않다.** 강한 변종은 전부 popularity 신호를
   공유(corr 0.73~0.83), 직교적인 순수 변종은 너무 약함. eq-blend가 모두 음수.
3. **grid-tuned blend weight는 stacker-trap이다.** 모든 축이 "win"을 w128=0.9~1.0에서만 보였는데, 이건 검증
   라벨로 1개 파라미터를 튜닝한 상방편향이고, 실제 public에서 무너진 logreg stacker(0.76245→0.75355)와 같은 위험.
   안전한 채택 기준은 solo 또는 동등가중(parameter-free) blend가 emb128을 넘는 것.

## 남은 단 하나의 "강+직교" 베팅

고전 CF는 천장. 남은 진짜 새 방향은 **강한 LightGCN 백본을 유지하면서 표현을 바꾸는** 그래프
self-supervised 계열뿐:
- **SGL / LightGCL** (graph contrastive, InfoNCE + edge dropout): 백본이 LightGCN이라 ~0.76 강함을 유지할
  가능성이 높고, contrastive 증강이 학습 표현을 바꿔 BPR과 부분 직교 가능 → "강+직교" 조건을 만족할 후보.
- 기대치는 보수적으로: oracle 천장이 +0.021이므로 현실적 blend 이득은 +0.004~0.008 수준. "큰 폭"이라기보다
  "의미 있는 마지막 한 걸음"으로 보는 게 정직하다.

## 정직한 결론

현재 0.77745는 CF 천장(oracle ~0.786)에 근접. 큰 폭(+0.02 이상) 개선은 이 데이터의 CF 구조상 가능성이
낮다. 남은 최선은 SGL/LightGCL로 강+직교 축을 1회 시도하는 것이며, 실패 시 0.77745(또는 8~16 seed 확장
앙상블)로 마무리하는 것이 합리적이다.
