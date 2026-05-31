# GPT-5.5 Pro 방향 탐색 — 구조적 레버 실측 라운드

**일자:** 2026-06-01 (KST)
**대회:** KMU RecSys '26 Steam (binary played 예측, Accuracy, uniform-LB 추종)
**기준 베이스라인:** emb128 4-seed LightGCN, uniform 0.76505 / public 0.77745
**원칙:** validation-first, uniform surrogate parameter-free 게이트, paired McNemar, noise-chasing 금지

---

## 배경: 왜 "새 encoder"가 아니라 "구조적 레버"인가

직전 라운드까지 BPR-LightGCN 패밀리의 **완전 포화**가 실측으로 확정됐다:

- corr_z(emb64, emb128) = 0.9747
- corr_z(emb128, emb192) = 0.986
- emb192 실제 public 0.77715 < emb128 0.77745 (capacity frontier도 noise)

corr 0.98은 "동일 graph signal + 동일 pairwise surrogate(BPR) → ranking 수렴"을 의미한다.
따라서 GPT-5.5 Pro에게 자기완결 프롬프트로 **이 대회 구조(익명 ID + positive-only + per-user 정확히 절반 positive + uniform LB 추종)를 직접 exploit하는 미개척 레버**를 요청했고,
음성 리스트(SimGCL/Turbo-CF/AlphaRec/DNS/MultiVAE/text/capacity/cross-blend)와 직교하는 후보 2개를 실측했다.

핵심 통찰(코드로 입증): `recsys_played_utils.predict_tophalf`가 이미 per-user top-K_u를 강제하므로,
monotone threshold/calibration은 ranking을 못 바꾼다(이득 0). 유효한 decision-rule 레버는
① ranking을 바꾸는 structural residual, ② exact-K set-level loss, ③ global assignment 뿐이다.

---

## 트랙 1: Exact-K conditional subset loss — **CLOSED (NO_GAIN)**

### 아이디어
encoder/graph는 emb128 그대로 두고 **loss geometry**만 변경. 대회 metric은 global AUC가 아니라
"user별 2K_u개 후보 중 정확히 K_u개 선택"이다. 이에 맞는 최대우도 목적함수는
conditional subset probability이고, 정규화 분모는 elementary symmetric polynomial:

```
L_u = -Σ_{i∈P_u} s_ui + log e_{K_u}(exp(s_uj) : j∈C_u)
```

`e_k`는 DP로 O(n·K) 계산, max-subtraction 수치 안정화.

### 수학 검증 (GPU 학습 전, 순수 stdlib, 5/5 통과)
| 검증 | 결과 |
|---|---|
| DP vs brute-force e_k (2000 케이스) | 오차 **7.1e-15** |
| K=1 subset loss == BPR (5000 케이스) | 오차 **3.5e-15** |
| 극단 스코어 overflow 가드 | 오차 0 |
| K≥2 subset ≠ BPR | subset 0.715 vs bpr 0.213 (별개 objective) |
| 경계 gradient = marginal inclusion prob | rank-K/K+1 = **0.50** (경계 집중 확인) |

→ 메커니즘은 진짜다: subset loss의 gradient는 rank-K_u/K_u+1 **경계 item에 최대 집중**하고,
이는 per-user accuracy를 실제로 결정하는 지점이다. BPR은 모든 pos-neg pair에 균등 분산한다.

### confound-controlled 실험 (단일 BPR init에서 3분기)
- **0. pretrained** (BPR 200ep): uniform **0.76205** ← canonical 0.762 경로 정확 재현
- **A. bpr_ft** (+40ep BPR, control): 0.76025
- **B. subset_ft** (+40ep subset loss, variant): 0.76025

| 지표 | 값 | 판정 |
|---|---|---|
| **격리된 Δ(subset − bpr_ft)** | **+0.00000** | loss geometry 효과 = 0 |
| Δ(bpr_ft − pretrained, 추가학습 효과) | −0.00180 | 추가 FT 자체는 미세 악화 |
| McNemar(B vs A): fixes / breaks | 73 / 73, **p=0.934** | 완전 동전던지기 |

### 결론
subset loss는 146개 행에서 예측을 재배치하지만(73 수정 + 73 파괴) **정확히 상쇄**된다.
근본 원인: pairs.csv K_u 분포의 **45%가 K=1**이고, K=1에선 subset = BPR(3.5e-15 오차 검증).
나머지 55%의 다른 gradient가 net-zero로 희석된다. **버그가 아니라 진짜 무신호.**

---

## 트랙 2: Temporal compatibility rerank — **CLOSED (직교적이나 비예측적)**

### 아이디어
`date`는 adjacency 밖의 유일한 edge 속성(topology가 아닌 timestamp 분포). 가설:
hidden positive는 user 활동 시기와 겹치고, uniform negative는 user 시간과 독립적으로 뽑힌다.

```
T_overlap(u,i) = Σ_t p_u(t) · log( p_i(t) / p_global(t) )
```

p_u, p_i 모두 **fold_train에서만** 구축(heldout positive 날짜 미사용 — 실제 test 누수 차단).
month-bin 88개, shrinkage β=10, popularity-trap 가드(T_resid: log-pop·degree에 residualize).

### 결과
- base emb128 4-seed ensemble uniform = **0.76505** = ref (base ranking 정합 확인)
- **corr(T, base_score) = −0.0401** ← LightGCN ranking과 진짜 직교 (목표한 비상관 달성)
- corr(T, log_pop) = −0.1216 ← popularity 재현도 아님 (trap 가드 통과)

| combiner | uniform acc | Δ vs base | changed precision | 판정 |
|---|---|---|---|---|
| T_only | 0.5176 | −0.24745 | 0.247 | REGRESS |
| rank_sum | 0.65863 | −0.10642 | 0.290 | REGRESS |
| rank_sum_resid | 0.67243 | −0.09262 | 0.302 | REGRESS |
| boundary_swap | 0.66803 | −0.09702 | 0.288 | REGRESS |

### 결론
이전 closure들이 "redundant(corr 0.8+)"였던 것과 **질적으로 다른** negative다.
T는 corr −0.04로 **엄밀히 독립**임을 검증했는데도 정확도가 없다(T_only 거의 동전던지기 0.518).
boundary_swap에서 T가 경계에 반대할 때 **71%가 틀린다**(precision 0.29).
→ GPT-5.5 Pro의 가설("positive=user 시기 겹침, negative=독립")이 **데이터로 반증**됨.
이 대회 구조는 temporal 축에 exploit 가능한 정보를 담고 있지 않다.
계수 튜닝은 precision 0.29를 76.5% base에 섞는 것이라 수학적으로 개선 여지가 없음(noise-chasing 회피).

---

## 트랙 3·4 처리

- **#4 hours confidence-weighted LightGCN**: 4×V100 병렬 실측 진행 중 (별도 기록).
- **#3 candidate-marginal quota assignment**: GPT-5.5 Pro 본인이 규정 리스크(negative generator
  역추정 → hidden label 복원 해석 여지) + hard-sampler trap 경고. 단독 진행하지 않고 사용자 결정점으로 보류.

---

## 메서드론적 시사점

이번 라운드는 **"redundant"를 넘어 "직교하지만 비예측적"이라는 더 깊은 negative**를 확보했다.
두 트랙 모두 음성 리스트와 진짜 직교(loss geometry / corr −0.04)임을 엄밀히 검증했음에도
uniform 게이트를 통과하지 못했다. 이는 단일 모델 패밀리의 포화를 넘어,
**이 대회 데이터 구조 자체가 (graph co-occurrence 외의) 추가 신호원을 거의 담고 있지 않다**는
직접 증거다. 남은 유효 후보군이 구조적으로 매우 좁다.
