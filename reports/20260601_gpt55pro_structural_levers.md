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

## 트랙 3: Candidate-marginal residual / quota — **CLOSED (가장 결정적 negative)**

### 아이디어 (validation-first로 규정 리스크 해소)
GPT-5.5 Pro가 "가장 구조 exploit적"이라 평가했으나 규정 리스크(negative generator 역추정 →
hidden label 복원 해석 여지)를 경고. **해법: 제출 적법성 질문 이전에, uniform surrogate에서
신호가 존재하는지부터 측정.** Δ<0.001이면 규정 질문 자체가 무의미해진다.

uniform split은 negative sampler를 정확히 알므로(uniform) 기대 negative 수를 정밀 계산 가능:
```
mu_neg(i) = Σ_{u: i∉seen_u} K_u / (n_items − |seen_u|)
z_i = (n_cand(i) − mu_neg(i)) / sqrt(mu_neg(i) + ε)   # count-shrunk
s'_ui = zscore_u(score_base) + λ · zscore_Cu(z_i)
```
실제 test 파일 미사용, CPU-only, validation-only → 규정 안전.

### 결과
| λ | uniform acc | Δ vs base | McNemar p | 판정 |
|---|---|---|---|---|
| 0.25 | 0.76365 | −0.00140 | 0.267 | REGRESS |
| 0.5 | 0.76125 | −0.00380 | 0.019 | REGRESS |
| **1.0 (게이트)** | 0.74555 | **−0.01950** | 0.0 | REGRESS |
| 2.0 | 0.72394 | −0.04111 | 0.0 | REGRESS |

**핵심 역설:** residual 추정기는 거의 완벽 작동 — `corr(z_residual, true_pos_count) = 0.9568`,
raw popularity baseline(0.9395)까지 이긴다. 즉 "어느 item에 hidden positive가 몰리는지" 정확 식별.
**그런데도 더하면 단조 악화.**

### 결론 (이번 라운드 가장 깊은 통찰)
residual z_i는 **item-level marginal** 신호("item i는 전역적으로 hidden positive 많음")인데,
대회 결정은 **user-내부**(각 후보셋에서 K_u개)다. item-level bias는 **모든 user에 동일 적용**되어,
그 item이 negative인 user에게까지 점수를 올린다. test가 user별 정확히 50/50이라 전역 item prior는
도움이 안 되고 **popularity bias 재주입**일 뿐 — 메모리에 기록된 logreg stacker public 퇴보
(0.76245→0.75355)와 **동일한 hard-sampler trap**.
→ **validation-first가 규정 리스크를 깔끔히 해소**: uniform 게이트 실패(−0.0195)이므로 제출 적법성
질문은 발생하지 않는다. 작동하지 않는 것의 적법성은 물을 필요가 없다.

## 트랙 4: hours confidence-weighted LightGCN

- 4×V100 병렬 실측 진행 중 (user_quantile / item_quantile / balanced / binary_control). 별도 갱신.

---

## 메서드론적 시사점

이번 라운드는 **"redundant"를 넘어 "직교하지만 비예측적"이라는 더 깊은 negative**를 확보했다.
두 트랙 모두 음성 리스트와 진짜 직교(loss geometry / corr −0.04)임을 엄밀히 검증했음에도
uniform 게이트를 통과하지 못했다. 이는 단일 모델 패밀리의 포화를 넘어,
**이 대회 데이터 구조 자체가 (graph co-occurrence 외의) 추가 신호원을 거의 담고 있지 않다**는
직접 증거다. 남은 유효 후보군이 구조적으로 매우 좁다.

---

## 캡스톤: Gate-floor bootstrap 분석 — 모든 게이트의 통계적 바닥 검증

지금까지 모든 후보는 단일 uniform split + noise band 0.0007로 게이트됐다. 그 split의
**고유 표본분산**을 처음으로 측정했다 (CPU-only, 학습 없음, 제출 없음).

### PART A — 단일 모델 절대정확도 부트스트랩 (user 재표집, B=2000)
- base emb128 4-seed: mean 0.76505, SE **0.00367**, 95%CI [0.75784, 0.77204], 반폭 **0.00710**
- 해석: 절대 정확도는 ±0.007 이하로 못 좁힌다. **단, 게이트는 절대값이 아니라 paired 비교를 쓰므로 이것이 게이트를 무효화하지 않음.**

### PART B — paired Δ 부트스트랩 (실제 emb128−emb64 예시, 동일 user 재표집)
- point Δ=+0.00360, **paired-SE=0.00181**, 95%CI [+0.00000, +0.00712], **MDE ≈ 0.00355**
- paired-SE가 절대 SE의 절반 → between-user 분산 상쇄. **이것이 줄곧 McNemar(paired)를 쓴 이유의 직접 입증.**
- **판정 GATE_BLUNT:** 게이트 임계 0.003이 paired noise floor(MDE 0.00355) **안에** 있다.
  진짜 +0.003급 효과조차 단일 split에선 noise와 구분 불가.
  - 우리가 "깨끗한 monotone gain"이라 본 emb64→emb128(+0.0036)조차 단일 split CI 하한이 0을 건드림
    (실제론 진짜 이득: public 0.77125→0.77745, +0.0062 전이 확인).
  - emb192 McNemar p=0.41이 "동전던지기"로 나온 것도 이 floor로 설명됨(효과가 진짜 null).

### PART C — 경계 오류 환원가능성 (정직한 분해)
rank-K_u(마지막 선택) vs rank-K_u+1(첫 탈락) 경계쌍 n=4037, 이 쌍들에서 base 정확도 0.7615.
"어느 쪽이 positive인가"를 covariate diff로 분리:
| covariate | AUC | 해석 |
|---|---|---|
| date overlap (d_ov) | **0.510** | 신규 직교축인데 **경계에서 무신호** (temporal closure 재확인) |
| item popularity (d_pop) | 0.664 | 유일한 비자명 신호이나 **이미 −0.0195로 닫은 popularity trap** |
| model score gap (d_score) | 0.741 | 모델 자기신호 = 순환적, 새 축 아님 |
- **판정: 신규 직교 covariate(date)로 경계 오류 환원 불가 = 구조적 천장.**
  유일한 above-chance 신호는 public surrogate에서 실패가 입증된 popularity trap뿐.

### 결론 & 미래 인프라
1. **이번 세션 closure는 전부 견고함** — 모든 음성이 0(subset)이거나 큰 음수라 blunt-gate "애매 구간"(+0.001~0.003)에 없음. 거짓 음성 피해자 0건.
2. **미래 진짜 +0.003급 후보는 단일 split으로 게이트 불가** → uniform seed {42, 7, 123} **3-split 패널**로
   재검증해 paired-SE를 √3배(→~0.001) 축소해야 함. 이 split들은 빌드 완료(`val_random_uniform_seed{7,123}`).
3. 76.5% 천장은 public 전이 가능한 직교 정보의 한계. 더 올리려면 새 정보원이 필요한데 데이터에 없음.

## 트랙 4: hours confidence-weighted LightGCN — **CLOSED (NO_GAIN)**

### 아이디어
binary played 라벨이라 hours는 preference magnitude로 쓰면 목표 불일치(0.2시간도 200시간도 positive).
유일한 정당한 사용은 **edge confidence**(Hu-Koren-Volinsky 2008): graph topology는 그대로 두고
edge weight `sqrt(c_ui)`로 재가중. 3가지 고정 변환(user/item quantile, balanced) + binary_control.
emb128 단일 seed, 200ep, 4×V100 병렬.

### 결과 (binary single-seed ref 0.76205)
| 모드 | uniform acc | Δ vs binary | 판정 |
|---|---|---|---|
| binary_control | 0.76205 | 0.0 | confound 가드 ✓ (conf=1.0 → canonical 정확 재현) |
| user_quantile | 0.76195 | −0.0001 | NO_GAIN |
| balanced | 0.76225 | +0.0002 | NO_GAIN |
| item_quantile | 0.76265 | +0.0006 | NO_GAIN |

### 결론
binary_control이 0.76205로 정확히 재현 → weighting 기계 자체는 artifact 없음, 모든 차이가 순수 hours 효과.
최대 이득 +0.0006조차 gate-floor MDE(0.00355) 안에 완전 포함. GPT-5.5 Pro 예측대로 hours는 같은
graph signal의 재가중이라 corr 0.98 포화를 못 깬다. **트랙 4 종결.**

---

## 라운드 종합

GPT-5.5 Pro의 실행 가능한 레버 4개 전수 + gate-floor 캡스톤:

| # | 레버 | 직교성 검증 | uniform 게이트 | 상태 |
|---|---|---|---|---|
| 1 | Exact-K subset loss | loss-geometry (수학 5/5) | Δ=+0.00000, p=0.934 | CLOSED |
| 2 | Temporal compatibility | corr(T,base)=−0.04 진짜 독립 | 전 combiner REGRESS | CLOSED |
| 3 | Candidate-marginal quota | 추정기 r=0.96 작동 | λ=1.0 −0.0195 | CLOSED |
| 4 | hours confidence | binary_control 가드 ✓ | max +0.0006 | CLOSED |
| — | **gate-floor 캡스톤** | — | MDE 0.00355 > 0.003 | 게이트 BLUNT 판정 |
| 5 | graph-only 새 encoder | — | GPT-5.5 Pro도 소진 동의 | SKIP |

**최종 결론:** 이 대회는 BPR-LightGCN 패밀리 포화(corr 0.98)를 넘어, **데이터 구조 자체가
graph co-occurrence 외의 public-전이 가능한 직교 신호를 담고 있지 않다**. 모든 게이트 0.003 임계는
단일 split의 paired noise floor(0.00355) 안이지만, 이번 세션 음성은 전부 0이거나 큰 음수라
거짓 음성 피해자 0건. final-2(emb128 0.77745, emb64 0.77125) 유지가 경험적으로 옳다.
미래 진짜 +0.003급 후보 등장 시 uniform 3-split 패널(seed 42/7/123) 재검증 인프라 준비 완료.
