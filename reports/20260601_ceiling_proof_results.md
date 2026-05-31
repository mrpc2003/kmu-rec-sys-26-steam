# Hermes 검증: Sisyphus 방향 1 (천장-증명) 실측 결과 — 2026-06-01 KST

**역할 분담:** Sisyphus(OpenCode/Opus 4.8) = adviser/방향 제안, Hermes = 최종 검증자(게이트).
**기준 best:** emb128 4-seed LightGCN, uniform 0.76505 / public 0.77745.

Sisyphus는 내 "구조적 천장" 주장에서 두 미검증 빈틈을 정확히 지목했다:
- **A.3 (boundary 측정):** `neither_correct_diagnosis.py`가 작성만 되고 한 번도 실행 안 됨 → 천장 주장이 미실행 스크립트 위에 서 있음. (확인됨: artifacts/reports에 실행 기록 0건)
- **A.2 (기하):** 닫힌 직교 시도는 전부 loss를 InfoNCE로 바꿨고(solo 붕괴), 닫힌 강한 시도는 전부 유클리드 내적 기하 공유. "ranking loss 유지 + 비유클리드 기하"는 미검증.

Sisyphus 권고 순서대로 **GPU 베팅 전에 CPU 천장-증명(direction 1)을 먼저** 실행했다.

---

## 측정 1 — neither-correct 진단 (21.36% 동시오답, 4272/19996행)

| 항목 | 결과 | 해석 |
|---|---|---|
| played:unplayed | 2137 : 2135 | **완벽 대칭** = 경계에서 편향 없는 동전던지기 |
| pop 분위 neither율 | Q0 0.171 → Q2 **0.260**(정점) → Q4 0.145 | 역U자(평탄 아님), range 0.116 |
| user-degree 분위 | 0.188~0.265 | 약한 증가 |
| candidate-set 크기 | 0.151(cset2) → 0.281(cset20) | 기계적(큰 set=경계 결정 多) |

**판정:** "구조적"이지만 이것만으로 천장/헤드룸 확정 불가. mid-pop 정점은 **intrinsic Bayes 난이도**(prior≈0.5)일 수 있어, 회복가능 신호인지 별도 검증 필요 → 측정 2.

## 측정 2 — boundary covariate 확장 (rank-K_u/K_u+1 경계쌍 n=4037)

**핵심 방법론 교정:** Sisyphus 초안은 `d_cooc`만 popularity 잔차화하고 `d_knn`은 누락 → 자동 verdict가
잔차화 안 된 d_knn(AUC 0.613)을 보고 **HEADROOM_EXISTS를 잘못 외침.** Hermes가 비대칭 버그를 잡아
d_knn도 동일하게 popularity 잔차화 추가.

| covariate | AUC | popularity 잔차화 | 판정 |
|---|---|---|---|
| d_logpop (알려진 trap) | 0.6775 | — 기준 | trap |
| d_cooc_raw | 0.674 | ❌ | ≈ d_logpop → **순수 인기** |
| d_knn (자동 verdict 트리거) | 0.613 | ❌ | **인기 위장** |
| **d_cooc_resid** | **0.534** | ✅ | CI[0.515,0.554] 유의하나 escalation bar(0.05) 미만 |
| **d_knn_resid** | **0.5037** | ✅ | CI[0.485,0.524] **거의 완전 chance** |

**판정: INTERMEDIATE (soft no-go).** 두 novel covariate 모두 popularity를 제거하면 chance로 붕괴.
candidate-marginal(r=0.96 추정기 → public −0.0195)과 **동일한 popularity-trap 패턴**.
잔차화 안 한 raw 신호가 above-chance인 건 전부 인기 재표현일 뿐.

---

## 종합 — A.3 빈틈은 닫혔다

- boundary 잔차는 **public 전이 가능한 비-인기 직교 신호를 담고 있지 않다**(intrinsic Bayes).
- d_cooc_resid 0.534의 미약한 속삭임조차 escalation bar 미만 + MDE(0.00355) 대비 무의미한 크기.
- 따라서 **decision-rule/covariate 축으로는 천장 확정.** Sisyphus의 direction 1 게이트는 soft no-go.

## 남은 단 하나 — A.2 (기하)는 아직 직접 미검증

boundary covariate 테스트는 "경계에 covariate 신호가 있나"의 **proxy**다. 하이퍼볼릭 기하 가설(A.2)은
covariate를 더하는 게 아니라 **같은 co-occurrence 그래프를 다른 공간에 재표현**하는 것이라 이 proxy로
완전히 기각되지 않는다. 단 d_cooc_resid=0.534(거의 chance)는 base가 conditional-cooc를 이미 거의 다
추출했음을 시사 → 하이퍼볼릭 기대치는 보수적.

**다음 행동 (validation-first):** 1.5일 풀 베팅이 아니라 **저비용 단일-seed emb64 하이퍼볼릭 probe + hard
reject 게이트**로 A.2를 직접·싸게 falsify한다. solo<floor(0.684) → 즉시 기각(SGL 운명 재확인).
solo 유지 & corr_z<0.9 & eq-blend>noise → 진짜 신호, 3-split 패널로 승격. eq-blend≤noise → "기하도 종결"
이라는 가장 강한 음성 확보. 구현은 Sisyphus(OpenCode), 게이트는 Hermes.
