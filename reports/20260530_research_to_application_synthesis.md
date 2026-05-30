# 연구 논문 → 적용 탐색 종합 (KMURecSys26 Steam, 2026-05-30 KST)

본 문서는 "Steam played 예측과 유사 task의 최신 연구 논문을 조사하여 적용하는 탐색 단계"의
결론을 통합한다. 각 논문 계열을 **실제 validation 결과**와 매핑하며, 모든 수치는 실행 로그/리포트 근거.
검증 기준선: LightGCN 단독 mean 0.63883 (3-split: sqrtpop/recent/popbin), public 0.76245.

---

## 1. 논문 계열별 탐색 결과 매핑

| 논문 계열 | 핵심 아이디어 | 적용 결과 (mean val) | Δ vs LightGCN | 판정 |
|---|---|---:|---:|---|
| **LightGCN** (He 2020) | 경량 GCN, BPR, feature transform/nonlinearity 제거 | **0.63883** | — | ✅ **채택·제출 (public 0.76245)** |
| Koren neighborhood / ItemKNN | item-item 유사도, BM25/TF-IDF 가중 | ~0.62 (Stage2 구성요소) | 음수 | 보조축 |
| EASE (Steck 2019) | closed-form item-item, λ 정규화 | ~0.62 (Stage2 구성요소) | 음수 | 보조축 |
| BPR-MF / ALS (Rendle, Hu) | implicit MF, popularity 보정 | ~0.62 (Stage2 구성요소) | 음수 | 보조축 |
| Stage2 mean-z blend | 위 축들의 z-score 앙상블 | 0.62563 | −0.0132 | LightGCN에 흡수 |
| Time-decay ItemKNN | 최근 상호작용 가중 (recency) | recent split 보조 | 약함 | 기각 |
| **리뷰 강화 추천** (text-aware) | 리뷰 텍스트 TF-IDF user-item cosine | 0.61/0.59/0.52 단독 | 음수 | 단독 기각 |
| **리뷰 축 직교성** (stacker 내부) | "텍스트는 상호작용과 직교" 주장 검증 | stacker 증분 **−0.00013** | 노이즈 | ❌ **직교성 미확인** |
| Stacking / 메타러닝 | base 모델 행별 게이팅 학습 | **0.64930** (정직 OOF) | **+0.0105** | ✅ **최고 보완 (후보 materialize)** |

---

## 2. 핵심 발견

### (a) LightGCN이 단일 최강 축
- 모든 split에서 Stage2 앙상블을 +0.011~0.015 능가, public transfer ratio 1.24로 검증.
- mid-pop(인기 중간) 구간에서 CF 신호가 가장 크게 기여 (diff 분석).

### (b) 리뷰 텍스트는 직교 신호가 아니다 (이 데이터에서)
- 리뷰 강화 추천 논문의 핵심 가정을 정직 GroupKFold로 검증 → 증분 −0.00013.
- 이유 추정: 테스트 pair에 text 없음 + train 리뷰 프로필이 이미 상호작용(played) 정보와 강하게 중복.
- → text 축은 test score 생성 비용을 들일 가치 없음. 기각.

### (c) Stacking이 oracle 갭의 일부를 정직하게 추출
- Oracle 상한 0.729 vs LightGCN 0.639 (행별 완벽 선택 시 +0.09 여지).
- logreg stacker가 정직 OOF +0.0105 추출 (누수 갭 3e-05으로 진짜 검증).
- 메타러너 가중치: Stage2 통합축(+0.87) + LightGCN(+0.32/+0.37) + 인기 과보정 억제(log_pop −0.42).

---

## 3. 검증 방법론 (정직성 보장)

- **per-user top-half 디코딩**: 대회 규칙(유저당 정확히 절반 played)을 그대로 반영한 canonical 평가.
- **3종 negative 샘플러 split**: sqrtpop(인기^0.5), recent(시간 tail), popbin(인기 bin-matched) — MNAR 통제.
- **유저 단위 GroupKFold**: within-user 피처 누수 차단. strat−group 갭으로 누수 정량화(3e-05).
- **transfer ratio 추적**: local Δ → public Δ 비율로 surrogate 신뢰도 확인(LightGCN=1.24).

---

## 4. 미해결 / 진행 중

| 항목 | 상태 |
|---|---|
| LightGCN 하이퍼파라미터 sweep (17 config) | 🔄 진행 중 (~01:40 KST). 초기 신호: L2 손해, L3+reg1e-3 선두(+0.0022) |
| stacker 테스트 후보 (pooled OOF +0.0091) | ⏳ 제출 승인 대기 |
| (미탐색) LightGCN seed 앙상블 | sweep 후 GPU 여유 시 검토 — 분산 감소로 +0.002~0.005 기대 |
| (미탐색) LightGCL / 그래프 contrastive | sweep best 확정 후 비교 검토 |

---

## 5. 결론

탐색 단계의 명확한 결론: **LightGCN(그래프 CF)이 이 task의 지배적 신호**이며,
리뷰 텍스트 등 비-상호작용 축은 정직 검증에서 직교 이득을 주지 못했다.
유일하게 검증된 보완은 **base 모델 stacking**(+0.0105 OOF)이고, 추가 이득은
(1) LightGCN 자체 강화(sweep/앙상블)와 (2) stacker 제출 검증에서 나올 것으로 본다.
