# 연구 논문 → 적용 탐색 종합 (KMURecSys26 Steam, 2026-05-30 KST)

본 문서는 "Steam played 예측과 유사 task의 최신 연구 논문을 조사하여 적용하는 탐색 단계"의
결론을 통합한다. 각 논문 계열을 **실제 validation 결과 + public 전이 결과**와 매핑하며,
모든 수치는 실행 로그/리포트 근거. 검증 기준선: LightGCN 단독 mean 0.63883
(3-split hard sampler: sqrtpop/recent/popbin), public **0.76245** (제출 anchor).

> ⚠ 본 종합은 두 번 갱신됨: (1) stacker가 정직 OOF에서 +0.0105로 "최고 보완"처럼 보였으나
> **public에서 회귀(0.75355)**했고, (2) 그 실패 원인을 규명하는 과정에서 **public surrogate가
> uniform split임**을 발견, seed 앙상블을 그 위에서 재검증해 **유일하게 견고한 보완**으로 확정했다.

---

## 1. 논문 계열별 탐색 결과 매핑

| 논문 계열 | 핵심 아이디어 | 적용 결과 | 판정 |
|---|---|---|---|
| **LightGCN** (He 2020) | 경량 GCN, BPR, feature transform/nonlinearity 제거 | hard mean 0.63883 · uniform 0.75445 · **public 0.76245** | ✅ **채택·제출 (anchor)** |
| Koren neighborhood / ItemKNN | item-item 유사도, BM25/TF-IDF 가중 | ~0.62 (Stage2 구성요소) | 보조축 |
| EASE (Steck 2019) | closed-form item-item, λ 정규화 | ~0.62 (Stage2 구성요소) | 보조축 |
| BPR-MF / ALS (Rendle, Hu) | implicit MF, popularity 보정 | ~0.62 (Stage2 구성요소) | 보조축 |
| Stage2 mean-z blend | 위 축들의 z-score 앙상블 | hard 0.62563 (−0.0132) | LightGCN에 흡수 |
| Time-decay ItemKNN | 최근 상호작용 가중 (recency) | recent split 보조, 약함 | 기각 |
| **리뷰 강화 추천** (text-aware) | 리뷰 텍스트 TF-IDF user-item cosine | 단독 0.61/0.59/0.52 · stacker 증분 **−0.00013** | ❌ **직교성 미확인, 기각** |
| Stacking / 메타러닝 | base 모델 행별 게이팅 학습 | 정직 OOF +0.0105 → **public −0.0089 (0.75355)** | ❌ **public 회귀, 기각** |
| **Seed 앙상블** (분산 감소) | 동일 config 다중 seed raw-score 평균 | hard +0.0041 · **uniform +0.00700 (0.76145)** | ✅ **견고 보완 확정 (제출 후보)** |

---

## 2. 핵심 발견

### (a) LightGCN이 단일 최강 축
- 모든 hard split에서 Stage2 앙상블을 +0.011~0.015 능가, public 0.76245로 검증.
- mid-pop(인기 중간) 구간에서 CF 신호가 가장 크게 기여 (diff 분석: Stage2와 15.5% 불일치, mid-pop 집중).

### (b) 리뷰 텍스트는 직교 신호가 아니다 (이 데이터에서)
- 리뷰 강화 추천 논문의 핵심 가정을 정직 GroupKFold로 검증 → 증분 −0.00013 (노이즈).
- 이유: 테스트 pair에 text 없음 + train 리뷰 프로필이 이미 상호작용(played) 정보와 강하게 중복. 기각.

### (c) ★ Stacking 실패와 그 진짜 원인 — 이번 탐색의 최대 교훈
- logreg stacker가 정직 GroupKFold OOF에서 +0.0105 (누수 갭 3e-05)로 **검증을 통과**했으나,
  실제 public에서 **0.76245 → 0.75355 (−0.0089)** 회귀. transfer ratio −0.98 (예측 +, 실제 −).
- **GroupKFold 통과는 필요조건이지 충분조건이 아니다.** 유저-행 누수는 막지만,
  **검증 negative 샘플러 자체에 대한 과적합은 못 잡는다.**
- 메타러너 top 피처: `score_blend_mean_z`(+0.87), `log_pop`(−0.42, 인기 down-weighting).
  이건 우리가 만든 hard 샘플러(sqrtpop/popbin/recent)의 인기-매칭 negative를 가르는 데는 도움되지만,
  실제 test의 negative 분포와 어긋난다.

### (d) ★ Public surrogate = uniform split (실패가 드러낸 결정적 발견)
- **같은 LightGCN이 negative 샘플러에 따라 0.555~0.754로 0.20 출렁인다** (같은 test pair):
  uniform 0.75445 · sqrtpop 0.67483 · recent 0.63963 · popbin 0.60202 · communitypop 0.57231 · recent+community 0.55551.
- 실제 public(0.76245)은 우리가 주력으로 쓴 sqrtpop이 아니라 **uniform split(0.75445)에 정확히 안착**.
- 메커니즘 테스트로 확증: `score − α·within_user_z(log_pop)` 적용 시 hard 샘플러에서는 개선(+0.003~0.006),
  **uniform에서는 어떤 down-weighting도 해롭다**(α=0.25에서 0.754→0.737). → stacker가 학습한 인기 보정이
  실제 test에서 순수하게 해로웠던 이유를 메커니즘 수준에서 설명. transfer ratio 1.24도 "증폭"이 아니라
  "sqrtpop surrogate가 실제 test보다 단지 어려웠을 뿐"으로 해소.

### (e) ★ Seed 앙상블이 uniform에서 견고하게 통과 (확정된 보완)
- stacker 실패 후 robust 복구 트랙으로, **동일 config(emb64 L3 reg1e-4)를 seed 42/123/2024/7로 학습해
  raw score 평균**. 검증 라벨/negative에서 아무것도 학습하지 않아 샘플러 과적합이 구조적으로 불가능.
- **진짜 surrogate(uniform)에서 재검증** — 이게 결정적 게이트:
  단일 seed 모두 anchor 초과(42=0.75445, 123=0.75735, 2024=0.75775, 7=0.75805),
  4-seed 앙상블 = **0.76145 = +0.00700 vs anchor** → 판정 **ROBUST_GAIN**.
- stacker(uniform 회귀)와 정확히 대조되는 거울상. "novelty가 아니라 surrogate에서의 견고성"이 채택 기준.

---

## 3. 검증 방법론 (정직성 보장 — 이번 세션에서 강화됨)

- **per-user top-half 디코딩**: 대회 규칙(유저당 정확히 절반 played)을 그대로 반영한 canonical 평가.
- **다중 negative 샘플러 split**: uniform / sqrtpop / recent / popbin / communitypop — MNAR 통제 + surrogate 식별.
- **유저 단위 GroupKFold**: within-user 피처 누수 차단(strat−group 갭 3e-05). 단, **이것만으로는 불충분**.
- **public-surrogate 게이트 (신규·핵심)**: 신뢰 모델 1회 제출로 어느 split이 public과 일치하는지 식별(=uniform),
  이후 모든 후보를 그 surrogate에서 게이트. 고정 threshold ladder(ROBUST_GAIN>+0.0005 / FLAT ±0.0005 / REGRESSION<−0.0005)로
  제출/보류/기각을 기계적으로 결정.
- **transfer ratio 추적**: local Δ → public Δ. 단, surrogate를 잘못 고르면 부호까지 뒤집힘(stacker −0.98이 실증).

---

## 4. 상태

| 항목 | 상태 |
|---|---|
| LightGCN 단일 (anchor) | ✅ 제출됨, public 0.76245, SHA `a3dbe04…` |
| logreg stacker (pooled OOF +0.0091) | ❌ 제출됨, public 0.75355 회귀 → 기각, 교훈 보존 |
| **Seed 앙상블 (4-seed)** | ✅ uniform +0.00700 ROBUST_GAIN, 후보 materialize (SHA `dcc578de…`), **제출 승인 대기** |
| LightGCN 하이퍼파라미터 sweep (17 config) | 🔄 마지막 emb256 config 진행 중. 완료 시 top config을 **uniform에서 재검증** 후에만 채택 |
| (미탐색) LightGCL / 그래프 contrastive | sweep·앙상블 확정 후 비교 검토 |

---

## 5. 결론

탐색 단계의 명확한 결론:

1. **LightGCN(그래프 CF)이 이 task의 지배적 신호**이며, 리뷰 텍스트 등 비-상호작용 축은
   정직 검증에서 직교 이득을 주지 못했다.
2. **학습형 보완(stacker/blend)은 위험하다** — GroupKFold를 통과해도 검증 negative 샘플러에
   과적합해 실제 public에서 반대로 작동할 수 있다(stacker −0.0089가 실증).
3. 이번 탐색의 최대 산출물은 단일 후보가 아니라 **방법론적 발견**이다:
   *어느 validation split이 public surrogate인지(=uniform)를 먼저 식별하고, 모든 후보를 그 위에서 게이트하라.*
4. 그 게이트를 통과한 유일한 견고 보완은 **seed 앙상블(+0.00700 on uniform, ROBUST_GAIN)** —
   검증 라벨을 학습하지 않아 stacker의 실패 모드를 구조적으로 회피한다. 현재 제출 승인 대기.
