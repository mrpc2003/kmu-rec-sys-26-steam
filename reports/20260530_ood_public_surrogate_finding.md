# OOD negative-sampler 견고성 — public surrogate 발견 (2026-05-30 KST)

## 동기

stacker가 정직한 GroupKFold OOF(+0.0091)를 통과하고도 public에서 실패(0.76245→0.75355)했다.
원인 가설은 "메타러너가 검증 negative 샘플러에 과적합"이었다. 이를 직접 검증하기 위해
단일 LightGCN(seed42, 검증된 0.76245 config)을 **negative 샘플러만 바꾼** 6개 split에서 평가했다.

## 결과: 샘플러에 따라 정확도가 0.20 출렁인다

| split (negative 샘플러) | LightGCN row_acc | per-user mean | public 0.76245 거리 |
|---|---:|---:|---:|
| **random_uniform** | **0.75445** | 0.77764 | **−0.00800** |
| random_sqrtpop (주 검증) | 0.67483 | 0.69584 | −0.08762 |
| recent_sqrtpop | 0.63963 | — | −0.12282 |
| random_popbin | 0.60202 | — | −0.16043 |
| random_communitypop | 0.57231 | 0.59145 | −0.19014 |
| recent_communitypop | 0.55551 | 0.57384 | −0.20694 |

동일 모델·동일 test pair인데 negative 구성만 바꾸면 0.555~0.754로 흔들린다.

## 핵심 결론

1. **실제 public(0.76245)은 uniform split(0.75445)에 압도적으로 가깝다.**
   히든 테스트 negative는 우리가 주력한 popularity-matched(sqrtpop/popbin)가 아니라
   **uniform에 가까운 분포**다.

2. **stacker 실패가 완전히 설명된다.** 메타러너는 sqrtpop/popbin/recent의 어려운
   인기-매칭 negative를 가르도록 popularity 보정(`log_pop` 가중 −0.42)을 학습했다.
   실제 test는 uniform에 가까워 그 보정이 역효과를 냈다.

3. **transfer ratio 1.24의 정체.** 신비한 증폭이 아니라, 주 검증(sqrtpop 0.675)이
   실제 test보다 단순히 더 어려웠을 뿐이다. local→public 갭은 난이도 차이였다.

## 검증 전략 변경 (이후 적용)

- **uniform split을 1차 public surrogate로 채택.** 후보의 public 방향은 uniform에서 먼저 본다.
- popularity-matched split(sqrtpop/popbin/communitypop)은 **stress 하한**으로만 사용
  (최악 negative 분포에서의 robustness 확인용).
- 후보를 신뢰하려면: uniform에서 개선 + 모든 샘플러에서 회귀 없음(robustness).
  단일 샘플러(특히 인기-매칭) OOF만으로 메타러너/blend를 제출하지 않는다.
- 이것이 stacker 실패가 가르쳐준, GroupKFold만으로 부족했던 정답 게이트다.

## 자산

- 리포트: `reports/20260530_lightgcn_ood_robustness.{json,md}`
- score: `artifacts/lightgcn_ood_robustness/{split}/lightgcn_scores.csv`
- 단일 LightGCN seed42, emb64 L3 reg1e-4 200ep (제출 0.76245와 동일 config)
