# Stacker 테스트 후보 — 제출 판단 리포트 (2026-05-30 KST)

## 후보 개요

| 항목 | 값 |
|---|---|
| 후보 파일 | `artifacts/stacker_20260530/test_candidate/candidate_stacker_logreg_emb64_L3_reg1e-04.csv` (gitignore, 로컬) |
| SHA256 | `ebd69b42548f4c48651905d54bd1e985d0bbbe57a7aa3db846b6ca05fdc637a0` |
| 방법 | LightGCN + Stage2 blend feature를 입력으로 한 logreg 메타러너 (per-user top-half 디코딩) |
| 학습 | 검증 3-split 전체 pool(59,988행, 14,208 user-group)로 단일 logreg 학습 |
| 행 수 | 19,998 / 라벨 1:1 (9,999/9,999) |

## Preflight (전부 통과)

| 체크 | 결과 |
|---|---|
| 컬럼 `ID,Label` | OK |
| 행 수 == pairs.csv | OK (19,998) |
| ID 0..N-1 연속 | OK |
| 라벨 binary {0,1} | OK |
| per-user top-half (bad_users) | 0 |
| 제출 LightGCN과 row diff | 1,560 (7.80%) — 경계 내 작은 변화 |

## 정직한 성능 추정 (pooled GroupKFold OOF)

| split | LightGCN | **stacker(pooled OOF)** | Δ |
|---|---:|---:|---:|
| random_sqrtpop | 0.67483 | 0.68214 | +0.0073 |
| recent_sqrtpop | 0.63963 | 0.64803 | +0.0084 |
| random_popbin | 0.60202 | 0.61352 | +0.0115 |
| **mean** | **0.63883** | **0.64790** | **+0.00907** |

- pooled 단일 메타러너는 split별 독립 stacker(+0.0105)보다 약간 보수적(+0.0091).
  이는 split 분포를 가로질러 일반화하는 더 현실적인 테스트 시나리오이므로 **신뢰도 높은 추정**.
- 3개 split 모두 양수, 누수 갭 3e-05(이전 검증) → 일반화 신호 확정.

## 메타러너 가중치 (해석)

상위 신호: `score_blend_mean_z`(+0.87, Stage2 통합축) > `log_pop`(−0.42, 인기 과보정 억제)
> `wz_score_blend_mean_z`(−0.42) > `wz_score_lightgcn`(+0.37) > `score_lightgcn`(+0.32).

→ Stage2 blend와 LightGCN을 **동시에** 활용하되, 유저 내 정규화(within-user z)와 인기 보정을 결합.
단일 모델로는 못 잡는 행별 게이팅을 학습함.

## Public 추정

- raw OOF 게인 +0.00907
- LightGCN transfer ratio 1.24 적용 시 → 약 **+0.011** 기대
- 추정 public: **0.772 ~ 0.774** (LightGCN 0.76245 기준)
- 위험: pooled 메타러너의 split→test 전이는 LightGCN 단독보다 1단계 더 복잡. transfer ratio가 동일하리란 보장은 없음.

## 제출 판단

- 일일 quota: 2/5 사용, **3 남음**
- 본 후보는 strict 게이트(정직 OOF 3/3 양수, preflight 통과, 작은 diff, 재현성 확보) 충족
- 단, 직전 메시지의 승인은 **LightGCN 1건**에 대한 것. 신규 파일은 **명시적 재승인 필요**(protocol).

**→ 제출하려면 승인을 주세요. 제출하지 않고 sweep 결과를 기다려 더 강한 base로 stacker를 재생성하는 선택도 가능합니다.**
