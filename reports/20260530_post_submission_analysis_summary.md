# 사후 분석 종합 리포트 — LightGCN 제출 (2026-05-30 KST)

본 문서는 KMURecSys26 Steam 대회에서 **LightGCN full-train 후보 제출(public 0.76245)** 이후
수행한 사후 분석 전 과정을 통합 정리한다. 모든 수치는 실제 실행 로그/리포트에 근거하며,
Kaggle 제출은 명시적 승인 1건(LightGCN)만 수행했다.

---

## 1. 제출 결과

| 항목 | 값 |
|---|---|
| 제출 파일 | `artifacts/lightgcn_20260530/test_full_train/candidate_lightgcn_full_train.csv` |
| SHA256 | `a3dbe043f0f8b781d8c35aea88b7a1f561fa7b705b34edf6c7b7d0451eceb2a6` |
| 제출 시각 | 2026-05-30 09:48 UTC (18:48 KST) |
| 상태 | `SubmissionStatus.COMPLETE` |
| **Public score** | **0.76245** |
| 이전 best (Stage2 blend) | 0.74594 |
| **Δ** | **+0.01651** |
| 일일 quota | 2/5 사용 (3 남음) |

학습 config: emb_dim=64, n_layers=3, lr=1e-3, reg=1e-4, batch=4096, epochs=200, seed=42 (V100, 1721.6s).

---

## 2. Transfer ratio 검증 (local → public)

| split | LightGCN | Stage2 | Δ |
|---|---:|---:|---:|
| random_sqrtpop | 0.6748 | 0.6597 | +0.0151 |
| recent_sqrtpop | 0.6396 | 0.6260 | +0.0136 |
| random_popbin | 0.6020 | 0.5908 | +0.0112 |
| **mean** | **0.63883** | **0.62550** | **+0.01330** |

- Public Δ +0.01651 / local mean Δ +0.01330 → **transfer ratio 1.24**
- 방향·크기 모두 local validation이 정확히 예측. 과적합·누수 징후 없음.
- 결론: validation harness가 public을 신뢰성 있게 대리(surrogate)함.

---

## 3. Diff 분석 — LightGCN vs Stage2 (전체 test pair)

- 전체 19,998행 중 라벨 불일치 3,098행 (**15.49%**)
- LightGCN이 Stage2 대비 promote(0→1) 1,549 / demote(1→0) 1,549 (균형)

### 아이템 인기 구간별 불일치율
| pop_bin | mean_pop | 불일치율 |
|---:|---:|---:|
| 0 | 19 | 12.9% |
| 1 | 34 | 18.2% |
| 2 | 65 | **23.9%** |
| 3 | 139 | 16.5% |
| 4 | 393 | 6.2% |

**해석**: 중간 인기(mean_pop≈65) 구간에서 LightGCN의 CF 신호가 가장 크게 기여.
초고인기(393)는 두 모델이 거의 동의 → popularity 신호만으로 충분.
promote된 아이템 평균 인기(69)가 demote(99)보다 낮음 → LightGCN이 덜 유명한 아이템을 더 잘 발굴.

---

## 4. 상호보완성(complementarity) 분석

| 항목 | mean (3 split) |
|---|---:|
| LightGCN 단독 | 0.63883 |
| Stage2 단독 | 0.62563 |
| 최적 고정 blend (w0.7~0.8 global-z) | 0.64106 |
| **Oracle 상한** (행별 완벽 선택) | **0.72893** |

- Oracle 상한이 LightGCN보다 **+0.09** 높음 → 행별 게이팅 여지 큼.
- 단, 고정 선형 blend는 +0.0022만 추출(oracle 갭의 극히 일부).
- crosstab: Stage2-only-right 행이 split당 1,549~2,214개 존재 → LightGCN이 놓치는 신호를 Stage2가 잡음.
- 결론: 행별 게이팅을 **학습하는 stacking 메타러너**가 다음 최고가치 실험.

---

## 5. Stacking 메타러너 — 핵심 발견

logreg/LightGBM 메타러너 (입력: LightGCN + Stage2 blend + within-user z/rank + pop/cand_count).

| split | LightGCN | 고정 blend | stack-logreg [strat] | **stack-logreg [group/정직]** |
|---|---:|---:|---:|---:|
| sqrtpop | 0.67483 | 0.67704 | 0.67944 | 0.67914 |
| recent | 0.63963 | 0.64053 | 0.64973 | 0.64923 |
| popbin | 0.60202 | 0.60562 | 0.61882 | 0.61952 |
| **mean** | 0.63883 | 0.64106 | 0.64933 | **0.64930** |

### 누수 검증 (가장 중요)
- **strat−group 누수 갭 = 3e-05** (사실상 0)
- 유저 단위 GroupKFold(같은 유저 candidate를 train/val에 절대 안 섞음)로 검증해도 게인 유지
- → **+0.01047 게인은 within-user 피처 누수가 아니라 실제 일반화 신호**
- 3개 split 모두 일관 양수: +0.0043 / +0.0096 / +0.0175

### Public 추정
- transfer ratio 1.24 적용 시 LightGCN public 0.76245 대비 약 **+0.012~0.015** 추가 기대

---

## 6. 진행 중 / 다음 단계

| 작업 | 상태 |
|---|---|
| LightGCN raw test-score 재생성 (stacker 입력용) | 🔄 진행 중 (~20:30 KST) |
| stacker 테스트 후보 materialize | ⏳ raw score 완료 대기 |
| 4-GPU 하이퍼파라미터 sweep (17 config) | 🔄 진행 중 (~01:40 KST) |

### stacker 후보 materialize 설계 (정합성)
- 메타러너(logreg)는 **검증 3-split 전체를 pool**해서 학습 → 테스트에 적용
- within-user z/rank는 스케일 불변이라 split-train→full-train base score 전이 안정적
- pooled GroupKFold OOF를 public 추정치로 함께 보고
- 제출 전 preflight(schema/ID/label/top-half/diff) + **명시적 승인** 필수

---

## 7. 재현성

- LightGCN 학습/추론: `scripts/lightgcn_train.py`, `scripts/lightgcn_fulltrain_save_scores.py`
- 분석: `scripts/post_submit_lightgcn_vs_stage2_diff.py`,
  `scripts/lightgcn_stage2_complementarity.py`, `scripts/lightgcn_stage2_stacker.py`
- sweep: `scripts/lightgcn_sweep_worker.py`, `scripts/lightgcn_sweep_aggregate.py`
- materialize: `scripts/materialize_stacker_candidate.py`
- W&B milestone: `lightgcn_public_0.76245_milestone` (project `kmu-rec-sys-26-steam`)
- 모든 리포트: `reports/20260530_*` (JSON+MD 쌍)
