# 직교축 탐색 종료 — 최종 라운드 음성 결과 (KMURecSys26 Steam, 2026-05-31 KST)

목표: 현재 best(emb128_L4_reg1e-3 4-seed 앙상블, public **0.77745**)를 넘는 새 축 확보.
결론을 먼저: **검증 가능한 모든 축이 소진**됐다. 이번 라운드의 마지막 두 베팅(MiniLM 의미축,
8-seed 확장)도 uniform 공개-대리 게이트를 통과하지 못했고, 이로써 final-2를 확정한다.

## 이번 라운드 추가 검증 (parameter-free 게이트: solo 또는 동등가중 blend가 emb128 초과해야 채택)

| 축 | 핵심 | solo (uniform) | corr_z vs emb128 | 50/50 z-blend | 판정 |
|---|---|---:|---:|---:|---|
| **MiniLM 의미 텍스트** | 사전학습 문장임베딩, user/item 리뷰 프로파일 cosine | **0.63853** | 0.461 | 0.73085 (**−0.0342**) | ❌ REJECT_WEAK |
| **8-seed 확장** | emb128 동일 config 4→8 seed raw-score 평균 (순수 분산축소) | seed별 0.7634~0.7651 | — | 8-seed=0.76465 (**−0.0004**) | ❌ TIED |

## 핵심 교훈 (직전 synthesis 재확증)

1. **텍스트 직교축도 "강+직교" 조건 실패.** MiniLM은 corr_z 0.461로 가장 직교적인 축 중 하나였지만
   solo 0.639로 popularity floor 0.684 미만. DirectAU(0.55)·TF-IDF와 동일한 실패 모드 —
   직교적이나 너무 약해 blend가 −0.034로 손해. 진단 그리드 최대(w=0.05, 0.76465)도 4-seed
   앙상블 0.76505를 못 넘으며, 가장 작은 weight에서만 미세 상승하는 전형적 stacker-trap.
2. **분산축소는 4-seed에서 이미 포화.** 신규 4개 seed의 개별 uniform 평균(0.7643)이 기존 4개
   (0.7631)보다 오히려 높았는데도 8-seed 앙상블(0.76465)이 4-seed(0.76505)를 못 넘었다.
   seed를 더 넣어도 앙상블 정확도는 평평 → 16/32 seed 확장도 무의미.
3. **CF 구조 천장 재확인.** oracle(emb64,emb128) 천장 +0.021, "neither correct" 21.4%는
   라벨 균형(coin-flip)·중간 인기 집중의 본질적 난이도. 큰 폭 개선은 이 데이터 구조상 불가능.

## 소진된 전체 축 목록

| 축 | 결과 | 비고 |
|---|---|---|
| ALS / EASE / ItemKNN (고전 CF) | redundant | corr 0.73~0.83, eq-blend 음수 |
| SGL (graph contrastive) | 0.515~0.641 | InfoNCE가 소규모 그래프 미세 ranking 파괴 |
| DirectAU (align+uniform) | 0.547~0.597 | uniformity 항이 유저별 소수 후보 ranking 파괴 |
| TF-IDF 텍스트 | 약함 | 어휘 중첩, standalone 실패 |
| MiniLM 의미 텍스트 | solo 0.639 / blend −0.034 | 직교적이나 floor 미만 |
| emb64↔emb128 z-blend | −0.0017 | corr 0.9747, BPR-LightGCN 패밀리 포화 |
| 8-seed 확장 | −0.0004 | 분산축소 4-seed에서 포화 |

## 최종 선택 (final-2, eCampus)

| 순위 | 모델 | public | uniform surrogate | 후보 파일 | SHA256 |
|---|---|---:|---:|---|---|
| #1 | emb128_L4_reg1e-3 4-seed 앙상블 | **0.77745** | 0.76505 | `candidate_lightgcn_emb128L4r3_seed_ens.csv` | `7e3191de…` (재현 확인) |
| #2 | emb64_L3_reg1e-4 4-seed 앙상블 | 0.77125 | 0.76145 | `candidate_lightgcn_seed_ens.csv` | `dcc578de…` |

두 후보 모두 재현성 패키지 검증 완료(`reports/20260530_ecampus_repro_emb128L4r3_077745.md`).
**새 제출 후보 없음** — 검증된 0.77745를 미검증 동급 후보로 교체하는 것은 리스크상 손해.

## 정직한 결론

현재 0.77745는 이 데이터의 CF 천장(oracle ~0.786)에 근접했고, parameter-free 게이트를 통과하는
직교 축은 더 이상 존재하지 않는다. 추가 GPU 탐색은 기대 이득이 noise 이하이므로 탐색을 종료하고
final-2를 확정한다.
