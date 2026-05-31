# Sisyphus 새 방향 탐색 — geometry & ceiling-proof 축 (KMURecSys26 Steam, 2026-06-01 KST)

**역할:** 경쟁 전략 adviser (validation-only 설계 제안, 실행/제출은 Hermes 게이트 후).
**기준 best:** emb128 L4 reg1e-3 4-seed LightGCN, uniform **0.76505** / public **0.77745**.
**원칙:** uniform = public surrogate, parameter-free 게이트, 3-split 패널(seed 42/7/123) + paired McNemar, noise-chasing 금지.

결론 먼저: **음성 리스트는 견고하다. 그러나 "구조적 천장" 주장에는 아직 측정되지 않은 두 개의 빈틈이 있다.** 모든 종결 실험은
(1) **loss를 바꿨거나**(SGL/DirectAU/XSimGCL → InfoNCE-uniformity 붕괴), (2) **유클리드 내적 기하를 공유했거나**(EASE/Turbo-CF/MultiVAE/ALS/capacity → corr 0.73~0.99 redundant),
(3) **비예측 축을 더했다**(temporal corr −0.04, hours, candidate-marginal popularity trap). 한 번도 검증되지 않은 것은 **"강한 ranking loss를 유지한 채 결정 기하(decision manifold)만 바꾸는"** 단일 레버, 그리고 **천장 주장 자체의 실측**이다. 이 둘이 이 라운드의 표적이다.

---

## Part A — first-principles 재구성: 진짜 벽은 어디에 있는가

### A.1 이 대회의 결정이론적 정체 (왜 현재 학습이 이미 거의 최적인가)

대회 결정은 **유저별 closed candidate set에서 정확히 K_u개 선택**이고, 채점은 set-accuracy다. 핵심 사실 4개를 결합하면:

- public test의 negative는 **uniform-unseen 분포**를 추종한다 (OOD finding: public 0.76245 ↔ uniform 0.75445, sqrtpop 0.675 전이 실패).
- `lightgcn_train.py`의 BPR sampler는 **정확히 uniform-unseen negative**를 뽑는다 (`neg = rng.integers(0, n_items)`, line 112).
- per-user top-half 디코딩(`predict_tophalf`)은 monotone 변환·calibration을 무력화한다 (GPT-5.5 Pro가 코드로 입증).
- 따라서 within-user에서 "실제 play vs uniform-random-unseen"을 가르는 분류기 = **그 negative 분포에 대한 likelihood-ratio 분류기**인데, BPR-uniform이 바로 그것이다.

→ **현재 백본은 "uniform-negative 분포에 대한 Bayes-optimal 스코어러"에 임베딩 용량이 허용하는 한 이미 도달**해 있다. 이것이 transfer ratio가 안정적이고, decision-rule 레버(exact-K, candidate-marginal, temporal)가 전부 0이거나 음수인 이유의 통합 설명이다. **남은 헤드룸은 두 곳뿐이다: (i) 더 나은 임베딩(=다른 기하), (ii) 잔차가 noise임을 증명(=천장 확정).** 그 외 모든 레버는 닫혔다.

### A.2 "포화" 주장의 load-bearing 가정 — 그리고 그 가정의 빈틈

종결 리포트들의 핵심 주장은 *"이 데이터의 강한 신호는 전부 co-occurrence이고, co-occurrence를 잡는 모든 모델은 같은 ranking으로 수렴한다(corr 0.97+)"* 이다. 그런데 이 명제가 실측으로 입증된 범위는 **유클리드 내적 패밀리에 한정**된다:

| 검증된 모델 | 기하/연산 | corr_z vs emb128 |
|---|---|---|
| LightGCN emb64/128/192/256/320 | 유클리드 내적, 대칭정규화 propagation | 0.9747~0.986 |
| EASE / Turbo-CF / ItemKNN / ALS / MultiVAE | 유클리드 item-item 선형/내적 | 0.73~0.85 |
| SGL / DirectAU / XSimGCL | 유클리드 + InfoNCE-uniformity (loss 변경) | 0.16~0.46 (직교하나 solo 붕괴) |

**한 번도 측정되지 않은 것:** "강한 ranking loss(BPR/triplet)는 유지하되 **결정 기하 또는 aggregation 연산자만** 비유클리드/비내적으로 바꾼 모델." 직교성을 만든 기존 시도(SGL/DirectAU)는 전부 **loss를 InfoNCE로 바꿔** solo 강도를 0.55로 무너뜨렸다. 기하만 바꾸고 ranking loss를 지키면 — solo 강도(~0.76)를 유지하면서 boundary 재배치가 달라질 수 있다. 이것이 "강+직교" 조건의 유일하게 미검증된 경로다.

### A.3 천장 주장의 두 번째 빈틈 — 측정되지 않은 boundary 정보

gate-floor PART C는 "신규 직교 covariate로 boundary 오류 환원 불가 = 구조적 천장"을 결론냈다. 그러나 코드(`gate_floor_bootstrap_analysis.py` line 228, 239)를 보면 boundary에서 **단 3개 covariate만** 테스트했다:

- `d_pop` (marginal popularity): AUC 0.664 — popularity trap (이미 닫힘)
- `d_ov` (date overlap): AUC 0.510 — 무신호
- `d_score` (모델 자기 점수차): AUC 0.741 — 순환

그리고 `NOVEL_ORTHOGONAL = ["d_ov"]`로 **날짜 하나만** "신규 직교"로 하드코딩했다. **테스트되지 않은 boundary covariate:** ① 유저 학습이력과 후보의 **비모수 conditional co-occurrence/Jaccard**(marginal popularity가 아닌, "이 유저의 게임들과 함께 플레이되는가"), ② **유저 최근접 이웃(kNN) consensus vote**(이 후보가 유사 유저들의 play-set에 더 자주 등장하는가), ③ **임베딩 norm/곡률 반경**. 또한 `neither_correct_diagnosis.py`는 작성돼 있으나 **실행 로그·리포트가 어디에도 없다** — 21.4% "neither-correct"가 intrinsic random인지(=진짜 천장) low-pop/low-degree에 집중된 구조적 오류인지(=cold-start-aware 헤드룸) **아직 측정 안 됨**. 천장 전체가 이 미실행 스크립트 위에 서 있다.

> **요지:** 음성 리스트는 닫혔다. 하지만 "더 짤 게 없다"는 결론은 (A.2) 비유클리드 기하와 (A.3) boundary 정보 실측이라는 두 미검증 전제에 의존한다. 정직한 다음 라운드는 이 둘을 표적한다.

---

## Part B — 방향 우선순위 (uniform 게이트 통과 가능성 × 음성 리스트 직교성)

### 🥇 방향 1 (최우선·CPU·결정적): 천장 실측 — neither-correct 진단 + boundary covariate 확장

**핵심 아이디어.** GPU를 쓰기 전에, "천장이 진짜인가"를 먼저 측정한다. 두 미실행 측정을 완료한다:
(a) `neither_correct_diagnosis.py`를 실제로 돌려 21.4% 동시-오답 행이 item-pop/user-degree/candidate-set-size에 평탄한지(→ intrinsic, 천장 확정) 저활동 구간에 집중되는지(→ 구조 헤드룸) 판정.
(b) `gate_floor` PART C를 **미검증 covariate 3종**으로 확장: conditional co-occurrence(Jaccard·BM25, popularity-residualized), kNN-consensus vote, embedding-norm gap.

**음성 리스트와 다른 점.** 이것은 새 모델이 아니라 **천장 주장의 증거 보강**이다. 기존 PART C는 marginal popularity(trap)·date(dead)·self-score(circular)만 봤다. conditional co-occurrence는 marginal popularity와 다르다 — popularity로 residualize한 뒤에도 boundary AUC가 0.5를 넘으면, "유저-내부 co-occurrence"라는 **popularity-trap이 아닌** 신호가 boundary에 존재한다는 직접 증거이고, 그건 EASE의 full-ranking redundancy(corr 0.83)와 별개 질문이다(전역 redundant ≠ boundary 무신호).

**uniform 게이트에서 개선이 나올 근거.** 직접 채택을 노리는 게 아니라 **go/no-go 게이트**다. (a)가 평탄+(b)가 전부 AUC≈0.5 → **천장 확정, GPU 중단, final-2 유지**(정직한 종결). (b)에서 conditional-cooc 또는 kNN-consensus가 AUC≥0.55 → boundary에 미활용 직교 신호 존재 → 방향 2/3로 escalate할 정당성 확보.

**검증 설계 (4×V100 불필요, CPU-only, ~10분).**
1. `python scripts/neither_correct_diagnosis.py` 실행 → `reports/`에 결과 저장. 판정: neither-rate가 pop/deg quintile 간 range < 0.05면 "intrinsic flat", > 0.10이면 "structured".
2. `gate_floor_bootstrap_analysis.py`를 복제해 PART C에 covariate 추가:
   - `d_cooc`: 후보 i에 대해 `Σ_{g∈hist_u} cooc(i,g)/sqrt(pop(i)pop(g))` (BM25 정규화), 그 뒤 `log_pop`에 대해 OLS-residualize → popularity 성분 제거.
   - `d_knn`: 유저 u의 임베딩-cosine top-50 이웃의 fold_train play-set에서 i의 빈도.
   - `d_norm`: `||emb_i|| − ||emb_j||` (곡률 대리).
   각각 boundary pair(n≈4037)에서 "어느 쪽이 positive인가" 분리 AUC 측정.
3. **판정 기준:** novel covariate(d_cooc_resid, d_knn) 중 하나라도 |AUC−0.5|≥0.05 & bootstrap CI가 0.5 미포함 → "HEADROOM_EXISTS". 전부 <0.03 → "CEILING_CONFIRMED".

**성공/실패.** 성공 = 명확한 go/no-go 신호. 실패 모드 없음(측정 자체가 산출물). 이 방향은 **반드시 방향 2/3보다 먼저** 실행해 GPU 낭비를 막는다.

**레퍼런스.** Ju et al., *How Does Message Passing Improve Collaborative Filtering?* (TAG-CF), NeurIPS 2024, arXiv:2404.08660 — graph-CF 이득의 상당 부분이 forward-pass 이웃 집계(=비모수 co-occurrence smoothing)임을 보임 → boundary에서 비모수 cooc를 따로 보는 동기.

---

### 🥇 방향 2 (최우선 GPU 베팅): 하이퍼볼릭/Lorentz CF — ranking loss 유지, 기하만 교체

**핵심 아이디어.** 백본 학습 신호(co-occurrence 그래프 + BPR/triplet ranking loss)는 **그대로** 두고, 임베딩 공간을 유클리드 → **Lorentz/Poincaré 하이퍼볼릭**으로 바꾼다. 스코어는 내적이 아니라 **음의 측지거리** `−d_L(u,i)`. 학습 loss는 InfoNCE가 아니라 **하이퍼볼릭 margin/triplet**(BPR의 하이퍼볼릭 대응)이라 solo 강도가 유지된다.

**왜 이 대회 구조에 맞는가.** ① 게임 popularity Gini 0.52 — 강한 계층(소수 head + 긴 tail). 하이퍼볼릭 공간은 **계층을 반경(norm)으로 자연 인코딩**해, head는 원점 근처·niche는 경계로 배치한다. ② public이 추종하는 **uniform negative는 tail로 치우친다**(uniform 후보 item-pop median 38 vs 실제 후보 62) — tail 변별이 더 중요한 분포다. 유클리드 내적은 tail에서 norm-collapse로 변별이 약하지만, 하이퍼볼릭 거리는 경계 근처에서 거리가 지수적으로 팽창해 tail item 간 분해능이 높다.

**왜 음성 리스트·corr 0.98 포화와 다른가 (메커니즘).** 닫힌 직교 시도(SGL/DirectAU/XSimGCL)는 전부 **loss를 InfoNCE-uniformity로 바꿔** median-4 ranking을 파괴했다(solo 0.55). 닫힌 강한 시도(EASE/Turbo-CF/ALS/capacity)는 전부 **유클리드 내적 기하를 공유**해 corr 0.73~0.99. 하이퍼볼릭은 **둘 다 아니다** — ranking loss를 유지(→ solo 강도 기대 ~0.76)하면서 결정 manifold가 근본적으로 다르다(→ corr 하락 기대). 곡률 κ는 mid-pop boundary item의 상대 순위를 유클리드와 다르게 재배치하되, 그것이 popularity down-weighting(=trap)이 **아니다**: 곡률 효과는 단조 인기 가중이 아니라 계층 깊이에 따른 비단조 재배치이므로 candidate-marginal(λ=1.0 −0.0195)의 실패 모드를 구조적으로 회피한다.

**uniform에서 parameter-free 개선이 나올 수 있는 근거.** HRCF/HICF(2021-22) 계열은 표준 벤치마크에서 LightGCN 대비 **tail/cold 구간에서 일관된 이득**을 보였다. 이 대회의 uniform-negative는 정확히 tail-heavy 평가다. solo가 floor(0.684)를 크게 넘고 corr_z가 0.9 미만이면, 동등가중 z-blend가 emb128 0.76505를 noise(0.0007) 이상 넘을 수 있다(parameter-free 채택 기준). best-of-2 oracle 천장 +0.021 안에서 현실적 blend 이득 +0.003~0.008을 노린다.

**검증 설계 (4×V100, 1~1.5일).**
1. **구현:** `scripts/lightgcn_hyperbolic.py` 신규(기존 `lightgcn_train.py` 구조 미러). `LightGCN.forward`의 propagation 후 임베딩을 Lorentz manifold로 expmap, 스코어 = `−d_L`(또는 Poincaré). loss = 하이퍼볼릭 triplet (margin m, uniform negative — 백본과 동일 sampler). `geoopt`(공개·MIT) RiemannianAdam 사용, 또는 TriplH 공개 구현 차용.
2. **차원:** 하이퍼볼릭은 저차원에서 효율적 → emb {32, 64, 128} 스윕(단일 seed42, uniform split). solo<0.684면 즉시 reject(InfoNCE류 운명 재확인).
3. **게이트 (sgl_gate.py 패턴 그대로):** solo accuracy, `corr_z` vs emb128, **동등가중 50/50 z-blend**. 채택 조건: solo>0.684 AND (solo>0.76505+0.0007 OR eq-blend>0.76505+0.0007). corr_z<0.9면 "직교 확보" 플래그.
4. **승격:** eq-blend가 1차 게이트 통과 시 → 4-seed 앙상블 → **3-split 패널**(`split_panel_aggregate.py` 확장, seed 42/7/123) + **paired McNemar**(gate_floor PART B 방식). between-split std 0.0027를 넘는 견고한 Δ만 채택.
5. **성공/실패 판정:** 성공 = 3-split 전부에서 eq-blend가 emb128 앙상블 초과 & McNemar p<0.05 & paired Δ가 MDE(0.00355) 초과. 실패 = solo<floor(즉시 종결) 또는 eq-blend≤noise(직교했으나 약함 → "geometry도 종결"이라는 강한 음성 확보, 그 자체로 가치).

**레퍼런스.**
- Yusupov, Rakhuba, Frolov, *Leveraging Geometric Insights in Hyperbolic Triplet Loss for Improved Recommendations* (TriplH), RecSys 2025, arXiv:2508.11978 — Lorentz triplet loss + adaptive item-item margin (이 설계의 직접 템플릿, 공개 코드).
- Yang et al., *Hgformer: Hyperbolic Graph Transformer for Collaborative Filtering*, ICML 2025 — 하이퍼볼릭 graph conv + cross-attention, interaction-only.
- Sun et al., *HICF / HRCF: Hyperbolic-aware / Hyperbolic Regularized CF*, WWW/WSDM 2021-22 — tail 이득의 사전 증거(보수적 기대치 근거).

---

### 🥈 방향 3 (2차 베팅): ranking-native graph attention (Rankformer) + 테스트타임 집계(TAG-CF)

**핵심 아이디어.** 대칭정규화 고정 propagation(LightGCN) 대신 **ranking objective에서 유도된 attention** 집계로 임베딩을 만든다. attention 가중은 유저별 이웃 중요도를 학습해, 모든 이웃을 평균하는 LightGCN propagation이 뭉개버리는 유저-특이 신호를 보존할 수 있다.

**음성 리스트와 다른 점.** 새 encoder는 GPT-5.5 Pro가 "소진 동의"했으나, 그 판단은 **graph contrastive(SGL/SimGCL/XSimGCL) 한정**이었다(InfoNCE 메커니즘 공유). Rankformer는 contrastive가 아니라 **ranking-loss 유도 attention**으로, 연산자(aggregation)가 다르다. 단 솔직히 직교성 리스크는 방향 2보다 높다 — 여전히 유클리드 내적 + co-occurrence라 corr가 높게 나올 수 있다(중간 신뢰).

**uniform 개선 근거.** TAG-CF(NeurIPS'24)는 graph 이득의 대부분이 forward-pass 집계임을 보이며 이를 **테스트타임에 parameter-free로** 옮긴다. 이는 거의 무비용 probe다(학습 불필요): 동결된 emb128에 테스트타임 이웃 집계를 1-스텝 추가 → eq-blend 게이트. 단 백본이 이미 L4 propagation이라 redundant일 가능성이 높다 → **저비용으로 "다른 집계도 redundant"를 빠르게 확정**하는 가치.

**검증 설계.**
1. **TAG-CF probe (CPU/GPU 경량, ~30분):** 동결 emb128에 테스트타임 1-hop 집계 적용 → uniform solo & eq-blend. corr_z·eq-blend가 noise 내면 즉시 "redundant" 종결.
2. **Rankformer (GPU, ~1일, TAG-CF가 신호 보이면만):** 공개 구현(arXiv:2503.16927) emb128 매칭, uniform negative, BPR-rank loss. 게이트는 방향 2와 동일(solo>floor, eq-blend, corr_z, 3-split + McNemar).
3. **성공/실패:** TAG-CF eq-blend≤noise면 Rankformer 생략 권고(연산자 축도 redundant). TAG-CF가 corr_z<0.9 & solo 유지면 Rankformer 본실험 정당.

**레퍼런스.**
- Ju et al., TAG-CF, NeurIPS 2024, arXiv:2404.08660 (공개 코드 snap-research/Test-time-Aggregation-for-CF).
- Chen et al., *Rankformer: A Graph Transformer for Recommendation based on Ranking Objective*, WWW 2025, arXiv:2503.16927 (공개 코드).

---

### ❌ 방향 4 (정직한 종결): decision-rule / global-assignment 레이어 — 구조적으로 닫힘

**왜 더 짤 게 없는가.** A.1에서 보였듯 채점은 per-user 독립 + within-user ranking이다. 따라서 (i) cross-user calibration은 within-user 순위에 영향 없음(무의미), (ii) item-level prior(candidate-marginal λ=1.0)는 user-내부 결정과 불일치해 popularity trap(−0.0195), (iii) set-level exact-K loss는 K=1이 45%라 net-zero(Δ=+0.00000, p=0.934). 세 변종이 서로 다른 각도에서 같은 벽을 쳤다. global optimal-transport 할당(GORACS류)도 결국 item/그룹 prior라 (ii)로 환원된다. **이 축은 3중으로 닫혔다 — 재시도 금지.** (단 방향 1의 boundary 측정에서 conditional-cooc가 신호를 보이면, 그건 decision-rule이 아니라 더 나은 임베딩(방향 2)의 근거로 해석한다.)

---

### ❌ 방향 5 (정직한 종결): seed/capacity/contrastive-loss 패밀리 — 실측 포화

**왜 더 짤 게 없는가.** seed 4→8 (0.76465, tied), capacity emb192/256/320 (단봉, emb192 public 0.77715<0.77745 실측), cross-capacity blend (corr 0.986, McNemar p=0.69), SGL/SimGCL/XSimGCL/DirectAU (InfoNCE solo 붕괴), Turbo-CF/MultiVAE (corr 0.78~0.85). 분산축소·용량·유클리드-contrastive는 전부 게이트 안. **재시도 금지.** 방향 2/3가 실패하면 여기로 회귀하지 말고 final-2를 확정한다.

---

## Part C — 우선순위 종합 매트릭스

| # | 방향 | 직교성 (vs 음성) | uniform 통과 가능성 | 비용 | 우선순위 |
|---|---|---|---|---|---|
| 1 | 천장 실측 (neither-correct + boundary covariate 확장) | N/A (천장 증거) | go/no-go 게이트 | CPU ~10분 | **즉시 (선행 필수)** |
| 2 | 하이퍼볼릭/Lorentz CF (TriplH) | **높음** (기하 교체, loss 유지) | 중 (tail-heavy uniform에 정합) | 4×V100 ~1.5일 | **높음 (방향1이 HEADROOM이면)** |
| 3 | Rankformer + TAG-CF (다른 집계 연산자) | 중 (연산자 다름, 내적 공유) | 중하 | TAG-CF ~30분 → Rankformer ~1일 | 중 (방향2 후 또는 병행 probe) |
| 4 | decision-rule / global assignment | — | **0 (3중 종결)** | — | **종결, 재시도 금지** |
| 5 | seed/capacity/contrastive | — | **0 (실측 포화)** | — | **종결, 재시도 금지** |

**권고 실행 순서:**
1. **방향 1을 먼저** 실행(CPU). `CEILING_CONFIRMED`면 → **정직하게 여기서 멈추고 final-2(0.77745/0.77125) 확정.** GPU 추가 투입 근거 없음.
2. 방향 1이 `HEADROOM_EXISTS`(conditional-cooc 또는 kNN-consensus가 boundary에서 above-chance)면 → **방향 2(하이퍼볼릭)에 GPU 1.5일** 투입. 병행으로 방향 3의 TAG-CF probe(30분)로 "연산자 축도 redundant인가"를 싸게 확정.
3. 방향 2가 3-split 패널 + McNemar를 통과하면 final-2 교체 후보로 Hermes 게이트. 통과 못 하면 → "geometry 축도 종결"이라는 가장 강한 음성을 확보하고 final-2 유지.

## Part D — 정직한 메타 판단

이 대회는 emb128 0.77745가 CF oracle 천장(~0.786)에 근접했고, 음성 리스트는 견고하다. 내가 제시하는 것은 "큰 폭 개선"의 약속이 **아니다** — 현실적 상방은 +0.003~0.008(oracle 천장 제약). 그러나 정직한 종결을 위해서는 두 미측정 전제를 닫아야 한다: **(A.2) 비유클리드 기하가 corr 0.97 포화를 깨는가, (A.3) boundary 잔차가 정말 intrinsic random인가.** 방향 1은 후자를 CPU로 즉시 판정하고, 방향 2는 전자를 검증된 게이트로 시험한다. 둘 다 음성이면 — 그때 비로소 *"이 데이터 구조에서 더 짜낼 것이 없다"*는 결론이 **두 미검증 전제까지 닫은** 완결된 종결이 된다. 근거 없는 낙관이 아니라, 종결을 완성하기 위한 마지막 두 측정이다.

SISYPHUS_DIRECTIONS_DONE: reports/20260601_sisyphus_new_directions.md
