# KMURecSys26 Steam — 전체 탐색 종합 (2026-05-31 KST, 최종)

best(public **0.77745**, emb128 L4 reg1e-3 4-seed LightGCN 앙상블)를 넘기 위한 전 방향 탐색의
최종 종합. 결론을 먼저: **검증 가능한 모든 축이 소진**됐고, 유일한 양성 신호(백본 capacity)는
통계적으로 유의하지 않았다. final-2는 변동 없음.

## 1. 평가 원칙 (모든 판정의 기준)

- **public 대리 = uniform-negative 검증 split.** 단일 LightGCN이 uniform 0.75445 → public
  0.76245로 전이됐고, sqrtpop(0.675)/popbin(0.602)은 전이 실패. 모든 후보는 uniform에서 게이트.
- **채택 기준 = parameter-free.** solo 또는 동등가중(50/50 z) blend가 emb128 앙상블 0.76505를
  단일-seed noise band(0.0007) 이상 초과해야 채택. grid-tuned blend weight는 stacker-trap.
- **제출은 우현의 명시적 1파일 승인 필요.** 자동 제출 없음.

## 2. 추천 알고리즘 4대 패러다임 — 전부 증거 기반 종결

| 패러다임 | 멤버 | uniform 결과 | 판정 |
|---|---|---|---|
| ① 그래프/BPR | LightGCN, SGL, SimGCL, **XSimGCL**, DirectAU, **DNS** | XSimGCL λ↑ 단조악화 0.74→0.62; DNS pool↑ 단조붕괴 0.68→0.63 | ❌ InfoNCE-uniformity가 median-4 ranking 파괴 / DNS는 popularity-skew |
| ② item-item 선형 | ALS/WMF, EASE, ItemKNN, **Turbo-CF** | Turbo-CF solo 0.742 but corr_z 0.85, blend −0.0068 | ❌ 전부 corr 0.73~0.85 redundant |
| ③ 텍스트·LM 의미 | TF-IDF, MiniLM, **AlphaRec** | solo 0.64~0.65 < floor 0.684 | ❌ 익명 gameID + 노이즈 리뷰라 본질적 약함 |
| ④ latent VAE 재구성 | **MultiVAE** | solo 0.730, corr_z 0.798(=EASE), blend −0.0085 | ❌ EASE의 비선형 사촌, redundant |

추가 배제(증거/구조): diffusion RecSys(arXiv 2505.09364가 잘튜닝 베이스라인이 초과 입증),
GBDT/logreg stacker(public 회귀 0.76245→0.75355), LLM listwise reranker(test에 텍스트 없음),
GeoCF(2410.03064; item-metadata geometry 의존 → 익명 gameID에 N/A).

## 3. 백본 capacity frontier — 유일한 양성 신호, 그러나 비유의

원래 hparam sweep은 emb256까지 돌렸지만 **하드 샘플러에서만** 평가 → public 대리(uniform) 미검증.
이를 처음으로 uniform에서 측정한 결과 **단봉 곡선**:

| config (single seed42) | uniform | vs emb128 single |
|---|---:|---:|
| emb128 | 0.76205 | 기준 |
| **emb192** | **0.76665** | **+0.00460** (정점) |
| emb256 | 0.76215 | +0.00010 |
| emb320 | 0.76145 | −0.00060 |

emb192 4-seed 앙상블 = **0.76615** vs emb128 앙상블 0.76505 = **+0.0011**(noise 1.6배). 그러나
seed42(+0.0046)는 high outlier였고 신규 3개 seed는 개별로 emb128 앙상블보다 약간 아래.
**paired McNemar**(불일치 660행, emb192 341승 vs emb128 319승, net +22): **p=0.4137 → 통계적
동전던지기.**

**실측 확정 (우현 승인 제출):** emb192 후보(SHA `1b3a6056…`)를 실제 public LB에 제출 → **0.77715**,
emb128 0.77745 대비 **−0.0003로 더 낮음.** paired McNemar 판정이 실측으로 정확히 확증됨 — surrogate
의 +0.0011은 noise였고 capacity frontier가 **실제 LB에서도 종결.** emb128이 backbone capacity
sweet spot.

## 3-b. 사후분석 파생: surrogate 신뢰 해상도 + cross-capacity 블렌드 종결

**surrogate 신뢰 해상도 ≈ Δuniform 0.003.** 전이 관계: emb64→128 Δuni +0.0036 → Δpub +0.0062
(부호 일치), 그러나 emb128→192 Δuni +0.0011 → Δpub −0.0003 (**부호 역전**). 즉 uniform Δ가 0.003
미만이면 surrogate만으로 제출 결정 불가 → 반드시 paired 검정으로 게이트.

**cross-capacity 블렌드(emb128⊕emb192) 종결.** emb128(0.77745)·emb192(0.77715)가 실측 동급이고
3.40% 행에서 다르길래 다양성 블렌드를 마지막 미검증 레버로 검증 → **corr_z(128,192)=0.9864**(거의
동일), 50/50 z-blend uniform 0.76545 = +0.0004(noise 미만), paired McNemar p=0.69(동전던지기) →
**NO_GAIN.** emb64↔128 corr 0.9747와 합쳐, BPR-LightGCN 패밀리는 차원을 바꿔도 거의 동일한 ranking을
내는 **완전 포화**가 직접 증명됨.

## 4. 근본 원인: 왜 SOTA가 안 먹히는가

2025-26 추천 SOTA는 **대규모-희소-텍스트/시퀀스** 전제. 이 대회는 정반대 — **소규모-조밀
(6710u×2437i, 165k), 유저당 median-4 1:1 균형 reranking, test에 텍스트/시퀀스 없음(익명 ID),
public이 uniform 분포 추종.** SOTA의 강점(롱테일/콜드스타트/시맨틱/hard-discrimination)이 전이될
구조적 이유가 약하다. 이것이 모든 축이 게이트를 통과 못 한 근본 배경이며, CF 천장(best-of-2
oracle ~0.786)도 SOTA 수준에서 재확증됐다.

## 5. final-2 (변동 없음)

| 순위 | 모델 | public | uniform | SHA256 |
|---|---|---:|---:|---|
| #1 | emb128 L4 reg1e-3 4-seed | **0.77745** | 0.76505 | `7e3191de…` (재현 확인) |
| #2 | emb64 L3 reg1e-4 4-seed | 0.77125 | 0.76145 | `dcc578de…` |

emb192 후보는 surrogate에서 비유의이므로 final-2 교체 근거 없음. eCampus 재현성 번들 완비.

## 6. 지속 탐색 (자동화)

주간 arXiv 모니터 cron(job_id `d9ef9fafb3d7`, 매주 월 00:00 UTC): 최신 cs.IR 추천 논문 스캔 →
소진 방법군 제거 → 이 대회 구조 부합 후보만 보고. 일회성 세션 실험이 아닌 durable 형태로
"지속적 최신논문 탐색" 구현.

## 정직한 최종 결론

검증 가능한 직교축·백본강화·SOTA 패밀리·capacity frontier를 전수 탐색했고, 유일한 양성 신호
(emb192)조차 paired 검정에서 비유의. 추가 GPU 탐색의 기대 이득은 noise 이하이므로 탐색을 종료하고,
새 신호는 주간 모니터에 위임한다. 제출 여부(emb192 일일 LB 실측)는 우현의 결정 사항으로 남긴다.
