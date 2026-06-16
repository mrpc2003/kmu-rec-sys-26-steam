# KMURecSys26 Steam 진행 현황 및 정체 지점 정리

- 작성 시각: 2026-06-12 19:01 KST
- 대상 저장소: `/opt/data/kaggle/kmu-rec-sys-26-steam`
- 대회: Kaggle `kmu-rec-sys-26-steam`
- 태스크: `userID, gameID` pair의 played 여부 이진 예측, Accuracy 평가
- 정리 범위: 로컬 repo의 `README.md`, `reports/`, `state/`, 최근 cron/OpenCode 결과, 현재 프로세스·cron 상태

## 1. 한 줄 결론

현재는 **새 후보를 만들 능력이 부족해서 멈춘 것이 아니라, 검증 가능한 거의 모든 독립 축을 돌려 본 뒤 strict gate를 통과하는 새 축이 더 나오지 않아 정체된 상태**다. Public LB 기준 최고점은 `candidate_rank_blend_emb128_emb192.csv`의 **0.77825**이고, 가장 깔끔한 재현형 백본은 emb128 LightGCN 4-seed ensemble의 **0.77745**다. 이후 residual/OTTO/ALS/boundary/TAGCF류 후보들은 일부 validation 양성 신호가 있었지만, fresh panel이나 실제 public에서 현재 best를 넘지 못했다.

## 2. 현재 최고 제출 및 기준점

| 구분 | 파일/모델 | Public | 핵심 해석 |
|---|---|---:|---|
| 현재 public best | `candidate_rank_blend_emb128_emb192.csv` | **0.77825** | emb128+emb192 rank-blend. Public은 +0.00080 vs emb128였지만, 제출 전 validation McNemar는 비유의라 “강한 증명”보다는 public-LB 정보/hedge에 가깝다. |
| 가장 안정적인 재현 백본 | `candidate_lightgcn_emb128L4r3_seed_ens.csv` | **0.77745** | emb128 L4 reg1e-3, 4 seeds(42/123/2024/7), raw-score mean, per-user top-half. SHA `7e3191de…`, 재현 스크립트로 byte-identical 검증된 기준 모델. |
| 이전 강한 기준 | `candidate_lightgcn_seed_ens.csv` | 0.77125 | emb64 L3 reg1e-4 4-seed. emb128로 넘어가며 public +0.00620 개선. |
| 첫 LightGCN anchor | `candidate_lightgcn_full_train.csv` | 0.76245 | 단일 emb64 LightGCN. uniform validation이 public과 맞는다는 판단의 출발점. |
| Stage2 CF blend | `candidate_score_blend_mean_z.csv` | 0.74594 | ItemKNN/EASE/ALS 계열 첫 제출. 이후 graph backbone이 우세함을 확인. |

현재 public best인 rank-blend는 점수만 보면 가장 높다. 다만 검증 통계는 emb128 단독보다 약하다. 최종 2개 제출을 고를 때는 “public 최고점”과 “재현·검증 안정성”을 따로 보고 결정해야 한다.

## 3. 지금까지 구축된 검증 체계

### 3.1 데이터 구조 파악

- train: 약 175,000 rows, 6,710 users, 2,437 games.
- test pairs: 19,998 rows, 4,737 users, 2,429 games.
- cold user/item row는 0으로 확인됐다.
- 모든 `pairs.csv` user의 candidate 수가 짝수이고, 구조상 전체 positive는 9,999개로 맞춰진다.
- 따라서 제출은 거의 모든 실험에서 **per-user top-half decoding**으로 변환했다. 각 user별 후보 중 상위 절반을 `Label=1`로 두는 방식이다.

### 3.2 validation split

초기에는 `sqrtpop` negative가 실제 candidate marginal과 맞아 보였지만, 제출 후 public LB와 비교하면서 결론이 바뀌었다.

- Public LB surrogate는 **uniform-negative validation split**이 가장 잘 맞았다.
- `sqrtpop`, `popbin`, `recent`는 primary gate가 아니라 robustness/stress split으로 낮췄다.
- 이유: actual pairs는 positive 50% + negative 50% 혼합이라 candidate 전체 marginal만 맞추면 negative 분포를 잘못 추정한다.
- 단일 LightGCN과 seed ensemble에서 uniform Δ가 public Δ로 잘 전이됐고, hard sampler 양성 신호는 public에서 자주 깨졌다.

### 3.3 strict gate

후반부 후보는 대체로 아래 기준으로 걸렀다.

- 3-split uniform panel에서 mean Δ가 충분히 커야 함. 후반 기준은 대체로 **+0.0015 이상**.
- split별 부호가 안정적이어야 함. 최소 split Δ가 음수면 탈락.
- positive split이 3/3이어야 함.
- paired exact/McNemar류 p-value가 유의해야 함.
- 이미 public에서 negative transfer가 난 계열은 quarantine 처리.
- full-test candidate/submission CSV는 strict gate 없이 만들지 않음. 단, 우현이 명시적으로 “한 번 태워보자”고 한 forced/manual-risk 제출은 별도 기록.

## 4. 주요 탐색 흐름

### 4.1 Stage2 CF baseline

초기에 ItemKNN BM25, EASE, ALS 계열 score와 z/rank/RRF blend를 만들었다. 첫 제출 `score_blend_mean_z`는 public 0.74594로 baseline보다 개선됐지만, 이후 LightGCN 계열에 크게 밀렸다.

의미는 있었다. 제공 baseline을 재현했고, per-user top-half 평가·검증 harness·후보 CSV preflight를 만들었기 때문이다. 이후 모든 실험은 이 harness 위에서 비교됐다.

### 4.2 LightGCN backbone

가장 큰 도약은 LightGCN이었다.

- 단일 emb64 LightGCN: public 0.76245.
- emb64 4-seed ensemble: public 0.77125.
- emb128 L4 reg1e-3 4-seed ensemble: public 0.77745.

이 구간에서 “public이 uniform split을 따라간다”는 경험적 근거가 강해졌다. emb128 4-seed는 현재까지도 가장 안정적인 백본으로 남아 있다.

### 4.3 capacity frontier

emb128 이후 더 큰 embedding을 실험했다.

| config | uniform 관찰 | public 관찰 | 판정 |
|---|---:|---:|---|
| emb128 4-seed | 0.76505 | 0.77745 | 기준 sweet spot |
| emb192 4-seed | 0.76615 | 0.77715 | uniform +0.0011은 noise였고 public은 -0.0003 |
| emb256/320 | emb192보다 낮음 | 제출 가치 없음 | capacity 초과/포화 |
| emb128⊕emb192 rank blend | validation은 약함 | 0.77825 | public best이지만 통계적 확신은 낮음 |

결론은 “차원을 더 키우면 이긴다”가 아니었다. emb192의 uniform 이득은 paired McNemar에서 비유의였고, 실제 public도 emb128보다 낮았다. rank-blend만 public에서 +0.00080을 얻었지만, 이 역시 private 우위의 강한 증거라고 보기는 어렵다.

### 4.4 stacker/GBDT/logreg류

LightGCN + Stage2 score를 feature로 넣은 logreg stacker는 OOF에서는 좋아 보였지만 public에서 크게 회귀했다.

- 제출: `candidate_stacker_logreg_emb64_L3_reg1e-04.csv`
- Public: 0.75355
- 당시 anchor 0.76245 대비 -0.00890
- 원인 가설: validation negative sampler artifact와 popularity down-weighting을 학습한 것. public의 near-uniform 구조에는 맞지 않았다.

이 실패 이후 meta-learner, validation-label 기반 gate, 과한 weight tuning은 강하게 경계하게 됐다.

### 4.5 논문/SOTA 계열

다음 계열을 넓게 검토하거나 직접 probe했다.

- LightGCN variants: SGL, DirectAU, DNS, xSimGCL류.
- item-item/linear: EASE, ALS/WMF, ItemKNN, Turbo-CF류.
- text/semantic: review TF-IDF, embedding 기반 semantic residual, item text/date/hour 통계.
- VAE/재구성: MultiVAE류.
- sequence/sequential: SASRec류는 set-membership/top-half 구조와 맞지 않아 약함.
- geometry: hyperbolic/Lorentz probe.
- K-aware/boundary: SL@K-lite, boundary vote, jackknife uncertainty, exact-K/boundary feature factory.
- full-test aggregation: TAGCF, boundary scoreblend, frontier z-blend.

대부분은 다음 둘 중 하나였다.

1. solo 성능이 LightGCN보다 약함.
2. LightGCN과 score/rank 상관이 너무 높아 blend 이득이 없음.

특히 hyperbolic geometry는 Euclidean LightGCN과 corr_z가 0.747로 꽤 달라 보였지만, solo 0.71734로 약했고 50/50 z-blend가 emb128을 -0.017 끌어내렸다. “다른 모델”이어도 “도움 되는 모델”은 아니었다.

## 5. 제출·자동화 이력

### 5.1 자동 제출 정책 변화

중간에 aggressive autorunner가 검증 양성 후보를 빠르게 제출하는 방향으로 바뀌었다. 이후 우현의 피드백에 맞춰 다음 guard를 강화했다.

- 동일 SHA candidate 재제출 금지.
- row diff가 너무 작은 near-duplicate 금지.
- 같은 family가 public에서 current best를 못 넘으면 quarantine.
- 5개를 몰아서 태우는 rapid quota burn 금지.
- 제출 후 post-analysis 필수.

### 5.2 public에서 확인된 실패 계열

| 계열 | 대표 public | current best 대비 | 해석 |
|---|---:|---:|---|
| ALS residual rankblend | 0.77785~0.77805 | -0.00020~-0.00040 | validation +0.0013대였지만 public 전이 실패. 같은 family quarantine. |
| boundary scoreblend | 0.77575~0.77755 | -0.00070~-0.00250 | boundary 후보는 public에서 강하게 깨짐. |
| frontier z-blend | 0.77715 | -0.00110 | capacity frontier 파생도 current best 못 넘김. |
| TAGCF full-test | 0.77615 | -0.00210 | validation 약양성 대비 public negative transfer. |
| forced OTTO | 0.77815 | -0.00010 | emb128 대비는 +0.00070였지만 current best rank-blend는 못 넘음. |

## 6. 현재 막힌 지점

### 6.1 새 독립 신호가 안 남았다

가장 큰 병목은 구현력이 아니라 **신호 포화**다. graph co-occurrence, popularity, item similarity, hours/text/date 통계가 이미 emb128 LightGCN의 ranking에 대부분 흡수되어 있다. feature-only AUC가 있어 보여도 base score와 상관이 높고, log-pop residual boundary에서 새 이득이 사라진다.

### 6.2 validation 양성 신호가 public/current-best를 못 넘는다

후반부에 나온 양성 신호들은 대부분 너무 작았다.

- rank-blend current-best residual atlas는 same-panel에서는 strict diagnostic pass처럼 보였다.
- 그러나 fresh independent confirmation에서 pre-registered row는 mean Δ +0.000800, min Δ -0.000300, positive 2/3, p=0.129로 strict fail.
- diagnostic best row도 mean Δ +0.001134로 +0.0015 기준에 못 미쳤다.

즉 “아예 신호가 없다”기보다는 “제출권을 쓸 만큼 안정적인 신호가 없다”가 정확하다.

### 6.3 stalled/incomplete probe가 있다

마지막 cron/OpenCode 점검에서 미완료로 남은 축은 두 개다.

| probe | 상태 | 막힌 증상 | 처리 |
|---|---|---|---|
| UserKNN gated residual fine-grid | `STALLED_INCOMPLETE` | 최종 JSON/MD report가 없고, 로그는 invalid-value warning 반복 뒤 metric 없이 끝난 상태 | 새 독립축이 아니므로 재실행하지 않음 |
| Jackknife uncertainty boundary expanded | `FAILED_INCOMPLETE_NO_METRIC_REPORT` | expanded report 파일이 없고, 로그는 12줄에서 seed123 invalid-value warning 뒤 종료 | smoke/probe 자체도 weak라 확장 재실행 근거 부족 |

이 둘은 “계산을 더 오래 돌리면 큰 후보가 나올 것”이라기보다, 이미 약하거나 중복된 방향에서 수치적으로 불안정한 채 멈춘 항목이다.

### 6.4 no-submit discovery loop도 새 축을 못 찾았다

마지막 `KMURecSys26 Steam OpenCode no-submit improvement-axis loop` 결과는 다음과 같다.

- 마지막 실행: 2026-06-07 18:25~18:33 KST 근처.
- verdict: `NO_SAFE_AXIS`.
- 새 validation-only probe launch: false.
- Kaggle submit: 없음.
- full-test candidate/submission CSV 생성: 없음.
- hidden labels/private answers/external Steam scraping: 없음.
- 결론: local surfaces가 closed/quarantined family, stalled/weak probe, unbounded training, submission-path 위험 중 하나에 걸려 새 bounded probe를 시작하지 않음.

이후 관련 cron은 멈춰 있다.

| job | 상태 | 의미 |
|---|---|---|
| `4d627b59804f` OpenCode no-submit improvement-axis loop | paused | 새 독립축을 계속 못 찾아 중지됨 |
| `272808a2bcca` aggressive autorunner watchdog | paused | strict candidate가 없어 submit-capable watchdog도 멈춤 |
| `d9ef9fafb3d7` weekly arXiv SOTA monitor | paused | 최신 논문 모니터도 현재는 중지 상태 |

현재 Hermes background process도 없고, repo 관련 live process도 확인되지 않았다. GPU 0/1/2는 idle, GPU3은 약 3.7GB memory만 보이지만 `nvidia-smi pmon`에 owner process가 잡히지 않는다.

### 6.5 repo checkpoint가 지저분하다

현재 git 상태도 정리되어 있지 않다.

- tracked modified: 3개
  - `reports/failed_axes.json`
  - `scripts/aggressive_quota_runner.py`
  - `state/aggressive_quota_runner_state.json`
- untracked entry: 약 2,003개
- 전체 porcelain entry: 2,006개

대부분은 실험 report/log/script 산출물이다. 이 상태에서 바로 commit/push하면 불필요한 artifact가 섞일 위험이 있다. 다음 작업 전에 “어떤 report/script만 보존할지”를 골라 checkpoint를 만드는 편이 안전하다.

## 7. 왜 여기서 정체됐는가

정체의 핵심은 세 가지다.

1. **Bayes ceiling에 가까운 구조**
   이 대회는 cold-start도 없고, item universe가 작으며, user별 candidate 수가 작고, 평가가 per-user top-half에 가깝다. LightGCN이 co-occurrence/popularity 신호를 거의 다 흡수하기 좋은 구조다.

2. **남은 신호가 대부분 base와 중복**
   text, hours, date, item popularity, co-visitation, ALS/EASE/ItemKNN score가 겉으로는 signal을 갖지만, emb128 LightGCN의 오류를 안정적으로 고치는 독립 residual로는 잘 남지 않았다.

3. **micro-delta 영역의 검증 불안정성**
   +0.0005~+0.0015 수준의 local gain은 split/seed/public에서 쉽게 뒤집힌다. emb192, ALS residual, OTTO, boundary/TAGCF가 모두 이 구간에서 흔들렸다. 지금 current best 0.77825를 넘기려면 매우 작은 행 수 차이를 맞혀야 하는데, local gate가 그 정도 해상도에서는 둔하다.

## 8. 지금 기준 선택지

### 선택지 A — 종료/최종 제출 정리

가장 현실적인 선택이다.

- Public 기준 후보: `candidate_rank_blend_emb128_emb192.csv` (0.77825).
- 안정 재현 후보: `candidate_lightgcn_emb128L4r3_seed_ens.csv` (0.77745).
- 필요 작업:
  - 최종 2개 후보를 확정.
  - 각 후보의 SHA, 생성 명령, seed, 환경, git 상태, 재현 스크립트를 eCampus 제출용으로 정리.
  - repo에서 보존할 report/script만 골라 commit.

### 선택지 B — no-submit loop만 재개

submit은 막고, 새 논문/새 아이디어만 감시하는 방식이다.

- paused 된 OpenCode no-submit loop 또는 arXiv monitor를 재개할 수 있다.
- 다만 최근 verdict가 연속 `NO_SAFE_AXIS`였으므로, 기대값은 낮다.
- 재개 조건은 “기존 closed family를 반복하지 않는 명확한 새 축”이어야 한다.

### 선택지 C — 마지막 forced public probe

권하지 않는다. 현재 남은 후보들은 대부분 weak-positive 또는 diagnostic-only다. forced OTTO도 0.77815로 current best에 -0.00010이었다. 제출권이 중요하면 더 태우는 것보다 final packaging이 낫다.

## 9. 보존해야 할 핵심 근거 파일

| 목적 | 파일 |
|---|---|
| validation harness | `reports/validation_harness_prototype_report.md` |
| 데이터 구조/후보 분포 EDA | `reports/20260601_deep_data_signature_eda.md` |
| 전체 탐색 종합 | `reports/20260531_full_exploration_synthesis.md` |
| rank-blend public best 제출 결과 | `reports/20260602T081244KST_rank_blend_submission_result.md` |
| autorun negative-transfer postmortem | `reports/20260602T210931KST_autorun_batch_similarity_postmortem.md` |
| OTTO independent confirmation | `reports/20260607T095549KST_otto_independent_uniform_confirmation.md` |
| forced OTTO public 결과 | `reports/20260607T114059KST_otto_forced_post_submission_analysis.md` |
| current-best residual atlas | `reports/20260607T125601KST_current_best_residual_atlas.md` |
| ALS independent confirmation | `reports/20260607T130533KST_current_best_als_independent_confirmation.md` |
| 마지막 no-safe-axis 상태 | `reports/20260607T183323KST_improvement_axis_cron_status.md` |
| 실패/격리 ledger | `reports/failed_axes.json` |
| autorunner state | `state/aggressive_quota_runner_state.json` |

## 10. 다음에 내가 바로 할 수 있는 일

1. 최종 제출 후보 2개를 기준으로 eCampus 재현 패키지/README를 정리.
2. 현재 untracked 2,003개 중 보존할 report/script만 선별해 git checkpoint 생성.
3. `failed_axes.json`과 `state/aggressive_quota_runner_state.json`을 사람이 읽기 쉬운 최종 ledger로 축약.
4. paused cron을 유지/삭제/재개할지 결정하고, 원하면 no-submit monitor만 안전하게 재개.

내 판단으로는 지금은 **새 실험을 더 돌릴 단계보다 final packaging과 checkpoint 정리 단계**에 가깝다. 단, 우현이 public #1 변동이나 leaderboard 상황 때문에 한 번 더 찾고 싶다면, 기존 계열 재반복이 아니라 “완전히 새 독립 신호”만 허용하는 no-submit 탐색으로 제한하는 게 안전하다.
