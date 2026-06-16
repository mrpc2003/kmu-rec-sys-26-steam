# KMURecSys26 Steam 진행 내용과 현재 막힌 지점

- 작성 시각: 2026-06-14 21:02 KST
- 저장소: `/opt/data/kaggle/kmu-rec-sys-26-steam`
- 기준 브랜치/커밋: `main`, `a11129f`
- 정리 범위: 지금까지의 제출·검증 리포트, `failed_axes.json`, 최근 pseudo-label/checkpoint/metadata feasibility 결과, 현재 cron/process/git 상태
- 이번 문서 작업: 기록 정리만 수행. Kaggle 제출, 새 candidate CSV 생성, full-test scoring은 하지 않음.

## 1. 한 줄 결론

현재 최고 public 점수는 `candidate_rank_blend_emb128_emb192.csv`의 **0.77825**이고, 가장 안정적으로 재현되는 백본은 emb128 LightGCN 4-seed ensemble의 **0.77745**다. 이후 제공 데이터 내부에서 시도한 residual, CF, graph filter, pseudo-label, checkpoint averaging 계열은 strict gate를 통과하지 못했다. 지금 막힌 핵심은 구현 부족이 아니라 **새 독립 신호 부족**이다. 외부 Steam metadata도 `gameID -> Steam appid` 매핑이 없어 바로 붙일 수 없고, 우현이 “하지마”라고 정했으므로 현 시점에서는 멈춰 둔 상태다.

## 2. 현재 기준 후보

| 구분 | 파일/모델 | Public | 역할 | 주의점 |
|---|---|---:|---|---|
| 현재 public best | `submissions/candidate_rank_blend_emb128_emb192.csv` | **0.77825** | final slot 후보 A. public 점수 보존용 | emb128+emb192 rank-blend. validation 통계는 강하지 않음. preflight 기준 mean Δ `+0.00083`, Fisher p `0.3421`, strict gate false. |
| 안정 재현 백본 | `artifacts/lightgcn_emb128L4r3_fulltest/test_candidate/candidate_lightgcn_emb128L4r3_seed_ens.csv` | **0.77745** | final slot 후보 B. 재현성 기준 모델 | emb128 L4 reg1e-3, seeds `42/123/2024/7`, raw-score mean, per-user top-half. SHA256 `7e3191de...c195`, byte-identical 재현 확인. |
| 이전 기준 | `candidate_lightgcn_seed_ens.csv` | 0.77125 | emb64 4-seed 기준선 | emb128로 넘어가며 public +0.00620 개선. |
| 첫 graph anchor | `candidate_lightgcn_full_train.csv` | 0.76245 | uniform surrogate 신뢰의 출발점 | 단일 emb64 LightGCN. |
| Stage2 CF blend | `candidate_score_blend_mean_z.csv` | 0.74594 | 초기 baseline 개선 | 이후 LightGCN 계열에 밀림. |

emb128 재현 명령은 아래처럼 기록되어 있다.

```bash
UV_LINK_MODE=copy uv run --with numpy --with pandas --with scipy \
  python3 scripts/reproduce_submission_emb128.py --verify-existing
```

검증 파일: `reports/20260601_ecampus_repro_emb128_verification.json`

## 3. 검증 체계가 어떻게 잡혔는지

### 3.1 데이터와 decoding

- train: 약 175,000 rows, 6,710 users, 2,437 games.
- test pairs: 19,998 rows, 4,737 users, 2,429 games.
- hidden test는 played/non-played가 1:1.
- `pairs.csv`의 각 user 후보 수가 짝수라서, 거의 모든 후보는 user별 상위 절반을 `Label=1`로 두는 **per-user top-half decoding**으로 제출했다.
- cold user/item은 사실상 없어서, graph co-occurrence 기반 모델이 강하게 먹히는 구조다.

### 3.2 public surrogate

초기에는 candidate marginal을 보고 `sqrtpop` negative가 더 그럴듯해 보였지만, 제출 결과를 비교하면서 결론이 바뀌었다.

- public LB는 **uniform-negative validation**을 가장 잘 따라갔다.
- hard sampler(`sqrtpop`, `popbin`, `recent`)는 primary gate가 아니라 stress/robustness 확인용으로 낮췄다.
- hard sampler에서 좋아지는 popularity down-weighting 계열은 public에서 자주 깨졌다.

### 3.3 strict gate

후반부 후보는 대체로 아래 기준을 넘겨야 submit 후보로 봤다.

- 3개 uniform split에서 mean Δ가 충분히 커야 함. 후반 기준은 대체로 `+0.0015` 이상.
- split별 부호가 안정적이어야 함. min split Δ가 음수면 탈락.
- positive split이 3/3이어야 함.
- paired exact/McNemar류 p-value가 유의해야 함.
- 이미 public negative transfer가 난 family는 quarantine.
- strict gate 없이 full-test candidate CSV를 만들지 않음. 우현이 명시적으로 승인한 forced/manual-risk 제출은 별도 기록.

## 4. 지금까지 진행한 큰 흐름

### 4.1 Stage2 CF baseline

ItemKNN BM25, EASE, ALS score를 만들고 z/rank/RRF blend를 검증했다. 첫 제출 `candidate_score_blend_mean_z.csv`는 public 0.74594로 baseline보다 나았지만, 이후 LightGCN에 크게 밀렸다. 이 단계에서 validation split, per-user top-half decoding, 제출 preflight, 리포트 기록 방식이 만들어졌다.

### 4.2 LightGCN 백본 확립

가장 큰 성능 상승은 LightGCN에서 나왔다.

| 모델 | Public | 의미 |
|---|---:|---|
| 단일 emb64 LightGCN | 0.76245 | graph backbone anchor |
| emb64 4-seed ensemble | 0.77125 | seed ensemble의 큰 상승 확인 |
| emb128 L4 reg1e-3 4-seed ensemble | 0.77745 | 현재 안정 재현 백본 |
| emb128+emb192 rank-blend | 0.77825 | 현재 public best, 다만 통계적 확신은 약함 |

emb192는 uniform에서 잠깐 좋아 보였지만 public은 emb128보다 `-0.00030` 낮았다. capacity를 더 키우는 방향은 sweet spot을 지난 것으로 봤다.

### 4.3 stacker와 residual 계열

LightGCN + Stage2 score를 feature로 넣은 logreg stacker는 OOF에서는 좋아 보였지만 public에서 크게 무너졌다.

- 제출: `candidate_stacker_logreg_emb64_L3_reg1e-04.csv`
- Public: 0.75355
- 당시 anchor 0.76245 대비 `-0.00890`
- 해석: validation negative sampler artifact와 popularity down-weighting을 배운 쪽에 가까움.

이 실패 뒤로 meta-learner, validation-label 기반 weight tuning, popularity residual은 강하게 보수적으로 다뤘다.

### 4.4 넓은 모델/논문 계열 탐색

다음 계열을 직접 구현하거나, validation-only probe와 OpenCode adviser로 확인했다.

- LightGCN variants: SGL, DirectAU, DNS, xSimGCL류.
- item-item/linear: EASE, ALS/WMF, ItemKNN, GF-CF/Turbo-CF류.
- text/semantic: review TF-IDF, embedding residual, item text/date/hour 통계.
- sequence: SASRec류. 이 대회는 시간순 다음 아이템 예측보다 set-membership/top-half 문제에 가까워 약했다.
- geometry: hyperbolic/Lorentz probe. emb128과 달라 보였지만 solo가 약해 blend 이득이 없었다.
- boundary/K-aware: SL@K-lite, boundary vote, jackknife uncertainty, exact-K/boundary feature factory.
- full-test aggregation: TAGCF, boundary scoreblend, frontier z-blend.

대부분은 solo 성능이 낮거나, LightGCN과 rank/score 상관이 너무 높아 blend가 새 오류를 고치지 못했다.

### 4.5 자동 제출과 public negative transfer

중간에 aggressive autorunner가 strict 후보를 빠르게 태우는 방향으로 움직였다. 이후 public negative transfer가 반복되면서 guard를 강화했다.

- 동일 SHA 재제출 금지.
- near-duplicate row diff 후보 금지.
- 같은 family가 current best를 못 넘으면 quarantine.
- 5개 몰아 태우는 rapid quota burn 금지.
- 제출 후 post-analysis 필수.

대표적인 public 실패/약한 후보는 아래와 같다.

| 계열 | 대표 결과 | 판정 |
|---|---:|---|
| ALS residual rankblend | 0.77785~0.77805 | current best 0.77825 미달. validation 양성 신호가 public으로 충분히 전이되지 않음. |
| boundary scoreblend/TAGCF | 0.77575~0.77755 | public negative transfer. |
| frontier z-blend | 0.77715 | capacity frontier 파생도 current best 미달. |
| forced OTTO | 0.77815 | emb128보다는 +0.00070였지만 current best에는 `-0.00010`. |

## 5. 최근에 닫은 축

### 5.1 EASE/ItemKNN wide audit

- report: `reports/20260612T213950KST_uniform_wide_ease_itemknn_aggregate.md`
- best: `score_itemknn_bm25_sum`
- best mean accuracy: `0.742849`
- emb128 기준선보다 크게 낮아서 underpowered family로 닫음.

### 5.2 GF-CF spectral panel

- report: `reports/20260612T214616KST_gfcf_uniform_panel_probe.md`
- best solo mean: `0.757285`
- best 50/50 blend mean: `0.763786`
- emb128 uniform 기준 `0.76505` 미달.
- seed42 smoke near-tie가 3-split에서는 살아남지 못했다.

### 5.3 pseudo-label transduction: margin 0

- report: `reports/20260612T2312KST_pseudolabel_transduction_aggregate.md`
- setting: teacher top-1 candidate를 pseudo-positive edge로 추가, emb128 student 재학습.
- runs: 12, splits: 3.

```text
mean student acc : 0.761927
mean teacher acc : 0.761986
mean Δ           : -0.000058
min Δ            : -0.001500
max Δ            : +0.001900
positive runs    : 6 / 12
pseudo precision : 0.8442
gate pass        : false
```

평균 lift가 사실상 0이어서 후보화하지 않았다.

### 5.4 checkpoint/SWA-like prediction averaging

- report: `reports/20260613T0106KST_checkpoint_avg_aggregate.md`
- best variant: `score_avg_last3_160_200`

```text
mean acc        : 0.762052
baseline        : 0.761986
mean Δ          : +0.000067
min Δ           : -0.000300
max Δ           : +0.000500
positive splits : 1 / 3
gate pass       : false
```

late checkpoint averaging은 noise scale만 회수했다. SWA/checkpoint family는 현재 백본 안에서는 닫았다.

### 5.5 margin-filtered pseudo-label

- report: `reports/20260613T0246KST_pseudolabel_margin_aggregate.md`
- 목적: pseudo-label precision을 높이면 teacher bias 문제가 줄어드는지 확인.

```text
margin 1.5
mean student acc : 0.761236
mean teacher acc : 0.761986
mean Δ           : -0.000750
min Δ            : -0.002801
max Δ            : +0.001700
positive runs    : 4 / 12
pseudo precision : 0.9109
gate pass        : false

margin 2.5
mean Δ           : -0.001134
min Δ            : -0.004701
positive runs    : 1 / 12
pseudo precision : 0.9444
gate pass        : false
```

precision을 0.91/0.94까지 올려도 평균이 더 나빠졌다. pseudo-label은 새 신호를 더하는 쪽이 아니라 teacher bias를 다시 주입하거나 graph training boundary를 흔드는 쪽으로 봐야 한다. 이 family는 현재 설정에서 닫았다.

### 5.6 OpenCode 후속 판정

OpenCode Hephaestus 경로로 최신 결과를 넣고 다시 adviser를 돌렸다.

- prompt: `reports/20260613T0605KST_opencode_post_pseudo_margin_axis_prompt.md`
- raw output: `reports/20260613T0605KST_opencode_post_pseudo_margin_axis_raw_text.md`
- verdict: `NO_SAFE_INTERNAL_AXIS`

내부 제공 데이터만으로 바로 띄울 만한 strict-gate probe가 없다는 판정이다. Hermes 쪽 검증 결과도 같은 방향이다.

### 5.7 metadata feasibility

우현이 한 번 승인한 뒤, item metadata join 가능성만 안전하게 확인했다.

- report: `reports/20260613T082844KST_metadata_approval_feasibility.md`
- `gameID` 예시: `g35322304`, `g49368897`, `g73495588`, `g68047320`
- `g` 뒤 숫자를 Steam Store `appid` 후보로 넣었지만 5개 모두 `success=false`.
- `data/` 안에 `gameID -> Steam appid` 매핑 파일 없음.

그래서 item metadata probe를 바로 시작할 수 없다. 리뷰 문장 매칭, user profile/owned-games 조회, 외부 리뷰 수집은 익명화 해제나 외부 사용자 데이터 사용으로 넘어갈 수 있어 멈췄다. 이후 우현이 “그럼 하지마”라고 했으므로, metadata 역매핑/수집 축은 현 상태에서 중단한다.

## 6. 현재 막힌 지점

### 6.1 `NO_SAFE_INTERNAL_AXIS`

가장 큰 막힘은 새 내부 신호가 남지 않았다는 점이다. graph co-occurrence, item popularity, item similarity, hours/text/date 통계가 emb128 LightGCN ranking에 많이 흡수되어 있다. 후반부 후보들은 LightGCN과 상관이 높거나, 독립성이 있어도 solo 성능이 너무 낮았다.

### 6.2 micro-delta 검증 한계

현재 best를 넘기려면 public에서 수십~수백 row 수준의 작은 차이를 맞혀야 한다. 그런데 local validation의 `+0.0005~+0.0015` 신호는 split/seed/public에서 자주 뒤집혔다. emb192, ALS residual, OTTO, boundary/TAGCF가 모두 이 구간에서 흔들렸다.

### 6.3 public negative transfer family가 늘어남

OOF나 same-panel diagnostic에서 좋아 보인 축이 public에서 current best를 못 넘긴 사례가 많다. 그래서 “약한 양성 신호”만으로 새 submission을 만들면 제출권만 소모할 가능성이 크다.

### 6.4 metadata join key 부재

외부 Steam item metadata 자체는 남은 실질적인 새 정보 축이었지만, 제공 데이터의 `gameID`가 Steam appid가 아니다. 공식 매핑 파일이 없으면 안전하게 join할 수 없다. 역매핑은 규칙 리스크가 있어 지금은 하지 않는다.

### 6.5 stalled/incomplete probe는 재개 근거가 약함

이전 정리에서 남은 incomplete 항목은 아래 둘이다.

| probe | 상태 | 막힌 증상 | 현재 처리 |
|---|---|---|---|
| UserKNN gated residual fine-grid | `STALLED_INCOMPLETE` | 최종 metric report 없이 warning 뒤 종료 | 새 독립축이 아니므로 재실행하지 않음 |
| Jackknife uncertainty boundary expanded | `FAILED_INCOMPLETE_NO_METRIC_REPORT` | expanded report 없음, weak smoke 계열 | 확장 재실행 근거 부족 |

계산을 더 오래 돌리면 새 후보가 나오는 축이라기보다, 이미 약하거나 중복된 방향에서 수치적으로 불안정했던 항목이다.

### 6.6 repo checkpoint가 많이 지저분함

현재 git 상태는 바로 commit/push하기 좋지 않다.

```text
tracked_changed = 4
untracked       = 2052
total           = 2056
```

대부분은 실험 리포트, 로그, submit snapshot, probe script다. 이 상태에서 `git add .`를 하면 CSV snapshot이나 불필요한 산출물이 섞일 위험이 크다. 다음 repo 작업은 보존할 report/script를 좁게 고르는 checkpoint 정리가 먼저다.

## 7. 현재 안전·운영 상태

2026-06-14 21:02 KST 확인 기준이다.

```text
submissions/*.csv count : 23
project process         : 없음
Kaggle submit process   : 없음
GPU pmon owner PID      : 없음
```

GPU memory는 일부 남아 있지만 `nvidia-smi pmon` 기준 실행 중인 project PID는 없다.

```text
GPU0: 880 / 32768 MiB, util 0
GPU1: 536 / 32768 MiB, util 1
GPU2:   4 / 32768 MiB, util 0
GPU3: 506 / 32768 MiB, util 0
```

관련 cron 상태:

| job | 상태 | 메모 |
|---|---|---|
| `4d627b59804f` KMURecSys26 OpenCode no-submit loop | paused | 최신 closure 반영됨. 새 승인/새 축 없으면 반복하지 않도록 중지. |
| `272808a2bcca` aggressive autorunner watchdog | paused | strict candidate가 없어 submit-capable watchdog도 정지. |
| `d9ef9fafb3d7` weekly arXiv SOTA monitor | paused | 최신 논문 모니터도 현재 중지. |

이번 문서 작성 중에도 Kaggle 제출, full-test candidate 생성, external metadata 수집은 없었다.

## 8. 지금 기준 선택지

### A. final packaging / eCampus 재현 패키지 정리

가장 현실적인 다음 단계다.

- final slot A: `candidate_rank_blend_emb128_emb192.csv` — public best 0.77825.
- final slot B: emb128 LightGCN 4-seed — 재현 안정형 0.77745.
- 해야 할 일:
  - 두 후보의 SHA256, 생성 명령, seed, 환경, 데이터 fingerprint, preflight 결과 정리.
  - eCampus 제출용 재현 README/스크립트 확인.
  - 보존할 report/script만 좁게 stage해서 git checkpoint 생성.

### B. repo hygiene checkpoint

실험을 더 돌리기보다 현재 산출물을 정리하는 작업이다.

- `reports/failed_axes.json`을 최종 ledger로 정돈.
- 핵심 report만 보존 목록으로 묶기.
- untracked 2052개 중 commit 대상과 local-only 대상을 분리.
- README/README.ko의 일부 오래된 설명을 현재 결론에 맞게 갱신.

### C. 공식 mapping이 제공될 때만 metadata probe 재개

`gameID -> Steam appid` 매핑이 공식적으로 제공되거나, 대회 측에서 명확히 허용한 경우에만 다시 연다. 그때도 바로 full-test candidate를 만들지 않고 validation-only side-information probe부터 시작한다. 현 상태에서는 우현 지시에 따라 하지 않는다.

### D. forced public probe

권하지 않는다. 남은 후보는 대부분 weak-positive 또는 diagnostic-only다. 제출권이 중요하면 더 태우는 것보다 final packaging이 낫다.

## 9. 핵심 근거 파일

| 목적 | 파일 |
|---|---|
| 이전 종합 정리 | `reports/20260612T1901KST_progress_and_blockers_summary.md` |
| current public best 제출 결과 | `reports/20260602T081244KST_rank_blend_submission_result.md` |
| rank-blend preflight refresh | `reports/20260612_rank_blend_emb128_emb192_preflight_refresh.json` |
| emb128 재현 검증 | `reports/20260601_ecampus_repro_emb128_verification.json` |
| 실패/격리 ledger | `reports/failed_axes.json` |
| pseudo-label margin0 결과 | `reports/20260612T2312KST_pseudolabel_transduction_aggregate.md` |
| checkpoint averaging 결과 | `reports/20260613T0106KST_checkpoint_avg_aggregate.md` |
| margin-filtered pseudo-label 결과 | `reports/20260613T0246KST_pseudolabel_margin_aggregate.md` |
| pseudo-label family closure | `reports/20260613T0606KST_pseudolabel_family_closed_no_safe_axis.md` |
| metadata feasibility | `reports/20260613T082844KST_metadata_approval_feasibility.md` |

## 10. 짧은 작업 메모

지금은 “더 세게 돌리면 뚫릴 것”보다는 “이미 나온 best를 제출/재현 패키지로 안전하게 묶을 때”에 가깝다. 새 실험을 재개하려면 기존 closed family를 반복하지 않는 새 정보가 필요하다. 현재 조건에서 그 후보는 공식 mapping이 딸린 metadata뿐인데, 그 경로는 지금 중단되어 있다.
