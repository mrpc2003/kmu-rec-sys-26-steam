# KMU RecSys 26 Steam

Kaggle [`kmu-rec-sys-26-steam`](https://www.kaggle.com/competitions/kmu-rec-sys-26-steam) 추천시스템 대회 작업 저장소입니다.

이 저장소는 **재현 가능한 코드, 검증 리포트, 실험 추적 설정, 공개 가능한 요약 산출물**만 보관합니다. 원본 Kaggle 데이터, 대용량 score/artifact, W&B local run blob, 제출 후보 CSV, credential은 커밋하지 않습니다.

## 대회 요약

자세한 대회 설명·제출/윤리 규칙·재현성 기준은 [`docs/competition_brief_and_rules.md`](docs/competition_brief_and_rules.md)에 정리합니다.

- Task: Steam 리뷰 데이터 기반 `userID, gameID` pair의 `played` 여부를 예측하는 이진 추천 문제
- Train data: `data/train.json`에 약 **175,000건**의 리뷰가 있으며, 각 인스턴스는 `userID`, `gameID`, `text`, `date`, `hours`, `hours_transformed=log2(1+hours)` 필드를 포함합니다.
- Test pairs: 대회 설명상 `pairs_Played.csv`, 현재 로컬 파일명은 `data/pairs.csv`입니다. 각 row의 `userID, gameID`에 대해 played 예측값을 제출합니다.
- 제출 형식: `ID,Label`, `Label ∈ {0,1}`
- 평가 지표: Accuracy. 테스트셋은 played/non-played가 **정확히 1:1**이며, Public LB는 테스트셋 절반만 반영하고 전체 점수는 대회 종료 후 공개됩니다.
- Baseline: `baseline.py`는 user 무관 popularity baseline, `baseline_bpr.py`는 popularity + BPR baseline입니다.
- 현재 로컬 관찰: `pairs.csv`의 각 user 후보 수가 짝수이고, 후보별 상위 절반을 `Label=1`로 내는 top-half ranking 제출이 자연스럽습니다.
- 주요 검증 전략: 실제 test 구조를 반영하기 위해 user 단위 후보군을 유지하고, unseen negative sampling은 uniform보다 `sqrt(popularity)`/pop-bin이 더 현실적인 surrogate로 취급합니다.

## 현재 작업 원칙

- Kaggle 제출은 사용자 명시 승인 후 **한 파일씩만** 수행합니다.
- Steam 리뷰 직접 수집, 리버스엔지니어링, hidden label 외부 획득, private test 역추적 등은 금지합니다.
- 공개되어 누구나 사용할 수 있는 pretrained 모델은 사용할 수 있으나, 사용 근거와 재현 조건을 기록합니다.
- 최종 선택 가능한 제출은 2개이며, 선택한 제출은 eCampus에 동일 결과 재현 코드/조건/실행 절차를 제출해야 하므로 후보 CSV SHA256, 생성 명령, seed, 데이터 fingerprint, 환경, Git 상태를 기록합니다.
- validation 우선으로 실험하고, W&B에는 `no-submit` tag를 유지합니다.
- 기본 W&B artifact mode는 `summary`입니다. 큰 CSV 업로드가 필요할 때만 `full`을 명시합니다.
- raw data와 생성 캐시는 `.gitignore`로 제외합니다.
- GitHub에는 코드/리포트 중심으로 올리고, `data/`, `artifacts/`, `submissions/`, `wandb_runs/`는 로컬 전용으로 유지합니다.

## 주요 디렉터리

| 경로 | 내용 | Git |
|---|---|---|
| `docs/` | 대회 설명, 운영/윤리/재현성 규칙 | tracked |
| `scripts/` | validation split, scoring, blending, W&B logging, EDA 스크립트 | tracked |
| `reports/` | EDA/validation/W&B/OpenCode/제출 preflight 리포트 | tracked, 단 대용량 CSV preview 일부 제외 |
| `data/` | 로컬 Kaggle 원본 데이터 | ignored |
| `artifacts/` | 로컬 score/artifact 산출물 | ignored |
| `submissions/` | 수동 제출용 CSV 보관 위치 | ignored |
| `wandb_runs/` | W&B local cache/run blobs | ignored |

## 현재 검증 결과 요약

Stage2 기준으로 ItemKNN BM25, EASE, ALS CF score를 만들고, user별 후보군 안에서 score ranking을 top-half `Label=1`로 변환했습니다. 현재 첫 제출 후보는 validation surrogate에서 가장 안정적으로 높은 `score_blend_mean_z`입니다.

| 검증 split / family | best score column | row accuracy | per-user mean accuracy |
|---|---:|---:|---:|
| Random sqrt-pop / Stage2 blend | `score_blend_mean_z` | 0.659732 | 0.675421 |
| Random sqrt-pop / ItemKNN+EASE | `score_itemknn_bm25_top3` | 0.650130 | 0.667937 |
| Random sqrt-pop / ALS CF | `score_als_f32_it30_alpha20_popa2` | 0.650930 | 0.670208 |
| Recent holdout / Stage2 blend | `score_blend_mean_z` | 0.626025 | 0.629962 |
| Random pop-bin / Stage2 blend | `score_blend_mean_z` | 0.590818 | 0.606171 |
| Random uniform / Proto | `score_itemknn_sum` | 0.740648 | 0.764331 |

Uniform split은 **실제 public LB의 직접 surrogate**임이 OOD 게이트와 메커니즘 테스트로 확인됐습니다(상세: `reports/20260530_ood_public_surrogate_finding.md`, `reports/20260530_popcorr_mechanism.md`). seed 앙상블의 uniform Δ +0.00700이 실제 public Δ +0.00880으로 전이비 1.26으로 전이됐고, 이는 단일 LightGCN의 1.24와 일치합니다. 따라서 후보 게이트는 sqrt-pop/recent보다 **uniform split을 1차 surrogate**로 사용하고, hard 샘플러는 robustness floor로만 활용합니다.

## 첫 제출 후보

- 후보 파일: `artifacts/scores/test_pairs_full_train_stage2_blend/prediction_csv/candidate_score_blend_mean_z.csv`
- SHA256: `5f93cf1be4066c1bb28dac846a0ba3849807b01e367dd9ca810a73146d458d34`
- 행 수: `19,998`
- 컬럼: `ID,Label`
- Label 분포: `0=9,999`, `1=9,999`
- ID: `0..19997` 연속
- User별 top-half 제약 위반: `0` users
- 공식 `played_bpr.csv` 대비 다른 행: `3,566`
- 공식 popularity baseline 대비 다른 행: `4,876`

## 제출 기록

| KST timestamp | submitted file | public score | status | report |
|---|---|---:|---|---|
| `20260530T124312KST` | `candidate_score_blend_mean_z.csv` | **0.74594** | `SubmissionStatus.COMPLETE` | `reports/20260530T124312KST_submission_analysis.md` |

첫 제출 결과는 random sqrt-pop validation surrogate `0.659732`보다 높게 나왔습니다. 따라서 다음 단계는 단순 popularity/BPR 재제출보다, 현재 public anchor를 기준으로 같은 top-half ranking 제약을 유지하면서 blend/sequence/graph 계열을 재검증하는 방향이 좋습니다.

## 최신 연구 기반 탐색

- Paper-guided synthesis: `reports/20260530_paper_guided_recsys_exploration.md`
- ArXiv search raw results: `reports/20260530_arxiv_paper_search_results.json`
- Feature feasibility probe: `reports/20260530_paper_guided_feature_probe.md`
- Review TF-IDF probe: `reports/20260530_review_tfidf_probe.md`
- AI-Q deep research (next-step): `reports/20260530_aiq_deep_next_curated.md` (raw: `reports/20260530_aiq_deep_next_raw.json`)
- Paper-guided next-step run: `reports/20260530_paper_guided_next_steps.md`, 해석: `reports/20260530_paper_guided_next_steps_summary.md`

현재 결론은 대형 LLM/graph 모델을 바로 구현하기보다, **PURL/CW loss류의 implicit-feedback 보정**, **ICPNS류의 community-aware negative sampling/validation**, **TFPS류의 time-decay confidence**를 먼저 탐색하는 것입니다. 첫 번째 next-step 라운드에서는 time-decay ItemKNN이 가장 안정적인 단일 axis로, review pseudo-category는 community/popularity가 통제된 stress split의 보완축으로 확인되었고, CW-lite 구현은 디버깅이 필요한 상태입니다.

## W&B

- Entity: `mrpc2003-kookmin-university`
- Project: `kmu-rec-sys-26-steam`
- Project URL: https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam

기존 산출물 후처리 기록 예시:

```bash
cd /opt/data/kaggle/kmu-rec-sys-26-steam
export HOME=/opt/data/home
export WANDB_DIR=/opt/data/kaggle/kmu-rec-sys-26-steam/wandb_runs

env -u VIRTUAL_ENV UV_NO_ACTIVE_VENV=1 HOME=/opt/data/home WANDB_DIR="$WANDB_DIR" \
  uv run --with wandb python scripts/log_wandb_results.py \
    --score-dir artifacts/scores/val_random_sqrtpop_seed42_stage2_blend \
    --run-name-prefix 20260530-example- \
    --group 20260530-existing-results \
    --tags validation-harness,stage2,no-submit \
    --artifact-mode summary
```

## 로컬 실행 예시

```bash
cd /opt/data/kaggle/kmu-rec-sys-26-steam

env -u VIRTUAL_ENV UV_NO_ACTIVE_VENV=1 \
  uv run --with numpy --with pandas --with scipy --with wandb \
  python scripts/score_popularity_itemknn_ease.py \
    --split-dir artifacts/validation/val_random_sqrtpop_seed42 \
    --out-dir artifacts/scores/example_run \
    --methods popularity,itemknn_bm25,ease,ease_htr \
    --ease-lambdas 300,1000 \
    --wandb \
    --wandb-tags validation,stage2,no-submit \
    --wandb-artifact-mode summary
```

## 제출 preflight 예시

```bash
cd /opt/data/kaggle/kmu-rec-sys-26-steam
export HOME=/opt/data/home

python3 - <<'PY'
import csv
from pathlib import Path
p = Path('artifacts/scores/test_pairs_full_train_stage2_blend/prediction_csv/candidate_score_blend_mean_z.csv')
with p.open(newline='') as f:
    rows = list(csv.DictReader(f))
assert rows and list(rows[0].keys()) == ['ID', 'Label']
assert len(rows) == 19998
assert [int(r['ID']) for r in rows] == list(range(len(rows)))
assert {int(r['Label']) for r in rows} <= {0, 1}
print('ok')
PY
```

## 보안/업로드 제외

커밋 전 다음을 확인합니다.

```bash
git status --short
git diff --cached --stat
```

업로드 제외 대상:

- `data/`
- `artifacts/`
- `submissions/`
- `wandb_runs/`
- `.sisyphus/`
- `.env`, `.netrc`, token/key/credential 파일

## 제출 결과

| 일자(KST) | 후보 | Public | Δ vs 이전 best | 비고 |
|---|---|---:|---:|---|
| 2026-05-30 12:44 | Stage2 mean-z blend (BM25 + EASE + ALS) | 0.74594 | — | 첫 제출 |
| 2026-05-30 18:48 | LightGCN full-train (BPR 200ep, emb64 L3 reg1e-4, seed42) | 0.76245 | +0.01651 | anchor, 재현 SHA `a3dbe04…`, transfer ratio 1.24 |
| 2026-05-30 21:03 | logreg stacker (LightGCN+Stage2, within-user z/rank) | 0.75355 | −0.00890 | ❌ 회귀, 검증 negative 샘플러 과적합 (`reports/20260530_stacker_submission_post_analysis.md`) |
| 2026-05-31 01:01 | 4-seed LightGCN 앙상블 (emb64 L3 reg1e-4, seeds 42/123/2024/7) | 0.77125 | +0.00880 | uniform 게이트 +0.00700 → public +0.00880(전이비 1.26), 재현 SHA `dcc578de…` (`reports/20260530_ecampus_repro_seed_ens_077125.md`) |
| 2026-05-31 02:?? | **4-seed LightGCN 앙상블 (emb128 L4 reg1e-3, seeds 42/123/2024/7)** | **0.77745** | **+0.00620** | 🏆 최종 best, uniform 게이트 +0.0036 vs emb64 → public +0.0062(전이비 1.72), 투영 ~0.776 적중, 재현 SHA `7e3191de…` (`reports/20260530_ecampus_repro_emb128L4r3_077745.md`) |

상세: `reports/20260530T184752KST_lightgcn_full_train_post_analysis.md`,
`reports/20260530_stacker_submission_post_analysis.md`,
`reports/20260530_seed_ensemble_uniform_gate.md`,
`reports/20260530_research_to_application_synthesis.md`

