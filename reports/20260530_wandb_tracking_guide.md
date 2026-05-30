# KMU RecSys 26 Steam — W&B 연동 및 기록 규칙

## 1. 연동 상태

- W&B CLI: `/opt/data/home/.local/bin/wandb`, 확인 버전 `0.26.1`.
- Python W&B smoke/API 확인: `wandb.Api()`가 `mrpc2003` 계정, 기본 entity `mrpc2003-kookmin-university`로 인증됨을 확인했다. API key/토큰은 출력하지 않았다.
- 기본 기록 위치: `entity=mrpc2003-kookmin-university`, `project=kmu-rec-sys-26-steam`.
- 프로젝트 URL: https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam
- 모든 현재 기록은 **Kaggle 제출 없이(no-submit)** validation/test 산출물만 W&B에 남긴 것이다.
- 주의: 초기 코드 버그(`Run.group` setter 사용) 때문에 `hnirbtqq` run이 `failed`로 1개 생겼고, 수정 후 정상 run으로 재기록했다. 실제 검증 대상 19개 run은 전부 `finished` 상태다.

## 2. 추가/수정한 로깅 파일

| 파일 | 역할 |
|---|---|
| `scripts/wandb_recsys_utils.py` | W&B lazy import 기반 공통 유틸. score/evaluation/validation summary를 metrics, config, table, artifact로 기록. `evaluation_summary.json`과 `blend_evaluation_summary.json` 모두 지원. prediction CSV 요약은 streaming CSV reader로 rows/Label=1/SHA256만 계산한다. |
| `scripts/log_wandb_results.py` | 이미 생성된 validation/score 디렉터리를 W&B에 후처리 기록하는 CLI. `--artifact-mode summary/full/none` 지원. |
| `scripts/score_popularity_itemknn_ease.py` | 기존 scoring 실행 후 `--wandb`를 붙이면 같은 run에서 metrics/config/artifact를 바로 기록하도록 옵션 추가. 기본 artifact mode는 `summary`. |

## 3. 기본 실행 방법

### 3.1 기존 결과를 W&B에 후처리 기록

```bash
cd /opt/data/kaggle/kmu-rec-sys-26-steam
export HOME=/opt/data/home
export WANDB_DIR=/opt/data/kaggle/kmu-rec-sys-26-steam/wandb_runs
env -u VIRTUAL_ENV UV_NO_ACTIVE_VENV=1 HOME=/opt/data/home WANDB_DIR="$WANDB_DIR" \
  uv run --with wandb python scripts/log_wandb_results.py \
    --validation-root artifacts/validation \
    --score-dir artifacts/scores/val_random_sqrtpop_seed42_stage2_blend \
    --run-name-prefix 20260530-example- \
    --group 20260530-existing-results \
    --tags validation-harness,stage2,no-submit \
    --artifact-mode summary
```

### 3.2 새 scoring/prototype 실행과 동시에 기록

```bash
cd /opt/data/kaggle/kmu-rec-sys-26-steam
export HOME=/opt/data/home
export WANDB_DIR=/opt/data/kaggle/kmu-rec-sys-26-steam/wandb_runs
env -u VIRTUAL_ENV UV_NO_ACTIVE_VENV=1 HOME=/opt/data/home WANDB_DIR="$WANDB_DIR" \
  uv run --with numpy --with pandas --with scipy --with wandb \
  python scripts/score_popularity_itemknn_ease.py \
    --split-dir artifacts/validation/val_random_sqrtpop_seed42 \
    --out-dir artifacts/scores/<new_run_name> \
    --methods popularity,itemknn_bm25,ease,ease_htr \
    --ease-lambdas 300,1000 \
    --wandb \
    --wandb-run-name <new_run_name> \
    --wandb-tags validation,stage2,no-submit \
    --wandb-artifact-mode summary
```

## 4. 앞으로 반드시 기록할 항목

### Config

- competition/version: `kmu-rec-sys-26-steam`, no-submit 여부, 실행 스크립트명
- split: holdout 방식(`random/recent`), negative sampler(`uniform/sqrtpop/popbin`), seed, candidate count/user count
- data hashes: 입력 `train_interactions.csv`, `candidates.csv`, 결과 summary/prediction CSV SHA256
- model/method: popularity, itemKNN/BM25/TF-IDF, EASE lambda/weighting, ALS factors/iterations/alpha/pop-adjust, blend recipe
- runtime: Python/패키지 버전, 실행 host, output directory

### Metrics

- validation: `best/row_accuracy`, `best/per_user_mean_accuracy`, `best_score_col`, `predicted_positive_total`, `true_positive_total`
- score별 비교: `score/<score_col>/row_accuracy`, `score/<score_col>/per_user_mean_accuracy`
- baseline: README raw popularity baseline accuracy/positive count
- validation split safety: fold-train overlap, missing user/item rows, candidate rows/users
- test/unlabeled: prediction file count, 각 후보 CSV의 rows/Label=1 count/SHA256

### Artifacts

- 기본 `summary`: `evaluation_summary.json/md` 또는 `blend_evaluation_summary.json/md`만 업로드한다.
- 필요할 때만 `full`: `candidate_scores.csv`와 `prediction_csv/*.csv`까지 업로드한다. 파일이 커지므로 public/private 후보 제출 파일을 무분별하게 올리지 않는다.
- raw Kaggle data, credential, API key, `.netrc`, W&B local binary run blobs는 업로드/커밋하지 않는다.

## 5. 이번에 실제 기록된 finished run

| run | state | best_score_col | best_row_acc | URL |
|---|---|---|---:|---|
| `20260530-wandb-setup-validation-splits` | finished | `` |  | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/j6bgvwk1 |
| `20260530-wandb-setup2-val_random_sqrtpop_seed42_proto` | finished | `score_ease_lambda1000` | 0.646329 | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/l9fen68z |
| `20260530-wandb-setup2-val_random_popbin_seed42_proto` | finished | `score_ease_lambda300` | 0.563013 | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/lz9xqxrj |
| `20260530-wandb-setup2-val_recent_sqrtpop_seed42_proto` | finished | `score_itemknn_top3` | 0.619424 | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/hlyd7idr |
| `20260530-wandb-setup2-test_pairs_full_train_proto` | finished | `` |  | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/4ytpir0v |
| `20260530-all-summary-test_pairs_full_train_stage2_blend` | finished | `` |  | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/rjz3713y |
| `20260530-all-summary-test_pairs_full_train_stage2_cf` | finished | `` |  | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/nm05u48j |
| `20260530-all-summary-test_pairs_full_train_stage2_itemease` | finished | `` |  | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/l4wl5dnz |
| `20260530-all-summary-val_random_popbin_seed42_stage2_blend` | finished | `score_blend_mean_z` | 0.590818 | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/7oyc6ju7 |
| `20260530-all-summary-val_random_popbin_seed42_stage2_cf` | finished | `score_als_f32_it30_alpha20` | 0.581716 | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/q2pxuai7 |
| `20260530-all-summary-val_random_popbin_seed42_stage2_itemease` | finished | `score_itemknn_bm25_top3` | 0.569914 | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/ygwpvm6t |
| `20260530-all-summary-val_random_sqrtpop_seed42_proto_htr` | finished | `score_ease_lambda1000` | 0.646329 | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/z4t2zsqt |
| `20260530-all-summary-val_random_sqrtpop_seed42_stage2_blend` | finished | `score_blend_mean_z` | 0.659732 | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/ox15hv3h |
| `20260530-all-summary-val_random_sqrtpop_seed42_stage2_cf` | finished | `score_als_f32_it30_alpha20_popa2` | 0.650930 | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/98z7q043 |
| `20260530-all-summary-val_random_sqrtpop_seed42_stage2_itemease` | finished | `score_itemknn_bm25_top3` | 0.650130 | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/x1mfhaz7 |
| `20260530-all-summary-val_random_uniform_seed42_proto` | finished | `score_itemknn_sum` | 0.740648 | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/x5jl9uxa |
| `20260530-all-summary-val_recent_sqrtpop_seed42_stage2_blend` | finished | `score_blend_mean_z` | 0.626025 | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/mdspq2fq |
| `20260530-all-summary-val_recent_sqrtpop_seed42_stage2_cf` | finished | `score_als_htr_f32_it30_alpha20_popa1` | 0.614923 | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/ceatm3el |
| `20260530-all-summary-val_recent_sqrtpop_seed42_stage2_itemease` | finished | `score_itemknn_bm25_top3` | 0.622825 | https://wandb.ai/mrpc2003-kookmin-university/kmu-rec-sys-26-steam/runs/841gn6t5 |

## 6. 검증 로그

- W&B run state API 검증: `reports/20260530_wandb_run_state_verification.json` — 19개 run 모두 `finished`.
- 초기 validation/prototype 기록 stdout: `reports/20260530_wandb_initial_logging_stdout.json`, `reports/20260530_wandb_score_logging_stdout.json`.
- 전체 기존 결과 summary-only 기록 stdout: `reports/20260530_wandb_all_existing_summary_stdout.json`.
- OpenCode read-only review: `reports/opencode_reviews/20260530_wandb_logging_review_plan.log` — `VERDICT: PASS`.
- 문법 확인: `python3 -m py_compile scripts/wandb_recsys_utils.py scripts/log_wandb_results.py scripts/score_popularity_itemknn_ease.py`.
- 보안 스캔: 수정 스크립트/리포트 대상 API key/token/password 패턴 0건.

## 7. 운영 원칙

- 새 실험은 가능한 한 실행 스크립트에 `--wandb`를 붙여 실험 시작 시점부터 config/metrics/artifacts를 남긴다.
- 과거 산출물이나 외부 스크립트 결과는 `scripts/log_wandb_results.py`로 후처리 기록한다.
- Kaggle submission은 계속 사용자 승인 전 금지한다. W&B run 이름과 tags에는 `no-submit`을 유지한다.
- W&B credential은 `/opt/data/home` 기준으로만 사용하고, 절대 출력/커밋하지 않는다.