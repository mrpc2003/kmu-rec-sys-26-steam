# KMU RecSys 26 Steam

Kaggle `kmu-rec-sys-26-steam` 추천시스템 대회 작업 저장소입니다.

이 저장소는 **코드, 검증 리포트, 실험 추적 설정**을 보관합니다. 원본 Kaggle 데이터, 대용량 score/artifact, W&B local run blob, 제출 후보 CSV, credential은 커밋하지 않습니다.

## 현재 작업 원칙

- Kaggle 제출은 사용자 명시 승인 전 금지합니다.
- validation 우선으로 실험하고, W&B에는 `no-submit` tag를 유지합니다.
- 기본 W&B artifact mode는 `summary`입니다. 큰 CSV 업로드가 필요할 때만 `full`을 명시합니다.
- raw data와 생성 캐시는 `.gitignore`로 제외합니다.

## 주요 디렉터리

| 경로 | 내용 |
|---|---|
| `scripts/` | validation split, scoring, blending, W&B logging, EDA 스크립트 |
| `reports/` | EDA/validation/W&B/OpenCode 검증 리포트 |
| `data/` | 로컬 Kaggle 원본 데이터. Git 제외 |
| `artifacts/` | 로컬 score/artifact 산출물. Git 제외 |
| `wandb_runs/` | W&B local cache/run blobs. Git 제외 |

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
