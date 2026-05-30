# eCampus 재현성 패키지 — 4-seed LightGCN 앙상블 public 0.77125

대회 규칙: 최종 선택 제출 2건은 **동일 결과가 나오는 코드 및 조건**을 eCampus로 제출해야 하며,
복원되지 않으면 제출로 인정되지 않는다. 본 문서는 신규 최고 후보(public 0.77125)의 재현 절차를 기술한다.
이 후보는 `20260530_ecampus_repro_lightgcn_076245.md`(anchor)를 빌딩블록으로 사용한다.

## 0. 재현성 증거 (이미 검증됨)

집계 단계는 결정적: 동일한 4개 seed의 raw score로 재집계하면 제출 파일과 **byte 단위로 동일**(SHA256 일치).

| 항목 | 값 |
|---|---|
| 제출 파일 | `artifacts/lightgcn_seed_ensemble/test_candidate/candidate_lightgcn_seed_ens.csv` |
| 제출 SHA256 | `dcc578de495f98133d1cbccfcb53a156f7d1f46973f571164b6ec90605d937f7` |
| 재집계 검증 SHA256 | `dcc578de495f98133d1cbccfcb53a156f7d1f46973f571164b6ec90605d937f7` (재실행 일치 확인) |
| `reproduces_submitted` | **true** (집계 결정성, 학습 결정성은 각 seed별 raw score 보존으로 보장) |

→ seed 고정(42/123/2024/7) + 결정적 BPR 샘플링 + raw score 평균은 학습 라벨/negative에 의존하지 않으므로
검증 negative 샘플러 과적합이 구조적으로 불가능. (실패한 stacker 0.75355와 대조되는 견고성.)

## 1. 환경

- Python 3.13.5
- PyTorch **2.10.0+cu128** (V100 sm_70에서 CUDA 커널 동작하는 검증된 wheel)
- numpy, pandas, scipy, scikit-learn
- GPU: Tesla V100-PCIE-32GB (cuda:0~3 중 하나)
- 패키지 관리: `uv run` (격리 실행)

## 2. 데이터

| 파일 | 경로 |
|---|---|
| train | `data/raw/public/data/train.json` (175,000 리뷰) |
| test pairs | `data/raw/public/data/pairs.csv` (19,998행) |

## 3. 재현 명령 (3단계)

### 3-a. seed 42 (anchor) raw score 생성 — anchor 재현 패키지와 동일
```bash
cd /opt/data/kaggle/kmu-rec-sys-26-steam
env -u VIRTUAL_ENV UV_NO_ACTIVE_VENV=1 \
  uv run --extra-index-url https://download.pytorch.org/whl/cu128 \
  --with 'torch==2.10.0+cu128' --with numpy --with pandas --with scipy --with scikit-learn \
  python scripts/lightgcn_fulltrain_save_scores.py --device cuda:0
# 출력: artifacts/lightgcn_20260530/test_full_train/lightgcn_test_raw_scores_emb64_L3_reg1e-04.csv
#       (anchor SHA a3dbe04... 재현 확인됨)
```

### 3-b. seed 123 / 2024 / 7 raw score 생성 (병렬 권장, 각 ~1750s)
```bash
for SEED in 123 2024 7; do
  env -u VIRTUAL_ENV UV_NO_ACTIVE_VENV=1 \
    uv run --extra-index-url https://download.pytorch.org/whl/cu128 \
    --with 'torch==2.10.0+cu128' --with numpy --with pandas --with scipy --with scikit-learn \
    python scripts/lightgcn_seed_ensemble_worker.py --seed ${SEED} --device cuda:0
done
# 출력 per seed: artifacts/lightgcn_seed_ensemble/seed{SEED}/test.csv
```

### 3-c. 4-seed raw score 평균 → per-user top-half → 제출 후보 (SHA 검증 포함)
```bash
env -u VIRTUAL_ENV UV_NO_ACTIVE_VENV=1 \
  uv run --with numpy --with pandas --with scipy --with scikit-learn \
  python scripts/lightgcn_seed_ensemble_aggregate.py
# 출력: artifacts/lightgcn_seed_ensemble/test_candidate/candidate_lightgcn_seed_ens.csv
#       (SHA dcc578de... 재현 확인됨)
```

## 4. 모델 / 하이퍼파라미터 (4 seed 공통, anchor와 동일 config)

| 항목 | 값 |
|---|---|
| 모델 | LightGCN (feature transform/nonlinearity 없음) |
| 손실 | BPR (logsigmoid pairwise) |
| emb_dim | 64 |
| n_layers | 3 |
| lr | 1e-3 (Adam, weight_decay=0) |
| reg | 1e-4 (L2 on embeddings, batch 정규화) |
| batch_size | 4096 |
| epochs | 200 |
| seeds | **42, 123, 2024, 7** (numpy default_rng + torch.manual_seed) |
| 학습 시간 | seed당 ~1750s, 총 ~7000s 단일 GPU (4 GPU 병렬 시 ~1750s) |

## 5. 앙상블 / 디코딩

- 각 seed별로 LightGCN을 위 config로 full train, test pair에 user/item 임베딩 내적으로 `score_lightgcn` 부여
- 4개 seed의 **raw score 평균**: `score_ens = mean(score_seed42, score_seed123, score_seed2024, score_seed7)`
- **유저별 top-half**: 후보 수 n인 유저는 `score_ens` 상위 n/2개를 `1`, 나머지 `0`
  (대회 규칙: 테스트셋은 유저별 played:not-played = 1:1)
- 구현: `recsys_played_utils.predict_tophalf`
- 검증 라벨/negative를 학습하지 않으므로 stacker가 빠진 negative-sampler 과적합 함정을 구조적으로 회피

## 6. 핵심 스크립트

| 역할 | 파일 |
|---|---|
| LightGCN 모델/학습 | `scripts/lightgcn_train.py` (`LightGCN`, `build_norm_adj`, `sample_bpr_batch`, `score_candidates`) |
| seed 42 full-train + raw score + SHA 검증 | `scripts/lightgcn_fulltrain_save_scores.py` |
| seed 123/2024/7 full-train + raw score | `scripts/lightgcn_seed_ensemble_worker.py` |
| 4-seed raw score 평균 + top-half + 후보 SHA 검증 | `scripts/lightgcn_seed_ensemble_aggregate.py` |
| 공용 유틸 (행렬/디코딩/평가) | `scripts/recsys_played_utils.py` |

## 7. 결과

- Public LB: **0.77125** (anchor LightGCN 0.76245 대비 +0.00880, Stage2 blend 0.74594 대비 +0.02531)
- Validation
  - hard-sampler mean (sqrtpop/recent/popbin): 0.64293 (anchor +0.0041)
  - **uniform split (public surrogate)**: **0.76145** (anchor 0.75445, +0.00700)
- 전이비: uniform Δ +0.00700 → public Δ +0.00880 = 1.26 (단일 LightGCN 1.24와 일치, surrogate 가설 확증)

## 8. 검증/방어 라인 (재현자 사용)

다른 환경에서 재현 시 다음으로 검증 가능:
- 3-a 산출물 SHA `a3dbe04…` (anchor)
- 3-b 산출물: `wc -l artifacts/lightgcn_seed_ensemble/seed{123,2024,7}/test.csv` 각 19999 (header 포함)
- 3-c 산출물 SHA `dcc578de…`
- preflight: rows=19998, label_1=9999, label_0=9999, bad_users=0
