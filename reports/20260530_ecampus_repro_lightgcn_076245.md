# eCampus 재현성 패키지 — LightGCN public 0.76245

대회 규칙: 최종 선택 제출 2건은 **동일 결과가 나오는 코드 및 조건**을 eCampus로 제출해야 하며,
복원되지 않으면 제출로 인정되지 않는다. 본 문서는 LightGCN 후보(public 0.76245)의 재현 절차를 기술한다.

## 0. 재현성 증거 (이미 검증됨)

동일 config로 독립 재학습한 결과가 제출 파일과 **byte 단위로 동일**(SHA256 일치)함을 확인:

| 항목 | 값 |
|---|---|
| 제출 파일 | `candidate_lightgcn_full_train.csv` |
| 제출 SHA256 | `a3dbe043f0f8b781d8c35aea88b7a1f561fa7b705b34edf6c7b7d0451eceb2a6` |
| 재학습 검증 SHA256 | `a3dbe043f0f8b781d8c35aea88b7a1f561fa7b705b34edf6c7b7d0451eceb2a6` |
| `reproduces_submitted` | **true** |
| 검증 메타 | `artifacts/lightgcn_20260530/test_full_train/raw_save_meta_emb64_L3_reg1e-04.json` |

→ seed 고정(42) + 결정적 BPR 샘플링으로 재현이 보장됨.

## 1. 환경

- Python 3.13.5
- PyTorch **2.10.0+cu128** (V100 sm_70에서 CUDA 커널 동작하는 검증된 wheel)
- numpy, pandas, scipy, scikit-learn
- GPU: Tesla V100-PCIE-32GB (`cuda:0`)
- 패키지 관리: `uv run` (격리 실행)

## 2. 데이터

| 파일 | 경로 |
|---|---|
| train | `data/raw/public/data/train.json` (175,000 리뷰) |
| test pairs | `data/raw/public/data/pairs.csv` (19,998행) |

## 3. 재현 명령

### 3-a. 제출 후보 CSV 재생성 (SHA 검증 포함)
```bash
cd /opt/data/kaggle/kmu-rec-sys-26-steam
env -u VIRTUAL_ENV UV_NO_ACTIVE_VENV=1 \
  uv run --extra-index-url https://download.pytorch.org/whl/cu128 \
  --with 'torch==2.10.0+cu128' --with numpy --with pandas --with scipy --with scikit-learn \
  python scripts/lightgcn_fulltrain_save_scores.py --device cuda:0
# 출력: reproduces_submitted: True (SHA == a3dbe04...)
```

## 4. 모델 / 하이퍼파라미터

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
| seed | 42 (numpy default_rng + torch.manual_seed) |
| 학습 시간 | ~1722s (V100) |
| final loss | 0.177485 |

## 5. 예측 디코딩

- 각 test pair에 user/item 임베딩 내적으로 `score_lightgcn` 부여
- **유저별 top-half**: 후보 수 n인 유저는 점수 상위 n/2개를 `1`, 나머지 `0`
  (대회 규칙: 테스트셋은 유저별 played:not-played = 1:1)
- 구현: `recsys_played_utils.predict_tophalf`

## 6. 핵심 스크립트

| 역할 | 파일 |
|---|---|
| LightGCN 모델/학습 | `scripts/lightgcn_train.py` (`LightGCN`, `build_norm_adj`, `sample_bpr_batch`, `score_candidates`) |
| full-train + raw score + SHA 검증 | `scripts/lightgcn_fulltrain_save_scores.py` |
| 공용 유틸 (행렬/디코딩/평가) | `scripts/recsys_played_utils.py` |

## 7. 결과

- Public LB: **0.76245** (이전 best Stage2 blend 0.74594 대비 +0.01651)
- Validation (3-split mean): 0.63883 (transfer ratio 1.24)
