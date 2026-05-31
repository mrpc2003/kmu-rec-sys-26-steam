# KMURecSys26 Steam — eCampus 최종 제출 재현성 번들 (final-2)

본 번들은 대회 규칙(최종 선택 제출 2건은 **동일 예측을 재현하는 코드/조건** 제출 필수)에 따라
final-2 두 후보의 완전한 재현 경로를 담는다. 두 후보 모두 결정적(deterministic)이며 SHA256 검증됨.

## final-2 후보

| 순위 | 모델 | public | 후보 CSV | SHA256 |
|---|---|---:|---|---|
| #1 | emb128_L4_reg1e-3 4-seed LightGCN 앙상블 | **0.77745** | `candidates/candidate_lightgcn_emb128L4r3_seed_ens.csv` | `7e3191dead2d85637dbf01f7ad50aeb61394c6b709959f2c2850bc53c430c195` |
| #2 | emb64_L3_reg1e-4 4-seed LightGCN 앙상블 | 0.77125 | `candidates/candidate_lightgcn_seed_ens.csv` | `dcc578de495f98133d1cbccfcb53a156f7d1f46973f571164b6ec90605d937f7` |

## 공통 환경

- Python 3.13.5 / PyTorch **2.10.0+cu128** (Tesla V100 sm_70에서 CUDA 커널 동작 검증 wheel)
- numpy, pandas, scipy, scikit-learn / 패키지 격리 실행 `uv run`
- 데이터(대회 제공, 번들 미포함): `data/raw/public/data/train.json` (175k 리뷰), `data/raw/public/data/pairs.csv` (19,998행)

## 모델 / 디코딩 (공통)

- LightGCN (feature transform/비선형 없음), BPR 손실(logsigmoid pairwise), Adam lr 1e-3, batch 4096, 200 epochs
- seed: 42 / 123 / 2024 / 7 (numpy default_rng + torch.manual_seed)
- **4-seed raw score 평균** → **유저별 top-half** 디코딩 (후보 수 n인 유저는 상위 n/2를 1, 나머지 0)
- #1: emb_dim=128, n_layers=4, reg=1e-3  |  #2: emb_dim=64, n_layers=3, reg=1e-4

## 재현 절차

각 후보의 상세 단계·검증 라인은 다음 문서 참조:
- #1: `docs/20260530_ecampus_repro_emb128L4r3_077745.md`
- #2: `docs/20260530_ecampus_repro_seed_ens_077125.md`

핵심 스크립트(`scripts/`): `lightgcn_train.py`(모델/학습), `lightgcn_fulltest_param.py`(#1 파라미터화 full-train),
`emb128L4r3_test_aggregate.py`(#1 집계+SHA), `lightgcn_seed_ensemble_worker.py`/`lightgcn_seed_ensemble_aggregate.py`(#2),
`recsys_played_utils.py`(공용 유틸/디코딩/평가).

## 선택 근거 (탐색 종료)

검증 가능한 모든 직교축(고전 CF: ALS/EASE/ItemKNN, 그래프 contrastive: SGL, loss 교체: DirectAU,
텍스트: TF-IDF/MiniLM 의미축, seed 8개 확장)이 parameter-free uniform 공개-대리 게이트를 통과 못 했다.
상세: `docs/20260531_orthogonal_search_closed.md`. 따라서 검증된 두 후보를 final-2로 확정한다.
