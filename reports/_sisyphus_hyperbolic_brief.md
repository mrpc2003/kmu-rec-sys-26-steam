# Sisyphus 구현 임무: 하이퍼볼릭 LightGCN probe (A.2 기하 가설, cheap falsification)

너는 코딩 워커다. 아래 스크립트 하나를 구현하고 단일-seed probe를 실행한다. Hermes(나)가 코드의 수학·수치 정확성과 게이트를 최종 검증한다(직전 라운드에서 Hermes가 너의 boundary 스크립트 d_knn 잔차화 누락 버그를 잡았다 — 같은 수준의 검증이 다시 적용된다).

## 배경 (왜 이 실험이 유일하게 미검증인가)
emb128 4-seed LightGCN이 uniform 0.76505 / public 0.77745로 best. 닫힌 모든 직교 시도(SGL/DirectAU/XSimGCL)는 loss를 InfoNCE로 바꿔 solo 강도를 붕괴시켰고(solo~0.55<floor 0.684), 닫힌 모든 강한 시도(EASE/Turbo-CF/ALS/MultiVAE/capacity)는 유클리드 내적 기하를 공유해 corr_z 0.73~0.99 redundant. **단 한 번도 검증 안 된 조합 = "강한 ranking loss(triplet/BPR)는 유지하되 결정 기하만 비유클리드(Lorentz 하이퍼볼릭)로."** 이것이 A.2 가설이고 이번 표적이다.

내 prior는 낮다 — boundary covariate 검증에서 popularity-residualized conditional co-occurrence AUC가 0.534(거의 chance)라, base가 경계 co-occurrence를 이미 거의 다 추출했다. 하이퍼볼릭은 새 정보가 아니라 같은 그래프의 다른 공간 재표현이다. 그러나 게임 popularity가 강한 계층(Gini 0.52)이고 public이 추종하는 uniform negative가 tail-heavy라, 하이퍼볼릭의 계층 인코딩(head=원점 근처, niche=경계, tail 거리 지수 팽창)이 boundary 재배치를 다르게 할 수 있다 — 이를 싸게 falsify한다.

## 절대 금지 (safety contract)
- Kaggle 제출 금지, submission 파일 생성 금지.
- 숨은 라벨 역추정/외부 수집 금지.
- 기존 `scripts/*.py`, `reports/*.md` 파괴적 수정 금지. 너의 산출물은 **신규 파일 2개만**: `scripts/lightgcn_hyperbolic.py`, `artifacts/hyperbolic/.../summary.json`(스크립트가 씀).
- `git commit`, `git push` 금지 (Hermes가 검증 후 커밋).
- full 1.5일 베팅 금지. **단일 seed42, emb64만** 먼저 (cheap probe).
- delegation에 빠져 "awaiting review"로 멈추지 말 것. 직접 구현·실행하고 sentinel로 끝낼 것.

## 구현 사양: scripts/lightgcn_hyperbolic.py
기존 `scripts/lightgcn_train.py` 구조를 미러하되 기하만 교체. 권장 패턴(HGCF/표준 하이퍼볼릭 LightGCN):
1. 임베딩은 Lorentz manifold 위에 존재 (geoopt 0.5.1 `geoopt.Lorentz()`, 설치 확인됨).
2. **Graph propagation:** 원점 기준 logmap으로 tangent space로 → 대칭정규화 인접행렬로 이웃 평균(기존 `build_norm_adj` 재사용 가능) → expmap으로 manifold 복귀. (LightGCN의 layer-mean을 tangent space에서 수행.)
3. **Score:** 내적이 아니라 **음의 Lorentz 측지거리** `score(u,i) = -d_L(u_emb, i_emb)`.
4. **Loss:** InfoNCE 아님. **하이퍼볼릭 triplet/BPR** — `-logsigmoid(score(u,pos) - score(u,neg))` 형태(거리 기반), negative는 **uniform-unseen**(기존 `sample_bpr_batch`와 동일 sampler, `neg = rng.integers(0,n_items)`). geoopt `RiemannianAdam` 사용.
5. 수치 안정화: Lorentz distance의 arccosh 인자 clamp(>=1+eps), expmap 시 norm clamp. (Hermes가 이 부분을 중점 검증한다.)
6. CLI: `--emb-dim 64 --n-layers 4 --epochs 200 --lr --reg --seed 42 --device cuda:0 --out-dir`. uniform split(`artifacts/validation/val_random_uniform_seed42`)에서 학습·스코어.
7. 평가: 기존 `evaluate_tophalf`로 uniform solo accuracy. 그리고 emb128 4-seed 앙상블 점수를 읽어(`scripts/split_panel_aggregate.py`의 EMB128 경로 또는 `headroom_uniform_analysis.py` 참고) **corr_z(within-user z-score)** 와 **50/50 z-blend accuracy**를 계산.
8. summary.json에 저장: solo_acc, corr_z_vs_emb128, eq_blend_acc, floor(0.684), emb128_ref(0.76505), noise(0.0007), tier 판정.

## 실행 (직접 돌려라)
```
cd /opt/data/kaggle/kmu-rec-sys-26-steam
HOME=/opt/data/home UV_LINK_MODE=copy uv run --python 3.13 --with "torch==2.10.0" --with geoopt --with numpy --with pandas --with scipy python3 scripts/lightgcn_hyperbolic.py --emb-dim 64 --n-layers 4 --epochs 200 --lr 1e-3 --reg 1e-4 --seed 42 --device cuda:0 --out-dir artifacts/hyperbolic
```

## hard-reject 게이트 (네가 summary에 tier로 기록)
- `solo_acc < 0.684` → **REJECT_FLOOR**: 기하가 ranking을 깸(SGL 운명 재확인). 즉시 종결.
- `solo_acc >= 0.684` AND `corr_z < 0.9` AND `eq_blend_acc > 0.76505+0.0007` → **SIGNAL_ESCALATE**: 직교+blend 이득 → 3-split 패널 승격 후보(Hermes가 게이트).
- 그 외(solo 유지하나 eq_blend ≤ noise, 또는 corr_z >= 0.9) → **GEOMETRY_REDUNDANT**: 기하도 종결(강한 음성, 그 자체로 가치).

## 출력
구현·실행을 마치면 summary.json 경로와 핵심 수치(solo_acc, corr_z, eq_blend, tier)를 출력하고, 마지막 줄에 정확히:
`SISYPHUS_HYPERBOLIC_DONE: <summary.json 경로> tier=<tier>`
