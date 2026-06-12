# KMURecSys26 Steam final-2 eCampus 재현 패키지

작성 시각: 2026-06-12 22:57 KST

작업 경로: `/opt/data/kaggle/kmu-rec-sys-26-steam`

패키지 경로: `artifacts/final_ecampus_package_20260612T2257KST`

이 문서는 최종 2개 후보를 eCampus 재현성 관점에서 고정하기 위한 체크포인트다. 이번 단계에서는 Kaggle 제출을 실행하지 않았다. 새 모델 후보를 만들기보다, 이미 public 점수와 재현성이 확인된 두 파일을 같은 기준으로 검증하고 보존하는 데 목적이 있다.

## 1. 현재 판단

- 내부 데이터만 쓰는 추가 improvement axis는 OpenCode 재검토까지 거쳐 `NO_SAFE_INTERNAL_AXIS`로 닫았다.
- 외부 Steam metadata는 아직 사용하지 않았다. 교수자 확인 전에는 final 후보에 넣지 않는다.
- final-2는 public 최고점 보존용 1개와 재현 안정형 1개로 나누어 두는 편이 가장 안전하다.

## 2. final slot 구성

| Slot | 후보 | 역할 | Public | 주요 파일 |
|---:|---|---|---:|---|
| 1 | `rank_blend_emb128_emb192` | public-best / manual-risk | 0.77825 | `artifacts/final_ecampus_package_20260612T2257KST/csv/final_slot1_rank_blend_emb128_emb192_LABEL.csv` |
| 2 | `lightgcn_emb128L4r3_4seed` | 안정 재현형 | 0.77745 | `artifacts/final_ecampus_package_20260612T2257KST/csv/final_slot2_lightgcn_emb128L4r3_4seed_LABEL.csv` |

Slot 1은 원 제출 파일이 `ID,Played` 헤더였고, 대회 로컬 규칙 문서는 `ID,Label` 형식을 요구한다. 그래서 패키지 안에는 두 파일을 모두 남겼다.

- 원 제출과 byte-identical한 보존본: `artifacts/final_ecampus_package_20260612T2257KST/csv/candidate_rank_blend_emb128_emb192_played.csv`
- eCampus/공식 형식용 header-normalized 본: `artifacts/final_ecampus_package_20260612T2257KST/csv/final_slot1_rank_blend_emb128_emb192_LABEL.csv`
- normalization은 두 번째 컬럼명만 `Played`에서 `Label`로 바꾼 것이다. ID 순서와 0/1 예측값은 바뀌지 않았다.

## 3. 검증 결과

### Slot 1 — rank-blend

- source submitted file: `submissions/candidate_rank_blend_emb128_emb192.csv`
- source SHA256: `835b8b90ce30116a3df7a7575e6ccdaec268af9c1acb01ca0c15c733b3152b2e`
- package `ID,Played` SHA256: `835b8b90ce30116a3df7a7575e6ccdaec268af9c1acb01ca0c15c733b3152b2e`
- package `ID,Label` SHA256: `1d38c3edf7afae2bbcde9f3d548aa92276323379af2c0c143b66844f25cef052`
- rows: 19,998
- Label=1 / Label=0: 9,999 / 9,999
- ID: 0부터 19,997까지 연속
- validation delta vs emb128: +0.00083 mean, split deltas `[0.0017, 0.0003, 0.0005]`
- Fisher combined p-value: 0.3421
- strict gate: false

Slot 1은 public best를 실제로 찍은 후보라는 장점이 있다. 반대로 내부 3-split 검증에서는 통계적으로 강한 승리까지는 아니므로, 재현 안정형이라기보다 public-best 보존용으로 보는 게 맞다.

### Slot 2 — emb128 LightGCN 4-seed

- source file: `artifacts/lightgcn_emb128L4r3_fulltest/test_candidate/candidate_lightgcn_emb128L4r3_seed_ens.csv`
- source/package SHA256: `7e3191dead2d85637dbf01f7ad50aeb61394c6b709959f2c2850bc53c430c195`
- rows: 19,998
- Label=1 / Label=0: 9,999 / 9,999
- ID: 0부터 19,997까지 연속
- seeds: `[42, 123, 2024, 7]`
- config: emb_dim=128, layers=4, reg=0.001, lr=0.001, epochs=200, batch=4096
- verify-existing 결과: SHA match `True`

Slot 2는 public 점수는 Slot 1보다 0.00080 낮지만, 생성 스크립트와 hash 검증 경로가 가장 깨끗하다. eCampus에서 “동일 결과 재현”을 설명하기에는 이 후보가 더 안정적이다.

## 4. 두 slot 간 차이

- Slot 1과 Slot 2의 예측 row diff: 368 / 19,998
- 비율: 0.018402

두 파일은 완전히 같은 후보가 아니다. final 2개 선택권을 쓴다면, public-best와 stable-backbone을 같이 보존하는 hedge가 된다.

## 5. 재현 명령

### Slot 1 재생성

```bash
UV_LINK_MODE=copy uv run --with numpy --with pandas python3 scripts/materialize_rank_blend_emb128_emb192.py \
  --out artifacts/final_ecampus_package_20260612T2257KST/csv/candidate_rank_blend_emb128_emb192_played.csv \
  --report artifacts/final_ecampus_package_20260612T2257KST/reports/rank_blend_emb128_emb192_preflight.json

python3 artifacts/final_ecampus_package_20260612T2257KST/scripts/normalize_rank_blend_header.py
```

첫 번째 명령은 원 제출과 같은 `ID,Played` 보존본을 만든다. 두 번째 명령은 eCampus/로컬 규칙에 맞춘 `ID,Label` 파일을 만든다. 두 번째 단계는 header만 바꾸며 예측값은 그대로 둔다.

### Slot 2 검증

```bash
UV_LINK_MODE=copy uv run --with numpy --with pandas --with scipy python3 scripts/reproduce_submission_emb128.py --verify-existing
```

검증 리포트는 `reports/20260601_ecampus_repro_emb128_verification.json`에 남는다.

## 6. 데이터 fingerprint

| 파일 | SHA256 |
|---|---|
| `train.json` | `2258fa550ae4835fe041d54bcbae4337c3bed8774396dee9ceafbe6a3041daa2` |
| `pairs.csv` | `58678f4a851709bc130a5c3f2a0a3a622ae8b41d74d255c7c517537a426c2bae` |

## 7. package manifest

기계적으로 확인할 수 있는 값은 다음 manifest에 모았다.

- `artifacts/final_ecampus_package_20260612T2257KST/MANIFEST.json`

manifest에는 각 CSV의 path, SHA256, header, row count, label balance, ID 연속성, slot 간 row diff, git 상태, 안전 상태가 들어 있다.

## 8. 남은 결정

1. Slot 1의 `ID,Played` 원 제출본과 `ID,Label` normalized 본 중 eCampus에 어느 파일명을 대표로 둘지 정해야 한다. 내용 예측은 같다.
2. 외부 Steam metadata를 쓸지는 아직 별도 승인 사안이다. 승인 전에는 이 final package에 넣지 않는다.
3. Git checkpoint는 별도 판단이 필요하다. 현재 repo 전체 dirty 파일 수가 많아서, 이번 패키지 파일만 선별 commit할지 먼저 고르는 게 안전하다.

## 9. 압축본

패키지 디렉터리는 압축본으로도 남겼다. 이 SHA256은 압축본 바깥의 리포트와 `.sha256` 파일에만 기록했다. 압축본 내부 파일을 바꾸면 archive hash도 달라지기 때문이다.

- archive: `artifacts/final_ecampus_package_20260612T2257KST.tar.gz`
- archive SHA256: `e0348d3369284f0b859a29aa6284c6cf84bbd5024b0a459206af85c562a1c7c4`
- sha256 sidecar: `artifacts/final_ecampus_package_20260612T2257KST.tar.gz.sha256`
