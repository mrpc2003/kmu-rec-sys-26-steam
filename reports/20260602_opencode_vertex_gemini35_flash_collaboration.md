# OpenCode Vertex AI Gemini 3.5 Flash 협업 검증 및 자문

- timestamp_kst: `2026-06-02T21:36:00+09:00`
- OpenCode model: `google-vertex/gemini-3.5-flash`
- Auth path: OpenCode `google-vertex` provider auth store / Vertex AI API key path; secret values are not recorded.
- Runtime smoke: explicit sentinel response verified before this advisory.
- Advisory raw log: `reports/20260602_opencode_vertex_gemini35_flash_recsys_advisory.jsonl`
- Raw log sha256: `4011750aec8fc8843e66865e5ad6d9d4566814eec9ab25f606b557db995df51d`

## Gemini 3.5 Flash read-only advisory output

I detect analysis - Act as a read-only collaborator to extract insights from the project's state and propose 5 distinct experimental axes to break the current performance ceiling. Proposing high-impact, non-redundant strategies.

---

### 1. README 및 사후 분석 기반 핵심 인사이트
*   **기존 격리 패밀리(ALS/BPR residual)의 실패 원인:** 최근 격리(Quarantine) 처리된 `rankblend_z_plus_score_als_...` 계열은 단순 하이퍼파라미터 및 가중치($w = 0.025 \sim 0.2$) 튜닝에 그쳤습니다. 이로 인해 Public Best 모델과의 Row-diff가 300 미만(일부는 100 대)으로 매우 낮아 제출 쿼타를 낭비했으며, 실제 다각화 효과 없이 대중성(Popularity) 및 커뮤니티 편향에만 고착되어 음수 전이(Negative Transfer)를 기록했습니다.
*   **차별화 방향성:** 단순 랭킹/점수 앙상블의 세부 튜닝을 멈추고, 데이터의 미사용 축(시간 흐름, 리뷰 텍스트, 플레이 시간 강도)을 명시적으로 반영하거나 모델의 목적 함수(Loss) 자체를 popularity-unbiased 형태로 재구축하여 **실질적인 예측 분포의 이질성(Row-diff > 500)**을 확보해야 합니다.

---

### 2. 다각화를 위한 5대 우선순위 실험 축 (Priority Experiment Axes)

#### 축 1: 시간 붕괴 신뢰도 기반 ItemKNN (TFPS / Time-Decay ItemKNN)
*   **접근법:** Steam 데이터 내의 `date` 필드를 활용해, 최근 소비된 아이템일수록 높은 가중치를 부여하는 Time-decay 가중치를 적용한 ItemKNN 모델을 구축하고 이를 앙상블 핵심 축으로 사용합니다.
*   **차별성:** 글로벌 임베딩 공간에서 정적인 선호도를 학습하는 LightGCN/ALS와 달리, 유저의 최근 소비 시퀀스 및 temporal dynamics를 국소적으로 포착하므로 추천 결과의 독립성이 매우 높습니다.
*   **제출 전 필수 검증 증거:**
    *   Uniform split 검증 셋에서 기존 Baseline ItemKNN 대비 **정확도 Delta > +0.003** 달성.
    *   현재 Public Best Ensemble 대비 **Row-diff > 600** 이상 확인되어 독립적 후보군 형성 증명.

#### 축 2: 인기 편향 제거 목적 함수 적용 Graph CF (PURL / CW Loss LightGCN)
*   **접근법:** 기존 LightGCN의 표준 BPR Loss 대신, 아이템 대중성 역수나 신뢰도 가중치를 반영하는 CW(Confidence-Weighted) Loss 또는 PURL(Popularity-Unbiased) Loss를 적용하여 학습을 진행합니다.
*   **차별성:** 사후 Rank Blend로 대중성을 제어하려는 시도와 달리, 학습 과정에서 헤비 유저/인기 아이템이 임베딩 공간을 지배하는 현상을 근본적으로 차단하므로 롱테일(Niche) 아이템 예측력을 향상시킵니다.
*   **제출 전 필수 검증 증거:**
    *   Uniform split 검증 세트 **Delta > +0.005** 확보.
    *   대중성 통제 스트레스 스플릿(Random pop-bin)에서 기존 LightGCN 대비 **Delta > +0.012** 이상의 압도적 우위 검증.
    *   최종 예측 라벨의 대중성 분포가 편향 완화되었음을 지표로 확인.

#### 축 3: 리뷰 텍스트 기반 시맨틱 카테고리 피처 및 TF-IDF 결합 (Text Feasibility Axis)
*   **접근법:** `train.json`에 포함된 약 175,000건의 리뷰 `text`에 대해 TF-IDF, 감성 분석 점수 혹은 경량 텍스트 임베딩을 추출하여 유저 선호 장르/게임 메커니즘 프로파일을 생성하고, 이를 Content-Based Filtering 스코어로 변환합니다.
*   **차별성:** 상호작용 매트릭스(User-Item Matrix) 정보만 사용하는 CF 계열 모델들과 완전히 다르게, 텍스트에 나타난 구체적인 유저 유즈케이스와 게임 특징을 매칭하므로 협업 필터링이 놓치는 cold-ish 게임 예측에 특화됩니다.
*   **제출 전 필수 검증 증거:**
    *   Text-based score 단독으로 Uniform split 검증 세트 Accuracy > 0.62 이상 달성 확인.
    *   LightGCN과의 단순 Rank Blend 시 Uniform split 검증 정확도가 기존 단일 모델 대비 **Delta > +0.004** 증가하고, **Row-diff > 800** 이상 유지.

#### 축 4: 커뮤니티 자각 부정 샘플러 적용 Graph 학습 (ICPNS / Community-Aware Negative Sampling)
*   **접근법:** 유저-아이템 이부 그래프를 커뮤니티 구조로 분할하고, 부정 샘플(Negative)을 추출할 때 동일 커뮤니티 내부의 미소비 아이템(Hard Negative) 위주로 샘플링을 진행하여 LightGCN을 미세 튜닝합니다.
*   **차별성:** 전체 아이템 풀에서 무작위로 negative를 뽑아 쉬운 구분선만 학습하는 기존 BPR과 달리, 유사 선호도 그룹 내에서의 정교한 선호 우선순위를 강제로 학습하도록 유도합니다.
*   **제출 전 필수 검증 증거:**
    *   Random uniform gate 기준 검증 성능 **Delta > +0.006** 이상 달성.
    *   검증 스플릿 대상 McNemar-Fisher p-value < 0.05 통과로 통계적 유의성 입증.
    *   Public Best 대비 **Row-diff > 550** 확보.

#### 축 5: 실제 플레이 시간 강도 가중 EASE/ALS CF (Playtime-weighted implicit feedback)
*   **접근법:** binary interaction 데이터 대신, 로그 플레이 시간인 `hours_transformed = log2(1+hours)` 필드를 Implicit feedback의 신뢰도 가중치(Confidence weight $C_{ui} = 1 + \alpha \cdot r_{ui}$)로 명시 반영하여 EASE 또는 ALS를 학습시킵니다.
*   **차별성:** 단순히 "플레이 여부(1/0)"만 반영한 기존 CF 모델들과 달리, 찍어먹어 본 게임(낮은 hours)과 수백 시간 플레이한 인생 게임(높은 hours)의 강도를 구별하여 유저 선호 경계를 정밀화합니다.
*   **제출 전 필수 검증 증거:**
    *   단일 Weighted EASE/ALS 모델이 이진 EASE/ALS 대비 Uniform 스플릿 검증 정확도 **Delta > +0.004** 달성.
    *   기존 binary-based 베스트 앙상블 대비 예측 순위의 상관계수(Correlation)가 0.75 이하로 떨어지며, **Row-diff > 700** 확인.

---

GEMINI35_FLASH_ADVISORY_DONE
