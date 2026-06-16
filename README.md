English | [한국어](README.ko.md)

<div align="center">

# 🎮 KMU RecSys 26 Steam

### Kaggle Steam played-prediction pipeline with LightGCN, validation audits, and reproducible reports

*Coursework competition workspace for binary user-game recommendation, keeping code and reports public while raw Kaggle data and submission CSVs stay local.*

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle" />
  <img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black" alt="W&B" />
  <img src="https://img.shields.io/badge/uv-DE5FE9?style=for-the-badge&logo=astral&logoColor=white" alt="uv" />
</p>

<p>
  <img src="https://img.shields.io/badge/Status-Public_Archive-2E7D32?style=flat-square" alt="Status" />
  <img src="https://img.shields.io/badge/Course-KMU_AI-1F4E79?style=flat-square" alt="Course" />
  <img src="https://img.shields.io/badge/Problem-Binary_RecSys-365F91?style=flat-square" alt="Problem" />
  <img src="https://img.shields.io/badge/Best_Public-0.77825-B79A57?style=flat-square" alt="Best public score" />
  <img src="https://img.shields.io/badge/Raw_Data-Excluded-555555?style=flat-square" alt="Raw data excluded" />
  <img src="https://img.shields.io/badge/License-Not_specified-lightgrey?style=flat-square" alt="License not specified" />
</p>

</div>

---

## 📑 Table of Contents

- [🧭 About](#-about)
- [🎯 Headline Results](#-headline-results)
- [🏗 Architecture](#-architecture)
- [🛠 Tech Stack](#-tech-stack)
- [🗂 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [📝 Reproducing](#-reproducing)
- [🔒 Public Safety](#-public-safety)
- [👤 Author](#-author)

## 🧭 About

This repository documents my work for Kaggle [`kmu-rec-sys-26-steam`](https://www.kaggle.com/competitions/kmu-rec-sys-26-steam), a binary recommendation task that predicts whether a given `userID, gameID` pair was played.

> TL;DR — The strongest public submission was a normalized rank blend of two LightGCN-family candidates, while the most stable reproducible backbone was an emb128 LightGCN 4-seed ensemble.

Only public-safe material is tracked: source code, validation scripts, audit reports, selected figures, and reproducibility notes. Raw Kaggle files, generated score matrices, W&B local runs, final submission CSVs, and credentials are intentionally excluded.

## 🎯 Headline Results

| Slot / role | Candidate | Public score | Rows | SHA256 | Evidence |
|---|---|---:|---:|---|---|
| **Final slot 1 / public-best preservation** | **Rank blend: emb128 + emb192** | **0.77825** | **19,998** | `1d38c3ed…` | [`final_slot1` report](reports/20260612T2308KST_final_slot1_kaggle_submission_result.md) |
| Final slot 2 / stable backbone | LightGCN emb128 L4 reg1e-3, 4 seeds | 0.77745 | 19,998 | `7e3191de…` | [`emb128` repro report](reports/20260530_ecampus_repro_emb128L4r3_077745.md) |
| Earlier anchor | LightGCN emb64 L3 reg1e-4, 4 seeds | 0.77125 | 19,998 | `dcc578de…` | [`seed ensemble` report](reports/20260530_ecampus_repro_seed_ens_077125.md) |
| First submitted baseline blend | BM25 + EASE + ALS mean-z blend | 0.74594 | 19,998 | `5f93cf1b…` | [`submission` report](reports/20260530T124312KST_submission_analysis.md) |

Important interpretation: the rank-blend row had the best public score, but its internal validation margin was weak. The emb128 LightGCN ensemble stayed as the safer reproducible backbone because its byte-identical regeneration and validation behavior were cleaner.

## 🏗 Architecture

```mermaid
flowchart LR
  subgraph DATA["📥 Data (local only)"]
    KAGGLE["Kaggle train.json / pairs.csv"]
    LOCAL["data/ ignored by Git"]
  end

  subgraph VAL["🧪 Validation"]
    SPLITS["user-level heldout splits"]
    NEG["uniform / sqrt-pop / pop-bin negatives"]
    TOPHALF["per-user top-half decode"]
  end

  subgraph MODELS["🤖 Models"]
    LGCN["LightGCN"]
    CF["ItemKNN / EASE / ALS / GF-CF"]
    PROBES["boundary, semantic, pseudo-label probes"]
  end

  subgraph AUDIT["📊 Evidence"]
    REPORTS["reports/*.md / *.json / figures"]
    PREFLIGHT["schema, SHA, label-balance checks"]
    PUBLIC["Kaggle public-score records"]
  end

  subgraph OUTPUT["📦 Local outputs"]
    SUBMIT["submissions/ ignored"]
    FINAL["final_package/ ignored"]
  end

  KAGGLE --> LOCAL --> SPLITS --> NEG --> TOPHALF
  TOPHALF --> LGCN
  TOPHALF --> CF
  LGCN --> PROBES
  CF --> PROBES
  LGCN --> PREFLIGHT
  PROBES --> REPORTS
  PREFLIGHT --> SUBMIT --> FINAL
  PUBLIC --> REPORTS
```

The repository keeps the decision trail in Git, not the raw competition files. Local-only data feeds validation and candidate generation; tracked reports record the tested axes, failed gates, public-score outcomes, and final reproducibility checks.

## 🛠 Tech Stack

| Role | Tools |
|---|---|
| Modeling | PyTorch, LightGCN, ItemKNN, EASE, ALS/WMF, GF-CF-style probes |
| Validation | NumPy, pandas, SciPy, user-level candidate splits, per-user top-half decoding |
| Experiment tracking | W&B summary logging, JSON/Markdown audit reports |
| Agent-assisted review | OpenCode / Hephaestus, AI-Q research notes, manual safety gates |
| Packaging | SHA256 preflight, final-slot reports, eCampus reproducibility manifests |

## 🗂 Project Structure

```text
kmu-rec-sys-26-steam/
├── README.md / README.ko.md          # public-facing bilingual overview
├── docs/                             # competition rules and operating notes
├── scripts/                          # ★ modeling, validation, materialization, and audit code
├── reports/                          # ★ tracked evidence: results, preflights, figures, decisions
├── state/                            # small runner state files that explain automation decisions
├── data/                             # ignored: raw Kaggle files
├── artifacts/                        # ignored: score matrices, model outputs, per-seed test scores
├── submissions/                      # ignored: generated Kaggle CSVs
├── final_package/                    # ignored: final label CSVs for external submission
├── wandb_runs/                       # ignored: W&B local cache
└── .gitignore                        # public-safety guardrails
```

## 🚀 Quick Start

Clone the public repository first:

```bash
git clone https://github.com/mrpc2003/kmu-rec-sys-26-steam.git
cd kmu-rec-sys-26-steam
```

Download the competition files from Kaggle, then place them in the local ignored data directory used by the scripts. The raw data is not redistributed in this repository.

```text
data/raw/public/data/train.json
data/raw/public/data/pairs.csv
```

Run a lightweight command check without submitting anything:

```bash
uv run --with numpy --with pandas --with scipy \
  python scripts/build_validation_splits.py --help
```

## 📝 Reproducing

The final stable backbone is reproduced through `scripts/reproduce_submission_emb128.py`. The default verification mode expects the local per-seed score files under `artifacts/`, which are excluded from Git because they are generated outputs.

```bash
uv run --with numpy --with pandas --with scipy \
  python scripts/reproduce_submission_emb128.py --verify-existing
```

For a full GPU rerun from the raw Kaggle data, use the same script with `--from-scratch`. This trains four deterministic LightGCN seeds and then regenerates the candidate CSV before checking the expected SHA.

```bash
uv run --python 3.13 \
  --with torch==2.10.0 --with numpy --with pandas --with scipy \
  python scripts/reproduce_submission_emb128.py --from-scratch --device cuda:0
```

No script in this repository performs a Kaggle submission as part of reproduction. Submission was handled separately after explicit approval and recorded in reports.

## 🔒 Public Safety

The repository was made public after a dedicated public-readiness audit:

- current forbidden prefixes: `0`
- historical forbidden prefixes after cleanup: `0`
- credential findings: `0`
- blobs over 100 MB: `0`
- fresh public clone verification: passed

See [`reports/20260616T2352KST_public_readiness_audit.md`](reports/20260616T2352KST_public_readiness_audit.md) for the audit record. A top-level code license has not been chosen yet, so public viewing is allowed but reuse terms are not specified.

## 👤 Author

[@mrpc2003](https://github.com/mrpc2003) — Kim Woohyun, Kookmin University AI major.

<div align="center">

<sub>Built with PyTorch, Kaggle validation discipline, and a lot of no-submit safety checks.</sub>

</div>
