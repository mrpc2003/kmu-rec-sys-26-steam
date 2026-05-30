"""DECISIVE GATE: evaluate the LightGCN seed ensemble on the UNIFORM split.

Why this matters
----------------
The seed ensemble (+0.0041 mean) was measured ONLY on the hard negative samplers
(sqrtpop / recent / popbin). But the OOD robustness gate showed the real Kaggle public
score (0.76245) tracks the UNIFORM split (seed42 uniform = 0.75445, the closest of all
six samplers). The hard-sampler gain is therefore NOT proof the ensemble helps on the
distribution that actually decides the leaderboard.

This script closes that gap: it averages the SAME verified config (emb64 L3 reg1e-4)
across seeds 42/123/2024/7 on the uniform split and checks whether the ensemble beats
the seed42 single model THERE.

Consistency guarantee
----------------------
All four seeds use the identical code path (lightgcn_train.py); train_interactions.csv
is md5-identical across uniform/sqrtpop, so only the scoring negatives differ.

  - seed42 uniform : artifacts/lightgcn_ood_robustness/val_random_uniform_seed42/lightgcn_scores.csv  (from OOD run)
  - seed{S} uniform: artifacts/lightgcn_uniform_eval/seed{S}/val_random_uniform_seed42/lightgcn_scores.csv

No Kaggle submission. Report-only gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import evaluate_tophalf  # noqa: E402

SPLIT = "val_random_uniform_seed42"
SEED42_PATH = ROOT / "artifacts/lightgcn_ood_robustness" / SPLIT / "lightgcn_scores.csv"
NEW_SEEDS = [123, 2024, 7]
SEED42_UNIFORM_REF = 0.75445  # from OOD robustness run (lightgcn_train.py, emb64 L3 reg1e-4)

OUT_JSON = ROOT / "reports/20260530_seed_ensemble_uniform_gate.json"
OUT_MD = ROOT / "reports/20260530_seed_ensemble_uniform_gate.md"


def new_seed_path(seed: int) -> Path:
    return ROOT / f"artifacts/lightgcn_uniform_eval/seed{seed}" / SPLIT / "lightgcn_scores.csv"


def main() -> None:
    if not SEED42_PATH.exists():
        raise FileNotFoundError(f"seed42 uniform scores missing: {SEED42_PATH}")
    base = pd.read_csv(SEED42_PATH)[["ID", "userID", "gameID", "Label", "score_lightgcn"]]
    base = base.rename(columns={"score_lightgcn": "score_seed42"})

    m = base
    seed_cols = ["score_seed42"]
    present = [42]
    missing = []
    for s in NEW_SEEDS:
        p = new_seed_path(s)
        if not p.exists():
            missing.append(s)
            continue
        d = pd.read_csv(p)[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": f"score_seed{s}"})
        m = m.merge(d, on="ID", how="inner")
        seed_cols.append(f"score_seed{s}")
        present.append(s)

    if missing:
        print(f"[wait] seeds not ready yet: {missing} (present={present})", flush=True)
        if len(present) < 2:
            print("Need at least 2 seeds to form an ensemble. Aborting.")
            return
        print(f"[partial] forming ensemble with available seeds {present}", flush=True)

    m["score_ens"] = m[seed_cols].mean(axis=1)

    per_seed = {}
    for c in seed_cols:
        summ, _ = evaluate_tophalf(m, c, label_col="Label", user_col="userID", id_col="ID")
        per_seed[c] = round(float(summ["row_accuracy"]), 5)
    ens_summ, _ = evaluate_tophalf(m, "score_ens", label_col="Label", user_col="userID", id_col="ID")
    ens_acc = round(float(ens_summ["row_accuracy"]), 5)

    gain_vs_seed42 = round(ens_acc - SEED42_UNIFORM_REF, 5)
    verdict = (
        "ROBUST_GAIN" if gain_vs_seed42 > 0.0005 else
        "FLAT" if gain_vs_seed42 >= -0.0005 else
        "REGRESSION"
    )

    summary = {
        "note": "Decisive gate: seed ensemble on uniform (public surrogate). No Kaggle submission.",
        "split": SPLIT,
        "seeds_present": present,
        "seeds_missing": missing,
        "rows": int(len(m)),
        "per_seed_acc": per_seed,
        "ensemble_acc": ens_acc,
        "seed42_uniform_ref": SEED42_UNIFORM_REF,
        "ens_vs_seed42_uniform": gain_vs_seed42,
        "verdict": verdict,
        "interpretation": {
            "ROBUST_GAIN": "Ensemble beats seed42 on the real public surrogate -> strong submit candidate.",
            "FLAT": "No meaningful change on uniform -> ensemble is safe but not clearly better; keep seed42 anchor.",
            "REGRESSION": "Ensemble hurts on uniform -> hard-sampler gain was an artifact; do NOT submit ensemble.",
        }[verdict],
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md = [
        "# Seed Ensemble — UNIFORM Gate (real public surrogate)\n",
        f"- split: `{SPLIT}`  (rows={len(m)})",
        f"- seeds present: {present}" + (f"  | missing: {missing}" if missing else ""),
        f"- seed42 uniform ref: **{SEED42_UNIFORM_REF}**",
        f"- ensemble uniform: **{ens_acc}**",
        f"- **Δ vs seed42 on uniform: {gain_vs_seed42:+.5f}**",
        f"- **verdict: {verdict}** — {summary['interpretation']}\n",
        "| seed | uniform row acc |",
        "|---|---:|",
    ]
    for c in seed_cols:
        md.append(f"| {c.replace('score_seed','')} | {per_seed[c]} |")
    md.append(f"| **ensemble** | **{ens_acc}** |")
    md.append(
        "\n## Why this gate\n"
        "The +0.0041 mean ensemble gain was measured on hard samplers only. "
        "The OOD gate proved public (0.76245) tracks the uniform split, so the gain must be "
        "confirmed HERE before the ensemble can be treated as a stronger candidate than the "
        "submitted seed42 LightGCN (public 0.76245)."
    )
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\n[UNIFORM GATE] ensemble={ens_acc} seed42_ref={SEED42_UNIFORM_REF} "
          f"Δ={gain_vs_seed42:+.5f} verdict={verdict}", flush=True)
    print(f"per-seed uniform: {per_seed}", flush=True)
    print(f"saved: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
