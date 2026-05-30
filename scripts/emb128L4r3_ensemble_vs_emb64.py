"""APPLES-TO-APPLES: emb128_L4_reg1e-3 4-seed ensemble vs emb64_L3_reg1e-4 4-seed ensemble.

Why this script
---------------
The sweep-best uniform gate flagged emb128_L4_reg1e-3 as "new strongest" — but that
compared a SINGLE seed (0.76205) against the emb64 4-SEED ensemble (0.76145). That is
unfair: single-seed uniform variance here is ~0.0007 (seeds 123/2024/7 spanned
0.75735-0.75805 for the emb64 config), so a +0.0006 edge is inside the noise. The honest
comparison is ensemble-vs-ensemble, same seeds {42,123,2024,7}, same uniform split.

  emb128_L4_reg1e-3 seeds:
    seed42 : artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03/val_random_uniform_seed42/lightgcn_scores.csv
    123/2024/7: artifacts/lightgcn_emb128L4r3_ens/seed{S}/val_random_uniform_seed42/lightgcn_scores.csv
  All columns: score_lightgcn (lightgcn_train.py canonical output).

Reference (current submitted candidate, emb64_L3_reg1e-4 ensemble): uniform 0.76145.

Decision: only treat emb128 as a real upgrade if its 4-seed ensemble beats the emb64
ensemble on uniform by MORE than the single-seed noise band (~+0.0007). Otherwise the two
are statistically tied and we keep the already-submitted emb64 ensemble (public 0.77125)
as the primary, and may hold emb128 as a candidate-2 only if clearly justified.

No Kaggle submission. Report-only.
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
EMB64_ENS_UNIFORM = 0.76145      # current submitted candidate (public 0.77125)
EMB128_SINGLE_SEED42 = 0.76205   # sweep-best single-seed uniform
ANCHOR_UNIFORM = 0.75445
NOISE_BAND = 0.0007              # measured single-seed uniform spread

SEED42_PATH = ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / SPLIT / "lightgcn_scores.csv"
NEW_SEEDS = [123, 2024, 7]

OUT_JSON = ROOT / "reports/20260530_emb128L4r3_ensemble_vs_emb64.json"
OUT_MD = ROOT / "reports/20260530_emb128L4r3_ensemble_vs_emb64.md"


def new_seed_path(seed: int) -> Path:
    return ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}" / SPLIT / "lightgcn_scores.csv"


def main() -> None:
    if not SEED42_PATH.exists():
        raise FileNotFoundError(f"emb128 seed42 uniform scores missing: {SEED42_PATH}")
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
        print(f"[wait] emb128 seeds not ready: {missing} (present={present})", flush=True)
        if len(present) < 4:
            print("Need all 4 seeds for the clean comparison. Aborting.")
            return

    m["score_ens"] = m[seed_cols].mean(axis=1)

    per_seed = {}
    for c in seed_cols:
        summ, _ = evaluate_tophalf(m, c, label_col="Label", user_col="userID", id_col="ID")
        per_seed[c] = round(float(summ["row_accuracy"]), 5)
    ens_summ, _ = evaluate_tophalf(m, "score_ens", label_col="Label", user_col="userID", id_col="ID")
    ens_acc = round(float(ens_summ["row_accuracy"]), 5)

    vs_emb64_ens = round(ens_acc - EMB64_ENS_UNIFORM, 5)
    if vs_emb64_ens > NOISE_BAND:
        verdict = (f"emb128_L4_reg1e-3 ensemble ({ens_acc}) beats emb64 ensemble "
                   f"({EMB64_ENS_UNIFORM}) by {vs_emb64_ens:+.5f} > noise {NOISE_BAND} "
                   f"-> genuine upgrade, consider as new primary.")
        tier = "UPGRADE"
    elif vs_emb64_ens < -NOISE_BAND:
        verdict = (f"emb128 ensemble ({ens_acc}) LOSES to emb64 ensemble by {vs_emb64_ens:+.5f} "
                   f"-> keep emb64 ensemble (already submitted, public 0.77125).")
        tier = "WORSE"
    else:
        verdict = (f"emb128 ensemble ({ens_acc}) vs emb64 ensemble ({EMB64_ENS_UNIFORM}) = "
                   f"{vs_emb64_ens:+.5f}, WITHIN the {NOISE_BAND} single-seed noise band -> "
                   f"statistically TIED. Keep emb64 ensemble as primary (already public 0.77125); "
                   f"emb128 ensemble is a viable candidate-2 (diversity) but not a clear upgrade.")
        tier = "TIED"

    summary = {
        "note": "emb128_L4_reg1e-3 4-seed ensemble vs emb64_L3_reg1e-4 4-seed ensemble on uniform. No submission.",
        "split": SPLIT,
        "seeds_present": present,
        "emb128_per_seed_uniform": per_seed,
        "emb128_ensemble_uniform": ens_acc,
        "emb64_ensemble_uniform_ref": EMB64_ENS_UNIFORM,
        "emb128_single_seed42_ref": EMB128_SINGLE_SEED42,
        "anchor_uniform": ANCHOR_UNIFORM,
        "emb128ens_vs_emb64ens": vs_emb64_ens,
        "emb128ens_vs_anchor": round(ens_acc - ANCHOR_UNIFORM, 5),
        "noise_band": NOISE_BAND,
        "tier": tier,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    md = ["# emb128_L4_reg1e-3 ensemble vs emb64 ensemble — UNIFORM (public surrogate)\n",
          f"- emb64 ensemble (submitted, public 0.77125): **{EMB64_ENS_UNIFORM}**",
          f"- emb128 single-seed42 (sweep gate): {EMB128_SINGLE_SEED42}",
          f"- anchor: {ANCHOR_UNIFORM}  |  single-seed noise band: ±{NOISE_BAND}",
          f"- **emb128 4-seed ensemble: {ens_acc}**",
          f"- **vs emb64 ensemble: {vs_emb64_ens:+.5f}** → **{tier}**",
          f"- verdict: {verdict}\n",
          "| seed | emb128 uniform acc |", "|---|---:|"]
    for c in seed_cols:
        md.append(f"| {c.replace('score_seed','')} | {per_seed[c]} |")
    md.append(f"| **ensemble** | **{ens_acc}** |")
    md.append("\n## Interpretation\n"
              "The sweep gate's single-seed 0.76205 vs the emb64 ensemble 0.76145 was an unfair "
              "comparison (single seed vs 4-seed mean). This evaluates emb128 as its own 4-seed "
              "ensemble so both sides have equal variance reduction. A within-noise result means "
              "the two configs are interchangeable on the surrogate; the emb64 ensemble stays "
              "primary because it is already validated on the real public LB (0.77125).")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(verdict)
    print(f"emb128 per-seed: {per_seed}")
    print(f"emb128 ensemble uniform = {ens_acc} | vs emb64 ens {vs_emb64_ens:+.5f} | tier={tier}")


if __name__ == "__main__":
    main()
