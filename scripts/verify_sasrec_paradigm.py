"""Independent verification + comparison harness for the SASRec paradigm sweep.

Does NOT trust each run's summary.json. Recomputes everything directly from the
score CSVs:
  - score validity (NaN fraction, unique values, range) -> not degenerate?
  - solo_acc (per-user top-half) via the canonical evaluate_tophalf
  - corr_z vs emb128 4-seed ensemble (within-user z)
  - eq_blend (50/50 within-user z) accuracy and Δ vs emb128 ref
Then prints a single comparison table across base + variants and assigns a tier
per the gate (FLOOR / GEOMETRY_REDUNDANT / SIGNAL_ESCALATE).

No GPU, no submission. Writes reports/20260601_sasrec_paradigm_verification.json.
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
SEEDS = [42, 123, 2024, 7]
FLOOR = 0.684
EMB128_REF = 0.76505
NOISE = 0.0007

RUNS = {
    "base_d64_L50":  ROOT / "artifacts/sasrec",
    "d128_L50":      ROOT / "artifacts/sasrec_variants/d128_L50",
    "d64_L100":      ROOT / "artifacts/sasrec_variants/d64_L100",
    "d64_L20":       ROOT / "artifacts/sasrec_variants/d64_L20",
}


def emb128_path(seed):
    if seed == 42:
        return ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / SPLIT / "lightgcn_scores.csv"
    return ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}" / SPLIT / "lightgcn_scores.csv"


def load_emb128():
    base = pd.read_csv(emb128_path(42))[["ID", "userID", "Label", "score_lightgcn"]].rename(columns={"score_lightgcn": "e42"})
    for s in SEEDS[1:]:
        d = pd.read_csv(emb128_path(s))[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": f"e{s}"})
        base = base.merge(d, on="ID", how="inner")
    base["emb128"] = base[[f"e{s}" for s in SEEDS]].mean(axis=1)
    return base[["ID", "userID", "Label", "emb128"]]


def zwu(df, col):
    g = df.groupby("userID")[col]
    return (df[col] - g.transform("mean")) / g.transform("std").replace(0, 1).fillna(1)


def find_scores(run_dir):
    hits = list(run_dir.rglob("lightgcn_scores.csv"))
    return hits[0] if hits else None


def main():
    ref = load_emb128()
    emb128_solo, _ = evaluate_tophalf(ref, "emb128", label_col="Label", user_col="userID", id_col="ID")
    emb128_solo = round(float(emb128_solo["row_accuracy"]), 5)

    rows = []
    for name, rdir in RUNS.items():
        sc = find_scores(rdir)
        if sc is None:
            rows.append({"run": name, "status": "PENDING (no scores csv yet)"})
            continue
        d = pd.read_csv(sc)
        s = d["score_lightgcn"]
        nan_frac = float(s.isna().mean())
        # treat the -1e9 cold sentinel as non-informative but finite
        finite = s.replace([-1e9], np.nan)
        valid = {
            "nan_frac": round(nan_frac, 4),
            "cold_sentinel_frac": round(float((s <= -1e8).mean()), 4),
            "unique_finite": int(finite.dropna().nunique()),
        }
        solo_summ, _ = evaluate_tophalf(d, "score_lightgcn", label_col="Label", user_col="userID", id_col="ID")
        solo = round(float(solo_summ["row_accuracy"]), 5)

        m = d[["ID", "userID", "Label", "score_lightgcn"]].merge(ref[["ID", "emb128"]], on="ID", how="inner")
        m["zs"] = zwu(m, "score_lightgcn")
        m["ze"] = zwu(m, "emb128")
        corr = round(float(np.corrcoef(m["zs"], m["ze"])[0, 1]), 4)
        m["blend"] = 0.5 * m["zs"] + 0.5 * m["ze"]
        eb, _ = evaluate_tophalf(m, "blend", label_col="Label", user_col="userID", id_col="ID")
        eqb = round(float(eb["row_accuracy"]), 5)
        d_blend = round(eqb - EMB128_REF, 5)

        if solo < FLOOR:
            tier = "REJECT_FLOOR"
        elif d_blend > NOISE and corr < 0.9:
            tier = "SIGNAL_ESCALATE"
        else:
            tier = "GEOMETRY_REDUNDANT"

        rows.append({
            "run": name, "status": "DONE", "validity": valid,
            "solo_acc": solo, "corr_z": corr, "eq_blend": eqb,
            "eq_blend_minus_ref": d_blend, "tier": tier,
        })

    out = {
        "note": "Independent verification of SASRec paradigm sweep (recomputed from CSVs).",
        "split": SPLIT, "floor": FLOOR, "emb128_ref": EMB128_REF,
        "emb128_solo_recomputed": emb128_solo, "noise": NOISE, "runs": rows,
    }
    (ROOT / "reports/20260601_sasrec_paradigm_verification.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"{'run':16s} {'solo':>8s} {'corr_z':>8s} {'eq_blend':>9s} {'Δvsref':>8s}  tier")
    print("-" * 70)
    print(f"{'emb128_4seed':16s} {emb128_solo:>8.5f} {'1.0000':>8s} {'—':>9s} {'—':>8s}  REF")
    for r in rows:
        if r["status"] != "DONE":
            print(f"{r['run']:16s} {r['status']}")
            continue
        print(f"{r['run']:16s} {r['solo_acc']:>8.5f} {r['corr_z']:>8.4f} {r['eq_blend']:>9.5f} "
              f"{r['eq_blend_minus_ref']:>+8.5f}  {r['tier']}")
    print("\nsaved: reports/20260601_sasrec_paradigm_verification.json")


if __name__ == "__main__":
    main()
