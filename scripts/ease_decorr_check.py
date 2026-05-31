"""Decorrelation check: pre-computed EASE / ItemKNN (uniform proto) vs emb128 ensemble.

EASE (closed-form item-item autoencoder) and ItemKNN have a DIFFERENT geometry than
LightGCN (graph CF) and ALS (MF), and unlike ALS they do not lean on popularity^alpha.
Scores already exist in artifacts/scores/val_random_uniform_seed42_proto/, so this is a
free CPU decorrelation probe — no retraining.

Applies the parameter-free rule: a column is a useful new axis only if it beats emb128
SOLO, or its EQUAL-WEIGHT (50/50) z-blend with emb128 beats emb128. A win that exists only
at a grid-tuned weight is the stacker trap and is reported as diagnostic-only.
No Kaggle submission.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import evaluate_tophalf

SPLIT = "val_random_uniform_seed42"
EMB128_ENS = 0.76505
NOISE = 0.0007

E128 = {
    42: ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / SPLIT / "lightgcn_scores.csv",
    123: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed123" / SPLIT / "lightgcn_scores.csv",
    2024: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed2024" / SPLIT / "lightgcn_scores.csv",
    7: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed7" / SPLIT / "lightgcn_scores.csv",
}
PROTO = ROOT / "artifacts/scores/val_random_uniform_seed42_proto/candidate_scores.csv"
COLS = ["score_ease_lambda100", "score_ease_lambda300", "score_ease_lambda1000",
        "score_itemknn_top3", "score_itemknn_htr_weighted"]


def ens128():
    base = None
    cols = []
    for s, p in E128.items():
        d = pd.read_csv(p)[["ID", "userID", "gameID", "Label", "score_lightgcn"]].rename(
            columns={"score_lightgcn": f"s{s}"})
        base = d if base is None else base.merge(d[["ID", f"s{s}"]], on="ID")
        cols.append(f"s{s}")
    base["s128"] = base[cols].mean(axis=1)
    return base[["ID", "userID", "gameID", "Label", "s128"]]


def acc(df, col):
    return float(evaluate_tophalf(df, col, label_col="Label", user_col="userID", id_col="ID")[0]["row_accuracy"])


def wz(df, c):
    g = df.groupby("userID")[c]
    return (df[c] - g.transform("mean")) / (g.transform("std") + 1e-9)


def main():
    base = ens128()
    base["z128"] = wz(base, "s128")
    proto = pd.read_csv(PROTO)
    if "ID" not in proto.columns:
        proto = proto.reset_index().rename(columns={"index": "ID"})
    print(f"emb128 ens uniform = {EMB128_ENS}  (parameter-free verdict; grid-max diag-only)\n")
    for col in COLS:
        if col not in proto.columns:
            print(f"[skip] {col} missing")
            continue
        m = base.merge(proto[["ID", col]], on="ID")
        if m[col].isna().any():
            m[col] = m[col].fillna(m[col].min())
        m["zc"] = wz(m, col)
        solo = acc(m, col)
        corr = float(m["zc"].corr(m["z128"]))
        m["zeq"] = 0.5 * m["z128"] + 0.5 * m["zc"]
        a_eq = acc(m, "zeq")
        best_w, best_blend = None, -1.0
        for w in np.linspace(0, 1, 21):
            m["zb"] = w * m["z128"] + (1 - w) * m["zc"]
            ab = acc(m, "zb")
            if ab > best_blend:
                best_blend, best_w = ab, float(w)
        verdict = ("STRONG_SOLO" if solo > EMB128_ENS + NOISE else
                   "NEW_AXIS" if a_eq > EMB128_ENS + NOISE else
                   "REDUNDANT")
        print(f"{col:32s}: solo={solo:.5f} corr={corr:.3f} eq50/50={a_eq:.5f} "
              f"(Δ{a_eq-EMB128_ENS:+.5f}) => {verdict}  [grid-max {best_blend:.5f}@w128={best_w:.2f}]")


if __name__ == "__main__":
    main()
