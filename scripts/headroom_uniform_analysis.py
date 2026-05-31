"""Headroom + decorrelation analysis on the uniform (public-surrogate) split.

Compares the emb64 4-seed ensemble vs emb128 4-seed ensemble:
- their uniform row accuracy
- within-user score-rank correlation (how diverse are the two signals?)
- per-row correctness overlap and the best-of-2 oracle upper bound

This tells us whether a DIFFERENT axis (loss / architecture) has room to help:
- low corr + large oracle headroom => a diverse model could lift the ensemble a lot
- high corr + tiny headroom        => more of the same backbone won't move much
No Kaggle submission.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import evaluate_tophalf, predict_tophalf

SPLIT = "val_random_uniform_seed42"


def load(p):
    return pd.read_csv(p)[["ID", "userID", "gameID", "Label", "score_lightgcn"]].rename(
        columns={"score_lightgcn": "s"})


E64 = {
    42: ROOT / "artifacts/lightgcn_ood_robustness" / SPLIT / "lightgcn_scores.csv",
    123: ROOT / "artifacts/lightgcn_uniform_eval/seed123" / SPLIT / "lightgcn_scores.csv",
    2024: ROOT / "artifacts/lightgcn_uniform_eval/seed2024" / SPLIT / "lightgcn_scores.csv",
    7: ROOT / "artifacts/lightgcn_uniform_eval/seed7" / SPLIT / "lightgcn_scores.csv",
}
E128 = {
    42: ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / SPLIT / "lightgcn_scores.csv",
    123: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed123" / SPLIT / "lightgcn_scores.csv",
    2024: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed2024" / SPLIT / "lightgcn_scores.csv",
    7: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed7" / SPLIT / "lightgcn_scores.csv",
}


def ens(files):
    base = None
    cols = []
    for s, p in files.items():
        d = load(p).rename(columns={"s": f"s{s}"})
        base = d if base is None else base.merge(d[["ID", f"s{s}"]], on="ID")
        cols.append(f"s{s}")
    base["s"] = base[cols].mean(axis=1)
    return base


def acc(df):
    return evaluate_tophalf(df, "s", label_col="Label", user_col="userID", id_col="ID")[0]["row_accuracy"]


def wz(df, c):
    g = df.groupby("userID")[c]
    return (df[c] - g.transform("mean")) / (g.transform("std") + 1e-9)


def correct_map(df, col):
    p = predict_tophalf(df.rename(columns={col: "sc"}), "sc", label_col=None, user_col="userID", id_col="ID")
    pp = p[["ID", "Pred"]].merge(df[["ID", "Label"]], on="ID")
    return dict(zip(pp.ID, (pp.Pred == pp.Label)))


def main():
    e64 = ens(E64)
    e128 = ens(E128)
    a64, a128 = acc(e64), acc(e128)
    print(f"emb64  ens uniform : {a64:.5f}")
    print(f"emb128 ens uniform : {a128:.5f}")

    m = e64[["ID", "userID", "Label", "s"]].rename(columns={"s": "s64"}).merge(
        e128[["ID", "s"]].rename(columns={"s": "s128"}), on="ID")
    m["z64"] = wz(m, "s64")
    m["z128"] = wz(m, "s128")
    print(f"corr(z64, z128)    : {m['z64'].corr(m['z128']):.4f}   (lower => more diverse)")

    # 50/50 z-blend accuracy: cheap test of whether even same-family diversity helps
    m["zblend"] = 0.5 * m["z64"] + 0.5 * m["z128"]
    ab = evaluate_tophalf(m, "zblend", label_col="Label", user_col="userID", id_col="ID")[0]["row_accuracy"]
    print(f"50/50 z-blend(64,128): {ab:.5f}  (Δ vs emb128 {ab - a128:+.5f})")

    c64 = correct_map(m, "s64")
    c128 = correct_map(m, "s128")
    ids = list(m.ID)
    N = len(ids)
    both = sum(1 for i in ids if c64[i] and c128[i])
    o64 = sum(1 for i in ids if c64[i] and not c128[i])
    o128 = sum(1 for i in ids if c128[i] and not c64[i])
    neither = sum(1 for i in ids if not c64[i] and not c128[i])
    oracle = (both + o64 + o128) / N
    print(f"rows: both={both/N:.4f} only64={o64/N:.4f} only128={o128/N:.4f} neither={neither/N:.4f}")
    print(f"best-of-2 oracle   : {oracle:.5f}  (headroom over emb128 = +{oracle - a128:.4f})")
    print(f"unreachable ceiling: {1-neither/N:.5f}  (rows where at least one is right)")


if __name__ == "__main__":
    main()
