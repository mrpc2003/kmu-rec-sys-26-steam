"""Quick decorrelation check: best ALS (hours-confidence WMF) vs emb128 ensemble on uniform.

ALS alone is weak on uniform (~0.71), but the question is whether its hours/popularity
signal is DECORRELATED enough from the BPR-LightGCN family to help the ensemble crack the
21.4% "neither correct" rows. Reports corr(z) and best within-user z-blend.
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
ALS_SCORES = ROOT / "artifacts/als_uniform/val_random_uniform_seed42/candidate_scores.csv"
ALS_COLS = ["score_als_f64_it30_alpha20_popa2", "score_als_f64_it30_alpha20",
            "score_als_htr_f64_it30_alpha20_popa2"]


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
    als = pd.read_csv(ALS_SCORES)
    print(f"emb128 ens uniform = {EMB128_ENS}\n")
    for col in ALS_COLS:
        if col not in als.columns:
            print(f"[skip] {col} missing")
            continue
        m = base.merge(als[["ID", col]], on="ID")
        m["zals"] = wz(m, col)
        solo = acc(m, col)
        corr = float(m["zals"].corr(m["z128"]))
        best_w, best_blend = None, -1.0
        for w in np.linspace(0, 1, 21):
            m["zb"] = w * m["z128"] + (1 - w) * m["zals"]
            ab = acc(m, "zb")
            if ab > best_blend:
                best_blend, best_w = ab, float(w)
        tag = "USEFUL" if best_blend > EMB128_ENS + NOISE else "redundant"
        print(f"{col}: solo={solo:.5f} corr_z={corr:.3f} "
              f"best_blend={best_blend:.5f} @w128={best_w:.2f} (Δ{best_blend-EMB128_ENS:+.5f}) {tag}")


if __name__ == "__main__":
    main()
