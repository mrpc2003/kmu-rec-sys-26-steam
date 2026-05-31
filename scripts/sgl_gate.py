"""SGL gate: parameter-free decorrelation/strength check vs emb128 ensemble on uniform.

Same logic as directau_gate.py (parameter-free 50/50 z-blend is the trustworthy verdict;
grid-max is diagnostic-only / stacker-risk), applied to the SGL lambda sweep.

A useful new axis must beat emb128 SOLO or via the EQUAL-WEIGHT z-blend. SGL is designed to
keep BPR strength (so solo should stay ~0.76, unlike DirectAU's 0.55) while the contrastive
term decorrelates it from plain BPR-LightGCN — so the key signal is: does solo stay strong
AND does corr drop below the 0.97 of the BPR family, with a positive eq-blend?
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
LAMBDAS = ["0.05", "0.1", "0.2", "0.5"]

E128 = {
    42: ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / SPLIT / "lightgcn_scores.csv",
    123: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed123" / SPLIT / "lightgcn_scores.csv",
    2024: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed2024" / SPLIT / "lightgcn_scores.csv",
    7: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed7" / SPLIT / "lightgcn_scores.csv",
}


def load(p):
    return pd.read_csv(p)[["ID", "userID", "gameID", "Label", "score_lightgcn"]].rename(
        columns={"score_lightgcn": "s"})


def ens128():
    base = None
    cols = []
    for s, p in E128.items():
        d = load(p).rename(columns={"s": f"s{s}"})
        base = d if base is None else base.merge(d[["ID", f"s{s}"]], on="ID")
        cols.append(f"s{s}")
    base["s128"] = base[cols].mean(axis=1)
    return base[["ID", "userID", "gameID", "Label", "s128"]]


def acc(df, col="s"):
    return float(evaluate_tophalf(df, col, label_col="Label", user_col="userID", id_col="ID")[0]["row_accuracy"])


def wz(df, c):
    g = df.groupby("userID")[c]
    return (df[c] - g.transform("mean")) / (g.transform("std") + 1e-9)


def main():
    base = ens128()
    base["z128"] = wz(base, "s128")
    print(f"refs: emb128_ens={EMB128_ENS}  noise=±{NOISE}")
    print("parameter-free 50/50 z-blend is the verdict number; grid-max is diagnostic-only.\n")
    rows = []
    for lam in LAMBDAS:
        p = ROOT / f"artifacts/sgl_uniform/lam{lam}" / SPLIT / "lightgcn_scores.csv"
        if not p.exists():
            print(f"[lam {lam}] not ready")
            continue
        d = load(p).rename(columns={"s": "ssgl"})
        m = base.merge(d[["ID", "ssgl"]], on="ID")
        m["zsgl"] = wz(m, "ssgl")
        solo = acc(m.rename(columns={"ssgl": "s"}))
        corr = float(m["zsgl"].corr(m["z128"]))
        m["zeq"] = 0.5 * m["z128"] + 0.5 * m["zsgl"]
        a_eq = acc(m.rename(columns={"zeq": "s"}))
        best_w, best_blend = None, -1.0
        for w in np.linspace(0, 1, 21):
            m["zb"] = w * m["z128"] + (1 - w) * m["zsgl"]
            ab = acc(m.rename(columns={"zb": "s"}))
            if ab > best_blend:
                best_blend, best_w = ab, float(w)
        verdict = ("STRONG_SOLO" if solo > EMB128_ENS + NOISE else
                   "NEW_AXIS" if a_eq > EMB128_ENS + NOISE else
                   "REDUNDANT")
        rows.append((lam, solo, corr, a_eq, best_w, best_blend, verdict))
        print(f"[lam {lam}] solo={solo:.5f} (vs128 {solo-EMB128_ENS:+.5f}) | corr_z={corr:.3f} | "
              f"eq-blend={a_eq:.5f} (vs128 {a_eq-EMB128_ENS:+.5f}) => {verdict}  "
              f"[grid-max {best_blend:.5f}@w128={best_w:.2f} diag]")
    if rows:
        print("\nsummary (sorted by eq-blend):")
        for lam, solo, corr, aeq, bw, bb, v in sorted(rows, key=lambda r: -r[3]):
            print(f"  lam={lam}: solo={solo:.5f} corr={corr:.3f} eq-blend={aeq:.5f} -> {v}")


if __name__ == "__main__":
    main()
