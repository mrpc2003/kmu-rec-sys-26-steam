"""DirectAU gate: is it a GENUINELY NEW axis vs the saturated BPR-LightGCN family?

The decision is NOT "does DirectAU beat emb128 alone" — it is whether DirectAU is
DECORRELATED enough from the BPR family to crack the 21.4% "neither correct" uniform rows.
The headroom analysis showed emb64/emb128 are 0.97 correlated and their blend HURTS, so a
useful new axis must have markedly lower correlation AND a positive blend gain.

For each DirectAU gamma run (uniform val scores), report:
- row accuracy (vs emb128 ensemble 0.76505, emb64 ensemble 0.76145)
- within-user z correlation with the emb128 ensemble (lower = more diverse)
- best z-blend accuracy over weights w in [0..1]: w*z(emb128) + (1-w)*z(directau)
- 3-way best-of oracle (emb64, emb128, directau) headroom

Verdict ladder:
- NEW_AXIS  : blend beats emb128 by > noise (0.0007) -> diverse & useful, pursue ensemble
- STRONG_SOLO: directau alone beats emb128 by > noise -> new primary candidate
- REDUNDANT : high corr / no blend gain -> same as BPR family, drop
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
EMB64_ENS = 0.76145
EMB128_ENS = 0.76505
NOISE = 0.0007
GAMMAS = ["0.5", "1.0", "2.0", "5.0"]

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


def load(p):
    return pd.read_csv(p)[["ID", "userID", "gameID", "Label", "score_lightgcn"]].rename(
        columns={"score_lightgcn": "s"})


def ens(files):
    base = None
    cols = []
    for s, p in files.items():
        d = load(p).rename(columns={"s": f"s{s}"})
        base = d if base is None else base.merge(d[["ID", f"s{s}"]], on="ID")
        cols.append(f"s{s}")
    base["s"] = base[cols].mean(axis=1)
    return base[["ID", "userID", "gameID", "Label", "s"]]


def acc(df, col="s"):
    return float(evaluate_tophalf(df, col, label_col="Label", user_col="userID", id_col="ID")[0]["row_accuracy"])


def wz(df, c):
    g = df.groupby("userID")[c]
    return (df[c] - g.transform("mean")) / (g.transform("std") + 1e-9)


def main():
    e128 = ens(E128).rename(columns={"s": "s128"})
    e64 = ens(E64).rename(columns={"s": "s64"})
    base = e128.merge(e64[["ID", "s64"]], on="ID")
    base["z128"] = wz(base, "s128")
    base["z64"] = wz(base, "s64")

    print(f"refs: emb64_ens={EMB64_ENS}  emb128_ens={EMB128_ENS}  noise=±{NOISE}\n")
    rows = []
    for g in GAMMAS:
        p = ROOT / f"artifacts/directau_uniform/g{g}" / SPLIT / "lightgcn_scores.csv"
        if not p.exists():
            print(f"[gamma {g}] not ready")
            continue
        d = load(p).rename(columns={"s": "sda"})
        m = base.merge(d[["ID", "sda"]], on="ID")
        m["zda"] = wz(m, "sda")
        a_solo = acc(m.rename(columns={"sda": "s"}))
        corr = float(m["zda"].corr(m["z128"]))
        # best blend weight
        best_w, best_blend = None, -1.0
        for w in np.linspace(0, 1, 21):
            m["zb"] = w * m["z128"] + (1 - w) * m["zda"]
            ab = acc(m.rename(columns={"zb": "s"}))
            if ab > best_blend:
                best_blend, best_w = ab, float(w)
        verdict = ("STRONG_SOLO" if a_solo > EMB128_ENS + NOISE else
                   "NEW_AXIS" if best_blend > EMB128_ENS + NOISE else
                   "REDUNDANT")
        rows.append((g, a_solo, corr, best_w, best_blend, verdict))
        print(f"[gamma {g}] solo={a_solo:.5f} (vs128 {a_solo-EMB128_ENS:+.5f}) | "
              f"corr_z={corr:.3f} | best_blend={best_blend:.5f} @w128={best_w:.2f} "
              f"(vs128 {best_blend-EMB128_ENS:+.5f}) | {verdict}")

    if rows:
        print("\nsummary (sorted by best_blend):")
        for g, asolo, corr, bw, bb, v in sorted(rows, key=lambda r: -r[4]):
            print(f"  g={g}: solo={asolo:.5f} corr={corr:.3f} blend={bb:.5f} -> {v}")


if __name__ == "__main__":
    main()
