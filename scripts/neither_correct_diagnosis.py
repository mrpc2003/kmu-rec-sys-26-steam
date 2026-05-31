"""Diagnose the 21.4% "neither correct" uniform rows: are they crackable or intrinsically random?

For each uniform candidate row, mark whether BOTH the emb64 and emb128 ensembles get it
wrong (per-user top-half decode). Then characterize those rows by:
- item popularity (train interaction count of the game)
- user training degree (how many games the user reviewed in train)
- whether the row's true label is 1 (played) or 0 (unplayed)

If "neither" rows are concentrated in low-popularity items / low-degree users, the CF signal
is simply absent there and NO model (incl. SGL) can crack them -> hard ceiling near oracle.
If they're spread evenly, a different representation might help. Honest upper-bound check.
No Kaggle submission.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import predict_tophalf, load_train_json

SPLIT = "val_random_uniform_seed42"
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
        d = pd.read_csv(p)[["ID", "userID", "gameID", "Label", "score_lightgcn"]].rename(
            columns={"score_lightgcn": f"s{s}"})
        base = d if base is None else base.merge(d[["ID", f"s{s}"]], on="ID")
        cols.append(f"s{s}")
    base["s"] = base[cols].mean(axis=1)
    return base[["ID", "userID", "gameID", "Label", "s"]]


def correct(df):
    p = predict_tophalf(df, "s", label_col=None, user_col="userID", id_col="ID")
    pp = p[["ID", "Pred"]].merge(df[["ID", "Label"]], on="ID")
    return dict(zip(pp.ID, (pp.Pred == pp.Label).astype(int)))


def main():
    e64 = ens(E64)
    e128 = ens(E128)
    m = e64[["ID", "userID", "gameID", "Label"]].copy()
    c64 = correct(e64)
    c128 = correct(e128)
    m["c64"] = m.ID.map(c64)
    m["c128"] = m.ID.map(c128)
    m["neither"] = ((m.c64 == 0) & (m.c128 == 0)).astype(int)

    tr = load_train_json(ROOT / "data/raw/public/data/train.json")
    item_pop = tr.groupby("gameID").size().to_dict()
    user_deg = tr.groupby("userID").size().to_dict()
    m["item_pop"] = m.gameID.map(item_pop).fillna(0).astype(int)
    m["user_deg"] = m.userID.map(user_deg).fillna(0).astype(int)

    N = len(m)
    nb = int(m.neither.sum())
    print(f"rows={N}  neither_correct={nb} ({nb/N:.4f})\n")

    # label split of neither rows
    nl = m[m.neither == 1]
    print(f"neither by true label: played(1)={int((nl.Label==1).sum())} "
          f"unplayed(0)={int((nl.Label==0).sum())}")

    # item popularity quintiles: neither-rate per bin
    m["pop_q"] = pd.qcut(m.item_pop, q=5, duplicates="drop", labels=False)
    print("\nneither-rate by item-popularity quintile (0=least popular):")
    g = m.groupby("pop_q").agg(n=("ID", "size"), neither=("neither", "mean"),
                               med_pop=("item_pop", "median"))
    print(g.round(4).to_string())

    # user degree quintiles
    m["deg_q"] = pd.qcut(m.user_deg, q=5, duplicates="drop", labels=False)
    print("\nneither-rate by user-degree quintile (0=least active):")
    g2 = m.groupby("deg_q").agg(n=("ID", "size"), neither=("neither", "mean"),
                                med_deg=("user_deg", "median"))
    print(g2.round(4).to_string())

    # candidate-set size per user (smaller sets = coin-flip-ish)
    csize = m.groupby("userID").size().rename("cset")
    m = m.merge(csize, on="userID")
    print("\nneither-rate by user candidate-set size:")
    g3 = m.groupby("cset").agg(n=("ID", "size"), neither=("neither", "mean"))
    print(g3.round(4).head(10).to_string())

    print("\nINTERPRETATION:")
    print("- If neither-rate is high & flat across pop/deg quintiles -> rows are intrinsically")
    print("  hard (no CF signal); SGL or any model can't crack them; ceiling ~oracle 0.786.")
    print("- If neither-rate spikes in low-pop/low-deg -> a cold-start-aware signal could help,")
    print("  but LightGCN already uses all interactions, so headroom is small.")


if __name__ == "__main__":
    main()
