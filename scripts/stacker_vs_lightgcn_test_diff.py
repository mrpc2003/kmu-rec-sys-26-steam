"""Diff: stacker test candidate vs submitted LightGCN — where does the stacker flip? (CPU-only)"""
import json, ast, collections
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
lgcn = pd.read_csv(ROOT/"artifacts/lightgcn_20260530/test_full_train/candidate_lightgcn_full_train.csv").rename(columns={"Label":"L_lg"})
stk  = pd.read_csv(ROOT/"artifacts/stacker_20260530/test_candidate/candidate_stacker_logreg_emb64_L3_reg1e-04.csv").rename(columns={"Label":"L_st"})
pairs = pd.read_csv(ROOT/"data/raw/public/data/pairs.csv")

m = lgcn.merge(stk, on="ID").merge(pairs, on="ID")
disagree = (m.L_lg != m.L_st)
n = len(m)

pop = collections.Counter()
with open(ROOT/"data/raw/public/data/train.json") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            pop[ast.literal_eval(line)["gameID"]] += 1
        except Exception:
            pass
m["item_pop"] = m.gameID.map(pop).fillna(0).astype(int)

promoted = m[(m.L_lg == 0) & (m.L_st == 1)]
demoted = m[(m.L_lg == 1) & (m.L_st == 0)]

print(f"total={n} disagree={int(disagree.sum())} ({100*disagree.mean():.2f}%)")
print(f"stacker promoted (lg0->st1): {len(promoted)}  demoted (lg1->st0): {len(demoted)}")
print(f"promoted item_pop mean={promoted.item_pop.mean():.1f} median={promoted.item_pop.median():.0f}")
print(f"demoted  item_pop mean={demoted.item_pop.mean():.1f} median={demoted.item_pop.median():.0f}")
print(f"all      item_pop mean={m.item_pop.mean():.1f} median={m.item_pop.median():.0f}")

m["pb"] = pd.qcut(m.item_pop, 5, duplicates="drop", labels=False)
print("\npop_bin  mean_pop      n   disagree   frac")
pop_rows = []
for b, sub in m.groupby("pb", observed=True):
    d = int(disagree.loc[sub.index].sum())
    print(f"  {int(b)}    {sub.item_pop.mean():7.0f}  {len(sub):5d}    {d:4d}   {d/len(sub):.4f}")
    pop_rows.append({"pop_bin": int(b), "mean_pop": round(float(sub.item_pop.mean()), 1),
                     "n": int(len(sub)), "disagree": d, "frac": round(d/len(sub), 4)})

m["cc"] = m.groupby("userID")["ID"].transform("size")
m["ccb"] = pd.cut(m.cc, [0, 2, 4, 6, 10, 40], labels=["2", "3-4", "5-6", "7-10", "11+"])
print("\ncand_bucket  rows   disagree   frac")
for b, sub in m.groupby("ccb", observed=True):
    d = int(disagree.loc[sub.index].sum())
    print(f"  {b:5s}     {len(sub):5d}    {d:4d}   {d/len(sub):.4f}")

out = {
    "total": int(n), "disagree": int(disagree.sum()), "disagree_frac": round(float(disagree.mean()), 4),
    "stacker_promoted": int(len(promoted)), "stacker_demoted": int(len(demoted)),
    "promoted_item_pop_mean": round(float(promoted.item_pop.mean()), 1),
    "demoted_item_pop_mean": round(float(demoted.item_pop.mean()), 1),
    "all_item_pop_mean": round(float(m.item_pop.mean()), 1),
    "by_item_pop_bin": pop_rows,
}
(ROOT/"reports/20260530_stacker_vs_lightgcn_test_diff.json").write_text(json.dumps(out, indent=2))
print("\nsaved reports/20260530_stacker_vs_lightgcn_test_diff.json")
