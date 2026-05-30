"""Post-submission diff analysis: LightGCN winning candidate vs Stage2 first submission.

Goal:
  - Where did LightGCN flip Stage2's labels?
  - What user-activity / item-popularity buckets drove the +0.01651 gain?
"""
from __future__ import annotations

import json
import collections
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
STAGE2 = ROOT / "artifacts/scores/test_pairs_full_train_stage2_blend/prediction_csv/candidate_score_blend_mean_z.csv"
LGCN = ROOT / "artifacts/lightgcn_20260530/test_full_train/candidate_lightgcn_full_train.csv"
PAIRS = ROOT / "data/raw/public/data/pairs.csv"
TRAIN_JSON = ROOT / "data/raw/public/data/train.json"
OUT_JSON = ROOT / "reports/20260530T184752KST_lightgcn_vs_stage2_diff.json"
OUT_MD = ROOT / "reports/20260530T184752KST_lightgcn_vs_stage2_diff.md"


def main() -> None:
    stage2 = pd.read_csv(STAGE2).rename(columns={"Label": "L_s2"})
    lgcn = pd.read_csv(LGCN).rename(columns={"Label": "L_lg"})
    pairs = pd.read_csv(PAIRS)
    m = stage2.merge(lgcn, on="ID").merge(pairs, on="ID")
    disagree = m.L_s2 != m.L_lg
    print(f"rows={len(m)} disagree_rows={int(disagree.sum())} ({100*disagree.mean():.2f}%)")

    # User candidate-count buckets
    user_idx = m.groupby("userID").indices
    rows = []
    for uid, idx in user_idx.items():
        n = len(idx)
        d = int(disagree.iloc[idx].sum())
        rows.append((uid, n, d))
    udf = pd.DataFrame(rows, columns=["userID", "n", "d"])
    udf["frac"] = udf.d / udf.n
    udf["bucket"] = pd.cut(udf.n, bins=[0, 2, 4, 6, 10, 40], labels=["2", "3-4", "5-6", "7-10", "11+"])
    g_user = udf.groupby("bucket", observed=True).agg(
        users=("userID", "size"), total_n=("n", "sum"), total_d=("d", "sum")
    )
    g_user["frac_dis"] = g_user.total_d / g_user.total_n
    print("\n== disagree by user candidate-count ==")
    print(g_user.round(4).to_string())

    # Item popularity from train.json
    import ast
    train_items: collections.Counter[str] = collections.Counter()
    with open(TRAIN_JSON) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = ast.literal_eval(line)
                train_items[r["gameID"]] += 1
            except Exception:
                pass
    m["item_pop"] = m.gameID.map(train_items).fillna(0).astype(int)
    m["pop_bin"] = pd.qcut(m.item_pop, q=5, duplicates="drop", labels=False)
    g_pop_rows = []
    for b, sub in m.groupby("pop_bin", observed=True):
        idx = sub.index
        d = int(disagree.loc[idx].sum())
        g_pop_rows.append({
            "pop_bin": int(b),
            "n": int(len(idx)),
            "dis": d,
            "frac_dis": d / len(idx),
            "mean_pop": float(sub.item_pop.mean()),
        })
    g_pop = pd.DataFrame(g_pop_rows).sort_values("pop_bin")
    print("\n== disagree by item popularity ==")
    print(g_pop.round(4).to_string(index=False))

    promoted = m[(m.L_s2 == 0) & (m.L_lg == 1)]
    demoted = m[(m.L_s2 == 1) & (m.L_lg == 0)]
    print(f"\npromoted s2=0→lg=1: {len(promoted)} mean_pop={promoted.item_pop.mean():.1f} median={promoted.item_pop.median()}")
    print(f"demoted  s2=1→lg=0: {len(demoted)} mean_pop={demoted.item_pop.mean():.1f} median={demoted.item_pop.median()}")
    print(f"all                 mean_pop={m.item_pop.mean():.1f} median={m.item_pop.median()}")

    out = {
        "rows": int(len(m)),
        "disagree_rows": int(disagree.sum()),
        "disagree_frac": float(disagree.mean()),
        "promoted_count": int(len(promoted)),
        "demoted_count": int(len(demoted)),
        "promoted_item_pop_mean": float(promoted.item_pop.mean()),
        "demoted_item_pop_mean": float(demoted.item_pop.mean()),
        "all_item_pop_mean": float(m.item_pop.mean()),
        "by_user_bucket": g_user.reset_index().to_dict(orient="records"),
        "by_item_pop_bin": g_pop.to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))

    md = []
    md.append("# Post-submission diff: LightGCN vs Stage2 (full test pairs)\n")
    md.append(f"- Rows: {len(m):,}")
    md.append(f"- Disagreement: {int(disagree.sum()):,} ({100*disagree.mean():.2f}%)")
    md.append(f"- LightGCN promoted (s2=0→lg=1): {len(promoted):,}")
    md.append(f"- LightGCN demoted  (s2=1→lg=0): {len(demoted):,}\n")
    md.append("## Disagreement by user candidate-count\n")
    md.append("| bucket | users | total candidates | disagree | frac |")
    md.append("|---|---:|---:|---:|---:|")
    for b, row in g_user.iterrows():
        md.append(f"| {b} | {int(row.users)} | {int(row.total_n)} | {int(row.total_d)} | {row.frac_dis:.4f} |")
    md.append("\n## Disagreement by item popularity (quintile)\n")
    md.append("| pop_bin | mean_pop | n | disagree | frac |")
    md.append("|---:|---:|---:|---:|---:|")
    for _, row in g_pop.iterrows():
        md.append(f"| {int(row.pop_bin)} | {row.mean_pop:.0f} | {int(row.n)} | {int(row.dis)} | {row.frac_dis:.4f} |")
    md.append("\n## Item popularity at flip points\n")
    md.append("| direction | n | mean item_pop | median item_pop |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| promoted (s2=0→lg=1) | {len(promoted)} | {promoted.item_pop.mean():.1f} | {promoted.item_pop.median():.0f} |")
    md.append(f"| demoted  (s2=1→lg=0) | {len(demoted)} | {demoted.item_pop.mean():.1f} | {demoted.item_pop.median():.0f} |")
    md.append(f"| all rows | {len(m)} | {m.item_pop.mean():.1f} | {m.item_pop.median():.0f} |")
    OUT_MD.write_text("\n".join(md))
    print(f"\nsaved: {OUT_JSON}\nsaved: {OUT_MD}")


if __name__ == "__main__":
    main()
