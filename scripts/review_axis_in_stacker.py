"""Does adding review-text features to the stacker improve HONEST OOF? (CPU-only)

Research motivation: review-enhanced recommendation papers claim text signals are
ORTHOGONAL to interaction signals. The standalone review axis is weak
(0.61/0.59/0.52 vs LightGCN 0.675/0.640/0.602), but the honest test is whether it
adds incremental value INSIDE the meta-learner.

Method: pooled validation rows across 3 splits. Compare two logreg stackers under
user-level GroupKFold (group = split::user, no within-user leakage):
  A. base   = LightGCN + Stage2 features (17 feats)  ← current best stacker
  B. +review = base + review_tfidf_cosine/item_review_count/user_review_count
               (+ within-user z/rank of each)

Only if B beats A on honest pooled OOF do we bother generating test review scores.
No Kaggle submission.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import evaluate_tophalf

SPLITS = ["val_random_sqrtpop_seed42", "val_recent_sqrtpop_seed42", "val_random_popbin_seed42"]
SEED = 42
N_FOLDS = 5
REVIEW_COLS = ["score_review_tfidf_user_item_cosine", "score_item_review_count", "score_user_review_count"]
OUT_JSON = ROOT / "reports/20260530_review_axis_in_stacker.json"
OUT_MD = ROOT / "reports/20260530_review_axis_in_stacker.md"


def within_user(df, col, kind):
    g = df.groupby("userID")[col]
    if kind == "z":
        return g.transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0.0)
    return g.rank(pct=True)


def load_split(split):
    lg = pd.read_csv(ROOT / f"artifacts/lightgcn_20260530/{split}/lightgcn_scores.csv")
    s2 = pd.read_csv(ROOT / f"artifacts/scores/{split}_stage2_blend/merged_blend_scores.csv")
    rv = pd.read_csv(ROOT / f"artifacts/review_tfidf_probe_20260530/{split}/review_tfidf_scores.csv")
    keep_s2 = ["ID", "score_blend_mean_z", "pop_count",
               "score_itemknn_bm25_top3", "score_ease_lambda1000",
               "score_als_f32_it30_alpha20_popa2"]
    keep_s2 = [c for c in keep_s2 if c in s2.columns]
    keep_rv = ["ID"] + [c for c in REVIEW_COLS if c in rv.columns]
    m = lg.merge(s2[keep_s2], on="ID", how="inner").merge(rv[keep_rv], on="ID", how="inner")
    m["split"] = split
    m["gid_user"] = split + "::" + m["userID"].astype(str)
    return m


def add_features(m, include_review):
    m = m.copy()
    m["cand_count"] = m.groupby("userID")["ID"].transform("size")
    m["log_pop"] = np.log1p(m["pop_count"]) if "pop_count" in m.columns else 0.0
    base = ["score_lightgcn", "score_blend_mean_z",
            "score_itemknn_bm25_top3", "score_ease_lambda1000",
            "score_als_f32_it30_alpha20_popa2"]
    review = [c for c in REVIEW_COLS if c in m.columns]
    cols = base + (review if include_review else [])
    feat = []
    for c in cols:
        if c not in m.columns:
            continue
        m[f"wz_{c}"] = within_user(m, c, "z")
        m[f"wr_{c}"] = within_user(m, c, "rank")
        feat += [c, f"wz_{c}", f"wr_{c}"]
    feat += ["log_pop", "cand_count"]
    m[feat] = m[feat].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return m, feat


def honest_oof(pooled, feat):
    X = pooled[feat].to_numpy(dtype=np.float64)
    y = pooled["Label"].to_numpy(dtype=int)
    groups = pooled["gid_user"].to_numpy()
    oof = np.zeros(len(pooled))
    for tr, va in GroupKFold(n_splits=N_FOLDS).split(X, y, groups=groups):
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        oof[va] = clf.predict_proba(sc.transform(X[va]))[:, 1]
    return oof


def per_split_acc(pooled, score_col):
    out = {}
    for s in SPLITS:
        sub = pooled[pooled.split == s]
        summ, _ = evaluate_tophalf(sub, score_col, label_col="Label", user_col="userID", id_col="ID")
        out[s] = round(float(summ["row_accuracy"]), 5)
    return out


def main():
    frames = [load_split(s) for s in SPLITS]
    pooled = pd.concat(frames, ignore_index=True)
    print(f"[pooled] rows={len(pooled)} users={pooled['gid_user'].nunique()}", flush=True)

    results = {}
    for variant, inc in [("base", False), ("plus_review", True)]:
        p2, feat = add_features(pooled, inc)
        p2["oof"] = honest_oof(p2, feat)
        acc = per_split_acc(p2, "oof")
        mean_acc = round(float(np.mean(list(acc.values()))), 5)
        results[variant] = {"n_features": len(feat), "per_split": acc, "mean": mean_acc}
        print(f"  [{variant:12s}] feats={len(feat)} mean_oof={mean_acc} {acc}", flush=True)

    lg_acc = per_split_acc(pooled.assign(), "score_lightgcn")
    mean_lg = round(float(np.mean(list(lg_acc.values()))), 5)
    delta = round(results["plus_review"]["mean"] - results["base"]["mean"], 5)
    summary = {
        "mean_lightgcn": mean_lg,
        "stacker_base_mean": results["base"]["mean"],
        "stacker_plus_review_mean": results["plus_review"]["mean"],
        "review_increment": delta,
        "review_helps": bool(delta > 0),
        "detail": results,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    md = ["# Review axis inside the stacker — honest GroupKFold OOF\n"]
    md.append(f"- LightGCN: {mean_lg}")
    md.append(f"- stacker base (LightGCN+Stage2): {results['base']['mean']}")
    md.append(f"- stacker +review: {results['plus_review']['mean']}")
    md.append(f"- **review increment: {delta:+.5f}** ({'HELPS' if delta>0 else 'no help'})\n")
    md.append("| variant | sqrtpop | recent | popbin | mean |")
    md.append("|---|---:|---:|---:|---:|")
    for v in ["base", "plus_review"]:
        ps = results[v]["per_split"]
        md.append(f"| {v} | {ps[SPLITS[0]]} | {ps[SPLITS[1]]} | {ps[SPLITS[2]]} | {results[v]['mean']} |")
    md.append("\n## Decision\n")
    if delta > 0.0005:
        md.append(f"- Review features add {delta:+.5f} honest OOF → generate test review scores "
                  "and include in the materialized stacker candidate.")
    else:
        md.append(f"- Review features add only {delta:+.5f} → NOT worth the extra test-score "
                  "generation / complexity. Keep the LightGCN+Stage2 stacker.")
    OUT_MD.write_text("\n".join(md))
    print(f"\nsaved: {OUT_JSON}")
    print(f"REVIEW INCREMENT: {delta:+.5f} (base={results['base']['mean']} +review={results['plus_review']['mean']})")


if __name__ == "__main__":
    main()
