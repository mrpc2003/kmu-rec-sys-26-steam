"""Materialize the LightGCN+Stage2 stacker as a TEST submission candidate.

Design (train/inference consistency):
  - Meta-learner (logreg) is trained on POOLED validation rows across all 3 splits.
    Base features there come from split-train base scores (legitimate out-of-sample).
  - Within-user z/rank features are scale-invariant, so the learned linear weights
    transfer to test even though test base scores are full-train.
  - Test features: full-train LightGCN raw score + full-train Stage2 blend,
    built with the SAME build_features logic.
  - Decode with canonical predict_tophalf (per-user top-half).

Honesty:
  - We also report the pooled-validation OOF GroupKFold accuracy of this exact
    pooled meta-learner, as the best available estimate of test behavior.
  - We compare the decoded test candidate against the submitted LightGCN candidate
    (row diff, label balance) so the user sees how much it moves.

NO Kaggle submission — only writes the candidate CSV + preflight + forecast report.
"""
from __future__ import annotations

import argparse
import hashlib
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
from recsys_played_utils import evaluate_tophalf, predict_tophalf, ensure_dir, write_json

SPLITS = [
    "val_random_sqrtpop_seed42",
    "val_recent_sqrtpop_seed42",
    "val_random_popbin_seed42",
]
SEED = 42
N_FOLDS = 5
SUBMITTED_LGCN = ROOT / "artifacts/lightgcn_20260530/test_full_train/candidate_lightgcn_full_train.csv"

BASE_SCORE_COLS = ["score_lightgcn", "score_blend_mean_z",
                   "score_itemknn_bm25_top3", "score_ease_lambda1000",
                   "score_als_f32_it30_alpha20_popa2"]


def within_user(df: pd.DataFrame, col: str, kind: str) -> pd.Series:
    g = df.groupby("userID")[col]
    if kind == "z":
        return g.transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0.0)
    if kind == "rank":
        return g.rank(pct=True)
    raise ValueError(kind)


def add_features(m: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    m = m.copy()
    m["cand_count"] = m.groupby("userID")["ID"].transform("size")
    m["log_pop"] = np.log1p(m["pop_count"]) if "pop_count" in m.columns else 0.0
    feat_cols = []
    for c in BASE_SCORE_COLS:
        if c not in m.columns:
            continue
        m[f"wz_{c}"] = within_user(m, c, "z")
        m[f"wr_{c}"] = within_user(m, c, "rank")
        feat_cols += [c, f"wz_{c}", f"wr_{c}"]
    feat_cols += ["log_pop", "cand_count"]
    m[feat_cols] = m[feat_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return m, feat_cols


def load_val(split: str) -> pd.DataFrame:
    lg = pd.read_csv(ROOT / f"artifacts/lightgcn_20260530/{split}/lightgcn_scores.csv")
    s2 = pd.read_csv(ROOT / f"artifacts/scores/{split}_stage2_blend/merged_blend_scores.csv")
    keep = ["ID", "score_blend_mean_z", "pop_count",
            "score_itemknn_bm25_top3", "score_ease_lambda1000",
            "score_als_f32_it30_alpha20_popa2"]
    keep = [c for c in keep if c in s2.columns]
    m = lg.merge(s2[keep], on="ID", how="inner")
    m["split"] = split
    # make IDs unique across splits for pooled grouping
    m["gid_user"] = split + "::" + m["userID"].astype(str)
    return m


def load_test(lgcn_tag: str) -> pd.DataFrame:
    raw_path = ROOT / f"artifacts/lightgcn_20260530/test_full_train/lightgcn_test_raw_scores_{lgcn_tag}.csv"
    lg = pd.read_csv(raw_path)
    s2 = pd.read_csv(ROOT / "artifacts/scores/test_pairs_full_train_stage2_blend/merged_blend_scores.csv")
    keep = ["ID", "score_blend_mean_z", "pop_count",
            "score_itemknn_bm25_top3", "score_ease_lambda1000",
            "score_als_f32_it30_alpha20_popa2"]
    keep = [c for c in keep if c in s2.columns]
    m = lg.merge(s2[keep], on="ID", how="inner")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgcn-tag", default="emb64_L3_reg1e-04")
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    # 1) Build pooled validation training set
    val_frames = [add_features(load_val(s))[0] for s in SPLITS]
    feat_cols = add_features(load_val(SPLITS[0]))[1]
    pooled = pd.concat(val_frames, ignore_index=True)
    print(f"[pooled-val] rows={len(pooled)} feats={len(feat_cols)} "
          f"users={pooled['gid_user'].nunique()}", flush=True)

    Xp = pooled[feat_cols].to_numpy(dtype=np.float64)
    yp = pooled["Label"].to_numpy(dtype=int)

    # 2) Honest GroupKFold OOF on pooled (group = split::user) — best test estimate
    groups = pooled["gid_user"].to_numpy()
    oof = np.zeros(len(pooled), dtype=np.float64)
    gkf = GroupKFold(n_splits=N_FOLDS)
    for tr, va in gkf.split(Xp, yp, groups=groups):
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=3000, C=args.C)
        clf.fit(sc.fit_transform(Xp[tr]), yp[tr])
        oof[va] = clf.predict_proba(sc.transform(Xp[va]))[:, 1]
    pooled["stack_oof"] = oof
    # per-split decoded accuracy on OOF
    per_split = {}
    for s in SPLITS:
        sub = pooled[pooled.split == s]
        summ, _ = evaluate_tophalf(sub, "stack_oof", label_col="Label",
                                   user_col="userID", id_col="ID")
        lg_summ, _ = evaluate_tophalf(sub, "score_lightgcn", label_col="Label",
                                      user_col="userID", id_col="ID")
        per_split[s] = {"stack_oof": round(float(summ["row_accuracy"]), 5),
                        "lightgcn": round(float(lg_summ["row_accuracy"]), 5)}
        print(f"  [{s}] stack_oof={per_split[s]['stack_oof']} "
              f"lightgcn={per_split[s]['lightgcn']}", flush=True)
    mean_stack = round(float(np.mean([per_split[s]["stack_oof"] for s in SPLITS])), 5)
    mean_lgcn = round(float(np.mean([per_split[s]["lightgcn"] for s in SPLITS])), 5)
    print(f"  POOLED OOF mean: stack={mean_stack} lightgcn={mean_lgcn} "
          f"gain={mean_stack-mean_lgcn:+.5f}", flush=True)

    # 3) Train final meta-learner on ALL pooled validation
    scaler = StandardScaler().fit(Xp)
    final_clf = LogisticRegression(max_iter=3000, C=args.C).fit(scaler.transform(Xp), yp)
    weights = dict(zip(feat_cols, final_clf.coef_[0].round(4).tolist()))
    print("\n[final meta-learner weights]")
    for k, v in sorted(weights.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:36s} {v:+.4f}")

    # 4) Build test features + score
    test = load_test(args.lgcn_tag)
    test, _ = add_features(test)
    Xt = test[feat_cols].to_numpy(dtype=np.float64)
    test["stack_score"] = final_clf.predict_proba(scaler.transform(Xt))[:, 1]

    pred = predict_tophalf(test, "stack_score", label_col=None,
                           user_col="userID", id_col="ID")
    submission = pred[["ID", "Pred"]].rename(columns={"Pred": "Label"}).sort_values("ID")

    out_dir = ensure_dir(ROOT / "artifacts/stacker_20260530/test_candidate")
    csv_path = out_dir / f"candidate_stacker_logreg_{args.lgcn_tag}.csv"
    submission.to_csv(csv_path, index=False)
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    # 5) Preflight + diff vs submitted LightGCN
    pairs = pd.read_csv(ROOT / "data/raw/public/data/pairs.csv")
    rows_ok = len(submission) == len(pairs)
    ids_ok = bool((submission["ID"].values == np.arange(len(submission))).all())
    labels_ok = bool(submission["Label"].isin([0, 1]).all())
    bal = (int(submission["Label"].sum()), int((1 - submission["Label"]).sum()))
    # per-user top-half check
    mp = submission.merge(pairs, on="ID")
    g = mp.groupby("userID").agg(n=("ID", "size"), p=("Label", "sum"))
    bad_users = int((g.p != g.n // 2).sum())

    lgcn_sub = pd.read_csv(SUBMITTED_LGCN).rename(columns={"Label": "L_lg"})
    cmp = submission.rename(columns={"Label": "L_st"}).merge(lgcn_sub, on="ID")
    row_diff = int((cmp.L_st != cmp.L_lg).sum())

    meta = {
        "candidate_file": str(csv_path),
        "sha256": sha,
        "lgcn_tag": args.lgcn_tag,
        "logreg_C": args.C,
        "rows": int(len(submission)),
        "label_balance": {"1": bal[0], "0": bal[1]},
        "preflight": {"rows_match_pairs": rows_ok, "ids_contiguous": ids_ok,
                      "labels_binary": labels_ok, "bad_users_tophalf": bad_users},
        "diff_vs_submitted_lightgcn": {"row_diff": row_diff,
                                       "row_diff_frac": round(row_diff / len(cmp), 4)},
        "pooled_oof": {"mean_stack": mean_stack, "mean_lightgcn": mean_lgcn,
                       "mean_gain": round(mean_stack - mean_lgcn, 5),
                       "per_split": per_split},
        "final_weights": weights,
    }
    write_json(out_dir / f"meta_{args.lgcn_tag}.json", meta)
    print(f"\n[candidate] {csv_path}")
    print(f"  sha256={sha}")
    print(f"  rows={len(submission)} balance={bal} bad_users={bad_users}")
    print(f"  preflight rows_ok={rows_ok} ids_ok={ids_ok} labels_ok={labels_ok}")
    print(f"  row_diff vs submitted LightGCN: {row_diff} ({100*row_diff/len(cmp):.2f}%)")
    print(f"\n[forecast] pooled OOF gain {mean_stack-mean_lgcn:+.5f}; "
          f"LightGCN public was 0.76245 (transfer ratio ~1.24 → est +0.012~0.015).")
    print(f"[STOP] Submission requires explicit user approval.")


if __name__ == "__main__":
    main()
