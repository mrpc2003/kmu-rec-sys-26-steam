#!/usr/bin/env python3
"""GBDT (LightGBM) ranker probe over engineered SYMMETRIC features — cross-split.

NEW PARADIGM (user explicitly requested FM / multi-model collaboration)
-----------------------------------------------------------------------
Instead of a single CF scorer, fuse features with a non-linear gradient-boosted tree.
This is the FM-family idea: combine the strong base score with engineered features the
order-free CF discards, and let the tree learn non-linear interactions.

Why this is NOT the failed logreg stacker
------------------------------------------
The logreg stacker regressed on public (0.76245 -> 0.75355) because it was trained on a
HARD-sampler (sqrtpop) OOF and learned popularity down-weighting that hurts the near-uniform
real test. Two corrections here:
  1. Train AND gate on UNIFORM splits only (the public-LB surrogate).
  2. CROSS-SPLIT training: fit on seed123 + seed7, evaluate on seed42 — the tree never sees
     the eval split's negative sampling, so it cannot memorize the sampler.
Plus GBDT non-linearity can capture interactions logreg cannot (the genuinely new element).

Feature design (within-user-VARYING only; symmetric, test-computable)
---------------------------------------------------------------------
Per-user top-half decode means user-level features are constant within a user and cannot
change the ranking, so only within-user-varying features matter:
  - lightgcn: emb128 4-seed mean raw score (the strong base)
  - lgcn_zwu: within-user z-score of lightgcn
  - lgcn_rank: within-user rank of lightgcn (0..1)
  - item_pop: log1p(item play count in that split's train)  [the popularity axis]
  - pop_zwu:  within-user z-score of item_pop
  - lgcn_x_pop: lightgcn * pop_zwu interaction
NO (user,item) review-side features (hours/text/date on the candidate) — those are the
candidate-marginal popularity trap and cannot exist for test pairs.

Gate (identical to hyperbolic / SASRec): solo_acc vs floor, corr_z vs emb128 ens,
50/50 z-blend. Promotion gated by Hermes on a 3-split panel + paired McNemar.

Validation-only. NO Kaggle submission.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import load_train_interactions, evaluate_tophalf, ensure_dir  # noqa: E402

SEEDS = [42, 123, 2024, 7]
FLOOR = 0.684
EMB128_REF = 0.76505
NOISE = 0.0007
TRAIN_SPLITS = ["val_random_uniform_seed123", "val_random_uniform_seed7"]
EVAL_SPLIT = "val_random_uniform_seed42"


def lgcn_path(split, seed):
    """emb128 4-seed lightgcn scores for a given split."""
    if split == EVAL_SPLIT:
        if seed == 42:
            return ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / split / "lightgcn_scores.csv"
        return ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}" / split / "lightgcn_scores.csv"
    # train splits use the split_panel materialization
    return ROOT / f"artifacts/split_panel_emb128/{split}/seed{seed}/lightgcn_scores.csv"


def item_pop(split):
    tr = load_train_interactions(ROOT / "artifacts/validation" / split / "train_interactions.csv")
    pop = tr.groupby("gameID").size()
    return pop


def zwu(df, col):
    g = df.groupby("userID")[col]
    return (df[col] - g.transform("mean")) / g.transform("std").replace(0, 1).fillna(1)


def rwu(df, col):
    return df.groupby("userID")[col].rank(pct=True)


def build_features(split):
    cand = pd.read_csv(ROOT / "artifacts/validation" / split / "candidates.csv")[
        ["ID", "userID", "gameID", "Label"]].copy()
    # lightgcn 4-seed mean
    base = pd.read_csv(lgcn_path(split, 42))[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": "s42"})
    for s in SEEDS[1:]:
        d = pd.read_csv(lgcn_path(split, s))[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": f"s{s}"})
        base = base.merge(d, on="ID", how="inner")
    base["lightgcn"] = base[[f"s{s}" for s in SEEDS]].mean(axis=1)
    cand = cand.merge(base[["ID", "lightgcn"]], on="ID", how="inner")
    # popularity
    pop = item_pop(split)
    cand["item_pop"] = np.log1p(cand["gameID"].map(pop).fillna(0).to_numpy())
    # within-user features
    cand["lgcn_zwu"] = zwu(cand, "lightgcn")
    cand["lgcn_rank"] = rwu(cand, "lightgcn")
    cand["pop_zwu"] = zwu(cand, "item_pop")
    cand["lgcn_x_pop"] = cand["lgcn_zwu"] * cand["pop_zwu"]
    return cand


FEATS = ["lightgcn", "item_pop", "lgcn_zwu", "lgcn_rank", "pop_zwu", "lgcn_x_pop"]


def main():
    out_dir = ensure_dir(ROOT / "artifacts/gbdt_ranker")

    # ---- training data: cross-split (seed123 + seed7) ----
    tr_frames = [build_features(sp) for sp in TRAIN_SPLITS]
    tr = pd.concat(tr_frames, ignore_index=True)
    # group = (split,user) so LambdaRank groups are per-user within each split
    tr["grp"] = tr.groupby(["userID"]).ngroup()  # users shared across splits -> still per-user
    # Use binary objective (robust, matches set-membership); ranker is sensitive to group defn.
    Xtr, ytr = tr[FEATS].to_numpy(), tr["Label"].to_numpy()

    params = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)
    dtrain = lgb.Dataset(Xtr, label=ytr)
    model = lgb.train(params, dtrain, num_boost_round=300)

    # ---- eval on seed42 ----
    ev = build_features(EVAL_SPLIT)
    ev["gbdt"] = model.predict(ev[FEATS].to_numpy())
    summ, _ = evaluate_tophalf(ev, "gbdt", label_col="Label", user_col="userID", id_col="ID")
    solo = round(float(summ["row_accuracy"]), 5)

    # corr_z + eq_blend vs lightgcn (emb128 4-seed) on the SAME eval rows
    ev["lgcn_zwu_eval"] = zwu(ev, "lightgcn")
    ev["gbdt_zwu"] = zwu(ev, "gbdt")
    corr = round(float(np.corrcoef(ev["gbdt_zwu"], ev["lgcn_zwu_eval"])[0, 1]), 4)
    ev["blend"] = 0.5 * ev["gbdt_zwu"] + 0.5 * ev["lgcn_zwu_eval"]
    eb, _ = evaluate_tophalf(ev, "blend", label_col="Label", user_col="userID", id_col="ID")
    eqb = round(float(eb["row_accuracy"]), 5)

    # lightgcn-alone solo on the same eval (sanity: should ~0.76505)
    lg_summ, _ = evaluate_tophalf(ev, "lightgcn", label_col="Label", user_col="userID", id_col="ID")
    lg_solo = round(float(lg_summ["row_accuracy"]), 5)

    d_solo = round(solo - lg_solo, 5)
    d_blend = round(eqb - EMB128_REF, 5)
    imp = dict(zip(FEATS, [int(x) for x in model.feature_importance(importance_type="gain")]))

    if solo < FLOOR:
        tier = "REJECT_FLOOR"
    elif d_solo > NOISE or (d_blend > NOISE and corr < 0.9):
        tier = "SIGNAL_ESCALATE"
    else:
        tier = "REDUNDANT"

    out = {
        "note": "GBDT (LightGBM) ranker over symmetric features, CROSS-SPLIT (train seed123+seed7, eval seed42). No submission.",
        "train_splits": TRAIN_SPLITS, "eval_split": EVAL_SPLIT, "features": FEATS,
        "gbdt_solo_acc": solo, "lightgcn_solo_on_eval": lg_solo, "emb128_ref": EMB128_REF,
        "gbdt_minus_lightgcn": d_solo,
        "corr_z_gbdt_vs_lightgcn": corr, "eq_blend_acc": eqb, "eq_blend_minus_ref": d_blend,
        "floor": FLOOR, "noise": NOISE, "feature_importance_gain": imp,
        "tier": tier,
    }
    (ROOT / "reports/20260601_gbdt_ranker_crosssplit.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    ev[["ID", "userID", "gameID", "Label", "gbdt"]].rename(columns={"gbdt": "score_lightgcn"}).to_csv(
        out_dir / "eval_seed42_scores.csv", index=False)

    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[GBDT] solo={solo} (lgcn {lg_solo}, Δ{d_solo:+}) corr_z={corr} eq_blend={eqb} (Δref {d_blend:+}) tier={tier}")
    print(f"[GBDT] feature gain: {imp}")


if __name__ == "__main__":
    main()
