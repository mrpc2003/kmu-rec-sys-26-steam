#!/usr/bin/env python3
"""Multi-model GBDT ranker — TRUE model collaboration (user-requested FM-style fusion).

Fuses multiple independent scorers with a non-linear LightGBM, cross-split:
  - lightgcn          : emb128 4-seed mean raw score (graph CF, the strong base)
  - score_itemknn_top3: item-CF neighbour score
  - score_ease_lambda300 : EASE closed-form score
  - score_pop_log     : log popularity
  + within-user z-score of each (within-user-varying signal that survives top-half decode)

Anti-trap (same as the LightGCN-only GBDT, which came out REDUNDANT corr_z 0.97):
  * train on seed123 + seed7, evaluate on seed42 (eval split's sampler never seen)
  * uniform splits only (public-LB surrogate)
  * symmetric, within-user-varying features only; NO candidate review-side features

Prior is strongly negative (Stage2 alone public 0.74594; logreg stacker 0.75355), but this
completes the user's explicit "multi-model collaboration" request and gates it rigorously.
Gate: solo vs floor, corr_z vs lightgcn, 50/50 z-blend vs emb128 ref. No Kaggle submission.
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
from recsys_played_utils import evaluate_tophalf, ensure_dir  # noqa: E402

SEEDS = [42, 123, 2024, 7]
FLOOR, EMB128_REF, NOISE = 0.684, 0.76505, 0.0007
TRAIN_SPLITS = ["val_random_uniform_seed123", "val_random_uniform_seed7"]
EVAL_SPLIT = "val_random_uniform_seed42"
STAGE2_COLS = ["score_itemknn_top3", "score_ease_lambda300", "score_pop_log"]


def lgcn_path(split, seed):
    if split == EVAL_SPLIT:
        if seed == 42:
            return ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / split / "lightgcn_scores.csv"
        return ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}" / split / "lightgcn_scores.csv"
    return ROOT / f"artifacts/split_panel_emb128/{split}/seed{seed}/lightgcn_scores.csv"


def zwu(df, col):
    g = df.groupby("userID")[col]
    return (df[col] - g.transform("mean")) / g.transform("std").replace(0, 1).fillna(1)


def build(split):
    cand = pd.read_csv(ROOT / "artifacts/validation" / split / "candidates.csv")[
        ["ID", "userID", "gameID", "Label"]].copy()
    # lightgcn 4-seed mean
    base = pd.read_csv(lgcn_path(split, 42))[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": "s42"})
    for s in SEEDS[1:]:
        d = pd.read_csv(lgcn_path(split, s))[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": f"s{s}"})
        base = base.merge(d, on="ID", how="inner")
    base["lightgcn"] = base[[f"s{s}" for s in SEEDS]].mean(axis=1)
    cand = cand.merge(base[["ID", "lightgcn"]], on="ID", how="inner")
    # stage2 scorers
    s2 = pd.read_csv(ROOT / f"artifacts/scores/{split}_proto/candidate_scores.csv")
    cand = cand.merge(s2[["ID"] + STAGE2_COLS], on="ID", how="inner")
    # within-user z for every raw scorer
    feats = ["lightgcn"] + STAGE2_COLS
    for c in feats:
        cand[f"z_{c}"] = zwu(cand, c)
    return cand, feats


def main():
    out_dir = ensure_dir(ROOT / "artifacts/gbdt_multimodel")
    tr_frames, feat_names = [], None
    for sp in TRAIN_SPLITS:
        f, feat_names = build(sp)
        tr_frames.append(f)
    tr = pd.concat(tr_frames, ignore_index=True)

    FEATS = feat_names + [f"z_{c}" for c in feat_names]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)
    model = lgb.train(params, lgb.Dataset(tr[FEATS].to_numpy(), label=tr["Label"].to_numpy()),
                      num_boost_round=300)

    ev, _ = build(EVAL_SPLIT)
    ev["gbdt"] = model.predict(ev[FEATS].to_numpy())
    solo = round(float(evaluate_tophalf(ev, "gbdt", "Label", "userID", "ID")[0]["row_accuracy"]), 5)
    lg_solo = round(float(evaluate_tophalf(ev, "lightgcn", "Label", "userID", "ID")[0]["row_accuracy"]), 5)

    ev["z_lg_eval"] = zwu(ev, "lightgcn")
    ev["z_gbdt"] = zwu(ev, "gbdt")
    corr = round(float(np.corrcoef(ev["z_gbdt"], ev["z_lg_eval"])[0, 1]), 4)
    ev["blend"] = 0.5 * ev["z_gbdt"] + 0.5 * ev["z_lg_eval"]
    eqb = round(float(evaluate_tophalf(ev, "blend", "Label", "userID", "ID")[0]["row_accuracy"]), 5)

    d_solo, d_blend = round(solo - lg_solo, 5), round(eqb - EMB128_REF, 5)
    imp = dict(zip(FEATS, [int(x) for x in model.feature_importance(importance_type="gain")]))
    imp = dict(sorted(imp.items(), key=lambda kv: -kv[1]))

    if solo < FLOOR:
        tier = "REJECT_FLOOR"
    elif d_solo > NOISE or (d_blend > NOISE and corr < 0.9):
        tier = "SIGNAL_ESCALATE"
    else:
        tier = "REDUNDANT"

    out = {
        "note": "Multi-model GBDT (lightgcn+itemknn+ease+pop) cross-split. User-requested model collaboration. No submission.",
        "train_splits": TRAIN_SPLITS, "eval_split": EVAL_SPLIT, "features": FEATS,
        "gbdt_solo_acc": solo, "lightgcn_solo_on_eval": lg_solo, "gbdt_minus_lightgcn": d_solo,
        "corr_z_gbdt_vs_lightgcn": corr, "eq_blend_acc": eqb, "eq_blend_minus_ref": d_blend,
        "emb128_ref": EMB128_REF, "floor": FLOOR, "noise": NOISE,
        "feature_importance_gain": imp, "tier": tier,
    }
    (ROOT / "reports/20260601_gbdt_multimodel_crosssplit.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    ev[["ID", "userID", "gameID", "Label", "gbdt"]].rename(columns={"gbdt": "score_lightgcn"}).to_csv(
        out_dir / "eval_seed42_scores.csv", index=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[MULTI-GBDT] solo={solo} (lgcn {lg_solo}, Δ{d_solo:+}) corr_z={corr} "
          f"eq_blend={eqb} (Δref {d_blend:+}) tier={tier}")
    print(f"[MULTI-GBDT] top feature gain: {list(imp.items())[:4]}")


if __name__ == "__main__":
    main()
