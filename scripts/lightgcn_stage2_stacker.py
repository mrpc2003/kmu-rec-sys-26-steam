"""Stacking meta-learner: LightGCN + Stage2 → row-level gating (CPU-only, no GPU).

Motivation: complementarity analysis showed oracle upper bound 0.729 vs
LightGCN 0.639 on validation. A learned per-row combination may extract more
of that gap than a fixed linear blend (which gave +0.0022).

Honesty: all reported accuracies are 5-fold OUT-OF-FOLD (OOF). The meta-learner
never predicts a row it was trained on, so the estimate generalizes across rows.
Base scores (LightGCN, Stage2) are trained on the split's train_interactions —
NOT on the candidate labels — so they are legitimate out-of-sample features.

Decoding uses canonical predict_tophalf/evaluate_tophalf (matches baseline).

Models: logistic regression (safe) + LightGBM (nonlinear gating).
No Kaggle submission — validation only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import evaluate_tophalf

SPLITS = [
    "val_random_sqrtpop_seed42",
    "val_recent_sqrtpop_seed42",
    "val_random_popbin_seed42",
]
LGCN_REF = {
    "val_random_sqrtpop_seed42": 0.67483,
    "val_recent_sqrtpop_seed42": 0.63963,
    "val_random_popbin_seed42":  0.60202,
}
FIXED_BLEND_REF = {  # from complementarity analysis (best fixed blend per split)
    "val_random_sqrtpop_seed42": 0.67704,
    "val_recent_sqrtpop_seed42": 0.64053,
    "val_random_popbin_seed42":  0.60562,
}
SEED = 42
N_FOLDS = 5

OUT_JSON = ROOT / "reports/20260530_lightgcn_stage2_stacker.json"
OUT_MD = ROOT / "reports/20260530_lightgcn_stage2_stacker.md"


def within_user(df: pd.DataFrame, col: str, kind: str) -> pd.Series:
    g = df.groupby("userID")[col]
    if kind == "z":
        return g.transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0.0)
    if kind == "rank":
        return g.rank(pct=True)
    raise ValueError(kind)


def build_features(split: str) -> tuple[pd.DataFrame, list[str]]:
    lg = pd.read_csv(ROOT / f"artifacts/lightgcn_20260530/{split}/lightgcn_scores.csv")
    s2 = pd.read_csv(ROOT / f"artifacts/scores/{split}_stage2_blend/merged_blend_scores.csv")
    keep = ["ID", "score_blend_mean_z", "pop_count",
            "score_itemknn_bm25_top3", "score_ease_lambda1000",
            "score_als_f32_it30_alpha20_popa2"]
    keep = [c for c in keep if c in s2.columns]
    m = lg.merge(s2[keep], on="ID", how="inner").copy()

    # candidate count per user
    m["cand_count"] = m.groupby("userID")["ID"].transform("size")
    m["log_pop"] = np.log1p(m["pop_count"]) if "pop_count" in m.columns else 0.0

    base_score_cols = ["score_lightgcn", "score_blend_mean_z",
                       "score_itemknn_bm25_top3", "score_ease_lambda1000"]
    if "score_als_f32_it30_alpha20_popa2" in m.columns:
        base_score_cols.append("score_als_f32_it30_alpha20_popa2")

    feat_cols = []
    for c in base_score_cols:
        if c not in m.columns:
            continue
        m[f"wz_{c}"] = within_user(m, c, "z")
        m[f"wr_{c}"] = within_user(m, c, "rank")
        feat_cols += [c, f"wz_{c}", f"wr_{c}"]
    feat_cols += ["log_pop", "cand_count"]

    m[feat_cols] = m[feat_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return m, feat_cols


def oof_predict(m: pd.DataFrame, feat_cols: list[str], model: str,
                fold_mode: str = "strat") -> np.ndarray:
    """fold_mode: 'strat' (row-level StratifiedKFold) or 'group' (user-level GroupKFold).

    'group'묶음은 같은 유저의 모든 candidate를 한 fold에 두어 within-user 피처 누수를 차단한다.
    """
    X = m[feat_cols].to_numpy(dtype=np.float64)
    y = m["Label"].to_numpy(dtype=int)
    oof = np.zeros(len(m), dtype=np.float64)
    if fold_mode == "strat":
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        folds = splitter.split(X, y)
    elif fold_mode == "group":
        groups = m["userID"].to_numpy()
        splitter = GroupKFold(n_splits=N_FOLDS)
        folds = splitter.split(X, y, groups=groups)
    else:
        raise ValueError(fold_mode)
    for tr, va in folds:
        if model == "logreg":
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xva = sc.transform(X[va])
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr, y[tr])
            oof[va] = clf.predict_proba(Xva)[:, 1]
        elif model == "lightgbm":
            import lightgbm as lgb
            dtr = lgb.Dataset(X[tr], label=y[tr])
            params = dict(objective="binary", learning_rate=0.05, num_leaves=31,
                          min_data_in_leaf=100, feature_fraction=0.8,
                          bagging_fraction=0.8, bagging_freq=1, seed=SEED,
                          num_threads=8, verbose=-1)
            booster = lgb.train(params, dtr, num_boost_round=300)
            oof[va] = booster.predict(X[va])
        else:
            raise ValueError(model)
    return oof


def decode(m: pd.DataFrame, score_col: str) -> float:
    summary, _ = evaluate_tophalf(m, score_col, label_col="Label",
                                  user_col="userID", id_col="ID")
    return float(summary["row_accuracy"])


def main():
    results = {}
    fold_modes = ["strat", "group"]
    for split in SPLITS:
        m, feat_cols = build_features(split)
        print(f"\n== {split} ==  rows={len(m)} feats={len(feat_cols)} "
              f"users={m['userID'].nunique()}", flush=True)

        lg_acc = decode(m, "score_lightgcn")
        row = {"rows": len(m), "n_features": len(feat_cols),
               "lightgcn": round(lg_acc, 5),
               "lightgcn_ref": LGCN_REF[split],
               "fixed_blend_ref": FIXED_BLEND_REF[split]}

        for fold_mode in fold_modes:
            for model in ["logreg", "lightgbm"]:
                key = f"stack_{model}_{fold_mode}"
                try:
                    m[key] = oof_predict(m, feat_cols, model, fold_mode=fold_mode)
                    acc = decode(m, key)
                    row[f"{key}_oof"] = round(acc, 5)
                    row[f"{key}_vs_lgcn"] = round(acc - lg_acc, 5)
                    print(f"  [{fold_mode:5s}] {model:8s}: OOF={acc:.5f}  "
                          f"Δvs_LGCN={acc-lg_acc:+.5f}", flush=True)
                except Exception as e:
                    row[f"{key}_error"] = str(e)
                    print(f"  [{fold_mode:5s}] {model:8s}: ERROR {e}", flush=True)

        results[split] = row

    # Aggregate
    def mean_of(key):
        vals = [results[s].get(key) for s in SPLITS if results[s].get(key) is not None]
        return round(float(np.mean(vals)), 5) if vals else None

    summary = {
        "mean_lightgcn": mean_of("lightgcn"),
        "mean_fixed_blend": round(float(np.mean(list(FIXED_BLEND_REF.values()))), 5),
        "mean_stack_logreg_strat": mean_of("stack_logreg_strat_oof"),
        "mean_stack_lightgbm_strat": mean_of("stack_lightgbm_strat_oof"),
        "mean_stack_logreg_group": mean_of("stack_logreg_group_oof"),
        "mean_stack_lightgbm_group": mean_of("stack_lightgbm_group_oof"),
        "splits": results,
    }
    # Honest best uses GROUP (user-level) OOF only — strat may leak via within-user feats
    honest_cands = {k: summary[k] for k in
                    ["mean_lightgcn", "mean_fixed_blend",
                     "mean_stack_logreg_group", "mean_stack_lightgbm_group"]
                    if summary.get(k) is not None}
    best = max(honest_cands.items(), key=lambda kv: kv[1])
    summary["best_honest_method"] = {"name": best[0], "mean_acc": best[1]}
    # leakage diagnostic: strat - group gap
    summary["leakage_gap_logreg"] = (
        round(summary["mean_stack_logreg_strat"] - summary["mean_stack_logreg_group"], 5)
        if summary.get("mean_stack_logreg_strat") and summary.get("mean_stack_logreg_group")
        else None
    )
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    md = ["# LightGCN + Stage2 Stacking Meta-Learner (OOF validation)\n"]
    md.append("Honest evaluation uses **user-level GroupKFold** (group); row-level "
              "StratifiedKFold (strat) is shown only as a leakage diagnostic — "
              "within-user z/rank features can leak across rows of the same user.\n")
    md.append(f"- mean LightGCN: **{summary['mean_lightgcn']}**")
    md.append(f"- mean fixed-blend: {summary['mean_fixed_blend']}")
    md.append(f"- mean stack-logreg [group/honest]: {summary['mean_stack_logreg_group']}")
    md.append(f"- mean stack-lightgbm [group/honest]: {summary['mean_stack_lightgbm_group']}")
    md.append(f"- mean stack-logreg [strat/diag]: {summary['mean_stack_logreg_strat']}")
    md.append(f"- mean stack-lightgbm [strat/diag]: {summary['mean_stack_lightgbm_strat']}")
    md.append(f"- logreg strat−group leakage gap: {summary['leakage_gap_logreg']}")
    md.append(f"- **best honest method: {best[0]} = {best[1]}**\n")
    md.append("| split | LightGCN | fixed | lr-strat | lr-group | lgbm-strat | lgbm-group |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in SPLITS:
        r = results[s]
        md.append(
            f"| {s.replace('val_','').replace('_seed42','')} "
            f"| {r.get('lightgcn','-')} | {r.get('fixed_blend_ref','-')} "
            f"| {r.get('stack_logreg_strat_oof','-')} | {r.get('stack_logreg_group_oof','-')} "
            f"| {r.get('stack_lightgbm_strat_oof','-')} | {r.get('stack_lightgbm_group_oof','-')} |"
        )
    md.append("\n## Interpretation\n")
    honest_gain = best[1] - summary["mean_lightgcn"]
    if best[0].startswith("mean_stack") and honest_gain > 0:
        md.append(f"- Honest (GroupKFold) stacking beats LightGCN by {honest_gain:+.5f} "
                  f"→ a full-data stacker is worth materializing as a submission candidate (pending approval).")
    else:
        md.append("- On honest user-level GroupKFold, stacking does NOT beat LightGCN alone "
                  "→ the StratifiedKFold gain was leakage from within-user features. "
                  "Keep LightGCN single-axis; focus on stronger base axes (sweep / new families).")
    if summary["leakage_gap_logreg"] and summary["leakage_gap_logreg"] > 0.003:
        md.append(f"- ⚠ Large strat−group gap ({summary['leakage_gap_logreg']:+.5f}) confirms "
                  "within-user feature leakage in the row-level CV.")
    OUT_MD.write_text("\n".join(md))
    print(f"\nsaved: {OUT_JSON}\nsaved: {OUT_MD}")
    print(f"\nBEST HONEST: {best[0]} = {best[1]} (LightGCN={summary['mean_lightgcn']}, "
          f"honest_gain={honest_gain:+.5f})")
    print(f"leakage_gap_logreg(strat-group) = {summary['leakage_gap_logreg']}")


if __name__ == "__main__":
    main()
