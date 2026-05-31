#!/usr/bin/env python3
"""Independent re-computation of DIN set-encoder probe metrics.
Does NOT trust summary.json. Recomputes solo_acc, corr_z, eq_blend directly
from score CSVs against the canonical emb128 uniform-seed42 reference.
Validation-only. No Kaggle submission."""
import sys
import glob
import numpy as np
import pandas as pd

SPLIT = "val_random_uniform_seed42"

DIN_FILES = {
    "DIN_d64": "artifacts/din_set/din_d64_L64_seed42/%s/lightgcn_scores.csv" % SPLIT,
    "DIN_d128": "artifacts/din_set_variants/d128_L64/din_d128_L64_seed42/%s/lightgcn_scores.csv" % SPLIT,
}

# Candidate emb128 reference score files (find the one that yields ~0.76505 solo)
EMB128_CANDIDATES = sorted(glob.glob("artifacts/lightgcn_sweep_uniform_eval/emb128_*/%s/lightgcn_scores.csv" % SPLIT))


def per_user_tophalf_acc(df, score_col):
    """Each user: predict top-half by score as positive. Row accuracy + per-user.
    Assumes Label in {0,1}, each user has even candidates, exactly half positive."""
    df = df.copy()
    # rank within user by score desc; top (n/2) -> predicted positive
    preds = np.zeros(len(df), dtype=int)
    for uid, g in df.groupby("userID"):
        n = len(g)
        k = n // 2
        idx = g["score"].values.argsort()[::-1]  # desc
        local_pred = np.zeros(n, dtype=int)
        local_pred[idx[:k]] = 1
        preds[g.index.values] = local_pred
    df["pred"] = preds
    row_acc = (df["pred"].values == df["Label"].values).mean()
    pu = df.groupby("userID").apply(lambda x: (x["pred"].values == x["Label"].values).mean())
    return row_acc, pu.mean(), int(df["pred"].sum()), int(df["Label"].sum())


def load_scores(path):
    df = pd.read_csv(path)
    # score column may be 'score_lightgcn'
    sc = "score_lightgcn" if "score_lightgcn" in df.columns else [c for c in df.columns if c.startswith("score")][0]
    df = df.rename(columns={sc: "score"})
    return df[["ID", "userID", "gameID", "Label", "score"]]


def zscore_within_user(df):
    """z-score the score within each user (for fair blending)."""
    out = df.copy()
    z = np.zeros(len(df))
    for uid, g in df.groupby("userID"):
        v = g["score"].values.astype(float)
        mu, sd = v.mean(), v.std()
        z[g.index.values] = (v - mu) / sd if sd > 1e-12 else 0.0
    out["z"] = z
    return out


def integrity(df, name):
    s = df["score"].values
    n_nan = int(np.isnan(s).sum())
    n_inf = int(np.isinf(s).sum())
    nunique = int(pd.Series(s).nunique())
    print(f"  [{name}] rows={len(df)} nan={n_nan} inf={n_inf} unique_scores={nunique} "
          f"min={np.nanmin(s):.4f} max={np.nanmax(s):.4f}")
    degenerate = (n_nan > 0) or (n_inf > 0) or (nunique < 100)
    return degenerate


# 1. find canonical emb128 reference (solo ~0.76505)
print("=== locating canonical emb128 reference ===")
ref_df = None
ref_path = None
for p in EMB128_CANDIDATES:
    d = load_scores(p)
    ra, _, _, _ = per_user_tophalf_acc(d, "score")
    print(f"  {p.split('/')[1]}: solo={ra:.5f}")
    if abs(ra - 0.76505) < 0.0005:
        ref_df, ref_path = d, p
if ref_df is None:
    # fall back to best
    best = max(EMB128_CANDIDATES, key=lambda p: per_user_tophalf_acc(load_scores(p), "score")[0])
    ref_df, ref_path = load_scores(best), best
print(f"  -> using reference: {ref_path}")
ref_solo, _, _, _ = per_user_tophalf_acc(ref_df, "score")
print(f"  -> reference solo_acc = {ref_solo:.5f}")
ref_z = zscore_within_user(ref_df).set_index("ID")["z"]

# 2. verify each DIN variant
print("\n=== DIN variant independent verification ===")
for name, path in DIN_FILES.items():
    print(f"\n[{name}] {path}")
    d = load_scores(path)
    deg = integrity(d, name)
    solo, pu_solo, pred_pos, true_pos = per_user_tophalf_acc(d, "score")
    dz = zscore_within_user(d).set_index("ID")
    # align on ID
    common = dz.index.intersection(ref_z.index)
    a = dz.loc[common, "z"].values
    b = ref_z.loc[common].values
    corr = float(np.corrcoef(a, b)[0, 1])
    # equal-weight z blend
    blend_df = dz.loc[common].copy()
    blend_df["score"] = a + b
    blend_acc, _, _, _ = per_user_tophalf_acc(blend_df.reset_index(), "score")
    print(f"  RECOMPUTED solo_acc={solo:.5f}  corr_z={corr:.4f}  eq_blend={blend_acc:.5f}")
    print(f"  pred_pos={pred_pos} true_pos={true_pos} match={pred_pos==true_pos}")
    print(f"  Δ(eq_blend - ref_solo) = {blend_acc - ref_solo:+.5f}")
    print(f"  degenerate={deg}")
