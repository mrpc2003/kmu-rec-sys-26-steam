#!/usr/bin/env python3
"""Multi-sampler cross-fitted residual invariance audit (GPT-5.5 Round-2 final experiment).

Tests whether ANY train-only symmetric item/interaction prior, added on top of LightGCN
within-user scores, produces a per-user top-half accuracy gain that is SIGN-STABLE across
four negative samplers (uniform / sqrtpop / popbin / communitypop). The four samplers share
EXACTLY the same held-out positives (verified 9998/9998); only negatives differ. A feature that
encodes true preference should help regardless of negative composition; a popularity/sampler
artifact will help on uniform but flip/collapse on hard samplers -> rejected.

Decision rule (a feature/model is a REAL signal only if):
  1. uniform-gate delta > 0  (it's the public LB surrogate)
  2. delta has the SAME sign (>0) on ALL FOUR samplers at the SAME lambda  (sampler-invariant)
  3. uniform delta exceeds noise band (>= +0.0007)

Validation-only. No Kaggle submission. CPU only."""
import numpy as np
import pandas as pd

SCORES = {
    "uniform":      "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03/val_random_uniform_seed42/lightgcn_scores.csv",
    "sqrtpop":      "artifacts/lightgcn_20260530/val_random_sqrtpop_seed42/lightgcn_scores.csv",
    "popbin":       "artifacts/lightgcn_20260530/val_random_popbin_seed42/lightgcn_scores.csv",
    "communitypop": "artifacts/lightgcn_ood_robustness/val_random_communitypop_seed42/lightgcn_scores.csv",
}
TRAIN = "artifacts/validation/val_random_uniform_seed42/train_interactions.csv"
NOISE = 0.0007


def load_scores(path):
    df = pd.read_csv(path)
    sc = [c for c in df.columns if c.startswith("score")][0]
    return df.rename(columns={sc: "score"})[["ID", "userID", "gameID", "Label", "score"]]


def zscore(x):
    x = x.astype(float)
    mu, sd = np.nanmean(x), np.nanstd(x)
    return (x - mu) / sd if sd > 1e-12 else np.zeros_like(x)


def within_user_z(df, col):
    out = np.zeros(len(df))
    v = df[col].values.astype(float)
    for uid, idx in df.groupby("userID").indices.items():
        s = v[idx]
        mu, sd = s.mean(), s.std()
        out[idx] = (s - mu) / sd if sd > 1e-12 else 0.0
    return out


def tophalf_acc_from_score(score, user_idx_groups, label):
    pred = np.zeros(len(score), dtype=np.int8)
    for idx in user_idx_groups:
        n = len(idx); k = n // 2
        order = idx[np.argsort(score[idx])[::-1]]
        pred[order[:k]] = 1
    return (pred == label).mean()


# ---- build train-only item & user aggregates ----
print("=== building train-only aggregates ===")
tr = pd.read_csv(TRAIN)
tr["date_ord"] = pd.to_datetime(tr["date"], errors="coerce").map(lambda d: d.toordinal() if pd.notnull(d) else np.nan)
tr["funny_pos"] = (tr["found_funny"].fillna(0) > 0).astype(float)

item_agg = tr.groupby("gameID").agg(
    it_hours_mean=("hours_transformed", "mean"),
    it_hours_med=("hours_transformed", "median"),
    it_hours_std=("hours_transformed", "std"),
    it_textlen_mean=("text_len", "mean"),
    it_ea_frac=("early_access", "mean"),
    it_funny_rate=("funny_pos", "mean"),
    it_count=("gameID", "size"),
    it_date_mean=("date_ord", "mean"),
).fillna(0.0)
user_agg = tr.groupby("userID").agg(
    us_hours_mean=("hours_transformed", "mean"),
    us_textlen_mean=("text_len", "mean"),
    us_count=("userID", "size"),
    us_date_mean=("date_ord", "mean"),
).fillna(0.0)
item_agg["it_logcount"] = np.log1p(item_agg["it_count"])
user_agg["us_logcount"] = np.log1p(user_agg["us_count"])
print(f"  items={len(item_agg)} users={len(user_agg)}")

# candidate features that change WITHIN-USER ranking: item-level + interaction.
# (pure user-level features are constant within a user -> cannot change top-half -> excluded as solo)
ITEM_FEATS = ["it_hours_mean", "it_hours_med", "it_hours_std", "it_textlen_mean",
              "it_ea_frac", "it_funny_rate", "it_logcount", "it_date_mean"]
INTER_FEATS = ["inter_hours_absdiff", "inter_hours_prod", "inter_date_absdiff"]
ALL_FEATS = ITEM_FEATS + INTER_FEATS

# ---- attach features to each sampler's candidate frame ----
def attach(df):
    df = df.merge(item_agg, left_on="gameID", right_index=True, how="left")
    df = df.merge(user_agg, left_on="userID", right_index=True, how="left")
    df = df.fillna(0.0)
    df["inter_hours_absdiff"] = np.abs(df["us_hours_mean"] - df["it_hours_mean"])
    df["inter_hours_prod"] = df["us_hours_mean"] * df["it_hours_mean"]
    df["inter_date_absdiff"] = np.abs(df["us_date_mean"] - df["it_date_mean"])
    return df


frames, groups, labels, base_z, feat_z = {}, {}, {}, {}, {}
for name, path in SCORES.items():
    d = attach(load_scores(path))
    frames[name] = d
    groups[name] = [np.asarray(idx) for idx in d.groupby("userID").indices.values()]
    labels[name] = d["Label"].values.astype(np.int8)
    base_z[name] = within_user_z(d, "score")
    feat_z[name] = {f: zscore(d[f].values) for f in ALL_FEATS}

# base solo per sampler
print("\n=== base LightGCN solo per sampler ===")
base_acc = {}
for name in SCORES:
    base_acc[name] = tophalf_acc_from_score(base_z[name], groups[name], labels[name])
    print(f"  {name:13s} base={base_acc[name]:.5f}")

# ---- single-feature lambda sweep with sign-invariance gate ----
LAMBDAS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, -0.05, -0.1, -0.2]
print("\n=== single-feature multi-sampler invariance audit ===")
print("  (PASS requires: uniform delta>=+%.4f AND all-4-sampler delta>0 at same lambda)\n" % NOISE)
any_pass = False
for f in ALL_FEATS:
    best = None
    # pick lambda by best uniform delta
    for lam in LAMBDAS:
        adj = base_z["uniform"] + lam * feat_z["uniform"][f]
        acc = tophalf_acc_from_score(adj, groups["uniform"], labels["uniform"])
        d = acc - base_acc["uniform"]
        if best is None or d > best[1]:
            best = (lam, d)
    lam, ud = best
    # evaluate that lambda on all samplers
    deltas = {}
    for name in SCORES:
        adj = base_z[name] + lam * feat_z[name][f]
        deltas[name] = tophalf_acc_from_score(adj, groups[name], labels[name]) - base_acc[name]
    all_pos = all(v > 0 for v in deltas.values())
    passes = (ud >= NOISE) and all_pos
    any_pass = any_pass or passes
    tag = "*** PASS ***" if passes else ("uniform+ but flips" if ud >= NOISE else "no uniform gain")
    print(f"  {f:20s} lam={lam:+.2f} uni={deltas['uniform']:+.5f} "
          f"sqrt={deltas['sqrtpop']:+.5f} pop={deltas['popbin']:+.5f} "
          f"comm={deltas['communitypop']:+.5f}  {tag}")

# ---- integrated residual model (cross-fitted ridge over ALL features) ----
print("\n=== integrated residual model (cross-fitted, user-split) ===")
from numpy.linalg import lstsq

def build_X(name):
    d = frames[name]
    cols = [base_z[name]] + [feat_z[name][f] for f in ALL_FEATS]
    return np.column_stack(cols)

# Cross-fit ON UNIFORM: split users into 2 halves, train residual weights on half A, apply to B & all samplers.
rng = np.random.default_rng(42)
u_uniform = frames["uniform"]["userID"].values
uniq_users = np.unique(u_uniform)
rng.shuffle(uniq_users)
half = set(uniq_users[:len(uniq_users)//2])
mask_A = np.array([u in half for u in u_uniform])

Xu = build_X("uniform")
yu = labels["uniform"].astype(float)
# fit logistic-ish via ridge on centered target within the model (linear probe), train on half A
lam_ridge = 1.0
XA, yA = Xu[mask_A], yu[mask_A]
W = np.linalg.solve(XA.T @ XA + lam_ridge * np.eye(XA.shape[1]), XA.T @ (yA - yA.mean()))
print(f"  fitted weights (base + {len(ALL_FEATS)} feats): base_w={W[0]:+.4f}")
for f, w in zip(ALL_FEATS, W[1:]):
    print(f"    {f:20s} w={w:+.4f}")

# evaluate integrated model: held-out half B (uniform, cross-fitted) + all samplers
print("\n  integrated-model deltas vs base:")
idxB = np.where(~mask_A)[0]                       # uniform users NOT used to fit W
dB = frames["uniform"].iloc[idxB].reset_index(drop=True)
gB = [np.asarray(v) for v in dB.groupby("userID").indices.values()]
labB = labels["uniform"][idxB]
for name in SCORES:
    X = build_X(name)
    s = X @ W
    if name == "uniform":
        acc_full = tophalf_acc_from_score(s, groups["uniform"], labels["uniform"])
        sB = s[idxB]
        accB = tophalf_acc_from_score(sB, gB, labB)
        baseB = tophalf_acc_from_score(base_z["uniform"][idxB], gB, labB)
        print(f"  {name:13s} full={acc_full - base_acc['uniform']:+.5f}  "
              f"heldoutB={accB - baseB:+.5f} (base_B={baseB:.5f})  <- cross-fitted, the honest number")
    else:
        acc = tophalf_acc_from_score(s, groups[name], labels[name])
        print(f"  {name:13s} delta={acc - base_acc[name]:+.5f}")

print("\nVERDICT:", "AT LEAST ONE FEATURE PASSED — escalate" if any_pass else
      "NO single feature is sampler-invariant; check integrated model heldoutB above.")
print("DONE_INVARIANCE_AUDIT")
