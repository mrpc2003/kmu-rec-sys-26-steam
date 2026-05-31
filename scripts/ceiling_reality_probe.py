#!/usr/bin/env python3
"""Empirical ceiling-reality probe for kmu-rec-sys-26-steam.

Tests whether the ~21.4% simultaneous mis-ranking is IRREDUCIBLE label noise
or partially RECOVERABLE, using a panel of structurally diverse independent
models already scored on val_random_uniform_seed42.

Method (confident-learning / cross-model agreement style):
  - For each model, compute per-user top-half prediction (the actual metric rule).
  - Build [n_rows x n_models] correctness matrix.
  - Concentrate error mass: rows correct by 0%..100% of DIVERSE models.
  - Of the best model's errors, how many does at least one diverse model fix
    (contested = potentially recoverable) vs none (truly hard = irreducible)?
  - Oracle ceilings: majority vote (realizable-ish) and per-row oracle (upper bound).
  - Does any TRAIN-ONLY covariate (item popularity, user degree) separate the
    contested set? If not -> even recoverable disagreement is not addressable.

Validation-only. No Kaggle submission."""
import glob
import numpy as np
import pandas as pd

SPLIT = "val_random_uniform_seed42"

# Structurally diverse panel: one strong representative per family.
DIVERSE = {
    "lightgcn_emb128": "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03/%s/lightgcn_scores.csv" % SPLIT,
    "directau_g1": "artifacts/directau_uniform/g1.0/%s/lightgcn_scores.csv" % SPLIT,
    "xsimgcl_l0.02": "artifacts/xsimgcl_uniform/lam0.02_eps0.1/%s/lightgcn_scores.csv" % SPLIT,
    "sgl_l0.05": "artifacts/sgl_uniform/lam0.05/%s/lightgcn_scores.csv" % SPLIT,
    "dns_pool1": "artifacts/dns_uniform/pool1/%s/lightgcn_scores.csv" % SPLIT,
    "hyperbolic": "artifacts/hyperbolic_es130/hyp_emb64_L4_reg0.0001_seed42/%s/lightgcn_scores.csv" % SPLIT,
    "sasrec_L50": "artifacts/sasrec/sasrec_d64_b2_h2_L50_seed42/%s/lightgcn_scores.csv" % SPLIT,
    "din_d64": "artifacts/din_set/din_d64_L64_seed42/%s/lightgcn_scores.csv" % SPLIT,
    "emb192": "artifacts/capacity_uniform/emb192_L4_r3/%s/lightgcn_scores.csv" % SPLIT,
}
# Homogeneous LightGCN seed panel (for contrast: how much agreement is just seed noise)
HOMO = sorted(glob.glob("artifacts/lightgcn_emb128L4r3_ens/seed*/%s/lightgcn_scores.csv" % SPLIT))

TRAIN_INTER = "artifacts/validation/%s/train_interactions.csv" % SPLIT


def load(path):
    df = pd.read_csv(path)
    sc = "score_lightgcn" if "score_lightgcn" in df.columns else [c for c in df.columns if c.startswith("score")][0]
    return df.rename(columns={sc: "score"})[["ID", "userID", "gameID", "Label", "score"]]


def tophalf_pred(df):
    """Per-user top-half -> predicted positive. Returns array aligned to df order."""
    pred = np.zeros(len(df), dtype=np.int8)
    sc = df["score"].values
    for uid, idx in df.groupby("userID").indices.items():
        n = len(idx)
        k = n // 2
        order = idx[np.argsort(sc[idx])[::-1]]
        pred[order[:k]] = 1
    return pred


# --- load base (truth + key) from the strongest model file ---
base = load(DIVERSE["lightgcn_emb128"]).reset_index(drop=True)
key = base[["ID", "userID", "gameID", "Label"]].copy()
n = len(key)
label = key["Label"].values.astype(np.int8)
print(f"rows={n} users={key['userID'].nunique()} positives={label.sum()} (should be exactly half)")

# --- build correctness matrix for diverse panel ---
def build_correct(files_dict):
    names, cols = [], []
    for name, path in files_dict.items():
        d = load(path)
        d = key.merge(d[["ID", "score"]], on="ID", how="left")
        assert d["score"].isna().sum() == 0, f"{name} missing rows"
        pred = tophalf_pred(d)
        correct = (pred == label).astype(np.int8)
        names.append(name)
        cols.append(correct)
        print(f"  {name:18s} row_acc={correct.mean():.5f}")
    return names, np.vstack(cols).T  # [n_rows, n_models]

print("\n=== DIVERSE panel (one per family) ===")
dnames, D = build_correct(DIVERSE)
nm = D.shape[1]
frac_correct = D.mean(axis=1)  # per-row fraction of models correct

print("\n=== HOMOGENEOUS LightGCN-seed panel ===")
hfiles = {f"lgcn_seed_{i}": p for i, p in enumerate(HOMO)}
hnames, H = build_correct(hfiles)
hfrac = H.mean(axis=1)

# --- error-mass concentration (diverse) ---
print("\n=== Error-mass concentration (DIVERSE panel) ===")
bins = np.linspace(0, 1, nm + 1)
hist = np.histogram(frac_correct, bins=np.arange(nm + 2) / (nm) - 0.5/nm)[0]
for k in range(nm + 1):
    cnt = int((np.round(frac_correct * nm).astype(int) == k).sum())
    tag = "ALL-WRONG (irreducible?)" if k == 0 else ("ALL-RIGHT (trivial)" if k == nm else "contested")
    print(f"  {k}/{nm} models correct: {cnt:6d} rows ({100*cnt/n:5.2f}%)  {tag}")

all_wrong = int((np.round(frac_correct * nm).astype(int) == 0).sum())
all_right = int((np.round(frac_correct * nm).astype(int) == nm).sum())
contested = n - all_wrong - all_right
print(f"\n  ALL-WRONG  = {all_wrong} ({100*all_wrong/n:.2f}%)  <- floor of irreducibility")
print(f"  ALL-RIGHT  = {all_right} ({100*all_right/n:.2f}%)")
print(f"  CONTESTED  = {contested} ({100*contested/n:.2f}%)  <- max disagreement-recoverable mass")

# --- best model's errors: rescuable vs truly-hard ---
best_idx = int(np.argmax(D.mean(axis=0)))
best_correct = D[:, best_idx]
best_err = (best_correct == 0)
n_best_err = int(best_err.sum())
# rescuable = best wrong but >=1 diverse model right
some_right = (D.sum(axis=1) >= 1)
rescuable = int((best_err & some_right).sum())
truly_hard = int((best_err & ~some_right).sum())
print(f"\n=== Best diverse model = {dnames[best_idx]} (acc {best_correct.mean():.5f}) ===")
print(f"  best-model errors            = {n_best_err} ({100*n_best_err/n:.2f}%)")
print(f"  ...rescuable (>=1 other right)= {rescuable} ({100*rescuable/n_best_err:.1f}% of errors)")
print(f"  ...truly hard (all wrong)    = {truly_hard} ({100*truly_hard/n_best_err:.1f}% of errors)")

# --- oracle ceilings ---
# NOTE: naive majority/oracle can break per-user exact-half; report as diagnostic upper bounds only.
maj_pred_correct = (D.mean(axis=1) >= 0.5).astype(np.int8)  # row correct if majority of models correct
majority_acc = maj_pred_correct.mean()
oracle_acc = (D.max(axis=1)).mean()  # per-row: correct if ANY model correct
print(f"\n=== Oracle ceilings (diagnostic, may violate exact-half) ===")
print(f"  best single diverse model    = {best_correct.mean():.5f}")
print(f"  majority-correct rate        = {majority_acc:.5f}  (Δ {majority_acc-best_correct.mean():+.5f})")
print(f"  per-row ANY-model-correct     = {oracle_acc:.5f}  (Δ {oracle_acc-best_correct.mean():+.5f})  <- absolute upper bound")

# --- homogeneous contrast ---
h_all_wrong = int((np.round(hfrac * H.shape[1]).astype(int) == 0).sum())
print(f"\n=== Homogeneous-seed contrast ===")
print(f"  seed-panel ALL-WRONG = {h_all_wrong} ({100*h_all_wrong/n:.2f}%)  vs diverse ALL-WRONG {100*all_wrong/n:.2f}%")
print(f"  (if diverse ALL-WRONG ~= seed ALL-WRONG -> diversity adds no rescue -> irreducible)")

# --- does a TRAIN-ONLY covariate separate contested rows? ---
print("\n=== Train-only covariate separation of contested set ===")
try:
    tr = pd.read_csv(TRAIN_INTER)
    item_col = "gameID" if "gameID" in tr.columns else tr.columns[1]
    user_col = "userID" if "userID" in tr.columns else tr.columns[0]
    item_pop = tr.groupby(item_col).size()
    user_deg = tr.groupby(user_col).size()
    key2 = key.copy()
    key2["item_pop"] = key2["gameID"].map(item_pop).fillna(0).values
    key2["user_deg"] = key2["userID"].map(user_deg).fillna(0).values
    contested_mask = (np.round(frac_correct * nm).astype(int) > 0) & (np.round(frac_correct * nm).astype(int) < nm)
    allwrong_mask = (np.round(frac_correct * nm).astype(int) == 0)
    for cov in ["item_pop", "user_deg"]:
        v = key2[cov].values.astype(float)
        print(f"  {cov:9s}: contested median={np.median(v[contested_mask]):.1f}  "
              f"all-wrong median={np.median(v[allwrong_mask]):.1f}  "
              f"all-right median={np.median(v[np.round(frac_correct*nm).astype(int)==nm]):.1f}")
        # correlation of covariate with frac_correct (does easiness track popularity?)
        c = np.corrcoef(np.log1p(v), frac_correct)[0, 1]
        print(f"             corr(log1p({cov}), frac_correct) = {c:+.4f}")
except Exception as e:
    print(f"  (covariate step skipped: {e})")

print("\nDONE_CEILING_PROBE")
