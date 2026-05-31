#!/usr/bin/env python3
"""Boundary covariate expansion — disambiguate intrinsic-Bayes vs recoverable headroom.

Sisyphus direction 1(b). The neither-correct diagnosis showed errors are NOT flat across
popularity (inverted-U, peak at mid-pop) — but "structured" could be intrinsic Bayes difficulty
(mid-pop prior ~0.5) rather than recoverable signal. This script adds NOVEL orthogonal covariates
the original gate_floor PART C never tested, on the rank-K/K+1 boundary pairs:

  d_cooc_resid : conditional co-occurrence (does candidate co-play with THIS user's library),
                 BM25-normalized, THEN OLS-residualized against log-popularity. This is the key
                 test: marginal popularity (d_pop, AUC 0.664) is the proven trap; conditional
                 co-occurrence RESIDUALIZED of popularity is a different question — "user-internal
                 co-occurrence" signal independent of global popularity. Global redundancy with
                 EASE (corr 0.83) does NOT imply boundary no-signal.
  d_knn        : user-user Jaccard kNN consensus — fraction of the user's top-K most similar users
                 (by play-history Jaccard) who played the candidate in fold_train.
  d_cooc_raw   : the same co-occurrence WITHOUT residualization (control — should track d_pop).

Decision (Sisyphus gate):
  any novel covariate (d_cooc_resid, d_knn) with |AUC-0.5| >= 0.05 AND bootstrap CI excluding 0.5
    -> HEADROOM_EXISTS  (boundary carries non-popularity orthogonal signal -> escalate to geometry bet)
  all novel covariates |AUC-0.5| < 0.03
    -> CEILING_CONFIRMED (boundary residual is intrinsic; no public-transferable orthogonal signal)

CPU-only. Reuses emb128 ensemble scores + fold_train. No training. No Kaggle submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import load_train_interactions, ensure_dir, write_json  # noqa: E402

SPLIT = "val_random_uniform_seed42"
EMB128 = {
    42:   ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / SPLIT / "lightgcn_scores.csv",
    123:  ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed123" / SPLIT / "lightgcn_scores.csv",
    2024: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed2024" / SPLIT / "lightgcn_scores.csv",
    7:    ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed7" / SPLIT / "lightgcn_scores.csv",
}
KNN_K = 50
RNG = np.random.default_rng(20260601)


def base_ensemble():
    base = None; cols = []
    for s, p in EMB128.items():
        d = pd.read_csv(p)[["ID", "userID", "gameID", "Label", "score_lightgcn"]].rename(
            columns={"score_lightgcn": f"s{s}"})
        base = d if base is None else base.merge(d[["ID", f"s{s}"]], on="ID")
        cols.append(f"s{s}")
    base["s"] = base[cols].mean(axis=1)
    return base[["ID", "userID", "gameID", "Label", "s"]]


def auc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    pos = labels == 1; neg = labels == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    ss = scores[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (ranks[order[i]] + ranks[order[j]]) / 2
        i = j + 1
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def auc_bootstrap_ci(scores, labels, B=1000):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    n = len(scores); aucs = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        a = auc(scores[idx], labels[idx])
        if a == a:
            aucs.append(a)
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    out = ensure_dir(ROOT / "artifacts/boundary_covariate" / SPLIT)
    base = base_ensemble()
    fold = load_train_interactions(ROOT / "artifacts/validation" / SPLIT / "train_interactions.csv")

    # id maps from fold_train
    items = np.sort(fold["gameID"].unique()); i2r = {g: r for r, g in enumerate(items)}
    users = np.sort(fold["userID"].unique()); u2r = {u: r for r, u in enumerate(users)}
    n_items, n_users = len(items), len(users)
    rows = fold["userID"].map(u2r).to_numpy()
    cols = fold["gameID"].map(i2r).to_numpy()
    R = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_users, n_items))
    item_pop = np.asarray(R.sum(axis=0)).ravel()  # interaction count per item
    user_hist = {u: set(R.indices[R.indptr[u]:R.indptr[u + 1]].tolist()) for u in range(n_users)}

    # item-item co-occurrence (sparse): C = R^T R  (n_items x n_items), zero diagonal
    print("[bcov] building item-item co-occurrence...", flush=True)
    C = (R.T @ R).tocsr().astype(np.float32)
    C.setdiag(0); C.eliminate_zeros()

    # user-user cosine kNN on binary play history (for consensus vote)
    print("[bcov] building user-user similarity (cosine on binary history)...", flush=True)
    deg = np.sqrt(np.asarray(R.multiply(R).sum(axis=1)).ravel()) + 1e-9
    Rn = R.multiply(1.0 / deg[:, None]).tocsr()           # row-normalized
    # we compute top-K neighbors lazily per boundary user to bound memory

    # --- rank within user under base score; extract boundary pairs ---
    b = base.sort_values(["userID", "s", "ID"], ascending=[True, False, True], kind="mergesort").copy()
    b["rank"] = b.groupby("userID").cumcount() + 1
    ku = b.groupby("userID")["Label"].sum().astype(int).to_dict()
    b["Ku"] = b["userID"].map(ku)
    sel = b[b["rank"] == b["Ku"]][["userID", "gameID", "Label"]].rename(
        columns={"gameID": "g_in", "Label": "lab_in"})
    rej = b[b["rank"] == b["Ku"] + 1][["userID", "gameID", "Label"]].rename(
        columns={"gameID": "g_out", "Label": "lab_out"})
    bnd = sel.merge(rej, on="userID", how="inner")
    bnd = bnd[(bnd["lab_in"] + bnd["lab_out"]) == 1].copy()  # exactly-one-positive pairs
    bnd["y"] = (bnd["lab_in"] == 1).astype(int)              # 1 if the SELECTED side is the positive
    print(f"[bcov] informative boundary pairs n={len(bnd)}", flush=True)

    def cooc_score(uid, gid):
        ur = u2r.get(uid); ir = i2r.get(gid)
        if ur is None or ir is None:
            return 0.0
        hist = user_hist.get(ur, set())
        if not hist:
            return 0.0
        col = C.getrow(ir)
        idx = col.indices; val = col.data
        s = 0.0
        pi = item_pop[ir] + 1.0
        for k, g in enumerate(idx):
            if g in hist:
                s += val[k] / np.sqrt(pi * (item_pop[g] + 1.0))   # BM25-ish normalization
        return float(s)

    def knn_consensus(uid, gid):
        ur = u2r.get(uid); ir = i2r.get(gid)
        if ur is None or ir is None:
            return 0.0
        sims = (Rn @ Rn.getrow(ur).T).toarray().ravel()  # cosine sim to all users
        sims[ur] = -1.0
        if KNN_K < len(sims):
            nbr = np.argpartition(sims, -KNN_K)[-KNN_K:]
        else:
            nbr = np.arange(len(sims))
        played = 0; tot = 0
        for v in nbr:
            if sims[v] <= 0:
                continue
            tot += 1
            if ir in user_hist.get(v, set()):
                played += 1
        return played / tot if tot else 0.0

    print("[bcov] scoring boundary covariates (cooc + knn)...", flush=True)
    recs = []
    for _, r in bnd.iterrows():
        u = r["userID"]
        cin = cooc_score(u, r["g_in"]);  cout = cooc_score(u, r["g_out"])
        kin = knn_consensus(u, r["g_in"]); kout = knn_consensus(u, r["g_out"])
        pin = item_pop[i2r[r["g_in"]]] if r["g_in"] in i2r else 0.0
        pout = item_pop[i2r[r["g_out"]]] if r["g_out"] in i2r else 0.0
        recs.append({"y": r["y"], "d_cooc_raw": cin - cout, "d_knn": kin - kout,
                     "d_logpop": np.log(pin + 1) - np.log(pout + 1)})
    bd = pd.DataFrame(recs)

    # residualize d_cooc_raw against d_logpop (remove popularity component)
    X = np.column_stack([np.ones(len(bd)), bd["d_logpop"].to_numpy()])
    coef_c, *_ = np.linalg.lstsq(X, bd["d_cooc_raw"].to_numpy(), rcond=None)
    bd["d_cooc_resid"] = bd["d_cooc_raw"].to_numpy() - X @ coef_c
    # CRITICAL: also residualize d_knn against popularity — popular items appear in MANY
    # neighbors' play-sets regardless of similarity, so raw d_knn is popularity-confounded
    # exactly like d_cooc_raw was. A fair novelty test requires the same residualization.
    coef_k, *_ = np.linalg.lstsq(X, bd["d_knn"].to_numpy(), rcond=None)
    bd["d_knn_resid"] = bd["d_knn"].to_numpy() - X @ coef_k

    # persist per-pair covariates so future residualization/analysis is free (no slow knn rerun)
    bd.to_csv(out / "boundary_pairs_covariates.csv", index=False)

    results = {}
    for cov in ["d_cooc_raw", "d_cooc_resid", "d_knn", "d_knn_resid", "d_logpop"]:
        a = auc(bd[cov].to_numpy(), bd["y"].to_numpy())
        lo, hi = auc_bootstrap_ci(bd[cov].to_numpy(), bd["y"].to_numpy(), B=1000)
        results[cov] = {"auc": round(a, 4), "ci95": [round(lo, 4), round(hi, 4)],
                        "abs_dev": round(abs(a - 0.5), 4), "ci_excludes_half": bool(lo > 0.5 or hi < 0.5)}
        print(f"[bcov] {cov:14s} AUC={a:.4f} CI=[{lo:.4f},{hi:.4f}] |dev|={abs(a-0.5):.4f} excl0.5={results[cov]['ci_excludes_half']}", flush=True)

    # NOVELTY GATE: a covariate counts as recoverable orthogonal headroom ONLY if it clears
    # chance AFTER popularity residualization (raw versions are popularity in disguise = the trap).
    novel = ["d_cooc_resid", "d_knn_resid"]
    headroom = [c for c in novel if results[c]["abs_dev"] >= 0.05 and results[c]["ci_excludes_half"]]
    all_flat = all(results[c]["abs_dev"] < 0.03 for c in novel)
    if headroom:
        verdict = (f"HEADROOM_EXISTS: popularity-residualized covariate(s) {headroom} clear chance on "
                   f"the boundary -> non-popularity recoverable signal exists; escalate to geometry bet.")
    elif all_flat:
        verdict = ("CEILING_CONFIRMED: all popularity-residualized novel covariates (cooc-resid, "
                   "knn-resid) within 0.03 of chance on the boundary -> residual error is intrinsic "
                   "Bayes difficulty, not recoverable with public-transferable orthogonal info. Keep "
                   "final-2, no GPU bet. Raw d_cooc/d_knn above chance are popularity re-expression "
                   "(d_logpop AUC matches), i.e. the proven trap.")
    else:
        verdict = ("INTERMEDIATE: popularity-residualized novel covariates between 0.03 and 0.05 from "
                   "chance -> weak/ambiguous; soft no-go below the 0.05 escalation bar.")

    payload = {"split": SPLIT, "n_boundary_pairs": int(len(bd)), "knn_k": KNN_K,
               "covariate_auc": results, "novel_covariates": novel,
               "headroom_covariates": headroom, "verdict": verdict,
               "note": "CPU-only boundary covariate expansion (Sisyphus dir 1b). No training. No submission."}
    write_json(out / "summary.json", payload)
    print("\n" + "=" * 80, flush=True)
    print(f"[bcov] VERDICT: {verdict}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
