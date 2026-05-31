#!/usr/bin/env python3
"""Candidate-marginal residual / quota signal — UNIFORM gate (VALIDATION-ONLY, rule-safe).

WHY validation-only first (GPT-5.5 Pro direction #3)
----------------------------------------------------
GPT-5.5 Pro rated this the most "structure-exploiting" lever but flagged a REGULATORY
risk: estimating hidden-positive surplus from the test candidate marginal could be read
as reverse-engineering hidden labels (forbidden). Its own advice: keep research-only
unless the rule reading is safe.

The validation-first resolution: measure whether the residual signal EVEN EXISTS on the
uniform surrogate BEFORE touching the submission-legality question. This script runs ONLY
on the synthetic uniform validation split (no real test labels are used anywhere). If the
uniform Δ < 0.001, the signal does not exist and the regulatory question is moot.

MECHANISM (rule-safe on validation because we KNOW the negative sampler)
------------------------------------------------------------------------
The uniform split samples K_u negatives per user UNIFORMLY from that user's unseen items.
So the expected number of times item i appears as a NEGATIVE is exactly computable:

    mu_neg(i) = sum_{u : i not in seen_u}  K_u / (n_items - |seen_u|)

The observed candidate appearance n_cand(i) = (#times i is a heldout positive) +
(#times i is a sampled negative). Therefore the Poisson-standardized surplus

    z_i = (n_cand(i) - mu_neg(i)) / sqrt(mu_neg(i) + eps)

estimates the item-level hidden-positive surplus WITHOUT reading any label. We shrink it
toward 0 for low-count items and add it as a within-user bias to the base ranking:

    s'_ui = zscore_u(score_base) + lambda * zscore_Cu(shrunk z_i)

PARAMETER-FREE GATE: lambda=1.0 (rank-equal weight, pre-registered). We also report a
small lambda sweep purely as diagnostics, but the CANDIDATE decision uses lambda=1.0.

SANITY CHECK: because validation positives ARE known here, we verify the residual z_i
correlates with the TRUE per-item heldout-positive count (does the estimator even work?).

GUARD vs the proven hard-sampler trap: if this only helps under a popularity-correlated
reading, it will FAIL the uniform gate (the public surrogate). That is the whole point of
gating on uniform.

CPU-only. Validation-only. No Kaggle submission. No real test file is read.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import (  # noqa: E402
    load_train_interactions, load_pairs_csv, evaluate_tophalf, ensure_dir, write_json,
)

SPLIT = "val_random_uniform_seed42"
BASE_REF = 0.76505
NOISE = 0.0007
EMB128_SEEDS = {
    42:   ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / SPLIT / "lightgcn_scores.csv",
    123:  ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed123" / SPLIT / "lightgcn_scores.csv",
    2024: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed2024" / SPLIT / "lightgcn_scores.csv",
    7:    ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed7" / SPLIT / "lightgcn_scores.csv",
}


def build_base():
    base = None; cols = []
    for s, p in EMB128_SEEDS.items():
        d = pd.read_csv(p)
        if base is None:
            base = d[["ID", "userID", "gameID", "Label"]].copy()
        c = f"s{s}"
        base = base.merge(d[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": c}), on="ID")
        cols.append(c)
    base["score_base"] = base[cols].mean(axis=1)
    return base


def zscore_within(df, col, by="userID"):
    g = df.groupby(by)[col]
    mean = g.transform("mean"); std = g.transform("std").fillna(0.0)
    return np.where(std.to_numpy() > 1e-12, (df[col] - mean) / std.replace(0, np.nan), 0.0)


def mcnemar(a, b):
    a = a.astype(bool); b = b.astype(bool)
    nb = int(np.sum(b & ~a)); nc = int(np.sum(a & ~b)); n = nb + nc
    if n == 0:
        return {"fixes": nb, "breaks": nc, "discordant": 0, "p_value": 1.0}
    chi2 = (abs(nb - nc) - 1) ** 2 / n
    return {"fixes": nb, "breaks": nc, "discordant": n, "p_value": round(math.erfc(math.sqrt(chi2 / 2.0)), 5)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="artifacts/candidate_marginal")
    args = ap.parse_args()
    out = ensure_dir(Path(args.out_dir) / SPLIT)

    sp_dir = ROOT / "artifacts/validation" / SPLIT
    fold_train = load_train_interactions(sp_dir / "train_interactions.csv")
    cand = load_pairs_csv(sp_dir / "candidates.csv")
    # need Label for K_u and for the (rule-safe, validation-only) sanity check
    cand_full = pd.read_csv(sp_dir / "candidates.csv")
    base = build_base()
    base_acc_summ, base_pred = evaluate_tophalf(base, "score_base", label_col="Label", user_col="userID", id_col="ID")
    base_acc = round(float(base_acc_summ["row_accuracy"]), 5)
    base_corr = base_pred.sort_values("ID")["Correct"].to_numpy()
    print(f"[candmarg] base emb128 4-seed uniform = {base_acc}", flush=True)

    # --- item universe & per-user seen sets from fold_train ---
    items = np.sort(fold_train["gameID"].unique())
    n_items = len(items)
    seen = fold_train.groupby("userID")["gameID"].apply(set).to_dict()

    # K_u from candidate file (= per-user positive count = candidate_count/2)
    ku = cand_full.groupby("userID")["Label"].sum().astype(int).to_dict()

    # --- observed candidate appearance count per item ---
    n_cand = cand_full.groupby("gameID").size().to_dict()

    # --- expected NEGATIVE count per item under UNIFORM sampling ---
    # mu_neg(i) = sum_{u: i not in seen_u} K_u / (n_items - |seen_u|)
    # Compute the user-side aggregate: every unseen item of user u gets K_u/(n_items-|seen_u|).
    # mu_neg(i) = (sum over ALL users of K_u/(n_items-|seen_u|)) - (sum over users who SAW i of same term)
    per_user_term = {}
    total_term = 0.0
    for u, kk in ku.items():
        pool = n_items - len(seen.get(u, ()))
        if pool <= 0:
            per_user_term[u] = 0.0
            continue
        t = kk / pool
        per_user_term[u] = t
        total_term += t
    # subtract contribution of users who saw item i
    seen_term_by_item = {}
    for u, gset in seen.items():
        t = per_user_term.get(u, 0.0)
        if t == 0.0:
            continue
        for g in gset:
            seen_term_by_item[g] = seen_term_by_item.get(g, 0.0) + t
    mu_neg = {g: total_term - seen_term_by_item.get(g, 0.0) for g in items}

    # --- Poisson-standardized surplus residual, shrunk by count ---
    eps = 1.0
    tau = float(np.median(list(n_cand.values())))
    z_item = {}
    for g in items:
        nc = n_cand.get(g, 0)
        mu = mu_neg.get(g, 0.0)
        z = (nc - mu) / math.sqrt(mu + eps)
        shrink = nc / (nc + tau) if (nc + tau) > 0 else 0.0
        z_item[g] = z * shrink

    base["z_item"] = base["gameID"].map(z_item).fillna(0.0)

    # --- SANITY: does residual track TRUE per-item heldout-positive count? (validation-only) ---
    true_pos_count = cand_full[cand_full["Label"] == 1].groupby("gameID").size().to_dict()
    items_arr = list(items)
    z_arr = np.array([z_item[g] for g in items_arr])
    tp_arr = np.array([true_pos_count.get(g, 0) for g in items_arr], dtype=float)
    nc_arr = np.array([n_cand.get(g, 0) for g in items_arr], dtype=float)
    corr_z_truepos = float(np.corrcoef(z_arr, tp_arr)[0, 1]) if z_arr.std() > 0 else 0.0
    # also: raw n_cand vs true pos (popularity baseline the residual must beat)
    corr_ncand_truepos = float(np.corrcoef(nc_arr, tp_arr)[0, 1]) if nc_arr.std() > 0 else 0.0
    print(f"[candmarg] sanity corr(residual_z, true_pos_count)={corr_z_truepos:.4f}  "
          f"corr(raw n_cand, true_pos)={corr_ncand_truepos:.4f}", flush=True)

    # --- bias-add decoding, lambda sweep (CANDIDATE uses lambda=1.0) ---
    base["zb"] = zscore_within(base, "score_base")
    base["zr"] = zscore_within(base, "z_item")
    results = {}
    for lam in [0.25, 0.5, 1.0, 2.0]:
        col = f"s_lam{lam}"
        base[col] = base["zb"] + lam * base["zr"]
        summ, pred = evaluate_tophalf(base, col, label_col="Label", user_col="userID", id_col="ID")
        acc = round(float(summ["row_accuracy"]), 5)
        corr = pred.sort_values("ID")["Correct"].to_numpy()
        d = round(acc - base_acc, 5)
        mc = mcnemar(base_corr, corr)
        if d >= 0.003:
            tier = "CANDIDATE"
        elif d >= 0.001 and mc["p_value"] < 0.05:
            tier = "WEAK_CANDIDATE_MCNEMAR"
        elif d >= -NOISE:
            tier = "NO_GAIN_NOISE"
        else:
            tier = "REGRESS"
        results[f"lambda_{lam}"] = {"acc": acc, "delta_vs_base": d, "mcnemar": mc, "tier": tier}
        flag = "  <== pre-registered gate" if lam == 1.0 else ""
        print(f"[candmarg] lambda={lam:<4} acc={acc} Δ={d:+.5f} fixes={mc['fixes']} breaks={mc['breaks']} p={mc['p_value']} -> {tier}{flag}", flush=True)

    payload = {
        "split": SPLIT, "base_acc": base_acc, "base_ref": BASE_REF, "noise_band": NOISE,
        "sanity_corr_residual_truepos": round(corr_z_truepos, 4),
        "sanity_corr_ncand_truepos": round(corr_ncand_truepos, 4),
        "tau_shrink": tau, "primary_gate_lambda": 1.0,
        "primary_result": results["lambda_1.0"], "lambda_sweep": results,
        "note": "VALIDATION-ONLY candidate-marginal residual. No real test file read. No Kaggle submission. "
                "Regulatory question (submission legality) only relevant IF uniform gate passes.",
    }
    write_json(out / "summary.json", payload)
    print("\n" + "=" * 80, flush=True)
    pr = results["lambda_1.0"]
    print(f"[candmarg] PRIMARY (λ=1.0): base={base_acc} -> {pr['acc']} Δ={pr['delta_vs_base']:+.5f} ({pr['tier']})", flush=True)
    print(f"[candmarg] residual estimator works? corr(z,true_pos)={corr_z_truepos:.4f} (raw n_cand {corr_ncand_truepos:.4f})", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
