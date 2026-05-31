#!/usr/bin/env python3
"""Temporal compatibility reranking for the emb128 4-seed LightGCN — UNIFORM gate.

WHY this is genuinely orthogonal to the saturated BPR-LightGCN family
---------------------------------------------------------------------
LightGCN / EASE / GF-CF / MultiVAE all read the binary interaction graph's
co-occurrence structure. corr_z(emb64,128,192) >= 0.97 proves they converge to the
same neighbour/eigen-ordering. The `date` field is the ONLY interaction attribute
that lives OUTSIDE adjacency: it is an edge-timestamp distribution, not a topology.

Hypothesis: a hidden positive item i (the user actually reviewed it) tends to be
temporally compatible with the user's known activity window, while a UNIFORM negative
is drawn independently of the user's time profile. So:

    T_overlap(u,i) = sum_t  p_u(t) * log( p_i(t) / p_global(t) )

is the expected log-likelihood-ratio of "item i active during the user's active months
vs the global baseline". Built ONLY from fold-train (the heldout positive's own date is
removed), exactly mirroring the real test where positive dates are unknown.

THREE COMBINERS (parameter-free first):
  base          : emb128 4-seed ensemble (raw-mean of score_lightgcn). REFERENCE = 0.76505.
  T_only        : temporal score alone (diagnostic, expected weak).
  rank_sum      : within each user, pick K_u smallest (rank_LGCN + rank_T). No coefficient.
  boundary_swap : only swap the rank-K_u / rank-K_u+1 pair when temporal disagrees.
  T_resid       : residualize T_overlap against log-popularity & degree, THEN rank_sum.
                  (popularity-trap guard: pure T_overlap correlates with item breadth.)

POPULARITY-TRAP GUARD: popular items are active across all months -> broad p_i(t) ->
high overlap with everyone. That is just popularity re-encoded, which we proved is the
hard-sampler trap that hurts the uniform/public test. T_resid removes that component.

Gate (established thresholds):
  uniform Δ(combiner − base) >= 0.003                     -> CANDIDATE
  0.001 <= Δ < 0.003 AND paired McNemar p < 0.05          -> WEAK CANDIDATE
  Δ < 0.001                                               -> NOISE / CLOSE
Plus changed-decision precision: of rows whose decision flipped, fraction that became
correct must exceed 0.55 to be meaningful (else the flips are noise).

CPU-only. Validation-only. No Kaggle submission.
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
BASE_REF = 0.76505           # emb128 4-seed ensemble uniform (public 0.77745)
NOISE = 0.0007
EMB128_SEEDS_RAWMEAN = {
    42:   ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / SPLIT / "lightgcn_scores.csv",
    123:  ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed123" / SPLIT / "lightgcn_scores.csv",
    2024: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed2024" / SPLIT / "lightgcn_scores.csv",
    7:    ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed7" / SPLIT / "lightgcn_scores.csv",
}


def build_base_ensemble() -> pd.DataFrame:
    """Raw-mean of 4 seeds on score_lightgcn — the canonical 0.76505 ensemble."""
    base = None
    cols = []
    for s, p in EMB128_SEEDS_RAWMEAN.items():
        if not p.exists():
            raise FileNotFoundError(f"missing emb128 seed{s} scores: {p}")
        d = pd.read_csv(p)
        if base is None:
            base = d[["ID", "userID", "gameID", "Label"]].copy()
        col = f"s{s}"
        base = base.merge(d[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": col}), on="ID")
        cols.append(col)
    base["score_base"] = base[cols].mean(axis=1)
    return base


def build_temporal_profiles(fold_train: pd.DataFrame, beta: float):
    """Month-binned shrunk distributions. Returns (month_index, p_global, item_mat, user_mat, item_pop, user_deg, i2r, u2r)."""
    ft = fold_train.copy()
    ft["ym"] = ft["date"].dt.year * 12 + ft["date"].dt.month
    months = np.sort(ft["ym"].unique())
    m2c = {int(m): j for j, m in enumerate(months)}
    T = len(months)

    # global
    g = np.zeros(T, dtype=np.float64)
    gc = ft["ym"].value_counts()
    for m, c in gc.items():
        g[m2c[int(m)]] = c
    p_global = g / g.sum()
    p_global = np.clip(p_global, 1e-9, None)
    p_global = p_global / p_global.sum()

    # per-item raw month counts
    items = np.sort(ft["gameID"].unique())
    i2r = {g_: r for r, g_ in enumerate(items)}
    item_counts = np.zeros((len(items), T), dtype=np.float64)
    for (gid, ym), c in ft.groupby(["gameID", "ym"]).size().items():
        item_counts[i2r[gid], m2c[int(ym)]] += c
    item_pop = item_counts.sum(axis=1)  # total reviews per item
    # shrink toward global: p_i(t) = (c_i(t) + beta*p_global(t)) / (c_i + beta)
    item_p = (item_counts + beta * p_global[None, :]) / (item_pop[:, None] + beta)

    # per-user month counts (weighting distribution, also shrunk so sparse users fall back to global)
    users = np.sort(ft["userID"].unique())
    u2r = {u_: r for r, u_ in enumerate(users)}
    user_counts = np.zeros((len(users), T), dtype=np.float64)
    for (uid, ym), c in ft.groupby(["userID", "ym"]).size().items():
        user_counts[u2r[uid], m2c[int(ym)]] += c
    user_deg = user_counts.sum(axis=1)
    user_p = (user_counts + beta * p_global[None, :]) / (user_deg[:, None] + beta)
    user_p = user_p / user_p.sum(axis=1, keepdims=True)

    log_ratio_item = np.log(item_p) - np.log(p_global)[None, :]  # [I, T]
    return p_global, log_ratio_item, user_p, item_pop, user_deg, i2r, u2r


def temporal_overlap(df: pd.DataFrame, log_ratio_item, user_p, i2r, u2r) -> np.ndarray:
    """T_overlap(u,i) = sum_t p_u(t) * log(p_i(t)/p_global(t))."""
    out = np.zeros(len(df), dtype=np.float64)
    uids = df["userID"].to_numpy()
    gids = df["gameID"].to_numpy()
    for n in range(len(df)):
        ur = u2r.get(uids[n]); ir = i2r.get(gids[n])
        if ur is None or ir is None:
            out[n] = 0.0
            continue
        out[n] = float(np.dot(user_p[ur], log_ratio_item[ir]))
    return out


def within_user_rank(df: pd.DataFrame, score_col: str) -> np.ndarray:
    """Rank descending within user (1=best), deterministic ID tie-break."""
    tmp = df[["ID", "userID", score_col]].copy()
    tmp = tmp.sort_values(["userID", score_col, "ID"], ascending=[True, False, True], kind="mergesort")
    tmp["r"] = tmp.groupby("userID").cumcount() + 1
    return tmp.sort_values("ID")["r"].to_numpy()


def eval_col(df, col):
    summ, pred = evaluate_tophalf(df, col, label_col="Label", user_col="userID", id_col="ID")
    return round(float(summ["row_accuracy"]), 5), pred.sort_values("ID")["Correct"].to_numpy()


def mcnemar(a, b):
    a = a.astype(bool); b = b.astype(bool)
    nb = int(np.sum(b & ~a)); nc = int(np.sum(a & ~b)); n = nb + nc
    if n == 0:
        return {"fixes": nb, "breaks": nc, "discordant": 0, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(nb - nc) - 1) ** 2 / n
    return {"fixes": nb, "breaks": nc, "discordant": n, "chi2": round(chi2, 4),
            "p_value": round(math.erfc(math.sqrt(chi2 / 2.0)), 5)}


def changed_precision(base_corr, new_corr):
    """Of rows whose correctness flipped, fraction that became correct (gain)."""
    flipped = base_corr != new_corr
    n = int(flipped.sum())
    if n == 0:
        return {"changed": 0, "gain_fraction": None}
    gains = int(np.sum((~base_corr.astype(bool)) & new_corr.astype(bool)))
    return {"changed": n, "gains": gains, "losses": n - gains,
            "gain_fraction": round(gains / n, 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beta", type=float, default=10.0, help="pre-registered shrinkage prior strength")
    ap.add_argument("--out-dir", default="artifacts/temporal_compat")
    args = ap.parse_args()

    out = ensure_dir(Path(args.out_dir) / SPLIT)
    sp_dir = ROOT / "artifacts/validation" / SPLIT
    fold_train = load_train_interactions(sp_dir / "train_interactions.csv")
    base = build_base_ensemble()
    base_acc, base_corr = eval_col(base, "score_base")
    print(f"[temporal] base emb128 4-seed ensemble uniform = {base_acc} (ref {BASE_REF})", flush=True)

    p_global, log_ratio_item, user_p, item_pop, user_deg, i2r, u2r = build_temporal_profiles(fold_train, args.beta)
    print(f"[temporal] months={len(p_global)} items={len(i2r)} users={len(u2r)} beta={args.beta}", flush=True)

    df = base.copy()
    df["T"] = temporal_overlap(df, log_ratio_item, user_p, i2r, u2r)

    # orthogonality diagnostic
    corr_T_base = float(np.corrcoef(df["T"], df["score_base"])[0, 1])
    # popularity of candidate items (from fold_train)
    df["log_pop"] = np.log([item_pop[i2r[g]] + 1.0 if g in i2r else 1.0 for g in df["gameID"]])
    corr_T_pop = float(np.corrcoef(df["T"], df["log_pop"])[0, 1])
    print(f"[temporal] corr(T, base_score)={corr_T_base:.4f}  corr(T, log_pop)={corr_T_pop:.4f}", flush=True)

    # ranks
    df["rank_base"] = within_user_rank(df, "score_base")
    df["rank_T"] = within_user_rank(df, "T")

    # --- T_only ---
    t_acc, t_corr = eval_col(df, "T")

    # --- rank_sum: lower (rank_base + rank_T) is better -> use negative as score ---
    df["score_ranksum"] = -(df["rank_base"] + df["rank_T"]).astype(float)
    rs_acc, rs_corr = eval_col(df, "score_ranksum")

    # --- T_resid: residualize T on [log_pop, log(user_deg)] then rank_sum ---
    X = np.column_stack([
        np.ones(len(df)),
        df["log_pop"].to_numpy(),
        np.log([user_deg[u2r[u]] + 1.0 if u in u2r else 1.0 for u in df["userID"]]),
    ])
    yT = df["T"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, yT, rcond=None)
    df["T_resid"] = yT - X @ coef
    df["rank_Tresid"] = within_user_rank(df, "T_resid")
    df["score_ranksum_resid"] = -(df["rank_base"] + df["rank_Tresid"]).astype(float)
    rsr_acc, rsr_corr = eval_col(df, "score_ranksum_resid")

    # --- boundary_swap: only flip rank_base==K_u (last selected) with rank_base==K_u+1
    #     (first rejected) when T prefers the rejected one. K_u from Label sum. ---
    ku = df.groupby("userID")["Label"].sum().astype(int).to_dict()
    df["Ku"] = df["userID"].map(ku)
    df["score_bswap"] = df["score_base"].astype(float)
    # find per-user boundary pair by rank_base
    sel = df[df["rank_base"] == df["Ku"]][["userID", "ID", "T", "score_base"]].rename(
        columns={"ID": "id_in", "T": "T_in", "score_base": "s_in"})
    rej = df[df["rank_base"] == df["Ku"] + 1][["userID", "ID", "T", "score_base"]].rename(
        columns={"ID": "id_out", "T": "T_out", "score_base": "s_out"})
    bnd = sel.merge(rej, on="userID", how="inner")
    # swap when temporal prefers the rejected (T_out > T_in)
    swaps = bnd[bnd["T_out"] > bnd["T_in"]]
    # raise rejected above selected: assign rejected the selected's score + tiny eps, selected drops
    eps = 1e-6
    score_map = dict(zip(df["ID"], df["score_bswap"]))
    for _, r in swaps.iterrows():
        s_in = score_map[r["id_in"]]
        score_map[r["id_in"]] = s_in - eps
        score_map[r["id_out"]] = s_in + eps
    df["score_bswap"] = df["ID"].map(score_map)
    bs_acc, bs_corr = eval_col(df, "score_bswap")

    results = {}
    for name, acc, corr in [
        ("T_only", t_acc, t_corr),
        ("rank_sum", rs_acc, rs_corr),
        ("rank_sum_resid", rsr_acc, rsr_corr),
        ("boundary_swap", bs_acc, bs_corr),
    ]:
        d = round(acc - base_acc, 5)
        mc = mcnemar(base_corr, corr)
        cp = changed_precision(base_corr, corr)
        if d >= 0.003:
            tier = "CANDIDATE"
        elif d >= 0.001 and mc["p_value"] < 0.05:
            tier = "WEAK_CANDIDATE_MCNEMAR"
        elif d >= -NOISE:
            tier = "NO_GAIN_NOISE"
        else:
            tier = "REGRESS"
        results[name] = {"acc": acc, "delta_vs_base": d, "mcnemar": mc,
                         "changed_precision": cp, "tier": tier}
        print(f"[temporal] {name:16s} acc={acc} Δ={d:+.5f} "
              f"fixes={mc['fixes']} breaks={mc['breaks']} p={mc['p_value']} "
              f"changed={cp.get('changed')} gain_frac={cp.get('gain_fraction')} -> {tier}", flush=True)

    payload = {
        "split": SPLIT, "beta": args.beta, "base_acc": base_acc, "base_ref": BASE_REF,
        "corr_T_base": round(corr_T_base, 4), "corr_T_logpop": round(corr_T_pop, 4),
        "noise_band": NOISE, "results": results,
        "note": "CPU-only temporal compatibility rerank. Validation-only. No Kaggle submission.",
    }
    write_json(out / "summary.json", payload)
    df[["ID", "userID", "gameID", "Label", "score_base", "T", "T_resid",
        "rank_base", "rank_T"]].to_csv(out / "temporal_scores.csv", index=False)

    best = max(results.items(), key=lambda kv: kv[1]["delta_vs_base"])
    print("\n" + "=" * 80, flush=True)
    print(f"[temporal] base={base_acc}  best combiner = {best[0]} Δ={best[1]['delta_vs_base']:+.5f} ({best[1]['tier']})", flush=True)
    print(f"[temporal] corr(T,base)={corr_T_base:.4f}  corr(T,pop)={corr_T_pop:.4f}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
