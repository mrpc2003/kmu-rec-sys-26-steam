#!/usr/bin/env python3
"""Turbo-CF / graph-filtering CF probe (Park et al., SIGIR'24) — UNIFORM gate.

HONEST PRIOR (low): Turbo-CF is presented as a new graph-signal-processing paradigm, but
mechanically it is an ITEM-ITEM linear scorer: score = R_u · F(P_item), same family as
EASE/ItemKNN which this project already found REDUNDANT with BPR-LightGCN (corr 0.73-0.83).
So the expected outcome is "strong but redundant". The decisive measurement is therefore the
CORRELATION vs emb128, not raw accuracy. If corr is ~0.8 like EASE -> close the GF family.
If (surprisingly) corr is low AND solo is strong -> genuine orthogonal find.

Value: training-free, deterministic, runs in seconds on CPU (no GPU contention with the
XSimGCL sweep), and definitively answers whether the 2024 graph-filtering CF SOTA adds an
orthogonal axis on this small dense balanced-reranking dataset.

Turbo-CF method implemented:
  1. R in {0,1}^{m x n}; symmetric-normalize: Rt = Du^{-1/2} R Di^{-1/2}
  2. item-item similarity P = Rt^T Rt ; edge-weight regulation P <- P^{power} (element-wise)
  3. polynomial low-pass filter (no matrix decomposition):
        linear : F = P
        poly2  : F = P @ (alpha*I + (1-alpha)*P)   (2nd-order LPF approx)
  4. score(u,i) = (R_u normalized) @ F   ; read only candidate (u,i) entries.

Sweeps power in {1.0, 2.0} x filter in {linear, poly2(alpha=0.3)}. Uniform split only.
Gate = parameter-free: beat emb128 4-seed 0.76505 by > noise 0.0007. No Kaggle submission.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import (  # noqa: E402
    build_user_item_matrix, load_train_interactions, load_pairs_csv,
    evaluate_tophalf, ensure_dir, write_json,
)

SPLIT = "val_random_uniform_seed42"
EMB128_SEED42_UNI = ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / SPLIT / "lightgcn_scores.csv"
EMB128_ENS_REF = 0.76505
POP_FLOOR = 0.684
NOISE = 0.0007
OUT = ensure_dir(ROOT / "artifacts/turbocf_uniform")
OUT_JSON = ROOT / "reports/20260531_turbocf_uniform_gate.json"
OUT_MD = ROOT / "reports/20260531_turbocf_uniform_gate.md"


def zscore_within_user(df, col, user_col="userID"):
    g = df.groupby(user_col)[col]
    mu = g.transform("mean"); sd = g.transform("std").replace(0, 1.0).fillna(1.0)
    return ((df[col] - mu) / sd).to_numpy()


def build_item_filter(R: sp.csr_matrix, power: float):
    """R: m x n binary. Return normalized Rt (m x n) and item-item P (n x n dense)."""
    du = np.asarray(R.sum(axis=1)).ravel()  # user degree
    di = np.asarray(R.sum(axis=0)).ravel()  # item degree
    du_inv = np.divide(1.0, np.sqrt(du), out=np.zeros_like(du), where=du > 0)
    di_inv = np.divide(1.0, np.sqrt(di), out=np.zeros_like(di), where=di > 0)
    Rt = sp.diags(du_inv) @ R @ sp.diags(di_inv)        # m x n symmetric-normalized
    P = (Rt.T @ Rt).toarray().astype(np.float32)         # n x n item-item similarity
    if power != 1.0:
        sign = np.sign(P)
        P = sign * (np.abs(P) ** power)
    return Rt.tocsr(), P


def score_pairs(Rt, F, u2i, i2i, cand, users_idx_ok):
    """score(u,i) = Rt_u (normalized user row) @ F[:, i]."""
    S = (Rt @ F)  # m x n dense-ish (n=2437 ok)
    if sp.issparse(S):
        S = S.toarray()
    scores = np.full(len(cand), np.nan, np.float32)
    for k, (uid, gid) in enumerate(cand[["userID", "gameID"]].astype(str).itertuples(index=False)):
        ui = u2i.get(uid); ii = i2i.get(gid)
        if ui is not None and ii is not None:
            scores[k] = S[ui, ii]
    return scores


def main():
    sp_dir = ROOT / "artifacts/validation" / SPLIT
    tr = load_train_interactions(sp_dir / "train_interactions.csv")
    cand = load_pairs_csv(sp_dir / "candidates.csv")
    mat, u2i, i2i, users, items = build_user_item_matrix(tr, binary=True)
    print(f"[turbocf] {SPLIT}: {len(users)}u {len(items)}i {mat.nnz}nnz", flush=True)

    e128 = pd.read_csv(EMB128_SEED42_UNI)[["ID", "score_lightgcn"]].rename(
        columns={"score_lightgcn": "score_cf"})

    configs = [("power1_linear", 1.0, "linear"), ("power2_linear", 2.0, "linear"),
               ("power1_poly2a0.3", 1.0, "poly2"), ("power2_poly2a0.3", 2.0, "poly2")]
    results = []
    for label, power, ftype in configs:
        Rt, P = build_item_filter(mat, power)
        if ftype == "linear":
            F = P
        else:  # poly2: F = P @ (alpha*I + (1-alpha)*P)
            a = 0.3
            F = P @ (a * np.eye(P.shape[0], dtype=np.float32) + (1 - a) * P)
        sc = score_pairs(Rt, F, u2i, i2i, cand, None)
        c = cand.copy(); c["score_turbo"] = sc
        fill = np.nanmin(sc[~np.isnan(sc)]) - 1.0 if np.any(~np.isnan(sc)) else -1.0
        c["score_turbo"] = c["score_turbo"].fillna(fill)
        summ, _ = evaluate_tophalf(c, "score_turbo", label_col="Label", user_col="userID", id_col="ID")
        acc = round(float(summ["row_accuracy"]), 5)

        m = c.merge(e128, on="ID", how="inner")
        m["zt"] = zscore_within_user(m, "score_turbo")
        m["zc"] = zscore_within_user(m, "score_cf")
        corr_z = float(np.corrcoef(m["zt"], m["zc"])[0, 1])
        m["zb"] = 0.5 * m["zt"] + 0.5 * m["zc"]
        b, _ = evaluate_tophalf(m, "zb", label_col="Label", user_col="userID", id_col="ID")
        blend = round(float(b["row_accuracy"]), 5)
        results.append({"config": label, "solo_uniform": acc, "corr_withinuser_z": round(corr_z, 4),
                        "blend50_uniform": blend, "blend_vs_ref": round(blend - EMB128_ENS_REF, 5)})
        print(f"  {label}: solo={acc} corr_z={corr_z:.3f} blend50={blend} "
              f"(vs ref {blend-EMB128_ENS_REF:+.5f})", flush=True)

    results.sort(key=lambda r: r["solo_uniform"], reverse=True)
    best = results[0]
    best_blend = max(results, key=lambda r: r["blend50_uniform"])
    if best["solo_uniform"] < POP_FLOOR and best_blend["blend_vs_ref"] <= NOISE:
        tier = "REJECT"
        verdict = (f"best solo {best['solo_uniform']} < floor {POP_FLOOR} and best blend "
                   f"{best_blend['blend50_uniform']} (vs ref {best_blend['blend_vs_ref']:+.5f}) "
                   f"within noise. GF/Turbo-CF behaves like EASE/ItemKNN -> graph-filtering CF "
                   f"family REDUNDANT, axis closed.")
    elif best_blend["blend_vs_ref"] > NOISE and best["corr_withinuser_z"] < 0.6:
        tier = "ADOPT_CHECK"
        verdict = (f"blend {best_blend['blend50_uniform']} beats ref by {best_blend['blend_vs_ref']:+.5f} "
                   f"with corr_z {best['corr_withinuser_z']} -> potential orthogonal find, escalate.")
    else:
        tier = "REDUNDANT"
        verdict = (f"strong-ish (best solo {best['solo_uniform']}) but corr_z {best['corr_withinuser_z']} "
                   f"high and blend {best_blend['blend50_uniform']} vs ref {best_blend['blend_vs_ref']:+.5f} "
                   f"-> same family as EASE/ItemKNN, redundant. No new axis.")

    summary = {"note": "Turbo-CF/GF graph-filtering CF probe. CPU. No submission.",
               "split": SPLIT, "emb128_ens_ref": EMB128_ENS_REF, "pop_floor": POP_FLOOR,
               "noise": NOISE, "results": results, "best_solo": best, "best_blend": best_blend,
               "tier": tier, "verdict": verdict}
    write_json(OUT_JSON, summary)
    md = ["# Turbo-CF / Graph-Filtering CF — UNIFORM gate (public surrogate)\n",
          f"- split `{SPLIT}` | emb128 4-seed ref **{EMB128_ENS_REF}** | floor {POP_FLOOR} | noise ±{NOISE}",
          f"- **tier: {tier}** — {verdict}\n",
          "| config | solo uniform | corr_z vs emb128 | 50/50 z-blend | blend vs ref |",
          "|---|---:|---:|---:|---:|"]
    for r in results:
        md.append(f"| {r['config']} | {r['solo_uniform']} | {r['corr_withinuser_z']} | "
                  f"{r['blend50_uniform']} | {r['blend_vs_ref']:+.5f} |")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\n[TURBOCF GATE] tier={tier} best_solo={best['solo_uniform']} "
          f"best_blend={best_blend['blend50_uniform']} corr_z={best['corr_withinuser_z']}", flush=True)
    print(f"saved: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
