#!/usr/bin/env python3
"""Validation-only rank aggregation and boundary rerank probe for KMURecSys26 Steam.

No Kaggle submission. Consumes saved validation score CSVs and writes a JSON/Markdown report.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "last_slot_rankagg"
REPORT = ROOT / "reports" / "20260601_last_slot_rankagg_probe.md"
JSON_OUT = OUT_DIR / "rankagg_seed42_summary.json"

SPLIT = "val_random_uniform_seed42"
SEEDS = [42, 7, 123, 2024]

EMB128_PATHS = {
    42: ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03/val_random_uniform_seed42/lightgcn_scores.csv",
    7: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed7/val_random_uniform_seed42/lightgcn_scores.csv",
    123: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed123/val_random_uniform_seed42/lightgcn_scores.csv",
    2024: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed2024/val_random_uniform_seed42/lightgcn_scores.csv",
}
EMB64_PATHS = {
    42: ROOT / "artifacts/lightgcn_ood_robustness/val_random_uniform_seed42/lightgcn_scores.csv",
    7: ROOT / "artifacts/lightgcn_uniform_eval/seed7/val_random_uniform_seed42/lightgcn_scores.csv",
    123: ROOT / "artifacts/lightgcn_uniform_eval/seed123/val_random_uniform_seed42/lightgcn_scores.csv",
    2024: ROOT / "artifacts/lightgcn_uniform_eval/seed2024/val_random_uniform_seed42/lightgcn_scores.csv",
}


def load_score_set(paths: Dict[int, Path], prefix: str) -> pd.DataFrame:
    base = None
    for seed, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        required = {"ID", "userID", "gameID", "Label", "score_lightgcn"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing {missing}")
        cols = ["ID", "userID", "gameID", "Label"]
        if base is None:
            base = df[cols].copy()
        else:
            if not base[cols].equals(df[cols]):
                raise ValueError(f"row mismatch for {path}")
        base[f"{prefix}_s{seed}"] = df["score_lightgcn"].astype(float).to_numpy()
    assert base is not None
    return base


def per_user_rank_features(df: pd.DataFrame, cols: List[str], descending: bool = True) -> pd.DataFrame:
    # Smaller rank is better. Use average rank for ties but scores are mostly unique.
    out = pd.DataFrame(index=df.index)
    for col in cols:
        out[f"rank_{col}"] = df.groupby("userID")[col].rank(ascending=not descending, method="average")
        n = df.groupby("userID")[col].transform("size")
        # percentile where 0 is best, 1 is worst
        out[f"pct_{col}"] = (out[f"rank_{col}"] - 1) / (n - 1).replace(0, np.nan)
    return out


def z_user(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("userID")[col]
    mu = g.transform("mean")
    sd = g.transform("std").replace(0, np.nan).fillna(1.0)
    return (df[col] - mu) / sd


def predict_tophalf(df: pd.DataFrame, score_col: str) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    # Stable deterministic tie-break: score desc, ID asc.
    for _, idx in df.groupby("userID", sort=False).groups.items():
        ids = np.asarray(list(idx))
        n = len(ids)
        k = int(df.loc[ids, "Label"].sum())  # validation: exact true positives per user
        if k * 2 != n:
            # Fallback to half if a future split lacks labels, but current validation should be exact.
            k = n // 2
        scores = df.loc[ids, score_col].to_numpy()
        row_ids = df.loc[ids, "ID"].to_numpy()
        order = np.lexsort((row_ids, -scores))
        pred[ids[order[:k]]] = 1
    return pred


def evaluate(df: pd.DataFrame, score_col: str, base_pred: np.ndarray | None = None) -> dict:
    pred = predict_tophalf(df, score_col)
    y = df["Label"].to_numpy(dtype=np.int8)
    acc = float((pred == y).mean())
    d = {"score": score_col, "acc": acc, "pred_pos": int(pred.sum())}
    if base_pred is not None:
        base_ok = base_pred == y
        cand_ok = pred == y
        b = int(np.sum(base_ok & ~cand_ok))
        c = int(np.sum(~base_ok & cand_ok))
        net = c - b
        # continuity-corrected McNemar chi-square approx, df=1 survival = erfc(sqrt(x/2))
        if b + c == 0:
            chi2 = 0.0
            p = 1.0
        else:
            chi2 = (max(abs(b - c) - 1, 0) ** 2) / (b + c)
            p = math.erfc(math.sqrt(chi2 / 2.0))
        d.update({"delta": acc - float(base_ok.mean()), "base_breaks": b, "candidate_fixes": c, "net_fixes": net, "mcnemar_chi2": chi2, "mcnemar_p": p, "flips": int(np.sum(pred != base_pred))})
    return d


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d128 = load_score_set(EMB128_PATHS, "e128")
    d64 = load_score_set(EMB64_PATHS, "e64")
    meta_cols = ["ID", "userID", "gameID", "Label"]
    if not d128[meta_cols].equals(d64[meta_cols]):
        raise ValueError("emb128 and emb64 row mismatch")
    df = d128.copy()
    for col in d64.columns:
        if col not in meta_cols:
            df[col] = d64[col]

    e128_cols = [f"e128_s{s}" for s in SEEDS]
    e64_cols = [f"e64_s{s}" for s in SEEDS]
    all_cols = e128_cols + e64_cols

    # Baseline and score-space controls.
    df["base_emb128_raw_mean"] = df[e128_cols].mean(axis=1)
    df["emb64_raw_mean"] = df[e64_cols].mean(axis=1)
    for col in all_cols:
        df[f"z_{col}"] = z_user(df, col)
    df["zmean_128_64_equal"] = df[[f"z_{c}" for c in all_cols]].mean(axis=1)
    df["zmean_128x2_64"] = (2 * df[[f"z_{c}" for c in e128_cols]].mean(axis=1) + df[[f"z_{c}" for c in e64_cols]].mean(axis=1)) / 3

    # Uncertainty-aware seed stability scores. These are fixed-grid LCB/UCB probes, not learned stackers.
    df["std_128"] = df[[f"z_{c}" for c in e128_cols]].std(axis=1).fillna(0.0)
    df["std_64"] = df[[f"z_{c}" for c in e64_cols]].std(axis=1).fillna(0.0)
    df["zmean_128"] = df[[f"z_{c}" for c in e128_cols]].mean(axis=1)
    df["zmean_64"] = df[[f"z_{c}" for c in e64_cols]].mean(axis=1)
    for lam in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
        tag = str(lam).replace('-', 'm').replace('.', 'p')
        df[f"uncert_128_lam{tag}"] = df["zmean_128"] + lam * df["std_128"]
        df[f"uncert_128_64_lam{tag}"] = 0.75 * df["zmean_128"] + 0.25 * df["zmean_64"] + lam * (0.75 * df["std_128"] + 0.25 * df["std_64"])

    # Rank aggregation. Negative rank-based scores so larger=better.
    rank_feats = per_user_rank_features(df, all_cols)
    for col in rank_feats.columns:
        df[col] = rank_feats[col]
    rank128 = [f"rank_{c}" for c in e128_cols]
    rank64 = [f"rank_{c}" for c in e64_cols]
    rankall = rank128 + rank64
    df["rankagg_128_mean"] = -df[rank128].mean(axis=1)
    df["rankagg_128_median"] = -df[rank128].median(axis=1)
    df["rankagg_128_64_mean"] = -df[rankall].mean(axis=1)
    df["rankagg_128_64_median"] = -df[rankall].median(axis=1)
    df["rankagg_128x2_64_mean"] = -(2 * df[rank128].mean(axis=1) + df[rank64].mean(axis=1)) / 3
    # RRF variants, larger better.
    for k in [1, 5, 10, 60]:
        df[f"rrf_128_64_k{k}"] = sum(1.0 / (k + df[r]) for r in rankall)
        df[f"rrf_128_k{k}"] = sum(1.0 / (k + df[r]) for r in rank128)

    # Majority vote top-half membership per model as a score; tie-break by baseline raw mean.
    for c in all_cols:
        pred = predict_tophalf(df, c)
        df[f"vote_{c}"] = pred
    df["vote_128_64_then_base"] = df[[f"vote_{c}" for c in all_cols]].sum(axis=1) + 1e-6 * z_user(df, "base_emb128_raw_mean")
    df["vote_128_then_base"] = df[[f"vote_{c}" for c in e128_cols]].sum(axis=1) + 1e-6 * z_user(df, "base_emb128_raw_mean")

    # Conservative boundary reranker: preserve baseline except if emb64/rank consensus strongly votes against boundary item.
    base_pred = predict_tophalf(df, "base_emb128_raw_mean")
    df["boundary_rule_score"] = df["base_emb128_raw_mean"].copy()
    # For each user, swap only rank-K and K+1 when baseline margin is small and emb64 4/4 prefers outside.
    changed = 0
    for user, idx in df.groupby("userID", sort=False).groups.items():
        ids = np.asarray(list(idx))
        y = df.loc[ids, "Label"].to_numpy()
        k = int(y.sum())
        if k <= 0 or k >= len(ids):
            continue
        scores = df.loc[ids, "base_emb128_raw_mean"].to_numpy()
        row_ids = df.loc[ids, "ID"].to_numpy()
        order = np.lexsort((row_ids, -scores))
        in_i = ids[order[k-1]]
        out_i = ids[order[k]]
        margin = df.at[in_i, "base_emb128_raw_mean"] - df.at[out_i, "base_emb128_raw_mean"]
        # user-local margin threshold as small absolute z gap; use rank-consensus not labels.
        emb64_pref = sum(df.at[out_i, c] > df.at[in_i, c] for c in e64_cols)
        emb128_minority_pref = sum(df.at[out_i, c] > df.at[in_i, c] for c in e128_cols)
        if margin < 0.10 and emb64_pref >= 4 and emb128_minority_pref >= 2:
            # Force out_i just above in_i without disturbing other ranks.
            mid = (df.at[in_i, "base_emb128_raw_mean"] + df.at[out_i, "base_emb128_raw_mean"]) / 2
            df.at[out_i, "boundary_rule_score"] = mid + 1e-5
            df.at[in_i, "boundary_rule_score"] = mid - 1e-5
            changed += 2

    score_cols = [
        "base_emb128_raw_mean", "emb64_raw_mean", "zmean_128_64_equal", "zmean_128x2_64",
        *[f"uncert_128_lam{str(lam).replace('-', 'm').replace('.', 'p')}" for lam in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]],
        *[f"uncert_128_64_lam{str(lam).replace('-', 'm').replace('.', 'p')}" for lam in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]],
        "rankagg_128_mean", "rankagg_128_median", "rankagg_128_64_mean", "rankagg_128_64_median", "rankagg_128x2_64_mean",
        "rrf_128_64_k1", "rrf_128_64_k5", "rrf_128_64_k10", "rrf_128_64_k60", "rrf_128_k1", "rrf_128_k5", "rrf_128_k10", "rrf_128_k60",
        "vote_128_64_then_base", "vote_128_then_base", "boundary_rule_score",
    ]
    base = predict_tophalf(df, "base_emb128_raw_mean")
    results = [evaluate(df, c, base) for c in score_cols]
    results = sorted(results, key=lambda x: (x["acc"], x["delta"]), reverse=True)
    best = results[0]
    df_out = df[meta_cols + ["base_emb128_raw_mean", "emb64_raw_mean"] + score_cols].copy()
    df_out.to_csv(OUT_DIR / "rankagg_seed42_scores.csv", index=False)
    summary = {"split": SPLIT, "rows": int(len(df)), "users": int(df.userID.nunique()), "boundary_changed_rows": changed, "results": results}
    JSON_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Last-slot validation-only probe: robust rank aggregation (seed42 uniform)\n")
    lines.append("**Safety:** validation_only=true · candidate_csv_written=false · kaggle_submit_executed=false\n")
    lines.append(f"Rows={len(df)}, users={df.userID.nunique()}, split={SPLIT}. Baseline is emb128 4-seed raw mean.\n")
    lines.append("## Top results\n")
    lines.append("| score | acc | delta_vs_base | flips | fixes | breaks | McNemar p |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in results[:12]:
        lines.append(f"| {r['score']} | {r['acc']:.5f} | {r.get('delta',0):+.5f} | {r.get('flips',0)} | {r.get('candidate_fixes',0)} | {r.get('base_breaks',0)} | {r.get('mcnemar_p',1):.4f} |\n")
    lines.append("\n## Gate verdict\n")
    gate = "REJECT"
    reason = "best non-baseline does not exceed +0.00355 MDE and/or McNemar p<0.05 gate"
    nonbase = [r for r in results if r["score"] != "base_emb128_raw_mean"]
    best_nonbase = nonbase[0]
    if best_nonbase["delta"] >= 0.00355 and best_nonbase["mcnemar_p"] < 0.05:
        gate = "ESCALATE_3SPLIT"
        reason = "single-split gain exceeds MDE with paired evidence; train/evaluate panel before any submission"
    elif best_nonbase["delta"] > 0 and best_nonbase["mcnemar_p"] < 0.05:
        gate = "WEAK_SIGNAL_PANEL_ONLY"
        reason = "statistical paired signal below MDE; panel required"
    lines.append(f"**{gate}** — {reason}. Best non-baseline: `{best_nonbase['score']}` delta={best_nonbase['delta']:+.5f}, p={best_nonbase['mcnemar_p']:.4f}.\n")
    REPORT.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "json": str(JSON_OUT), "best": best, "best_nonbase": best_nonbase, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
