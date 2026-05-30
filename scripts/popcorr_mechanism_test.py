"""Mechanism test: does a popularity-correction help on hard samplers but HURT on uniform?

The logreg stacker's top negative weight was log_pop (-0.42): it learned to down-weight
popular items. That gained on sqrtpop/popbin OOF but the public submission FAILED. The
OOD finding says public ~ uniform split. This script decisively tests the mechanism
WITHOUT rebuilding Stage2: take the existing LightGCN scores on every sampler split,
apply a popularity-correction `score_lightgcn - alpha * within_user_z(log_pop)`, and check
whether positive alpha (popularity down-weighting) helps the hard samplers but hurts uniform.

If yes: it explains the stacker failure at the mechanism level (popularity correction is
sampler-specific, and the true test is near-uniform where it is harmful).

CPU-only, no training, no Kaggle submission.
"""
from __future__ import annotations

import ast
import json
import collections
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import evaluate_tophalf

# (split, path to lightgcn_scores.csv)
SPLIT_SCORES = {
    "random_uniform":      ROOT / "artifacts/lightgcn_ood_robustness/val_random_uniform_seed42/lightgcn_scores.csv",
    "random_sqrtpop":      ROOT / "artifacts/lightgcn_20260530/val_random_sqrtpop_seed42/lightgcn_scores.csv",
    "recent_sqrtpop":      ROOT / "artifacts/lightgcn_20260530/val_recent_sqrtpop_seed42/lightgcn_scores.csv",
    "random_popbin":       ROOT / "artifacts/lightgcn_20260530/val_random_popbin_seed42/lightgcn_scores.csv",
    "random_communitypop": ROOT / "artifacts/lightgcn_ood_robustness/val_random_communitypop_seed42/lightgcn_scores.csv",
    "recent_communitypop": ROOT / "artifacts/lightgcn_ood_robustness/val_recent_communitypop_seed42/lightgcn_scores.csv",
}
ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0]
OUT_JSON = ROOT / "reports/20260530_popcorr_mechanism.json"
OUT_MD = ROOT / "reports/20260530_popcorr_mechanism.md"


def load_item_pop() -> collections.Counter:
    pop = collections.Counter()
    with open(ROOT / "data/raw/public/data/train.json") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pop[ast.literal_eval(line)["gameID"]] += 1
            except Exception:
                pass
    return pop


def within_user_z(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("userID")[col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0.0)


def std_z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def acc(df: pd.DataFrame, col: str) -> float:
    summ, _ = evaluate_tophalf(df, col, label_col="Label", user_col="userID", id_col="ID")
    return float(summ["row_accuracy"])


def main():
    pop = load_item_pop()
    results = {}
    for split, path in SPLIT_SCORES.items():
        if not path.exists():
            print(f"[skip] {split}: {path} missing", flush=True)
            continue
        df = pd.read_csv(path)
        df["item_pop"] = df["gameID"].map(pop).fillna(0).astype(int)
        df["log_pop"] = np.log1p(df["item_pop"])
        df["wz_log_pop"] = within_user_z(df, "log_pop")
        df["z_lgcn"] = std_z(df["score_lightgcn"])
        base = acc(df, "score_lightgcn")
        row = {"lightgcn_base": round(base, 5)}
        for a in ALPHAS:
            # popularity down-weighting: subtract popularity from the (z-scored) LightGCN score
            df["corr"] = df["z_lgcn"] - a * df["wz_log_pop"]
            row[f"alpha_{a}"] = round(acc(df, "corr"), 5)
        # best alpha and its delta vs base
        best_a = max(ALPHAS, key=lambda a: row[f"alpha_{a}"])
        row["best_alpha"] = best_a
        row["best_delta_vs_base"] = round(row[f"alpha_{best_a}"] - base, 5)
        results[split] = row
        print(f"[{split:20s}] base={base:.5f} "
              + " ".join(f"a{a}={row[f'alpha_{a}']:.5f}" for a in ALPHAS)
              + f"  best_a={best_a} Δ={row['best_delta_vs_base']:+.5f}", flush=True)

    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    md = ["# Popularity-correction mechanism test (per negative-sampler)\n"]
    md.append("Tests whether down-weighting popular items (the stacker's learned behavior, "
              "`log_pop` weight −0.42) helps hard samplers but hurts the near-uniform "
              "public-surrogate split.\n")
    md.append("`score = z(LightGCN) − alpha · within_user_z(log_pop)`; alpha>0 = popularity down-weighting.\n")
    cols = "".join(f" a={a} |" for a in ALPHAS)
    md.append(f"| split | base |{cols} best_a | Δ |")
    md.append("|---|---:|" + "---:|"*len(ALPHAS) + "---:|---:|")
    for s, r in results.items():
        cells = "".join(f" {r[f'alpha_{a}']:.5f} |" for a in ALPHAS)
        md.append(f"| {s} | {r['lightgcn_base']:.5f} |{cells} {r['best_alpha']} | {r['best_delta_vs_base']:+.5f} |")
    md.append("\n## Interpretation\n")
    uni = results.get("random_uniform", {})
    hard = [s for s in ["random_sqrtpop", "random_popbin", "random_communitypop"] if s in results]
    uni_helps = uni.get("best_alpha", 0) > 0 and uni.get("best_delta_vs_base", 0) > 0.0005
    hard_helps = any(results[s]["best_alpha"] > 0 and results[s]["best_delta_vs_base"] > 0.0005 for s in hard)
    if hard_helps and not uni_helps:
        md.append("- ✅ Mechanism CONFIRMED: popularity down-weighting (alpha>0) helps the hard "
                  "popularity-matched samplers but does NOT help (or hurts) the near-uniform "
                  "public-surrogate split. This is exactly why the logreg stacker — which "
                  "learned a −0.42 log_pop weight on hard-sampler validation — failed on the "
                  "near-uniform public test.")
    else:
        md.append(f"- Mechanism signal: uniform best_alpha={uni.get('best_alpha')} "
                  f"(Δ={uni.get('best_delta_vs_base')}); hard-sampler popularity-correction "
                  f"helps={hard_helps}. Interpret with the table above.")
    OUT_MD.write_text("\n".join(md))
    print(f"\nsaved: {OUT_JSON}\nsaved: {OUT_MD}")


if __name__ == "__main__":
    main()
