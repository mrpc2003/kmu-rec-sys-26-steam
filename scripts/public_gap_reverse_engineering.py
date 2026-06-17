#!/usr/bin/env python3
"""Reverse-engineer the validation/public gap for KMURecSys26 Steam.

Safety contract:
- validation-only / analysis-only;
- no Kaggle API calls;
- no test submission/candidate CSV materialization;
- outputs only reports/*.json and reports/*.md.

The experiment asks whether uniform validation is directionally right but too hard in
absolute terms. We fit a simple public-mimic row weighting on uniform validation:
positive rows keep weight 1, while negative rows are reweighted by fold-train item
popularity. The beta parameter is chosen so the emb128 4-seed backbone's mean
weighted accuracy matches its known public score (0.77745). We then read the
rank-blend delta under that same weighting and compare it with the public delta
0.77825 - 0.77745 = +0.00080.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VAL_ROOT = ROOT / "artifacts/validation_uniform_panel20_20260612T214626KST"
SCORE_ROOT = ROOT / "artifacts/boundary_v1_panel20_score_coverage"
REPORT_JSON = ROOT / "reports/20260617_public_gap_reverse_engineering.json"
REPORT_MD = ROOT / "reports/20260617_public_gap_reverse_engineering.md"
SEEDS = (42, 123, 2024, 7)
PUBLIC_EMB128 = 0.77745
PUBLIC_RANKBLEND = 0.77825
PUBLIC_DELTA = PUBLIC_RANKBLEND - PUBLIC_EMB128


def score_col(df: pd.DataFrame) -> str:
    for col in ("score_lightgcn", "score", "score_layermix_uniform"):
        if col in df.columns:
            return col
    raise ValueError(f"No known score column in {df.columns.tolist()}")


def ensemble(split: str, model: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    cols: list[str] = []
    for seed in SEEDS:
        p = SCORE_ROOT / model / split / f"seed{seed}" / "lightgcn_scores.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        d = pd.read_csv(p)
        sc = score_col(d)
        need = {"ID", "userID", "gameID", "Label", sc}
        if not need.issubset(d.columns):
            raise ValueError(f"Missing columns in {p}: {need - set(d.columns)}")
        col = f"score_{model}_seed{seed}"
        part = d[["ID", "userID", "gameID", "Label", sc]].rename(columns={sc: col})
        if merged is None:
            merged = part
        else:
            before = len(merged)
            merged = merged.merge(part[["ID", col]], on="ID", how="inner", validate="one_to_one")
            if len(merged) != before:
                raise RuntimeError(f"Row alignment changed for {p}: {before}->{len(merged)}")
        cols.append(col)
    assert merged is not None
    merged[f"score_{model}"] = merged[cols].mean(axis=1)
    return merged[["ID", "userID", "gameID", "Label", f"score_{model}"]].sort_values("ID", kind="mergesort")


def user_rank_high_is_good(df: pd.DataFrame, col: str) -> np.ndarray:
    ranks = np.zeros(len(df), dtype=np.float64)
    values = df[col].to_numpy(dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        ranks[idx[np.argsort(values[idx], kind="mergesort")]] = np.arange(len(idx), dtype=np.float64)
    return ranks


def predict_tophalf(df: pd.DataFrame, col: str) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    values = df[col].to_numpy(dtype=float)
    labels = df["Label"].to_numpy(dtype=np.int8)
    ids = df["ID"].to_numpy(dtype=np.int64)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        k = int(labels[idx].sum())
        order = np.lexsort((ids[idx], -values[idx]))
        pred[idx[order[:k]]] = 1
    return pred


def weighted_accuracy(correct: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    return float(np.dot(correct.astype(float), w) / max(float(w.sum()), 1e-12))


def split_frame(split: str) -> pd.DataFrame:
    candidates_path = VAL_ROOT / split / "candidates.csv"
    train_path = VAL_ROOT / split / "train_interactions.csv"
    if not candidates_path.exists() or not train_path.exists():
        raise FileNotFoundError(f"missing split files for {split}")
    cand = pd.read_csv(candidates_path)
    tr = pd.read_csv(train_path)
    pop = tr.groupby("gameID").size().astype(float)
    user_deg = tr.groupby("userID").size().astype(float)
    a = ensemble(split, "emb128")
    b = ensemble(split, "emb192")[["ID", "score_emb192"]]
    df = a.merge(b, on="ID", how="inner", validate="one_to_one")
    df = df.merge(cand[["ID", "source", "requested_pos_k", "actual_pos_k"]], on="ID", how="left", validate="one_to_one")
    if len(df) != len(cand):
        raise RuntimeError(f"split row mismatch {split}: {len(df)} vs {len(cand)}")
    df["item_pop"] = df["gameID"].map(pop).fillna(0.0).astype(float)
    df["log_item_pop"] = np.log1p(df["item_pop"].to_numpy(dtype=float))
    df["user_deg"] = df["userID"].map(user_deg).fillna(0.0).astype(float)
    df["rank_emb128"] = user_rank_high_is_good(df, "score_emb128")
    df["rank_emb192"] = user_rank_high_is_good(df, "score_emb192")
    df["score_rankblend"] = df["rank_emb128"] + df["rank_emb192"]
    for col in ("score_emb128", "score_emb192", "score_rankblend"):
        df[f"pred_{col}"] = predict_tophalf(df, col)
        df[f"correct_{col}"] = (df[f"pred_{col}"].to_numpy(dtype=np.int8) == df["Label"].to_numpy(dtype=np.int8)).astype(np.int8)
    return df


def negative_pop_weights(df: pd.DataFrame, beta: float) -> np.ndarray:
    y = df["Label"].to_numpy(dtype=np.int8)
    z = df["log_item_pop"].to_numpy(dtype=float)
    neg = y == 0
    pos = y == 1
    weights = np.ones(len(df), dtype=float)
    if neg.any():
        z_neg = z[neg]
        z_std = float(z_neg.std(ddof=0))
        if z_std <= 1e-12:
            raw = np.ones(int(neg.sum()), dtype=float)
        else:
            z_norm = (z_neg - float(z_neg.mean())) / z_std
            raw = np.exp(beta * z_norm)
        # Preserve the validation 1:1 positive/negative mass, changing only which negatives matter.
        raw *= max(float(pos.sum()), 1.0) / max(float(raw.sum()), 1e-12)
        weights[neg] = raw
    return weights


def summarize_feature_shift(df: pd.DataFrame, weights: np.ndarray) -> dict[str, Any]:
    y = df["Label"].to_numpy(dtype=np.int8)
    out: dict[str, Any] = {}
    for mask_name, mask in {"positive": y == 1, "negative": y == 0}.items():
        if not mask.any():
            continue
        for col in ("item_pop", "log_item_pop", "user_deg"):
            vals = df.loc[mask, col].to_numpy(dtype=float)
            w = weights[mask]
            out[f"{mask_name}_{col}_mean_unweighted"] = float(vals.mean())
            out[f"{mask_name}_{col}_mean_weighted"] = float(np.dot(vals, w) / max(float(w.sum()), 1e-12))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-splits", type=int, default=20)
    ap.add_argument("--report-json", default=str(REPORT_JSON))
    ap.add_argument("--report-md", default=str(REPORT_MD))
    args = ap.parse_args()

    split_dirs = sorted(
        [p for p in VAL_ROOT.glob("val_random_uniform_seed*") if p.is_dir()],
        key=lambda p: int(re.search(r"seed(\d+)$", p.name).group(1)) if re.search(r"seed(\d+)$", p.name) else 0,
    )[: args.max_splits]
    frames = []
    for p in split_dirs:
        # Require both emb128 and emb192 score coverage.
        if not (SCORE_ROOT / "emb128" / p.name).exists() or not (SCORE_ROOT / "emb192" / p.name).exists():
            continue
        frames.append(split_frame(p.name))
    if not frames:
        raise RuntimeError("No usable validation splits found")

    beta_grid = np.round(np.linspace(-4.0, 4.0, 161), 4)
    grid_rows = []
    for beta in beta_grid:
        split_rows = []
        for df in frames:
            w = negative_pop_weights(df, float(beta))
            row = {
                "split": str(df.attrs.get("split", "")),
                "emb128_acc": weighted_accuracy(df["correct_score_emb128"].to_numpy(), w),
                "emb192_acc": weighted_accuracy(df["correct_score_emb192"].to_numpy(), w),
                "rankblend_acc": weighted_accuracy(df["correct_score_rankblend"].to_numpy(), w),
            }
            split_rows.append(row)
        agg = pd.DataFrame(split_rows)
        grid_rows.append({
            "beta": float(beta),
            "emb128_mean": float(agg["emb128_acc"].mean()),
            "emb128_std": float(agg["emb128_acc"].std(ddof=0)),
            "emb192_mean": float(agg["emb192_acc"].mean()),
            "rankblend_mean": float(agg["rankblend_acc"].mean()),
            "rankblend_delta_vs_emb128": float((agg["rankblend_acc"] - agg["emb128_acc"]).mean()),
            "target_abs_error": abs(float(agg["emb128_acc"].mean()) - PUBLIC_EMB128),
            "public_delta_error": abs(float((agg["rankblend_acc"] - agg["emb128_acc"]).mean()) - PUBLIC_DELTA),
        })
    grid = pd.DataFrame(grid_rows).sort_values(["target_abs_error", "public_delta_error"], ascending=[True, True])
    best_beta = float(grid.iloc[0]["beta"])

    split_summaries = []
    unweighted_rows = []
    mimic_rows = []
    for df, p in zip(frames, split_dirs, strict=False):
        w0 = np.ones(len(df), dtype=float)
        wm = negative_pop_weights(df, best_beta)
        base0 = weighted_accuracy(df["correct_score_emb128"].to_numpy(), w0)
        rb0 = weighted_accuracy(df["correct_score_rankblend"].to_numpy(), w0)
        basem = weighted_accuracy(df["correct_score_emb128"].to_numpy(), wm)
        rbm = weighted_accuracy(df["correct_score_rankblend"].to_numpy(), wm)
        unweighted_rows.append({"split": p.name, "emb128": base0, "rankblend": rb0, "delta": rb0 - base0})
        mimic_rows.append({"split": p.name, "emb128": basem, "rankblend": rbm, "delta": rbm - basem})
        fs = summarize_feature_shift(df, wm)
        fs.update({"split": p.name, "unweighted_emb128": base0, "mimic_emb128": basem, "unweighted_rankblend_delta": rb0 - base0, "mimic_rankblend_delta": rbm - basem})
        split_summaries.append(fs)

    unweighted = pd.DataFrame(unweighted_rows)
    mimic = pd.DataFrame(mimic_rows)
    top_grid = grid.head(15).to_dict(orient="records")
    payload = {
        "safety": {
            "validation_only": True,
            "kaggle_submit_executed": False,
            "candidate_csv_written": False,
            "hidden_label_access": False,
            "external_steam_scraping": False,
        },
        "public_targets": {
            "emb128_public": PUBLIC_EMB128,
            "rankblend_public": PUBLIC_RANKBLEND,
            "rankblend_public_delta_vs_emb128": PUBLIC_DELTA,
        },
        "method": {
            "split_source": str(VAL_ROOT.relative_to(ROOT)),
            "score_source": str(SCORE_ROOT.relative_to(ROOT)),
            "fitted_parameter": "negative-row item-popularity beta; positives weight=1; negative total mass normalized to positive total mass",
            "interpretation": "negative beta upweights low-popularity negative rows and downweights high-popularity negatives",
        },
        "n_splits": len(frames),
        "best_beta": best_beta,
        "best_grid_row": grid.iloc[0].to_dict(),
        "top_grid_rows": top_grid,
        "unweighted_summary": {
            "emb128_mean": float(unweighted["emb128"].mean()),
            "rankblend_mean": float(unweighted["rankblend"].mean()),
            "rankblend_delta_mean": float(unweighted["delta"].mean()),
            "rankblend_delta_std": float(unweighted["delta"].std(ddof=0)),
        },
        "mimic_summary": {
            "emb128_mean": float(mimic["emb128"].mean()),
            "rankblend_mean": float(mimic["rankblend"].mean()),
            "rankblend_delta_mean": float(mimic["delta"].mean()),
            "rankblend_delta_std": float(mimic["delta"].std(ddof=0)),
        },
        "split_summaries": split_summaries,
        "verdict": "PUBLIC_GAP_PARTLY_EXPLAINED_BY_EASIER_LOW_POP_NEGATIVES" if best_beta < 0 else "PUBLIC_GAP_NOT_EXPLAINED_BY_LOW_POP_NEGATIVE_WEIGHTING",
    }

    out_json = Path(args.report_json)
    out_md = Path(args.report_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")

    lines = []
    lines.append("# Validation ↔ Public Gap Reverse Engineering\n")
    lines.append("Safety: validation-only analysis. No Kaggle submit, no hidden labels, no external Steam scraping, no candidate CSV.\n")
    lines.append("## Target\n")
    lines.append(f"- emb128 public: `{PUBLIC_EMB128:.5f}`")
    lines.append(f"- rankblend public: `{PUBLIC_RANKBLEND:.5f}`")
    lines.append(f"- public delta(rankblend-emb128): `{PUBLIC_DELTA:+.5f}`\n")
    lines.append("## Fitted public-mimic weighting\n")
    lines.append(f"- splits used: `{len(frames)}` uniform panel splits")
    lines.append(f"- fitted beta: `{best_beta:+.3f}`")
    lines.append("- beta meaning: negative rows are weighted by item popularity after z-normalization; beta < 0 means tail/low-pop negatives matter more.\n")
    lines.append("| metric | unweighted uniform | public-mimic weighted | public target |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| emb128 mean acc | {payload['unweighted_summary']['emb128_mean']:.6f} | {payload['mimic_summary']['emb128_mean']:.6f} | {PUBLIC_EMB128:.5f} |")
    lines.append(f"| rankblend mean acc | {payload['unweighted_summary']['rankblend_mean']:.6f} | {payload['mimic_summary']['rankblend_mean']:.6f} | {PUBLIC_RANKBLEND:.5f} |")
    lines.append(f"| rankblend Δ vs emb128 | {payload['unweighted_summary']['rankblend_delta_mean']:+.6f} | {payload['mimic_summary']['rankblend_delta_mean']:+.6f} | {PUBLIC_DELTA:+.5f} |\n")
    lines.append("## Top beta rows\n")
    lines.append("| beta | emb128 mean | rankblend mean | rankblend Δ | abs target err | Δ err |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in top_grid[:10]:
        lines.append(f"| {r['beta']:+.3f} | {r['emb128_mean']:.6f} | {r['rankblend_mean']:.6f} | {r['rankblend_delta_vs_emb128']:+.6f} | {r['target_abs_error']:.6f} | {r['public_delta_error']:.6f} |")
    lines.append("\n## Feature shift at fitted beta\n")
    fs = pd.DataFrame(split_summaries)
    for col in ["negative_item_pop_mean_unweighted", "negative_item_pop_mean_weighted", "negative_log_item_pop_mean_unweighted", "negative_log_item_pop_mean_weighted"]:
        if col in fs.columns:
            lines.append(f"- {col}: `{fs[col].mean():.4f}`")
    lines.append("\n## Interpretation\n")
    if best_beta < 0:
        lines.append("- Public absolute score is matched only after making uniform validation negatives easier/tail-heavier than the raw uniform panel.")
        lines.append("- This supports the hypothesis that the old uniform split got the direction roughly right but overstated hard/head-item negatives.")
    else:
        lines.append("- The simple low-pop negative-weighting family did not explain the public gap; a richer feature model is needed.")
    lines.append("- Even after calibration, rankblend's public-sized edge remains tiny. Treat this surrogate as a ranking/triage tool, not as evidence for blind submissions.\n")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "report_json": str(out_json.relative_to(ROOT)),
        "report_md": str(out_md.relative_to(ROOT)),
        "best_beta": best_beta,
        "unweighted_summary": payload["unweighted_summary"],
        "mimic_summary": payload["mimic_summary"],
        "verdict": payload["verdict"],
    }, indent=2))


if __name__ == "__main__":
    main()
