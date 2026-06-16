#!/usr/bin/env python3
"""GF-CF spectral filter probe on uniform validation panels.

Validation-only probe for KMURecSys26 Steam.  This script never reads full-test
labels, never writes a Kaggle submission/candidate CSV, and never calls Kaggle.

Implemented score family follows the standard GF-CF decomposition:

    R_tilde = D_u^-1/2 R D_i^-1/2
    P_tilde = R_tilde.T @ R_tilde
    F_idl(k) = D_i^-1/2 V_k V_k.T D_i^1/2
    score = R @ (P_tilde + gamma * F_idl(k))

where V_k are the top-k right singular vectors of R_tilde.  The goal is not to
claim a paper-perfect reproduction of every BSPM/FaGSP variant, but to test the
missing decomposition-based GF-CF axis that the earlier polynomial Turbo-CF
probe did not cover.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import (  # noqa: E402
    build_user_item_matrix,
    ensure_dir,
    evaluate_tophalf,
    load_pairs_csv,
    load_train_interactions,
)


def parse_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def safe_power(values: np.ndarray, exponent: float) -> np.ndarray:
    return np.power(np.maximum(values, 0.0), exponent)


def score_candidates(score_matrix: np.ndarray, candidates: pd.DataFrame, user_to_idx: dict, item_to_idx: dict) -> np.ndarray:
    out = np.full(len(candidates), np.nan, dtype=np.float32)
    for n, (uid, gid) in enumerate(candidates[["userID", "gameID"]].astype(str).itertuples(index=False)):
        ui = user_to_idx.get(uid)
        ii = item_to_idx.get(gid)
        if ui is not None and ii is not None:
            out[n] = score_matrix[ui, ii]
    if np.isnan(out).any():
        finite = out[np.isfinite(out)]
        fill = float(finite.min() - 1.0) if finite.size else -1.0
        out = np.nan_to_num(out, nan=fill)
    return out


def zscore_within_user(df: pd.DataFrame, col: str) -> np.ndarray:
    g = df.groupby("userID", sort=False)[col]
    mu = g.transform("mean")
    sd = g.transform("std").replace(0, 1.0).fillna(1.0)
    return ((df[col] - mu) / sd).to_numpy(dtype=float)


def load_emb128_reference(split_name: str) -> pd.DataFrame | None:
    """Best available emb128 reference scores for decorrelation/blend diagnostics."""
    candidates: list[Path] = []
    if split_name == "val_random_uniform_seed42":
        # Original emb128 L4 reg1e-3 seed42 sweep score.  This is not the 4-seed
        # ensemble but is useful for same-split correlation diagnostics.
        candidates.append(ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / split_name / "lightgcn_scores.csv")
        for seed in (42, 123, 2024, 7):
            candidates.append(ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}" / split_name / "lightgcn_scores.csv")
    else:
        for seed in (42, 123, 2024, 7):
            candidates.append(ROOT / "artifacts/split_panel_emb128" / split_name / f"seed{seed}" / "lightgcn_scores.csv")
    parts = []
    for p in candidates:
        if p.exists():
            d = pd.read_csv(p)
            score_col = "score_lightgcn" if "score_lightgcn" in d.columns else "score"
            parts.append(d[["ID", score_col]].rename(columns={score_col: f"score_ref_{len(parts)}"}))
    if not parts:
        return None
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on="ID", how="inner", validate="one_to_one")
    score_cols = [c for c in out.columns if c.startswith("score_ref_")]
    out["score_emb128_ref"] = out[score_cols].mean(axis=1)
    return out[["ID", "score_emb128_ref"]]


def run_one_split(split_dir: Path, ks: list[int], gammas: list[float], out_dir: Path, max_iter: int) -> list[dict[str, object]]:
    split_name = split_dir.name
    train_df = load_train_interactions(split_dir / "train_interactions.csv")
    cand = load_pairs_csv(split_dir / "candidates.csv")
    # Preserve labels for validation evaluation.
    if "Label" in pd.read_csv(split_dir / "candidates.csv", nrows=1).columns:
        cand = pd.read_csv(split_dir / "candidates.csv")

    R, user_to_idx, item_to_idx, _, _ = build_user_item_matrix(train_df, binary=True)
    R = R.astype(np.float64).tocsr()
    du = np.asarray(R.sum(axis=1)).ravel()
    di = np.asarray(R.sum(axis=0)).ravel()
    du_inv_sqrt = np.divide(1.0, np.sqrt(du), out=np.zeros_like(du, dtype=float), where=du > 0)
    di_inv_sqrt = np.divide(1.0, np.sqrt(di), out=np.zeros_like(di, dtype=float), where=di > 0)
    di_sqrt = np.sqrt(np.maximum(di, 0.0))

    Rt = sp.diags(du_inv_sqrt, format="csr") @ R @ sp.diags(di_inv_sqrt, format="csr")
    P = (Rt.T @ Rt).toarray().astype(np.float64)

    max_k = min(max(ks), min(Rt.shape) - 1)
    if max_k < 1:
        raise ValueError(f"max_k too small for {split_name}: {max_k}")
    # svds returns ascending singular values; sort descending.
    _, singular_values, vt = svds(Rt, k=max_k, which="LM", maxiter=max_iter, return_singular_vectors=True)
    order = np.argsort(singular_values)[::-1]
    V = vt[order].T  # items x max_k

    variant_scores: dict[str, np.ndarray] = {}
    S_linear = R @ P
    variant_scores["gfcf_linear_P"] = score_candidates(np.asarray(S_linear), cand, user_to_idx, item_to_idx)

    for k in ks:
        kk = min(k, max_k)
        Vk = V[:, :kk]
        # F_idl = D_i^-1/2 V_k V_k.T D_i^1/2.  Apply with column/row scaling
        # without materializing extra diagonal matrices.
        F = (di_inv_sqrt[:, None] * (Vk @ Vk.T)) * di_sqrt[None, :]
        S_idl = np.asarray(R @ F)
        variant_scores[f"gfcf_idl_k{kk}"] = score_candidates(S_idl, cand, user_to_idx, item_to_idx)
        for gamma in gammas:
            S = np.asarray(R @ (P + gamma * F))
            label = f"gfcf_P_plus_g{gamma:g}_idl_k{kk}"
            variant_scores[label] = score_candidates(S, cand, user_to_idx, item_to_idx)

    ref = load_emb128_reference(split_name)
    rows = []
    score_frame = cand[["ID", "userID", "gameID", "Label"]].copy()
    for label, scores in variant_scores.items():
        score_frame[label] = scores
        summary, _ = evaluate_tophalf(score_frame[["ID", "userID", "gameID", "Label", label]].copy(), label, label_col="Label", user_col="userID", id_col="ID")
        row = {
            "split": split_name,
            "variant": label,
            "row_accuracy": float(summary["row_accuracy"]),
            "per_user_mean_accuracy": float(summary["per_user_mean_accuracy"]),
        }
        if ref is not None:
            m = cand[["ID", "userID", "gameID", "Label"]].copy()
            m[label] = scores
            m = m.merge(ref, on="ID", how="inner", validate="one_to_one")
            m["z_gfcf"] = zscore_within_user(m, label)
            m["z_ref"] = zscore_within_user(m, "score_emb128_ref")
            row["corr_z_vs_emb128_ref"] = float(np.corrcoef(m["z_gfcf"], m["z_ref"])[0, 1])
            m["blend50"] = 0.5 * m["z_gfcf"] + 0.5 * m["z_ref"]
            blend_summary, _ = evaluate_tophalf(m[["ID", "userID", "gameID", "Label", "blend50"]].copy(), "blend50", label_col="Label", user_col="userID", id_col="ID")
            row["blend50_row_accuracy"] = float(blend_summary["row_accuracy"])
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("row_accuracy", ascending=False).to_csv(out_dir / f"{split_name}_gfcf_summary.csv", index=False)
    return rows


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def aggregate(rows: list[dict[str, object]], run_root: Path, out_json: Path, out_md: Path) -> None:
    by: dict[str, list[dict[str, object]]] = {}
    for r in rows:
        by.setdefault(str(r["variant"]), []).append(r)
    agg = []
    for variant, rs in by.items():
        if len(rs) < 1:
            continue
        vals = [float(r["row_accuracy"]) for r in rs]
        item = {
            "variant": variant,
            "n_splits": len(rs),
            "mean_acc": float(np.mean(vals)),
            "min_acc": float(np.min(vals)),
            "max_acc": float(np.max(vals)),
            "split_acc": {str(r["split"]): float(r["row_accuracy"]) for r in rs},
        }
        blend_vals = [float(r["blend50_row_accuracy"]) for r in rs if "blend50_row_accuracy" in r]
        corr_vals = [float(r["corr_z_vs_emb128_ref"]) for r in rs if "corr_z_vs_emb128_ref" in r]
        if blend_vals:
            item["mean_blend50_acc"] = float(np.mean(blend_vals))
            item["min_blend50_acc"] = float(np.min(blend_vals))
        if corr_vals:
            item["mean_corr_z_vs_emb128_ref"] = float(np.mean(corr_vals))
        agg.append(item)
    agg.sort(key=lambda x: (x.get("mean_blend50_acc", -1), x["mean_acc"], x["min_acc"]), reverse=True)
    payload = {
        "run_root": str(run_root),
        "safety": {
            "validation_only": True,
            "kaggle_submit_executed": False,
            "full_test_candidate_or_submission_csv_created": False,
            "hidden_private_labels_used": False,
            "external_steam_scraping_used": False,
        },
        "rows": agg,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# GF-CF spectral uniform panel probe",
        "",
        f"- run_root: `{display_path(run_root)}`",
        "- safety: validation-only; no Kaggle submit; no full-test candidate/submission CSV; no external Steam scraping",
        "",
        "| rank | variant | splits | mean acc | min acc | mean blend50 acc | min blend50 acc | mean corr_z |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(agg[:40], 1):
        lines.append(
            f"| {i} | `{r['variant']}` | {r['n_splits']} | {r['mean_acc']:.6f} | {r['min_acc']:.6f} | "
            f"{r.get('mean_blend50_acc', float('nan')):.6f} | {r.get('min_blend50_acc', float('nan')):.6f} | "
            f"{r.get('mean_corr_z_vs_emb128_ref', float('nan')):.4f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="val_random_uniform_seed42,val_random_uniform_seed7,val_random_uniform_seed123")
    ap.add_argument("--ks", default="32,64,128,256,512")
    ap.add_argument("--gammas", default="0.1,0.3,0.7,1.0")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--max-iter", type=int, default=2000)
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    ks = parse_ints(args.ks)
    gammas = parse_floats(args.gammas)
    if args.out_root:
        run_root = Path(args.out_root)
    else:
        run_root = ROOT / "artifacts/gfcf_uniform_panel_probe"
    run_root = ensure_dir(run_root)
    all_rows = []
    for split in splits:
        print(f"[gfcf] {split}", flush=True)
        all_rows.extend(run_one_split(ROOT / "artifacts/validation" / split, ks, gammas, run_root, args.max_iter))
    out_json = ROOT / "reports/gfcf_uniform_panel_probe.json"
    out_md = ROOT / "reports/gfcf_uniform_panel_probe.md"
    if args.out_root:
        stem = Path(args.out_root).name
        out_json = ROOT / "reports" / f"{stem}.json"
        out_md = ROOT / "reports" / f"{stem}.md"
    aggregate(all_rows, run_root, out_json, out_md)
    print(f"[done] {out_md}", flush=True)


if __name__ == "__main__":
    main()
