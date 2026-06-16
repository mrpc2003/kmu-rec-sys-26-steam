#!/usr/bin/env python3
"""Boundary v1 scored cross-fit evaluation on panel20 LightGCN score coverage (NO-SUBMIT).

Consumes validation score artifacts produced by
`scripts/boundary_v1_panel20_lightgcn_score_coverage.py` and evaluates two boundary-only
specialists:

- ridge logistic row scorer
- pairwise logistic utility scorer

It never reads hidden labels, never writes full-test/candidate CSVs, and never submits to
Kaggle.  Outputs are validation-only reports under `reports/`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import write_json  # noqa: E402

PANEL20_ROOT = ROOT / "artifacts/validation_uniform_panel20_20260612T214626KST"
SCORE_ROOT_DEFAULT = ROOT / "artifacts/boundary_v1_panel20_score_coverage"
OUT_DIR_DEFAULT = ROOT / "reports"
MODEL_SEEDS = [42, 123, 2024, 7]
EMB_DIMS = [128, 192]
DIFF_BANDS = [50, 100, 150, 300, 500, 850]
BOUNDARY_BAND = 3
TARGET_GATES = {
    ("top2", 100): 0.82,
    ("top2", 150): 0.72,
    ("top2", 300): 0.65,
    ("top2", 500): 0.604,
    ("top2", 850): 0.582,
    ("top1", 300): 0.85,
    ("top1", 500): 0.74,
    ("top1", 850): 0.74,
}
FEATURE_COLS = [
    "z_emb128",
    "z_emb192",
    "z_rankblend",
    "rank_emb128_norm",
    "rank_emb192_norm",
    "rank_disagreement_norm",
    "score_emb128_seed_std_z",
    "score_emb192_seed_std_z",
    "boundary_side",
    "boundary_distance",
    "user_boundary_margin",
    "log_user_degree",
    "log_item_degree",
    "log_candidate_count",
    "rankblend_rel_to_cutoff",
]


@dataclass
class SplitData:
    split: str
    frame: pd.DataFrame


def clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return None if not math.isfinite(x) else x
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def within_user_z(df: pd.DataFrame, col: str) -> np.ndarray:
    g = df.groupby("userID", sort=False)[col]
    mu = g.transform("mean").to_numpy(dtype=float)
    sd = g.transform(lambda s: float(s.std(ddof=0))).to_numpy(dtype=float)
    x = df[col].to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=float)
    mask = sd > 1e-12
    out[mask] = (x[mask] - mu[mask]) / sd[mask]
    out[~np.isfinite(out)] = 0.0
    return out


def global_z(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    sd = float(arr.std(ddof=0))
    if sd <= 1e-12:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / sd


def rank_high_is_good(df: pd.DataFrame, col: str) -> np.ndarray:
    ranks = np.zeros(len(df), dtype=float)
    values = df[col].to_numpy(dtype=float)
    for _, idx_raw in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx_raw)
        ranks[idx[np.argsort(values[idx], kind="mergesort")]] = np.arange(len(idx), dtype=float)
    return ranks


def top_half_pred(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    v = np.asarray(score, dtype=float)
    for _, idx_raw in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx_raw)
        k = len(idx) // 2
        if k <= 0:
            continue
        order = idx[np.argsort(v[idx], kind="mergesort")[::-1]]
        pred[order[:k]] = 1
    return pred


def score_file(score_root: Path, emb_dim: int, split: str, seed: int) -> Path:
    return score_root / f"emb{emb_dim}" / split / f"seed{seed}" / "lightgcn_scores.csv"


def discover_complete_splits(score_root: Path, splits: list[str]) -> list[str]:
    complete: list[str] = []
    for split in splits:
        ok = True
        for emb_dim in EMB_DIMS:
            for seed in MODEL_SEEDS:
                if not score_file(score_root, emb_dim, split, seed).exists():
                    ok = False
                    break
            if not ok:
                break
        if ok:
            complete.append(split)
    return complete


def load_split(score_root: Path, split: str) -> SplitData:
    merged: pd.DataFrame | None = None
    for emb_dim in EMB_DIMS:
        seed_cols: list[str] = []
        for seed in MODEL_SEEDS:
            path = score_file(score_root, emb_dim, split, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_csv(path)
            need = {"ID", "userID", "gameID", "Label", "score_lightgcn"}
            if not need.issubset(df.columns):
                raise ValueError(f"Missing columns in {path}: {need - set(df.columns)}")
            col = f"score_emb{emb_dim}_seed{seed}"
            part = df[["ID", "userID", "gameID", "Label", "score_lightgcn"]].rename(columns={"score_lightgcn": col})
            if merged is None:
                merged = part
            else:
                before = len(merged)
                check = merged[["ID", "userID", "gameID", "Label"]].merge(
                    part[["ID", "userID", "gameID", "Label"]],
                    on="ID",
                    how="inner",
                    suffixes=("_base", "_new"),
                    validate="one_to_one",
                )
                if len(check) != before:
                    raise RuntimeError(f"Row alignment changed in {path}")
                for c in ["userID", "gameID", "Label"]:
                    if not (check[f"{c}_base"].astype(str).to_numpy() == check[f"{c}_new"].astype(str).to_numpy()).all():
                        raise RuntimeError(f"Identity mismatch for {split} emb{emb_dim} seed{seed} col={c}")
                merged = merged.merge(part[["ID", col]], on="ID", validate="one_to_one")
            seed_cols.append(col)
        assert merged is not None
        merged[f"score_emb{emb_dim}"] = merged[seed_cols].mean(axis=1)
        merged[f"score_emb{emb_dim}_seed_std"] = merged[seed_cols].std(axis=1, ddof=0)
    assert merged is not None
    merged = merged.sort_values("ID", kind="mergesort").reset_index(drop=True)

    train = pd.read_csv(PANEL20_ROOT / split / "train_interactions.csv", usecols=["userID", "gameID"])
    user_deg = train.groupby("userID").size().astype(float)
    item_deg = train.groupby("gameID").size().astype(float)
    merged["user_degree"] = merged["userID"].map(user_deg).fillna(0.0).astype(float)
    merged["item_degree"] = merged["gameID"].map(item_deg).fillna(0.0).astype(float)

    merged["rank_emb128"] = rank_high_is_good(merged, "score_emb128")
    merged["rank_emb192"] = rank_high_is_good(merged, "score_emb192")
    merged["score_rankblend"] = merged["rank_emb128"] + merged["rank_emb192"]
    merged["z_emb128"] = within_user_z(merged, "score_emb128")
    merged["z_emb192"] = within_user_z(merged, "score_emb192")
    merged["z_rankblend"] = within_user_z(merged, "score_rankblend")
    merged["score_emb128_seed_std_z"] = global_z(merged["score_emb128_seed_std"])
    merged["score_emb192_seed_std_z"] = global_z(merged["score_emb192_seed_std"])

    cand_count = np.zeros(len(merged), dtype=int)
    k_values = np.zeros(len(merged), dtype=int)
    rankblend_pos = np.zeros(len(merged), dtype=int)
    boundary_dist = np.zeros(len(merged), dtype=float)
    user_margin = np.zeros(len(merged), dtype=float)
    anchor = np.zeros(len(merged), dtype=np.int8)
    score = merged["score_rankblend"].to_numpy(dtype=float)
    for _, idx_raw in merged.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx_raw)
        n = len(idx)
        k = n // 2
        order = idx[np.argsort(score[idx], kind="mergesort")[::-1]]
        if k > 0:
            anchor[order[:k]] = 1
        for pos, row_idx in enumerate(order):
            cand_count[row_idx] = n
            k_values[row_idx] = k
            rankblend_pos[row_idx] = pos + 1
            boundary_dist[row_idx] = abs((pos + 1) - (k + 0.5))
        if 0 < k < n:
            user_margin[order] = float(score[order[k - 1]] - score[order[k]])
    merged["candidate_count"] = cand_count
    merged["tophalf_k"] = k_values
    merged["rankblend_position"] = rankblend_pos
    merged["boundary_distance"] = boundary_dist
    merged["user_boundary_margin"] = np.nan_to_num(user_margin, nan=0.0)
    merged["anchor_pred"] = anchor
    merged["boundary_band"] = merged["boundary_distance"] <= float(BOUNDARY_BAND)
    denom = np.maximum(1, merged["candidate_count"].to_numpy(dtype=float) - 1.0)
    merged["rank_emb128_norm"] = merged["rank_emb128"].to_numpy(dtype=float) / denom
    merged["rank_emb192_norm"] = merged["rank_emb192"].to_numpy(dtype=float) / denom
    merged["rank_disagreement_norm"] = (merged["rank_emb192"] - merged["rank_emb128"]).to_numpy(dtype=float) / denom
    merged["boundary_side"] = merged["anchor_pred"].astype(float) * 2.0 - 1.0
    merged["rankblend_rel_to_cutoff"] = (merged["rankblend_position"].to_numpy(dtype=float) - (merged["tophalf_k"].to_numpy(dtype=float) + 0.5)) / np.maximum(1.0, merged["candidate_count"].to_numpy(dtype=float))
    merged["log_user_degree"] = np.log1p(merged["user_degree"].to_numpy(dtype=float))
    merged["log_item_degree"] = np.log1p(merged["item_degree"].to_numpy(dtype=float))
    merged["log_candidate_count"] = np.log1p(merged["candidate_count"].to_numpy(dtype=float))
    for col in FEATURE_COLS:
        merged[col] = np.nan_to_num(merged[col].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return SplitData(split=split, frame=merged)


def row_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float((np.asarray(y, dtype=np.int8) == np.asarray(pred, dtype=np.int8)).mean())


def fit_ridge(train_frames: list[pd.DataFrame]) -> Any:
    train = pd.concat([f[f["boundary_band"]] for f in train_frames], ignore_index=True)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, penalty="l2", solver="lbfgs", max_iter=500, class_weight="balanced"),
    )
    model.fit(train[FEATURE_COLS].to_numpy(dtype=float), train["Label"].to_numpy(dtype=int))
    return model


def fit_pairwise(train_frames: list[pd.DataFrame], max_pairs_per_user: int = 32) -> tuple[np.ndarray, float, list[float], list[float]]:
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    for frame in train_frames:
        b = frame[frame["boundary_band"]]
        for _, g in b.groupby("userID", sort=False):
            pos = g[g["Label"].astype(int) == 1]
            neg = g[g["Label"].astype(int) == 0]
            if pos.empty or neg.empty:
                continue
            pairs = 0
            for _, prow in pos.iterrows():
                pf = prow[FEATURE_COLS].to_numpy(dtype=float)
                for _, nrow in neg.iterrows():
                    nf = nrow[FEATURE_COLS].to_numpy(dtype=float)
                    diff = pf - nf
                    X_rows.append(diff)
                    y_rows.append(1)
                    X_rows.append(-diff)
                    y_rows.append(0)
                    pairs += 1
                    if pairs >= max_pairs_per_user:
                        break
                if pairs >= max_pairs_per_user:
                    break
    if not X_rows:
        raise RuntimeError("No pairwise rows were generated")
    X = np.vstack(X_rows)
    y = np.asarray(y_rows, dtype=int)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd <= 1e-12] = 1.0
    Xz = (X - mu) / sd
    clf = LogisticRegression(C=0.5, penalty="l2", solver="lbfgs", max_iter=500, class_weight="balanced")
    clf.fit(Xz, y)
    coef = clf.coef_[0] / sd
    intercept = float(clf.intercept_[0] - np.dot(coef, mu))
    return coef.astype(float), intercept, mu.tolist(), sd.tolist()


def greedy_diffband_pred(frame: pd.DataFrame, utility: np.ndarray, band: int) -> np.ndarray:
    pred = frame["anchor_pred"].to_numpy(dtype=np.int8).copy()
    max_swaps = band // 2
    if max_swaps <= 0:
        return pred
    proposals: list[tuple[float, int, int, str]] = []
    for uid, idx_raw in frame.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx_raw)
        in_band = idx[frame.loc[idx, "boundary_band"].to_numpy(dtype=bool)]
        selected = [i for i in in_band if pred[i] == 1]
        unselected = [i for i in in_band if pred[i] == 0]
        if not selected or not unselected:
            continue
        # best demotion for each promotion candidate is the selected row with lowest utility.
        selected_sorted = sorted(selected, key=lambda i: (utility[i], i))
        for b in unselected:
            a = selected_sorted[0]
            gain = float(utility[b] - utility[a])
            if gain > 0:
                proposals.append((gain, int(b), int(a), str(uid)))
    proposals.sort(reverse=True, key=lambda x: (x[0], -x[1], -x[2]))
    used_rows: set[int] = set()
    swaps = 0
    for gain, promote, demote, uid in proposals:
        if swaps >= max_swaps:
            break
        if promote in used_rows or demote in used_rows:
            continue
        if pred[promote] != 0 or pred[demote] != 1:
            continue
        pred[promote] = 1
        pred[demote] = 0
        used_rows.add(promote)
        used_rows.add(demote)
        swaps += 1
    return pred


def metrics_for_band(frame: pd.DataFrame, pred: np.ndarray, model_name: str, band: int, split: str) -> dict[str, Any]:
    y = frame["Label"].to_numpy(dtype=np.int8)
    anchor = frame["anchor_pred"].to_numpy(dtype=np.int8)
    ok = pred == y
    base_ok = anchor == y
    changed = pred != anchor
    fixes = int((~base_ok & ok & changed).sum())
    breaks = int((base_ok & ~ok & changed).sum())
    changed_n = int(changed.sum())
    return {
        "split": split,
        "model": model_name,
        "band_total_row_diff_target": int(band),
        "changed_rows": changed_n,
        "swaps": changed_n // 2,
        "fixes": fixes,
        "breaks": breaks,
        "net_gain_rows": fixes - breaks,
        "flip_precision": None if changed_n == 0 else fixes / changed_n,
        "anchor_accuracy": row_accuracy(y, anchor),
        "candidate_accuracy": row_accuracy(y, pred),
        "delta_accuracy": row_accuracy(y, pred) - row_accuracy(y, anchor),
    }


def evaluate_crossfit(splits: list[SplitData]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_base_rows: list[dict[str, Any]] = []
    for heldout_idx, heldout in enumerate(splits):
        train_frames = [s.frame for i, s in enumerate(splits) if i != heldout_idx]
        ridge = fit_ridge(train_frames)
        ridge_prob = ridge.predict_proba(heldout.frame[FEATURE_COLS].to_numpy(dtype=float))[:, 1]
        coef, intercept, _, _ = fit_pairwise(train_frames)
        pair_util = heldout.frame[FEATURE_COLS].to_numpy(dtype=float) @ coef + intercept
        y = heldout.frame["Label"].to_numpy(dtype=np.int8)
        anchor = heldout.frame["anchor_pred"].to_numpy(dtype=np.int8)
        split_base_rows.append(
            {
                "split": heldout.split,
                "rows": int(len(heldout.frame)),
                "users": int(heldout.frame["userID"].nunique()),
                "anchor_accuracy": row_accuracy(y, anchor),
                "anchor_errors": int((anchor != y).sum()),
                "boundary_band_rows": int(heldout.frame["boundary_band"].sum()),
            }
        )
        for model_name, utility in [("ridge_logistic", ridge_prob), ("pairwise_logistic", pair_util)]:
            for band in DIFF_BANDS:
                pred = greedy_diffband_pred(heldout.frame, utility, band)
                rows.append(metrics_for_band(heldout.frame, pred, model_name, band, heldout.split))
    curve = pd.DataFrame(rows)
    base = pd.DataFrame(split_base_rows)
    agg_rows: list[dict[str, Any]] = []
    for (model, band), g in curve.groupby(["model", "band_total_row_diff_target"], sort=True):
        gate_top2 = TARGET_GATES.get(("top2", int(band)))
        gate_top1 = TARGET_GATES.get(("top1", int(band)))
        mean_prec = float(g["flip_precision"].dropna().mean()) if g["flip_precision"].notna().any() else float("nan")
        agg_rows.append(
            {
                "model": model,
                "band_total_row_diff_target": int(band),
                "splits": int(len(g)),
                "mean_changed_rows": float(g["changed_rows"].mean()),
                "mean_flip_precision": mean_prec,
                "mean_net_gain_rows": float(g["net_gain_rows"].mean()),
                "mean_delta_accuracy": float(g["delta_accuracy"].mean()),
                "positive_split_ratio": float((g["net_gain_rows"] > 0).mean()),
                "worst_split_net_gain_rows": int(g["net_gain_rows"].min()),
                "best_split_net_gain_rows": int(g["net_gain_rows"].max()),
                "top2_gate_precision": gate_top2,
                "top2_gate_pass": bool(gate_top2 is not None and mean_prec >= gate_top2 and (g["net_gain_rows"] > 0).mean() >= 0.70 and int(g["net_gain_rows"].min()) >= -5),
                "top1_gate_precision": gate_top1,
                "top1_gate_pass": bool(gate_top1 is not None and mean_prec >= gate_top1 and (g["net_gain_rows"] > 0).mean() >= 0.80 and int(g["net_gain_rows"].min()) >= -5),
            }
        )
    agg = pd.DataFrame(agg_rows).sort_values(["top2_gate_pass", "mean_net_gain_rows", "mean_flip_precision"], ascending=[False, False, False], kind="mergesort")
    return curve, base, {"aggregate": agg}


def write_markdown(path: Path, complete_splits: list[str], base: pd.DataFrame | None, agg: pd.DataFrame | None, readiness: dict[str, Any]) -> None:
    lines = [
        "# boundary v1 scored split20 cross-fit eval",
        "",
        "- validation_only: true",
        "- no_kaggle_submit: true",
        "- candidate_csv_written: false",
        "- full_test_candidate_written: false",
        "",
        "## score coverage 상태",
        "",
        f"- complete scored splits: {len(complete_splits)}",
        f"- required minimum splits: {readiness['min_splits']}",
        f"- scored_boundary_eval_ready: {readiness['ready']}",
        "",
    ]
    if not readiness["ready"]:
        lines.extend([
            "아직 scored cross-fit 평가를 실행하지 않았다. panel20 score coverage가 최소 split 수에 도달해야 한다.",
            "",
            "## missing next",
            "",
            "```text",
            readiness.get("message", "score coverage incomplete"),
            "```",
        ])
    else:
        assert base is not None and agg is not None
        lines.extend([
            "## anchor 요약",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| mean anchor accuracy | {base['anchor_accuracy'].mean():.6f} |",
            f"| mean boundary band rows | {base['boundary_band_rows'].mean():.1f} |",
            "",
            "## aggregate diff-band 결과",
            "",
            "| model | band | mean precision | mean net rows | positive split ratio | worst split | top2 pass | top1 pass |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ])
        for _, r in agg.iterrows():
            lines.append(
                f"| {r['model']} | {int(r['band_total_row_diff_target'])} | {float(r['mean_flip_precision']):.3f} | "
                f"{float(r['mean_net_gain_rows']):.2f} | {float(r['positive_split_ratio']):.2f} | {int(r['worst_split_net_gain_rows'])} | "
                f"{bool(r['top2_gate_pass'])} | {bool(r['top1_gate_pass'])} |"
            )
        lines.extend([
            "",
            "## 판정",
            "",
            "top2/top1 pass가 모두 false면 full-test candidate를 만들지 않는다.",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-root", default=str(SCORE_ROOT_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--min-splits", type=int, default=20)
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    score_root = Path(args.score_root)
    out_dir = Path(args.out_dir)
    all_splits = sorted(p.name for p in PANEL20_ROOT.iterdir() if p.is_dir() and p.name.startswith("val_random_uniform_seed"))
    complete_splits = discover_complete_splits(score_root, all_splits)
    ready = len(complete_splits) >= args.min_splits or (args.allow_partial and len(complete_splits) >= 2)
    readiness = {
        "validation_only": True,
        "no_kaggle_submit": True,
        "candidate_csv_written": False,
        "score_root": str(score_root),
        "complete_splits": complete_splits,
        "complete_split_count": len(complete_splits),
        "required_all_splits": all_splits,
        "min_splits": args.min_splits,
        "allow_partial": args.allow_partial,
        "ready": ready,
        "message": "Need complete emb128+emb192 4-seed score files for more splits.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "boundary_v1_scored_split20_readiness.json", readiness)
    if not ready:
        write_markdown(out_dir / "boundary_v1_scored_split20_crossfit_eval.md", complete_splits, None, None, readiness)
        print(json.dumps(clean({"ready": False, **readiness}), indent=2, ensure_ascii=False))
        return
    splits = [load_split(score_root, s) for s in complete_splits]
    curve, base, bundle = evaluate_crossfit(splits)
    agg = bundle["aggregate"]
    curve_path = out_dir / "boundary_v1_diffband_precision_curve_scored.csv"
    base_path = out_dir / "boundary_v1_scored_split_base_metrics.csv"
    agg_path = out_dir / "boundary_v1_diffband_precision_curve_scored_aggregate.csv"
    curve.to_csv(curve_path, index=False)
    base.to_csv(base_path, index=False)
    agg.to_csv(agg_path, index=False)
    payload = {
        **readiness,
        "ready": True,
        "curve_csv": str(curve_path),
        "base_metrics_csv": str(base_path),
        "aggregate_csv": str(agg_path),
        "top2_any_pass": bool(agg["top2_gate_pass"].any()),
        "top1_any_pass": bool(agg["top1_gate_pass"].any()),
        "submission_readiness": "PASS_REVIEW_REQUIRED" if bool(agg["top2_gate_pass"].any()) else "FAIL__NO_SCORED_GATE_PASS",
    }
    write_json(out_dir / "boundary_v1_scored_split20_crossfit_eval.json", payload)
    write_markdown(out_dir / "boundary_v1_scored_split20_crossfit_eval.md", complete_splits, base, agg, payload)
    print(json.dumps(clean(payload), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
