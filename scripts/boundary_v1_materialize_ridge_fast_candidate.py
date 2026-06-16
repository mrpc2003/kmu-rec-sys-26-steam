#!/usr/bin/env python3
"""Materialize a forced/manual-risk boundary v1 ridge-fast candidate.

This uses only train-derived validation labels from the panel20 score coverage to train a
ridge row-utility model, applies it to full-test emb128/emb192 LightGCN score features,
and writes an `ID,Played` candidate.  The script does NOT submit to Kaggle.

Use only after an explicit manual-risk user request because the scored panel20 gate did
not pass (`FAIL__NO_SCORED_GATE_PASS`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from boundary_v1_scored_crossfit_eval import (  # noqa: E402
    FEATURE_COLS,
    SCORE_ROOT_DEFAULT,
    clean,
    discover_complete_splits,
    global_z,
    greedy_diffband_pred,
    load_split,
    rank_high_is_good,
    within_user_z,
)
from recsys_played_utils import load_train_interactions, write_json  # noqa: E402

PAIRS = ROOT / "data/raw/public/data/pairs.csv"
TRAIN_JSON = ROOT / "data/raw/public/data/train.json"
PANEL20_ROOT = ROOT / "artifacts/validation_uniform_panel20_20260612T214626KST"
CURRENT_BEST = ROOT / "submissions/candidate_rank_blend_emb128_emb192.csv"
OUT_DEFAULT = ROOT / "submissions/candidate_boundary_v1_ridge_fast_panel20_forced.csv"
REPORT_DEFAULT = ROOT / "reports/boundary_v1_ridge_fast_panel20_forced_candidate.json"
MODEL_SEEDS = [42, 123, 2024, 7]
EMB_DIMS = [128, 192]
BOUNDARY_BAND = 3
N_EXPECTED = 19998


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_col(df: pd.DataFrame) -> str:
    for col in ("score_lightgcn", "score"):
        if col in df.columns:
            return col
    raise ValueError(f"No score column in {df.columns.tolist()}")


def fulltest_score_file(emb_dim: int, seed: int) -> Path:
    return ROOT / f"artifacts/lightgcn_emb{emb_dim}L4r3_fulltest/seed{seed}/test.csv"


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


def build_fulltest_frame() -> pd.DataFrame:
    pairs = pd.read_csv(PAIRS)
    if len(pairs) != N_EXPECTED or list(pairs.columns) != ["ID", "userID", "gameID"]:
        raise ValueError(f"Unexpected pairs shape/columns: {len(pairs)} {pairs.columns.tolist()}")
    df = pairs.copy()
    for emb_dim in EMB_DIMS:
        seed_cols: list[str] = []
        for seed in MODEL_SEEDS:
            path = fulltest_score_file(emb_dim, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            part = pd.read_csv(path)
            sc = score_col(part)
            col = f"score_emb{emb_dim}_seed{seed}"
            part = part[["ID", sc]].rename(columns={sc: col})
            before = len(df)
            df = df.merge(part, on="ID", validate="one_to_one")
            if len(df) != before:
                raise RuntimeError(f"Row merge changed for {path}")
            seed_cols.append(col)
        df[f"score_emb{emb_dim}"] = df[seed_cols].mean(axis=1)
        df[f"score_emb{emb_dim}_seed_std"] = df[seed_cols].std(axis=1, ddof=0)
    train = load_train_interactions(TRAIN_JSON)
    user_deg = train.groupby("userID").size().astype(float)
    item_deg = train.groupby("gameID").size().astype(float)
    df["user_degree"] = df["userID"].map(user_deg).fillna(0.0).astype(float)
    df["item_degree"] = df["gameID"].map(item_deg).fillna(0.0).astype(float)

    df["rank_emb128"] = rank_high_is_good(df, "score_emb128")
    df["rank_emb192"] = rank_high_is_good(df, "score_emb192")
    df["score_rankblend"] = df["rank_emb128"] + df["rank_emb192"]
    df["z_emb128"] = within_user_z(df, "score_emb128")
    df["z_emb192"] = within_user_z(df, "score_emb192")
    df["z_rankblend"] = within_user_z(df, "score_rankblend")
    df["score_emb128_seed_std_z"] = global_z(df["score_emb128_seed_std"])
    df["score_emb192_seed_std_z"] = global_z(df["score_emb192_seed_std"])

    cand_count = np.zeros(len(df), dtype=int)
    k_values = np.zeros(len(df), dtype=int)
    rankblend_pos = np.zeros(len(df), dtype=int)
    boundary_dist = np.zeros(len(df), dtype=float)
    user_margin = np.zeros(len(df), dtype=float)
    anchor = np.zeros(len(df), dtype=np.int8)
    score = df["score_rankblend"].to_numpy(dtype=float)
    for _, idx_raw in df.groupby("userID", sort=False).indices.items():
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
    df["candidate_count"] = cand_count
    df["tophalf_k"] = k_values
    df["rankblend_position"] = rankblend_pos
    df["boundary_distance"] = boundary_dist
    df["user_boundary_margin"] = np.nan_to_num(user_margin, nan=0.0)
    df["anchor_pred"] = anchor
    df["boundary_band"] = df["boundary_distance"] <= float(BOUNDARY_BAND)
    denom = np.maximum(1, df["candidate_count"].to_numpy(dtype=float) - 1.0)
    df["rank_emb128_norm"] = df["rank_emb128"].to_numpy(dtype=float) / denom
    df["rank_emb192_norm"] = df["rank_emb192"].to_numpy(dtype=float) / denom
    df["rank_disagreement_norm"] = (df["rank_emb192"] - df["rank_emb128"]).to_numpy(dtype=float) / denom
    df["boundary_side"] = df["anchor_pred"].astype(float) * 2.0 - 1.0
    df["rankblend_rel_to_cutoff"] = (df["rankblend_position"].to_numpy(dtype=float) - (df["tophalf_k"].to_numpy(dtype=float) + 0.5)) / np.maximum(1.0, df["candidate_count"].to_numpy(dtype=float))
    df["log_user_degree"] = np.log1p(df["user_degree"].to_numpy(dtype=float))
    df["log_item_degree"] = np.log1p(df["item_degree"].to_numpy(dtype=float))
    df["log_candidate_count"] = np.log1p(df["candidate_count"].to_numpy(dtype=float))
    for col in FEATURE_COLS:
        df[col] = np.nan_to_num(df[col].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return df


def fit_final_model(score_root: Path, alpha: float) -> Any:
    all_splits = sorted(p.name for p in PANEL20_ROOT.iterdir() if p.is_dir() and p.name.startswith("val_random_uniform_seed"))
    complete = discover_complete_splits(score_root, all_splits)
    if len(complete) != 20:
        raise RuntimeError(f"Need 20 complete splits, got {len(complete)}")
    frames = [load_split(score_root, split).frame for split in complete]
    train = pd.concat([f[f["boundary_band"]] for f in frames], ignore_index=True)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=0))
    model.fit(train[FEATURE_COLS].to_numpy(dtype=float), train["Label"].to_numpy(dtype=float))
    return model


def read_pred(path: Path, name: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    col = "Played" if "Played" in d.columns else "Label"
    return d[["ID", col]].rename(columns={col: name})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-root", default=str(SCORE_ROOT_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--report", default=str(REPORT_DEFAULT))
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--band", type=int, default=300)
    args = ap.parse_args()

    model = fit_final_model(Path(args.score_root), alpha=args.alpha)
    frame = build_fulltest_frame()
    utility = model.predict(frame[FEATURE_COLS].to_numpy(dtype=float))
    pred = greedy_diffband_pred(frame, utility, int(args.band)).astype(int)
    out = frame[["ID"]].copy()
    out["Played"] = pred
    out = out.sort_values("ID", kind="mergesort").reset_index(drop=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    sha = sha256_file(out_path)

    per_user = frame[["ID", "userID"]].merge(out, on="ID", validate="one_to_one").groupby("userID")["Played"].agg(["sum", "count"])
    bad_users = int(((per_user["count"] % 2 != 0) | (per_user["sum"] != per_user["count"] // 2)).sum())
    anchor_diff = int((out["Played"].to_numpy(dtype=int) != frame["anchor_pred"].astype(int).to_numpy()).sum())
    comparisons: dict[str, Any] = {}
    for label, path in {
        "current_best_rank_blend": CURRENT_BEST,
    }.items():
        if path.exists():
            m = out.merge(read_pred(path, label), on="ID", validate="one_to_one")
            diff = int((m["Played"].astype(int) != m[label].astype(int)).sum())
            comparisons[label] = {"path": str(path), "row_diff": diff, "row_diff_frac": diff / len(out)}
    payload = {
        "candidate": "boundary_v1_ridge_fast_panel20_forced",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "FORCED manual-risk boundary v1 ridge-fast candidate; scored gate failed, user explicitly requested submission attempt.",
        "validation_gate": "FAIL__NO_SCORED_GATE_PASS",
        "source_eval": "reports/boundary_v1_scored_split20_fast_eval.md",
        "file": str(out_path),
        "sha256": sha,
        "alpha": args.alpha,
        "band": int(args.band),
        "preflight": {
            "rows": int(len(out)),
            "expected_rows": N_EXPECTED,
            "columns": out.columns.tolist(),
            "id_unique": bool(out["ID"].is_unique),
            "id_contiguous": bool((out["ID"].to_numpy() == np.arange(len(out))).all()),
            "labels_binary": bool(set(out["Played"].astype(int).unique()).issubset({0, 1})),
            "label_1": int(out["Played"].sum()),
            "label_0": int(len(out) - out["Played"].sum()),
            "bad_users_tophalf": bad_users,
            "row_diff_vs_anchor_rankblend": anchor_diff,
        },
        "comparisons": comparisons,
        "safety": {
            "kaggle_submit_executed_by_script": False,
            "external_metadata_used": False,
            "hidden_labels_used": False,
            "public_lb_feedback_used_in_policy_gate": True,
        },
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, clean(payload))
    print(json.dumps(clean(payload), indent=2, ensure_ascii=False), flush=True)
    pf = payload["preflight"]
    ok = (
        pf["rows"] == N_EXPECTED
        and pf["id_unique"]
        and pf["id_contiguous"]
        and pf["labels_binary"]
        and pf["label_1"] == pf["label_0"] == N_EXPECTED // 2
        and pf["bad_users_tophalf"] == 0
        and pf["row_diff_vs_anchor_rankblend"] > 0
    )
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
