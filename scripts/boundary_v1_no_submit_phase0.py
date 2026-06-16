#!/usr/bin/env python3
"""Boundary specialist v1 phase-0 calibration artifacts (NO-SUBMIT).

This script does not train a new model and does not write a Kaggle candidate CSV.
It creates the three planning/calibration artifacts requested for
`boundary_specialist_v1_rowflip_constrained`:

1. reports/boundary_public_failure_calibration.csv
2. reports/boundary_public_like_split_panel.json
3. reports/boundary_v1_diffband_precision_curve.csv

The goal is to make the public-negative-transfer history and row-budget math explicit
before any boundary-only specialist training or full-test materialization.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS_PATH = ROOT / "data/raw/public/data/pairs.csv"
TRAIN_PATH = ROOT / "data/raw/public/data/train.json"
CURRENT_BEST_PATH = ROOT / "submissions/candidate_rank_blend_emb128_emb192.csv"
PUBLIC_DENOM_EST = 9999
OUR_PUBLIC_BEST = 0.77825
TOP3_SCORE = 0.78085
TOP2_SCORE = 0.78095
TOP1_SCORE = 0.78795

EMB128_FILES = [ROOT / f"artifacts/lightgcn_emb128L4r3_fulltest/seed{s}/test.csv" for s in (42, 123, 2024, 7)]
EMB192_FILES = [ROOT / f"artifacts/lightgcn_emb192L4r3_fulltest/seed{s}/test.csv" for s in (42, 123, 2024, 7)]
SPLIT_PANEL20_SUMMARY = ROOT / "artifacts/validation_uniform_panel20_20260612T214626KST/validation_splits_summary.json"

# Public-failed / calibration candidates.  Public scores were verified with Kaggle CLI
# on 2026-06-14 KST.  validation_* fields are copied from local reports/Kaggle descriptions.
# No public/private labels are used.
CALIBRATION_CANDIDATES: list[dict[str, Any]] = [
    {
        "candidate_id": "boundary_scoreblend_z128_z192_z64_w-0.75",
        "family": "boundary_scoreblend",
        "path": "submissions/candidate_autorun_boundary_scoreblend_z128_z192_z64_w-0.75.csv",
        "public_score": 0.77755,
        "validation_delta": 0.000633,
        "validation_fixes": 841,
        "validation_breaks": 803,
        "source_note": "Kaggle submissions description; public negative-transfer boundary family",
    },
    {
        "candidate_id": "boundary_scoreblend_z128_z192_z64_w2",
        "family": "boundary_scoreblend",
        "path": "submissions/candidate_autorun_boundary_scoreblend_z128_z192_z64_w2.csv",
        "public_score": 0.77575,
        "validation_delta": 0.000433,
        "validation_fixes": 849,
        "validation_breaks": 823,
        "source_note": "Kaggle submissions description; public negative-transfer boundary family",
    },
    {
        "candidate_id": "frontier_z_w1920_w64-0.25",
        "family": "frontier_z_boundary",
        "path": "submissions/candidate_autorun_frontier_z_w1920_w64-0.25.csv",
        "public_score": 0.77715,
        "validation_delta": 0.000267,
        "validation_fixes": 770,
        "validation_breaks": 754,
        "source_note": "Kaggle submissions description; current-best frontier/boundary follow-up",
    },
    {
        "candidate_id": "otto_coplay_top5_reverse_recent_forced",
        "family": "otto_covisitation",
        "path": "submissions/candidate_otto_coplay_top5_reverse_recent_w0090_w0040_forced_20260607T114059KST.csv",
        "public_score": 0.77815,
        "validation_delta": 0.000667,
        "validation_fixes": None,
        "validation_breaks": None,
        "source_note": "Forced manual-risk OTTO; public near-miss but did not beat current best",
    },
    {
        "candidate_id": "als_htr_popa4_w0.025",
        "family": "als_residual_rankblend",
        "path": "submissions/candidate_autorun_rankblend_z_plus_score_als_htr_f32_it30_alpha20_popa4_w0.025.csv",
        "public_score": 0.77805,
        "validation_delta": 0.001300,
        "validation_fixes": None,
        "validation_breaks": None,
        "source_note": "README/ALS_htr residual public negative-transfer cluster",
    },
    {
        "candidate_id": "als_htr_popa4_w0.05",
        "family": "als_residual_rankblend",
        "path": "submissions/candidate_autorun_rankblend_z_plus_score_als_htr_f32_it30_alpha20_popa4_w0.05.csv",
        "public_score": 0.77805,
        "validation_delta": 0.001300,
        "validation_fixes": None,
        "validation_breaks": None,
        "source_note": "README/ALS_htr residual public negative-transfer cluster",
    },
    {
        "candidate_id": "als_htr_popa4_w0.1",
        "family": "als_residual_rankblend",
        "path": "submissions/candidate_autorun_rankblend_z_plus_score_als_htr_f32_it30_alpha20_popa4_w0.1.csv",
        "public_score": 0.77805,
        "validation_delta": 0.001300,
        "validation_fixes": None,
        "validation_breaks": None,
        "source_note": "README/ALS_htr residual public negative-transfer cluster",
    },
    {
        "candidate_id": "als_popa4_w0.2",
        "family": "als_residual_rankblend",
        "path": "submissions/candidate_autorun_rankblend_z_plus_score_als_f32_it30_alpha20_popa4_w0.2.csv",
        "public_score": 0.77795,
        "validation_delta": 0.001400,
        "validation_fixes": None,
        "validation_breaks": None,
        "source_note": "ALS residual public negative-transfer cluster",
    },
    {
        "candidate_id": "als_htr_popa4_w0.2",
        "family": "als_residual_rankblend",
        "path": "submissions/candidate_autorun_rankblend_z_plus_score_als_htr_f32_it30_alpha20_popa4_w0.2.csv",
        "public_score": 0.77785,
        "validation_delta": 0.001434,
        "validation_fixes": None,
        "validation_breaks": None,
        "source_note": "First aggressive ALS_htr residual public negative-transfer candidate",
    },
    {
        "candidate_id": "tagcf_seed2024_sym_a0.1_raw_zblend_bw0.5",
        "family": "tagcf_boundary",
        "path": "submissions/candidate_autorun_tagcf_fulltest_seed2024_sym_a0.1_raw_zblend_bw0.5.csv",
        "public_score": 0.77615,
        "validation_delta": 0.000767,
        "validation_fixes": None,
        "validation_breaks": None,
        "source_note": "TAGCF/boundary public negative-transfer family",
    },
    {
        "candidate_id": "emb128_emb64_zblend",
        "family": "capacity_blend",
        "path": "submissions/candidate_emb128_emb64_zblend.csv",
        "public_score": 0.77815,
        "validation_delta": None,
        "validation_fixes": None,
        "validation_breaks": None,
        "source_note": "Capacity/z-blend near-miss, useful for overlap calibration",
    },
]


def clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return None if not math.isfinite(x) else x
    return v


def label_col(df: pd.DataFrame) -> str:
    for c in ("Played", "Label"):
        if c in df.columns:
            return c
    raise ValueError(f"No Played/Label column in {df.columns.tolist()}")


def read_labels(path: Path, out_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    c = label_col(df)
    if "ID" not in df.columns:
        raise ValueError(f"Missing ID in {path}")
    return df[["ID", c]].rename(columns={c: out_col}).assign(**{out_col: lambda d: d[out_col].astype(int)})


def score_col(df: pd.DataFrame) -> str:
    for c in ("score_lightgcn", "score", "score_layermix_uniform"):
        if c in df.columns:
            return c
    raise ValueError(f"No score column in {df.columns.tolist()}")


def ensemble(files: list[Path], name: str) -> pd.DataFrame:
    out: pd.DataFrame | None = None
    for i, path in enumerate(files):
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        sc = score_col(df)
        need = {"ID", "userID", "gameID", sc}
        if not need.issubset(df.columns):
            raise ValueError(f"Missing columns in {path}: {need - set(df.columns)}")
        part = df[["ID", "userID", "gameID", sc]].rename(columns={sc: f"{name}_seed{i}"})
        if out is None:
            out = part
        else:
            before = len(out)
            out = out.merge(part[["ID", f"{name}_seed{i}"]], on="ID", validate="one_to_one")
            if len(out) != before:
                raise RuntimeError(f"Row count changed merging {path}")
    assert out is not None
    seed_cols = [c for c in out.columns if c.startswith(f"{name}_seed")]
    out[name] = out[seed_cols].mean(axis=1)
    out[f"{name}_std"] = out[seed_cols].std(axis=1, ddof=0)
    return out[["ID", "userID", "gameID", name, f"{name}_std"]]


def user_rank_high_is_good(df: pd.DataFrame, col: str) -> np.ndarray:
    ranks = np.zeros(len(df), dtype=np.float64)
    values = df[col].to_numpy(dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        ranks[idx[np.argsort(values[idx], kind="mergesort")]] = np.arange(len(idx), dtype=np.float64)
    return ranks


def load_pairs_with_boundary_features() -> pd.DataFrame:
    pairs = pd.read_csv(PAIRS_PATH)
    if list(pairs.columns) != ["ID", "userID", "gameID"]:
        raise ValueError(f"Unexpected pairs columns: {pairs.columns.tolist()}")
    emb128 = ensemble(EMB128_FILES, "score_emb128")
    emb192 = ensemble(EMB192_FILES, "score_emb192")[["ID", "score_emb192", "score_emb192_std"]]
    df = pairs.merge(emb128, on=["ID", "userID", "gameID"], validate="one_to_one").merge(
        emb192, on="ID", validate="one_to_one"
    )
    df["rank_emb128"] = user_rank_high_is_good(df, "score_emb128")
    df["rank_emb192"] = user_rank_high_is_good(df, "score_emb192")
    df["score_rank_blend"] = df["rank_emb128"] + df["rank_emb192"]
    pos_rank = np.zeros(len(df), dtype=int)
    cand_count = np.zeros(len(df), dtype=int)
    k_values = np.zeros(len(df), dtype=int)
    user_margin = np.zeros(len(df), dtype=float)
    boundary_distance = np.zeros(len(df), dtype=float)
    for _, idx_raw in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx_raw)
        values = df.loc[idx, "score_rank_blend"].to_numpy(dtype=float)
        order_local = np.argsort(values, kind="mergesort")[::-1]
        ordered_idx = idx[order_local]
        n = len(idx)
        k = n // 2
        for r, global_idx in enumerate(ordered_idx):
            pos_rank[global_idx] = r + 1
            cand_count[global_idx] = n
            k_values[global_idx] = k
            boundary_distance[global_idx] = abs((r + 1) - (k + 0.5))
        if 0 < k < n:
            user_margin[ordered_idx] = float(values[order_local[k - 1]] - values[order_local[k]])
        else:
            user_margin[ordered_idx] = float("nan")
    df["rankblend_position"] = pos_rank
    df["candidate_count"] = cand_count
    df["tophalf_k"] = k_values
    df["boundary_distance"] = boundary_distance
    df["user_boundary_margin"] = user_margin
    df["boundary_band_le_1"] = df["boundary_distance"] <= 1.0
    df["boundary_band_le_3"] = df["boundary_distance"] <= 3.0
    df["boundary_band_le_5"] = df["boundary_distance"] <= 5.0
    return df


def load_train_degrees() -> tuple[pd.Series, pd.Series]:
    rows: list[dict[str, str]] = []
    with TRAIN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = ast.literal_eval(line)
            rows.append({"userID": str(d["userID"]), "gameID": str(d["gameID"])})
    train = pd.DataFrame(rows)
    return train.groupby("userID").size().astype(int), train.groupby("gameID").size().astype(int)


def decile_codes(values: pd.Series) -> pd.Series:
    # Rank first to avoid qcut failures under ties.
    ranks = values.rank(method="first")
    return pd.qcut(ranks, q=10, labels=False, duplicates="drop").astype("Int64")


def summarize_counts(s: pd.Series) -> str:
    counts = s.value_counts(dropna=False).sort_index()
    return json.dumps({str(k): int(v) for k, v in counts.items()}, ensure_ascii=False, sort_keys=True)


def bucket_candidate_count(x: int) -> str:
    if x <= 2:
        return "02"
    if x <= 4:
        return "03-04"
    if x <= 6:
        return "05-06"
    if x <= 10:
        return "07-10"
    return "11+"


def create_failure_calibration(out_csv: Path, out_pairs_meta: pd.DataFrame) -> list[dict[str, Any]]:
    current = read_labels(CURRENT_BEST_PATH, "current")
    base = out_pairs_meta.merge(current, on="ID", validate="one_to_one")

    # Sets used to quantify overlap with previously failed boundary families.
    boundary_failed_ids: set[int] = set()
    for c in CALIBRATION_CANDIDATES:
        if c["family"] in {"boundary_scoreblend", "frontier_z_boundary", "tagcf_boundary"}:
            p = ROOT / c["path"]
            if p.exists():
                cand = read_labels(p, "cand")
                m = current.merge(cand, on="ID", validate="one_to_one")
                boundary_failed_ids.update(m.loc[m["current"] != m["cand"], "ID"].astype(int).tolist())

    rows: list[dict[str, Any]] = []
    for c in CALIBRATION_CANDIDATES:
        path = ROOT / c["path"]
        if not path.exists():
            rows.append({**c, "exists": False})
            continue
        cand = read_labels(path, "candidate")
        m = base.merge(cand, on="ID", validate="one_to_one")
        changed = m[m["current"] != m["candidate"]].copy()
        row_diff = int(len(changed))
        val_fixes = c.get("validation_fixes")
        val_breaks = c.get("validation_breaks")
        val_net = None if val_fixes is None or val_breaks is None else int(val_fixes) - int(val_breaks)
        val_precision = None if val_fixes is None or val_breaks is None else float(val_fixes) / float(int(val_fixes) + int(val_breaks))
        public_delta = float(c["public_score"] - OUR_PUBLIC_BEST)
        public_net_rows_est = public_delta * PUBLIC_DENOM_EST
        calibration_factor = None
        if val_net not in (None, 0):
            calibration_factor = public_net_rows_est / float(val_net)
        overlap = int(len(set(changed["ID"].astype(int).tolist()) & boundary_failed_ids))
        row = {
            "candidate_id": c["candidate_id"],
            "family": c["family"],
            "file": c["path"],
            "exists": True,
            "public_score": c["public_score"],
            "public_lb_feedback_used": True,
            "public_denominator_est": PUBLIC_DENOM_EST,
            "public_delta_vs_current_best": public_delta,
            "public_net_rows_est_at_den9999": public_net_rows_est,
            "validation_delta": c.get("validation_delta"),
            "validation_fixes": val_fixes,
            "validation_breaks": val_breaks,
            "validation_net_fixes_minus_breaks": val_net,
            "validation_flip_precision": val_precision,
            "validation_to_public_net_calibration_factor": calibration_factor,
            "row_diff_vs_current_best_fulltest": row_diff,
            "expected_public_changed_if_half": row_diff / 2.0,
            "changed_pct_boundary_le1": None if row_diff == 0 else float(changed["boundary_band_le_1"].mean()),
            "changed_pct_boundary_le3": None if row_diff == 0 else float(changed["boundary_band_le_3"].mean()),
            "changed_pct_boundary_le5": None if row_diff == 0 else float(changed["boundary_band_le_5"].mean()),
            "changed_mean_boundary_distance": None if row_diff == 0 else float(changed["boundary_distance"].mean()),
            "changed_mean_user_boundary_margin": None if row_diff == 0 else float(changed["user_boundary_margin"].mean()),
            "changed_mean_candidate_count": None if row_diff == 0 else float(changed["candidate_count"].mean()),
            "changed_candidate_count_bucket_counts": summarize_counts(changed["candidate_count_bucket"]) if row_diff else "{}",
            "changed_user_degree_decile_counts": summarize_counts(changed["user_degree_decile"]) if row_diff else "{}",
            "changed_item_degree_decile_counts": summarize_counts(changed["item_degree_decile"]) if row_diff else "{}",
            "overlap_with_boundary_failed_union_rows": overlap,
            "overlap_frac_of_changed_with_boundary_failed_union": None if row_diff == 0 else overlap / row_diff,
            "source_note": c["source_note"],
        }
        rows.append(row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clean(rows)).to_csv(out_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    return rows


def percentile_summary(values: list[float] | pd.Series) -> dict[str, float | int | None]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {k: None for k in ["count", "mean", "std", "min", "p25", "p50", "p75", "p90", "p95", "p99", "max"]}
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def create_split_panel(out_json: Path, pairs_meta: pd.DataFrame) -> dict[str, Any]:
    if not SPLIT_PANEL20_SUMMARY.exists():
        raise FileNotFoundError(SPLIT_PANEL20_SUMMARY)
    splits = json.loads(SPLIT_PANEL20_SUMMARY.read_text(encoding="utf-8"))
    candidate_rows = [int(s["candidate_rows"]) for s in splits]
    candidate_users = [int(s["candidate_users"]) for s in splits]
    heldout_pos = [int(s["heldout_positive_rows"]) for s in splits]
    skipped = [int(s["skipped_users"]) for s in splits]
    adjusted = [int(s["adjusted_users"]) for s in splits]
    test_user_counts = pairs_meta.groupby("userID").size().astype(int)
    item_exposure = pairs_meta.groupby("gameID").size().astype(int)
    payload = {
        "artifact": "boundary_public_like_split_panel",
        "created_by": "scripts/boundary_v1_no_submit_phase0.py",
        "status": "existing_metadata_panel20_available__score_coverage_not_expanded_to_30_50_yet",
        "validation_only": True,
        "no_kaggle_submit": True,
        "public_lb_feedback_used": True,
        "candidate_csv_written": False,
        "new_full_test_scoring_performed": False,
        "existing_full_test_score_artifacts_read": True,
        "external_metadata_used": False,
        "requested_future_panel_size": "30-50 public-like splits",
        "available_panel": {
            "path": str(SPLIT_PANEL20_SUMMARY.relative_to(ROOT)),
            "split_count": len(splits),
            "split_names": [s["name"] for s in splits],
            "holdout_modes": sorted({s["holdout"] for s in splits}),
            "negative_modes": sorted({s["negative"] for s in splits}),
            "candidate_rows_summary": percentile_summary(candidate_rows),
            "candidate_users_summary": percentile_summary(candidate_users),
            "heldout_positive_rows_summary": percentile_summary(heldout_pos),
            "skipped_users_summary": percentile_summary(skipped),
            "adjusted_users_summary": percentile_summary(adjusted),
            "all_candidate_counts_even": bool(all(bool(s.get("all_candidate_counts_even", False)) for s in splits)),
            "max_overlap_with_fold_train": int(max(int(s.get("overlap_with_fold_train", 0)) for s in splits)),
            "max_missing_user_rows_vs_fold_train": int(max(int(s.get("missing_user_rows_vs_fold_train", 0)) for s in splits)),
            "max_missing_item_rows_vs_fold_train": int(max(int(s.get("missing_item_rows_vs_fold_train", 0)) for s in splits)),
        },
        "test_pairs_reference": {
            "path": str(PAIRS_PATH.relative_to(ROOT)),
            "rows": int(len(pairs_meta)),
            "users": int(pairs_meta["userID"].nunique()),
            "items": int(pairs_meta["gameID"].nunique()),
            "per_user_candidate_count": percentile_summary(test_user_counts),
            "item_exposure_in_pairs": percentile_summary(item_exposure),
        },
        "intended_use": [
            "risk measurement panel, not score boasting",
            "full accuracy delta + boundary-only delta + flip precision + bucket collapse checks",
            "do not use repeated split tweaking as public-score tuning",
        ],
        "next_gap": "To train/evaluate boundary specialist on 30-50 splits, LightGCN/current-best proxy scores must be generated for additional splits; this script only records existing split metadata.",
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def required_precision(target_net_gain: float, public_changed: float) -> float | None:
    if public_changed <= 0 or target_net_gain > public_changed:
        return None
    return (1.0 + target_net_gain / public_changed) / 2.0


def create_diffband_curve(out_csv: Path, calibration_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bands = [50, 100, 150, 300, 500, 850]
    targets = {
        "top3_plus26": (TOP3_SCORE - OUR_PUBLIC_BEST) * PUBLIC_DENOM_EST,
        "top2_plus27": (TOP2_SCORE - OUR_PUBLIC_BEST) * PUBLIC_DENOM_EST,
        "top1_plus97": (TOP1_SCORE - OUR_PUBLIC_BEST) * PUBLIC_DENOM_EST,
    }
    rows: list[dict[str, Any]] = []
    for band in bands:
        public_changed = band / 2.0
        nearby = [r for r in calibration_rows if r.get("exists") and abs(float(r["row_diff_vs_current_best_fulltest"]) - band) <= max(75, band * 0.25)]
        public_net_values = [float(r["public_net_rows_est_at_den9999"]) for r in nearby if r.get("public_net_rows_est_at_den9999") is not None]
        val_precisions = [float(r["validation_flip_precision"]) for r in nearby if r.get("validation_flip_precision") is not None]
        row = {
            "public_denominator_est": PUBLIC_DENOM_EST,
            "public_lb_feedback_used": True,
            "band_total_row_diff": band,
            "expected_public_changed_M_if_half": public_changed,
            "required_precision_top3_gap": required_precision(targets["top3_plus26"], public_changed),
            "required_precision_top2_gap": required_precision(targets["top2_plus27"], public_changed),
            "required_precision_top1_gap": required_precision(targets["top1_plus97"], public_changed),
            "target_net_rows_top3": targets["top3_plus26"],
            "target_net_rows_top2": targets["top2_plus27"],
            "target_net_rows_top1": targets["top1_plus97"],
            "legacy_candidates_near_band": ";".join(str(r["candidate_id"]) for r in nearby),
            "legacy_public_net_rows_mean": None if not public_net_values else float(np.mean(public_net_values)),
            "legacy_public_net_rows_min": None if not public_net_values else float(np.min(public_net_values)),
            "legacy_public_net_rows_max": None if not public_net_values else float(np.max(public_net_values)),
            "legacy_validation_flip_precision_mean": None if not val_precisions else float(np.mean(val_precisions)),
            "interpretation": "requirements_only_phase0__no_boundary_specialist_trained_yet",
        }
        rows.append(row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clean(rows)).to_csv(out_csv, index=False)
    return rows


def fmt_optional(value: Any, fmt: str = ".3f") -> str:
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(x):
        return ""
    return format(x, fmt)


def fmt_precision_req(value: Any) -> str:
    text = fmt_optional(value, ".3f")
    return text if text else "불가능"


def create_markdown_summary(path: Path, calibration_rows: list[dict[str, Any]], split_panel: dict[str, Any], curve_rows: list[dict[str, Any]]) -> None:
    failed_boundary = [r for r in calibration_rows if r.get("family") in {"boundary_scoreblend", "frontier_z_boundary", "tagcf_boundary"}]
    lines = [
        "# boundary_specialist_v1_rowflip_constrained — phase-0 no-submit artifacts",
        "",
        "- validation_only: true",
        "- no_kaggle_submit: true",
        "- public_lb_feedback_used: true",
        "- candidate_csv_written: false",
        "- new_full_test_scoring_performed: false",
        "- existing_full_test_score_artifacts_read: true",
        "- external_metadata_used: false",
        "",
        "## 생성 파일",
        "",
        "```text",
        "reports/boundary_public_failure_calibration.csv",
        "reports/boundary_public_like_split_panel.json",
        "reports/boundary_v1_diffband_precision_curve.csv",
        "```",
        "",
        "## 핵심 해석",
        "",
        "이번 산출물은 boundary specialist 제출 후보가 아니라, 기존 boundary 계열의 public negative-transfer를 calibration set으로 고정하는 phase-0 기록이다.",
        "기존 boundary/frontier/TAGCF 계열은 validation에서 양수였지만 public에서 current best 0.77825를 넘지 못했다.",
        "따라서 v1은 후보 CSV 생성이 아니라 row-flip precision을 예측할 수 있는지부터 검증해야 한다.",
        "",
        "## 기존 boundary 계열 public 결과",
        "",
        "| candidate | public | public Δ | row diff | validation precision | overlap with failed boundary union |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in failed_boundary:
        val_precision = fmt_optional(r.get("validation_flip_precision"), ".3f")
        overlap_frac = fmt_optional(r.get("overlap_frac_of_changed_with_boundary_failed_union"), ".3f")
        lines.append(
            f"| `{r['candidate_id']}` | {float(r['public_score']):.5f} | {float(r['public_delta_vs_current_best']):+.5f} | "
            f"{int(r['row_diff_vs_current_best_fulltest'])} | "
            f"{val_precision} | "
            f"{overlap_frac} |"
        )
    lines.extend([
        "",
        "## split panel 상태",
        "",
        f"- existing split panel: {split_panel['available_panel']['split_count']} splits",
        "- status: score coverage를 30~50 split으로 확장한 상태는 아직 아님. 현재 파일은 risk measurement panel metadata다.",
        "",
        "## diff-band 요구 precision",
        "",
        "| total diff band | public changed est. | top2 precision req. | top1 precision req. |",
        "|---:|---:|---:|---:|",
    ])
    for r in curve_rows:
        top2 = r.get("required_precision_top2_gap")
        top1 = r.get("required_precision_top1_gap")
        lines.append(
            f"| {int(r['band_total_row_diff'])} | {float(r['expected_public_changed_M_if_half']):.1f} | "
            f"{fmt_precision_req(top2)} | "
            f"{fmt_precision_req(top1)} |"
        )
    lines.extend([
        "",
        "## 다음 단계",
        "",
        "1. 이 calibration을 기준으로 기존 boundary artifact bucket을 피할 수 있는 feature만 남긴다.",
        "2. ridge logistic / pairwise logistic cross-fit으로 boundary-only flip proposal을 평가한다.",
        "3. 300 diff band에서 2~3등권 net gain이 보이지 않으면 full-test candidate를 만들지 않는다.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(ROOT / "reports"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    pairs_meta = load_pairs_with_boundary_features()
    user_deg, item_deg = load_train_degrees()
    pairs_meta["user_degree"] = pairs_meta["userID"].map(user_deg).fillna(0).astype(int)
    pairs_meta["item_degree"] = pairs_meta["gameID"].map(item_deg).fillna(0).astype(int)
    pairs_meta["user_degree_decile"] = decile_codes(pairs_meta["user_degree"])
    pairs_meta["item_degree_decile"] = decile_codes(pairs_meta["item_degree"])
    pairs_meta["candidate_count_bucket"] = pairs_meta["candidate_count"].map(bucket_candidate_count)

    calibration_csv = out_dir / "boundary_public_failure_calibration.csv"
    split_panel_json = out_dir / "boundary_public_like_split_panel.json"
    curve_csv = out_dir / "boundary_v1_diffband_precision_curve.csv"
    md_summary = out_dir / "boundary_specialist_v1_phase0_summary.md"

    calibration_rows = create_failure_calibration(calibration_csv, pairs_meta)
    split_panel = create_split_panel(split_panel_json, pairs_meta)
    curve_rows = create_diffband_curve(curve_csv, calibration_rows)
    create_markdown_summary(md_summary, calibration_rows, split_panel, curve_rows)

    payload = {
        "created": [str(calibration_csv), str(split_panel_json), str(curve_csv), str(md_summary)],
        "validation_only": True,
        "no_kaggle_submit": True,
        "public_lb_feedback_used": True,
        "candidate_csv_written": False,
        "new_full_test_scoring_performed": False,
        "existing_full_test_score_artifacts_read": True,
        "external_metadata_used": False,
        "calibration_rows": len(calibration_rows),
        "split_count": split_panel["available_panel"]["split_count"],
    }
    print(json.dumps(clean(payload), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
