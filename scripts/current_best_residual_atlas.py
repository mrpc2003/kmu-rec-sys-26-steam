#!/usr/bin/env python3
"""Validation-only residual atlas around the current rankblend public-best anchor.

This script intentionally does NOT create candidate/submission CSVs and does NOT
call Kaggle.  It answers a narrower development question after multiple closed
axis loops: where does the current emb128+emb192 rankblend anchor still make
validation errors, and do simple train-only profile-compatibility features show
any split-stable residual direction worth an independent follow-up?

Outputs are JSON + Markdown reports only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import materialize_readme_rankblend_residual as rb  # noqa: E402

SPLITS = ["val_random_uniform_seed42", "val_random_uniform_seed7", "val_random_uniform_seed123"]
FEATURE_WEIGHTS = [-0.30, -0.20, -0.10, -0.05, 0.05, 0.10, 0.20, 0.30]
BOUNDARY_BANDS = [None, 1, 2, 3]
STRICT_MEAN_DELTA = 0.0015
STRICT_P = 0.05


def exact_two_sided_binom_p(fixes: int, breaks: int) -> float:
    n = fixes + breaks
    if n <= 0:
        return 1.0
    return float(binomtest(min(fixes, breaks), n, 0.5, alternative="two-sided").pvalue)


def nan_to_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    if isinstance(obj, tuple):
        return [nan_to_none(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def safe_z(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    out = np.zeros(len(arr), dtype=float)
    mask = np.isfinite(arr)
    if not mask.any():
        return out
    mu = float(np.nanmean(arr[mask]))
    sd = float(np.nanstd(arr[mask]))
    if sd > 1e-12:
        out[mask] = (arr[mask] - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out


def within_user_z(df: pd.DataFrame, col: str) -> np.ndarray:
    values = df[col].to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        vals = values[idx]
        mask = np.isfinite(vals)
        if not mask.any():
            continue
        mu = float(np.nanmean(vals[mask]))
        sd = float(np.nanstd(vals[mask]))
        if sd > 1e-12:
            out[idx[mask]] = (vals[mask] - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out


def base_rank_and_margin(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rank_desc = np.zeros(len(df), dtype=float)
    margin = np.zeros(len(df), dtype=float)
    scores = df["score_rankblend"].to_numpy(dtype=float)
    ids = df["ID"].to_numpy(dtype=np.int64)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        k = len(idx) // 2
        order = np.lexsort((ids[idx], -scores[idx]))
        rank_desc[idx[order]] = np.arange(1, len(idx) + 1, dtype=float)
        margin[idx] = np.abs(rank_desc[idx] - (k + 0.5))
    return rank_desc, margin


def top_half_pred(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    v = np.asarray(values, dtype=float)
    ids = df["ID"].to_numpy(dtype=np.int64)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        k = len(idx) // 2
        order = np.lexsort((ids[idx], -v[idx]))
        pred[idx[order[:k]]] = 1
    return pred


def eval_pred(y: np.ndarray, pred: np.ndarray, base_pred: np.ndarray) -> dict[str, Any]:
    ok = pred == y
    base_ok = base_pred == y
    fixes = int((ok & ~base_ok).sum())
    breaks = int((~ok & base_ok).sum())
    return {
        "accuracy": float(ok.mean()),
        "delta_vs_rankblend": float(ok.mean() - base_ok.mean()),
        "fixes": fixes,
        "breaks": breaks,
        "discordant": fixes + breaks,
        "changed": int((pred != base_pred).sum()),
        "p_exact": exact_two_sided_binom_p(fixes, breaks),
    }


def add_train_only_features(split: str, df: pd.DataFrame) -> pd.DataFrame:
    train_path = ROOT / "artifacts" / "validation" / split / "train_interactions.csv"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    train = pd.read_csv(train_path)
    train["date"] = pd.to_datetime(train["date"], errors="coerce")
    train["date_ord"] = train["date"].map(lambda x: float(x.toordinal()) if pd.notna(x) else np.nan)

    item = train.groupby("gameID", sort=False).agg(
        item_pop=("gameID", "size"),
        item_htr_mean=("hours_transformed", "mean"),
        item_hours_mean=("hours", "mean"),
        item_text_mean=("text_len", "mean"),
        item_funny_mean=("found_funny", "mean"),
        item_early_rate=("early_access", "mean"),
        item_date_mean=("date_ord", "mean"),
    ).reset_index()
    item["log_item_pop"] = np.log1p(item["item_pop"].astype(float))

    # For user profile, include average popularity of items the user played in fold-train.
    train_item_pop = train.merge(item[["gameID", "log_item_pop", "item_pop"]], on="gameID", how="left")
    user = train_item_pop.groupby("userID", sort=False).agg(
        user_deg=("gameID", "size"),
        user_htr_mean=("hours_transformed", "mean"),
        user_hours_mean=("hours", "mean"),
        user_text_mean=("text_len", "mean"),
        user_funny_mean=("found_funny", "mean"),
        user_early_rate=("early_access", "mean"),
        user_date_mean=("date_ord", "mean"),
        user_hist_log_item_pop_mean=("log_item_pop", "mean"),
        user_hist_item_pop_mean=("item_pop", "mean"),
    ).reset_index()
    user["log_user_deg"] = np.log1p(user["user_deg"].astype(float))

    out = df.merge(item, on="gameID", how="left", validate="many_to_one")
    out = out.merge(user, on="userID", how="left", validate="many_to_one")

    fill_cols = [c for c in out.columns if c.startswith("item_") or c.startswith("user_") or c.startswith("log_")]
    for col in fill_cols:
        if out[col].dtype.kind in "biufc":
            out[col] = out[col].astype(float)
            med = float(np.nanmedian(out[col].to_numpy(dtype=float))) if np.isfinite(out[col]).any() else 0.0
            out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(med)

    out["rel_log_item_pop_vs_user_hist"] = out["log_item_pop"] - out["user_hist_log_item_pop_mean"]
    out["compat_log_item_pop"] = -np.abs(out["rel_log_item_pop_vs_user_hist"])
    out["rel_item_htr_vs_user"] = out["item_htr_mean"] - out["user_htr_mean"]
    out["compat_item_htr"] = -np.abs(out["rel_item_htr_vs_user"])
    out["rel_item_text_vs_user"] = out["item_text_mean"] - out["user_text_mean"]
    out["compat_item_text"] = -np.abs(out["rel_item_text_vs_user"])
    out["rel_item_date_vs_user"] = out["item_date_mean"] - out["user_date_mean"]
    out["compat_item_date"] = -np.abs(out["rel_item_date_vs_user"])
    out["rel_item_early_vs_user"] = out["item_early_rate"] - out["user_early_rate"]
    out["compat_item_early"] = -np.abs(out["rel_item_early_vs_user"])

    out["emb192_minus_emb128"] = out["score_emb192"] - out["score_emb128"]
    out["rank_disagreement_abs"] = np.abs(out["rank_emb192"] - out["rank_emb128"])
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    base = [
        "log_item_pop",
        "item_htr_mean",
        "item_hours_mean",
        "item_text_mean",
        "item_funny_mean",
        "item_early_rate",
        "item_date_mean",
        "log_user_deg",
        "user_htr_mean",
        "user_text_mean",
        "rel_log_item_pop_vs_user_hist",
        "compat_log_item_pop",
        "rel_item_htr_vs_user",
        "compat_item_htr",
        "rel_item_text_vs_user",
        "compat_item_text",
        "rel_item_date_vs_user",
        "compat_item_date",
        "rel_item_early_vs_user",
        "compat_item_early",
        "emb192_minus_emb128",
        "rank_disagreement_abs",
    ]
    readme_axes = [c for c in rb.AXES if c in df.columns]
    return [c for c in base + readme_axes if c in df.columns]


def bucket_rows(df: pd.DataFrame, y: np.ndarray, base_pred: np.ndarray, margin: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_ok = base_pred == y
    specs: list[tuple[str, np.ndarray]] = []
    specs.append(("boundary_margin_le_1", margin <= 1))
    specs.append(("boundary_margin_le_2", margin <= 2))
    specs.append(("boundary_margin_le_3", margin <= 3))
    for col in ["log_item_pop", "log_user_deg", "rank_disagreement_abs", "compat_log_item_pop", "compat_item_htr", "item_htr_mean"]:
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if finite.sum() < 10:
            continue
        try:
            qs = np.nanquantile(vals[finite], [0.2, 0.4, 0.6, 0.8])
        except Exception:
            continue
        bins = np.digitize(vals, qs, right=False)
        for b in range(5):
            specs.append((f"{col}_q{b+1}", bins == b))
    for name, mask in specs:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            continue
        rows.append({
            "bucket": name,
            "rows": int(mask.sum()),
            "base_accuracy": float(base_ok[mask].mean()),
            "error_rate": float((~base_ok[mask]).mean()),
            "false_positive": int(((base_pred == 1) & (y == 0) & mask).sum()),
            "false_negative": int(((base_pred == 0) & (y == 1) & mask).sum()),
        })
    rows.sort(key=lambda r: (r["error_rate"], r["rows"]), reverse=True)
    return rows[:30]


def split_scan(split: str, weights: list[float], bands: list[int | None]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    df = rb.load_frame(split)
    df = add_train_only_features(split, df)
    rank_desc, margin = base_rank_and_margin(df)
    df["base_rank_desc"] = rank_desc
    df["base_boundary_margin"] = margin
    y = df["Label"].to_numpy(dtype=np.int8)
    base_score = df["score_rankblend"].to_numpy(dtype=float)
    base_pred = top_half_pred(df, base_score)
    base_ok = base_pred == y
    z_base = within_user_z(df, "score_rankblend")

    feats = feature_columns(df)
    rows: list[dict[str, Any]] = []
    fn_mask = (base_pred == 0) & (y == 1)
    fp_mask = (base_pred == 1) & (y == 0)
    feature_diagnostics = []
    for col in feats:
        vals = df[col].to_numpy(dtype=float)
        z = within_user_z(df, col)
        fn_mean = float(np.nanmean(vals[fn_mask])) if fn_mask.any() else float("nan")
        fp_mean = float(np.nanmean(vals[fp_mask])) if fp_mask.any() else float("nan")
        feature_diagnostics.append({
            "split": split,
            "feature": col,
            "fn_mean": fn_mean,
            "fp_mean": fp_mean,
            "fn_minus_fp": fn_mean - fp_mean if np.isfinite(fn_mean) and np.isfinite(fp_mean) else None,
        })
        for w in weights:
            for band in bands:
                if band is None:
                    mask_multiplier = 1.0
                    band_name = "all"
                else:
                    mask_multiplier = (margin <= band).astype(float)
                    band_name = f"band{band}"
                score = z_base + float(w) * z * mask_multiplier
                pred = top_half_pred(df, score)
                ev = eval_pred(y, pred, base_pred)
                rows.append({
                    "split": split,
                    "variant": f"zrankblend_plus_{col}_w{w:+.2f}_{band_name}",
                    "feature": col,
                    "weight": float(w),
                    "band": band,
                    **ev,
                })
    base_summary = {
        "split": split,
        "base_accuracy": float(base_ok.mean()),
        "rows": int(len(df)),
        "users": int(df["userID"].nunique()),
        "errors": int((~base_ok).sum()),
        "false_positive": int(((base_pred == 1) & (y == 0)).sum()),
        "false_negative": int(((base_pred == 0) & (y == 1)).sum()),
        "feature_count": len(feats),
        "top_error_buckets": bucket_rows(df, y, base_pred, margin),
        "feature_fn_fp_diagnostics": feature_diagnostics,
    }
    return base_summary, rows, feature_diagnostics


def aggregate_variant_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(str(r["variant"]), []).append(r)
    out: list[dict[str, Any]] = []
    for variant, rs in by.items():
        if len(rs) != len(SPLITS):
            continue
        deltas = [float(r["delta_vs_rankblend"]) for r in rs]
        fixes = int(sum(int(r["fixes"]) for r in rs))
        breaks = int(sum(int(r["breaks"]) for r in rs))
        p = exact_two_sided_binom_p(fixes, breaks)
        strict = (
            float(np.mean(deltas)) >= STRICT_MEAN_DELTA
            and float(np.min(deltas)) >= 0
            and int(sum(d > 0 for d in deltas)) == len(SPLITS)
            and fixes > breaks
            and p < STRICT_P
        )
        out.append({
            "variant": variant,
            "feature": rs[0]["feature"],
            "weight": rs[0]["weight"],
            "band": rs[0]["band"],
            "mean_delta_vs_rankblend": float(np.mean(deltas)),
            "min_delta_vs_rankblend": float(np.min(deltas)),
            "max_delta_vs_rankblend": float(np.max(deltas)),
            "positive_splits": int(sum(d > 0 for d in deltas)),
            "fixes": fixes,
            "breaks": breaks,
            "discordant": fixes + breaks,
            "pooled_p_exact": p,
            "changed": int(sum(int(r["changed"]) for r in rs)),
            "strict_diagnostic_pass": strict,
            "split_deltas": {str(r["split"]): float(r["delta_vs_rankblend"]) for r in rs},
        })
    out.sort(key=lambda r: (r["strict_diagnostic_pass"], r["mean_delta_vs_rankblend"], r["fixes"] - r["breaks"]), reverse=True)
    return out


def aggregate_feature_gaps(diags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = {}
    for d in diags:
        by.setdefault(str(d["feature"]), []).append(d)
    out: list[dict[str, Any]] = []
    for feature, ds in by.items():
        vals = [float(v) for d in ds for v in [d.get("fn_minus_fp")] if v is not None]
        if not vals:
            continue
        signs = [1 if v > 0 else -1 if v < 0 else 0 for v in vals]
        out.append({
            "feature": feature,
            "mean_fn_minus_fp": float(np.mean(vals)),
            "min_fn_minus_fp": float(np.min(vals)),
            "max_fn_minus_fp": float(np.max(vals)),
            "positive_splits": int(sum(s > 0 for s in signs)),
            "negative_splits": int(sum(s < 0 for s in signs)),
            "split_values": {str(d["split"]): d.get("fn_minus_fp") for d in ds},
        })
    out.sort(key=lambda r: (max(r["positive_splits"], r["negative_splits"]), abs(r["mean_fn_minus_fp"])), reverse=True)
    return out


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# Current-best residual atlas — {payload['timestamp_kst']}")
    lines.append("")
    lines.append("Validation-only diagnostic around `rank_blend_emb128_emb192` current-best style. No Kaggle submission, no full-test candidate CSV.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- verdict: `{payload['verdict']}`")
    lines.append(f"- strict_diagnostic_pass_count: `{payload['strict_diagnostic_pass_count']}`")
    lines.append(f"- tested_variants: `{payload['variant_count']}`")
    lines.append("")
    lines.append("## Base rankblend validation")
    lines.append("")
    for b in payload["base_summaries"]:
        lines.append(f"- {b['split']}: acc=`{b['base_accuracy']:.6f}`, errors=`{b['errors']}`, FP/FN=`{b['false_positive']}/{b['false_negative']}`")
    lines.append("")
    lines.append("## Top diagnostic residual variants")
    lines.append("")
    lines.append("| rank | variant | mean Δ | min Δ | pos | fixes/breaks | p | strict diag |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate(payload["top_variants"][:20], 1):
        lines.append(
            f"| {i} | `{r['variant']}` | {r['mean_delta_vs_rankblend']:+.6f} | {r['min_delta_vs_rankblend']:+.6f} | {r['positive_splits']}/3 | {r['fixes']}/{r['breaks']} | {r['pooled_p_exact']:.4g} | {r['strict_diagnostic_pass']} |"
        )
    lines.append("")
    lines.append("## Stable false-negative vs false-positive feature gaps")
    lines.append("")
    lines.append("Positive `FN-FP` means missed positives had larger feature values than false positives. This is diagnostic only, not a submission rule.")
    lines.append("")
    lines.append("| rank | feature | mean FN-FP | + splits | - splits |")
    lines.append("|---:|---|---:|---:|---:|")
    for i, r in enumerate(payload["feature_gap_summary"][:20], 1):
        lines.append(f"| {i} | `{r['feature']}` | {r['mean_fn_minus_fp']:+.6f} | {r['positive_splits']} | {r['negative_splits']} |")
    lines.append("")
    lines.append("## Highest-error buckets per split")
    for b in payload["base_summaries"]:
        lines.append("")
        lines.append(f"### {b['split']}")
        lines.append("| bucket | rows | base acc | error | FP | FN |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in b["top_error_buckets"][:10]:
            lines.append(f"| `{r['bucket']}` | {r['rows']} | {r['base_accuracy']:.4f} | {r['error_rate']:.4f} | {r['false_positive']} | {r['false_negative']} |")
    lines.append("")
    lines.append("## Safety flags")
    lines.append("")
    for k, v in payload["safety_flags"].items():
        lines.append(f"- {k}: `{str(v).lower()}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    args = ap.parse_args()

    import datetime as _dt
    kst = _dt.timezone(_dt.timedelta(hours=9), name="KST")
    ts = _dt.datetime.now(kst).strftime("%Y%m%dT%H%M%SKST")

    all_rows: list[dict[str, Any]] = []
    base_summaries: list[dict[str, Any]] = []
    feature_diags: list[dict[str, Any]] = []
    for split in SPLITS:
        base, rows, diags = split_scan(split, FEATURE_WEIGHTS, BOUNDARY_BANDS)
        base_summaries.append(base)
        all_rows.extend(rows)
        feature_diags.extend(diags)

    variants = aggregate_variant_rows(all_rows)
    strict_count = int(sum(bool(r["strict_diagnostic_pass"]) for r in variants))
    top = variants[0] if variants else None
    if strict_count:
        verdict = "STRICT_PASS_DIAGNOSTIC_NEEDS_INDEPENDENT_CONFIRMATION"
    elif top and top["mean_delta_vs_rankblend"] > 0 and top["positive_splits"] >= 2 and top["fixes"] > top["breaks"]:
        verdict = "WEAK_SIGNAL_DIAGNOSTIC"
    else:
        verdict = "REJECT_NO_STABLE_RESIDUAL"

    payload: dict[str, Any] = {
        "timestamp_kst": ts,
        "safety_flags": {
            "validation_only": True,
            "candidate_csv_written": False,
            "full_test_candidate_or_submission_csv_created": False,
            "kaggle_submit_executed": False,
            "hidden_labels_used": False,
            "private_answers_used": False,
            "external_steam_scraping_used": False,
            "credentials_or_tokens_printed": False,
            "quarantine_or_guard_logic_weakened": False,
            "git_stage_commit_push_executed": False,
        },
        "base": "rank_blend_emb128_emb192_public_best_style",
        "splits": SPLITS,
        "weights": FEATURE_WEIGHTS,
        "boundary_bands": BOUNDARY_BANDS,
        "base_summaries": base_summaries,
        "variant_count": len(variants),
        "strict_diagnostic_pass_count": strict_count,
        "verdict": verdict,
        "top_variants": variants[:50],
        "feature_gap_summary": aggregate_feature_gaps(feature_diags)[:50],
        "all_variants": variants,
    }
    payload = nan_to_none(payload)
    json_path = Path(args.json)
    md_path = Path(args.md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(md_path, payload)
    print(json.dumps({
        "json": str(json_path),
        "md": str(md_path),
        "verdict": verdict,
        "variant_count": len(variants),
        "strict_diagnostic_pass_count": strict_count,
        "top_variant": top,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
