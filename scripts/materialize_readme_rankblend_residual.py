#!/usr/bin/env python3
"""Materialize README-derived residual candidates on top of the current rank-blend public best.

This is an aggressive/manual-risk end-game script for KMURecSys26 Steam under the
autonomous submission policy. It combines the
current public-best style rank blend (emb128 4-seed + emb192 4-seed) with README-style
BPR/ALS/hour/pop axes, validates the fixed variants on the calibrated 3-split uniform
panel, and materializes exactly one top-half CSV when requested.

Safety:
- Uses only provided train/pairs and previously generated fold/full-train scores.
- Does not read hidden labels or scrape Steam.
- Does not call Kaggle submit; runner/preflight handles submission separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ["val_random_uniform_seed42", "val_random_uniform_seed7", "val_random_uniform_seed123"]
SEEDS = (42, 123, 2024, 7)
N_EXPECTED = 19998
PAIRS = ROOT / "data/raw/public/data/pairs.csv"
OUT_JSON_DEFAULT = ROOT / "reports/20260602_readme_rankblend_residual_materialization.json"
OUT_MD_DEFAULT = ROOT / "reports/20260602_readme_rankblend_residual_materialization.md"

AXES = [
    "score_als_htr_f32_it30_alpha20_popa8",
    "score_als_htr_f32_it30_alpha20_popa4",
    "score_als_f32_it30_alpha20_popa4",
    "score_als_f32_it30_alpha20_popa8",
    "score_bpr_f32_it100_popa4",
    "score_bpr_f32_it100_popa8",
]
WEIGHTS = (0.025, 0.05, 0.1, 0.2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_two_sided_binom_p(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    kk = min(k, n - k)
    logs = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0)
        for i in range(kk + 1)
    ]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(v - m) for v in logs)
    return min(1.0, 2.0 * tail)


def fisher_pvalue(p_values: list[float]) -> float:
    if not p_values:
        return 1.0
    try:
        from scipy.stats import chi2
        stat = -2.0 * sum(math.log(max(min(float(p), 1.0), 1e-300)) for p in p_values)
        return float(chi2.sf(stat, 2 * len(p_values)))
    except Exception:
        return float("nan")


def score_col(df: pd.DataFrame) -> str:
    for c in ("score_lightgcn", "score", "score_layermix_uniform"):
        if c in df.columns:
            return c
    raise ValueError(f"No score column in {df.columns.tolist()}")


def emb128_files(split: str | None) -> list[Path]:
    if split is None:
        return [ROOT / f"artifacts/lightgcn_emb128L4r3_fulltest/seed{s}/test.csv" for s in SEEDS]
    if split == "val_random_uniform_seed42":
        return [
            ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / split / "lightgcn_scores.csv",
            ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed123" / split / "lightgcn_scores.csv",
            ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed2024" / split / "lightgcn_scores.csv",
            ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed7" / split / "lightgcn_scores.csv",
        ]
    return [ROOT / f"artifacts/split_panel_emb128/{split}/seed{s}/lightgcn_scores.csv" for s in SEEDS]


def emb192_files(split: str | None) -> list[Path]:
    if split is None:
        return [ROOT / f"artifacts/lightgcn_emb192L4r3_fulltest/seed{s}/test.csv" for s in SEEDS]
    if split == "val_random_uniform_seed42":
        return [
            ROOT / "artifacts/capacity_uniform/emb192_L4_r3" / split / "lightgcn_scores.csv",
            ROOT / "artifacts/capacity_uniform/emb192_L4_r3_seed123" / split / "lightgcn_scores.csv",
            ROOT / "artifacts/capacity_uniform/emb192_L4_r3_seed2024" / split / "lightgcn_scores.csv",
            ROOT / "artifacts/capacity_uniform/emb192_L4_r3_seed7" / split / "lightgcn_scores.csv",
        ]
    return [ROOT / f"artifacts/split_panel_emb192/{split}/seed{s}/lightgcn_scores.csv" for s in SEEDS]


def ensemble(files: list[Path], name: str, has_label: bool) -> pd.DataFrame:
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {name} files: {missing}")
    first = pd.read_csv(files[0])
    sc = score_col(first)
    base_cols = ["ID", "userID", "gameID"] + (["Label"] if has_label and "Label" in first.columns else [])
    out = first[base_cols].copy()
    out[name] = first[sc].astype(float).to_numpy()
    for path in files[1:]:
        d = pd.read_csv(path)
        sc2 = score_col(d)
        part = d[["ID", sc2]].rename(columns={sc2: "_score"})
        before = len(out)
        out = out.merge(part, on="ID", how="inner", validate="one_to_one")
        if len(out) != before:
            raise RuntimeError(f"Row alignment changed while merging {path}: {before}->{len(out)}")
        out[name] += out.pop("_score").astype(float).to_numpy()
    out[name] /= len(files)
    return out


def within_user_rank_high(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    ranks = np.zeros(len(df), dtype=np.float64)
    v = np.asarray(values, dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        # Lowest score rank 0; highest rank n-1.
        ranks[idx[np.argsort(v[idx], kind="mergesort")]] = np.arange(len(idx), dtype=np.float64)
    return ranks


def within_user_z(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    tmp = pd.DataFrame({"userID": df["userID"].to_numpy(), "v": np.asarray(values, dtype=float)})
    g = tmp.groupby("userID", sort=False)["v"]
    mu = g.transform("mean").to_numpy(dtype=float)
    sd = g.transform(lambda s: float(s.std(ddof=0))).to_numpy(dtype=float)
    out = np.zeros(len(tmp), dtype=float)
    m = sd > 1e-12
    vals = tmp["v"].to_numpy(dtype=float)
    out[m] = (vals[m] - mu[m]) / sd[m]
    out[~np.isfinite(out)] = 0.0
    return out


def top_half_decode(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    v = np.asarray(values, dtype=float)
    ids = df["ID"].to_numpy(dtype=np.int64)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        k = len(idx) // 2
        # Highest score first, then lower ID as stable tie-break.
        order = np.lexsort((ids[idx], -v[idx]))
        pred[idx[order[:k]]] = 1
    return pred


def metric(df: pd.DataFrame, score: np.ndarray, base_pred: np.ndarray) -> dict[str, Any]:
    pred = top_half_decode(df, score)
    y = df["Label"].to_numpy(dtype=np.int8)
    ok = pred == y
    base_ok = base_pred == y
    fixes = int((~base_ok & ok).sum())
    breaks = int((base_ok & ~ok).sum())
    return {
        "accuracy": float(ok.mean()),
        "delta": float(ok.mean() - base_ok.mean()),
        "fixes": fixes,
        "breaks": breaks,
        "discordant": fixes + breaks,
        "changed": int((pred != base_pred).sum()),
        "p_exact": exact_two_sided_binom_p(fixes, fixes + breaks),
    }


def add_rankblend_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank_emb128"] = within_user_rank_high(df, df["score_emb128"].to_numpy(dtype=float))
    df["rank_emb192"] = within_user_rank_high(df, df["score_emb192"].to_numpy(dtype=float))
    df["score_rankblend"] = df["rank_emb128"] + df["rank_emb192"]
    df["z_rankblend"] = within_user_z(df, df["score_rankblend"].to_numpy(dtype=float))
    return df


def readme_scores_path(split: str | None) -> Path:
    if split is None:
        return ROOT / "artifacts/scores/test_pairs_full_train_readme_bprals/candidate_scores.csv"
    return ROOT / f"artifacts/scores/{split}_readme_bprals/candidate_scores.csv"


def load_frame(split: str | None) -> pd.DataFrame:
    has_label = split is not None
    e128 = ensemble(emb128_files(split), "score_emb128", has_label=has_label)
    e192 = ensemble(emb192_files(split), "score_emb192", has_label=has_label)[["ID", "score_emb192"]]
    if split is None:
        pairs = pd.read_csv(PAIRS)
        df = pairs.merge(e128, on=["ID", "userID", "gameID"], how="left", validate="one_to_one")
    else:
        df = e128
    df = df.merge(e192, on="ID", how="inner", validate="one_to_one")
    rp = readme_scores_path(split)
    if not rp.exists():
        raise FileNotFoundError(f"Missing README full/split score file: {rp}")
    axes = pd.read_csv(rp)
    keep = ["ID", *[c for c in AXES if c in axes.columns]]
    if len(keep) <= 1:
        raise ValueError(f"No expected README axes found in {rp}")
    df = df.merge(axes[keep], on="ID", how="inner", validate="one_to_one")
    df = df.sort_values("ID", kind="mergesort").reset_index(drop=True)
    if df[["score_emb128", "score_emb192", *keep[1:]]].isna().any().any():
        raise RuntimeError("NaN after score merge")
    return add_rankblend_base(df)


def variant_score(df: pd.DataFrame, axis: str, weight: float, mode: str) -> np.ndarray:
    axis_z = within_user_z(df, df[axis].to_numpy(dtype=float))
    if mode == "z":
        return df["z_rankblend"].to_numpy(dtype=float) + weight * axis_z
    if mode == "rank":
        rb_rank = within_user_rank_high(df, df["score_rankblend"].to_numpy(dtype=float))
        ax_rank = within_user_rank_high(df, df[axis].to_numpy(dtype=float))
        return rb_rank + weight * ax_rank
    raise ValueError(mode)


def variant_name(axis: str, weight: float, mode: str) -> str:
    return f"rankblend_{mode}_plus_{axis}_w{weight:g}"


def parse_variant(name: str) -> tuple[str, float, str]:
    m = re.fullmatch(r"rankblend_(z|rank)_plus_(.+)_w([0-9.]+)", name)
    if not m:
        raise ValueError(f"Cannot parse variant name: {name}")
    mode, axis, w = m.group(1), m.group(2), float(m.group(3))
    return axis, w, mode


def validate_variants() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        df = load_frame(split)
        base_pred = top_half_decode(df, df["score_rankblend"].to_numpy(dtype=float))
        y = df["Label"].to_numpy(dtype=np.int8)
        base_acc = float((base_pred == y).mean())
        base_rows.append({"split": split, "rankblend_base_acc": base_acc})
        present_axes = [a for a in AXES if a in df.columns]
        for axis in present_axes:
            for w in WEIGHTS:
                for mode in ("z", "rank"):
                    name = variant_name(axis, w, mode)
                    m = metric(df, variant_score(df, axis, w, mode), base_pred)
                    rows.append({"split": split, "variant": name, "axis": axis, "weight": w, "mode": mode, **m})
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)
    summary: list[dict[str, Any]] = []
    for name, rs in by_variant.items():
        deltas = [float(r["delta"]) for r in rs]
        fixes = int(sum(int(r["fixes"]) for r in rs))
        breaks = int(sum(int(r["breaks"]) for r in rs))
        ps = [float(r["p_exact"]) for r in rs]
        first = rs[0]
        summary.append({
            "variant": name,
            "axis": first["axis"],
            "weight": first["weight"],
            "mode": first["mode"],
            "mean_delta_vs_rankblend": float(np.mean(deltas)),
            "min_delta": float(np.min(deltas)),
            "max_delta": float(np.max(deltas)),
            "positive_splits": int(sum(d > 0 for d in deltas)),
            "fixes": fixes,
            "breaks": breaks,
            "discordant": fixes + breaks,
            "changed": int(sum(int(r["changed"]) for r in rs)),
            "pooled_p_exact": exact_two_sided_binom_p(fixes, fixes + breaks),
            "fisher_p": fisher_pvalue(ps),
            "split_rows": rs,
            "manual_risk_signal": bool(np.mean(deltas) > 0 and sum(d > 0 for d in deltas) >= 2 and fixes > breaks),
        })
    summary.sort(key=lambda x: (x["manual_risk_signal"], x["mean_delta_vs_rankblend"], x["fixes"] - x["breaks"]), reverse=True)
    return {
        "base": "rank_blend_emb128_emb192_public_best_style",
        "base_rows": base_rows,
        "variant_count": len(summary),
        "top_variants": summary[:25],
        "all_variants": summary,
    }


def preflight_submission(out: pd.DataFrame) -> dict[str, Any]:
    pairs = pd.read_csv(PAIRS)
    if list(out.columns) != ["ID", "Played"]:
        raise ValueError(f"Unexpected output columns: {out.columns.tolist()}")
    m = pairs[["ID", "userID"]].merge(out, on="ID", validate="one_to_one")
    per_user = m.groupby("userID", sort=False)["Played"].agg(["sum", "count"])
    return {
        "rows": int(len(out)),
        "expected_rows": N_EXPECTED,
        "columns": out.columns.tolist(),
        "id_unique": bool(out["ID"].is_unique),
        "id_contiguous": bool((out["ID"].to_numpy() == np.arange(len(out))).all()),
        "labels_binary": bool(set(out["Played"].astype(int).unique()).issubset({0, 1})),
        "label_1": int(out["Played"].sum()),
        "label_0": int(len(out) - out["Played"].sum()),
        "bad_users_tophalf": int(((per_user["count"] % 2 != 0) | (per_user["sum"] != per_user["count"] // 2)).sum()),
    }


def materialize(variant: str, out_path: Path) -> dict[str, Any]:
    axis, weight, mode = parse_variant(variant)
    df = load_frame(None)
    score = variant_score(df, axis, weight, mode)
    pred = top_half_decode(df, score)
    out = pd.DataFrame({"ID": df["ID"].astype(int).to_numpy(), "Played": pred.astype(int)}).sort_values("ID", kind="mergesort")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    pf = preflight_submission(out)
    sha = sha256_file(out_path)
    comparisons: dict[str, Any] = {}
    for label, path in {
        "rankblend_public_best": ROOT / "submissions/candidate_rank_blend_emb128_emb192.csv",
        "emb128_anchor": ROOT / "artifacts/lightgcn_emb128L4r3_fulltest/test_candidate/candidate_lightgcn_emb128L4r3_seed_ens.csv",
        "zblend_public_2": ROOT / "submissions/candidate_emb128_emb64_zblend.csv",
    }.items():
        if path.exists():
            d = pd.read_csv(path)
            lab = "Played" if "Played" in d.columns else "Label"
            cmp = out.merge(d[["ID", lab]].rename(columns={lab: label}), on="ID", validate="one_to_one")
            comparisons[label] = {
                "path": str(path),
                "row_diff": int((cmp["Played"].astype(int) != cmp[label].astype(int)).sum()),
                "row_diff_frac": float((cmp["Played"].astype(int) != cmp[label].astype(int)).mean()),
            }
        else:
            comparisons[label] = {"path": str(path), "exists": False}
    return {"file": str(out_path), "sha256": sha, "preflight": pf, "comparisons": comparisons}


def write_reports(payload: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# README rank-blend residual materialization", ""]
    lines.append(f"- base: `{payload.get('validation', {}).get('base', 'rank_blend_emb128_emb192')}`")
    top = payload.get("validation", {}).get("top_variants", [])[:10]
    lines += ["", "## Top validation variants", "", "| rank | variant | mean Δ vs rankblend | pos splits | fixes/breaks | pooled p | manual-risk |", "|---:|---|---:|---:|---:|---:|---|"]
    for i, r in enumerate(top, 1):
        lines.append(
            f"| {i} | `{r['variant']}` | {r['mean_delta_vs_rankblend']:+.6f} | {r['positive_splits']}/3 | {r['fixes']}/{r['breaks']} | {r['pooled_p_exact']:.4g} | {r['manual_risk_signal']} |"
        )
    if payload.get("materialized"):
        m = payload["materialized"]
        lines += ["", "## Materialized", "", f"- file: `{m['file']}`", f"- sha256: `{m['sha256']}`", f"- preflight: `{m['preflight']}`"]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--variant", default=None, help="Variant name to materialize. Use --auto-best to choose top manual-risk variant.")
    ap.add_argument("--auto-best", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=str(OUT_JSON_DEFAULT))
    ap.add_argument("--md", default=str(OUT_MD_DEFAULT))
    args = ap.parse_args()

    validation = validate_variants()
    chosen = args.variant
    if args.auto_best:
        chosen = None
        for r in validation["top_variants"]:
            if r.get("manual_risk_signal"):
                chosen = r["variant"]
                break
        if chosen is None:
            chosen = validation["top_variants"][0]["variant"]
    materialized = None
    if chosen:
        out = Path(args.out) if args.out else ROOT / "submissions" / f"candidate_autorun_{chosen}.csv"
        # keep filename manageable
        out = out.with_name(re.sub(r"[^A-Za-z0-9_.-]+", "_", out.name)[:180])
        materialized = materialize(chosen, out)
        materialized["variant"] = chosen
    payload = {
        "note": "Aggressive/manual-risk README residual on top of current rank-blend public-best style. No Kaggle submit inside this script.",
        "safety": {
            "hidden_label_access": False,
            "external_steam_scraping": False,
            "kaggle_submit_executed": False,
        },
        "validation": validation,
        "materialized": materialized,
    }
    write_reports(payload, Path(args.json), Path(args.md))
    print(json.dumps({
        "best": validation["top_variants"][0],
        "materialized": materialized,
        "json": args.json,
        "md": args.md,
    }, indent=2, ensure_ascii=False))

    if materialized:
        pf = materialized["preflight"]
        ok = (
            pf["rows"] == N_EXPECTED and pf["id_unique"] and pf["id_contiguous"] and pf["labels_binary"]
            and pf["label_1"] == pf["label_0"] == N_EXPECTED // 2 and pf["bad_users_tophalf"] == 0
        )
        raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
