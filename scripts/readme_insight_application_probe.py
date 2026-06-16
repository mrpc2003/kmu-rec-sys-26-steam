#!/usr/bin/env python3
"""Apply in-bundle README insights against the saturated LightGCN backbone.

The competition README recommends: popularity + CF hybrids, per-user top-half
ranking, validation tuning, LightGCN, hours_transformed confidence, ensembling,
and alpha tuning. Most were already used earlier; this script performs the
remaining end-game reconciliation: can exact README-style BPR/ALS/pop/hour axes
change the calibrated multi-split uniform gate when added to the current emb128
4-seed LightGCN backbone?

Safety: validation-only. It never reads hidden labels, never reads the real test
pairs, never writes a candidate/submission CSV, and never calls Kaggle APIs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("val_random_uniform_seed42", "val_random_uniform_seed7", "val_random_uniform_seed123")
SEEDS = (42, 123, 2024, 7)
BASE_EXPECTED = {
    "val_random_uniform_seed42": 0.7650530106021204,
    "val_random_uniform_seed7": 0.7609521904380876,
    "val_random_uniform_seed123": 0.7599519903980796,
}
MDE = 0.00355
OUT_JSON_DEFAULT = ROOT / "reports/20260602_readme_insight_application_probe.json"
OUT_MD_DEFAULT = ROOT / "reports/20260602_readme_insight_application_probe.md"


def exact_two_sided_binom_p(k: int, n: int) -> float:
    """Exact two-sided sign-test p-value for paired discordant fixes/breaks."""
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


def score_col(df: pd.DataFrame) -> str:
    for c in ("score_layermix_uniform", "score_lightgcn", "score"):
        if c in df.columns:
            return c
    raise ValueError(f"No score column in {df.columns.tolist()}")


def lightgcn_path(split: str, seed: int) -> Path:
    """Current final backbone: emb128 L4 reg1e-3, four validation seeds."""
    if split == "val_random_uniform_seed42":
        if seed == 42:
            return ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / split / "lightgcn_scores.csv"
        return ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}" / split / "lightgcn_scores.csv"
    return ROOT / f"artifacts/split_panel_emb128/{split}/seed{seed}/lightgcn_scores.csv"


def load_lightgcn_ensemble(split: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    cols: list[str] = []
    for seed in SEEDS:
        p = lightgcn_path(split, seed)
        if not p.exists():
            raise FileNotFoundError(f"Missing emb128 LightGCN score split={split} seed={seed}: {p}")
        d = pd.read_csv(p)
        sc = score_col(d)
        need = {"ID", "userID", "gameID", "Label", sc}
        if not need.issubset(d.columns):
            raise ValueError(f"Missing columns in {p}: {need - set(d.columns)}")
        col = f"e128_seed{seed}"
        part = d[["ID", "userID", "gameID", "Label", sc]].rename(columns={sc: col})
        if merged is None:
            merged = part
        else:
            before = len(merged)
            merged = merged.merge(part[["ID", col]], on="ID", how="inner", validate="one_to_one")
            if len(merged) != before:
                raise RuntimeError(f"Row alignment changed while merging {p}")
        cols.append(col)
    assert merged is not None
    merged["base_e128"] = merged[cols].mean(axis=1)
    return merged[["ID", "userID", "gameID", "Label", "base_e128"]].sort_values("ID", kind="mergesort").reset_index(drop=True)


def load_readme_axes(split: str) -> pd.DataFrame:
    p = ROOT / f"artifacts/scores/{split}_readme_bprals/candidate_scores.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing README BPR/ALS scores for {split}: {p}. "
            "Run scripts/score_bpr_als.py with the README-style parameters first."
        )
    d = pd.read_csv(p)
    need = {"ID", "userID", "gameID", "Label", "pop_count"}
    if not need.issubset(d.columns):
        raise ValueError(f"Missing columns in {p}: {need - set(d.columns)}")
    d = d.sort_values("ID", kind="mergesort").reset_index(drop=True).copy()
    # Derive pure README popularity variants from the same fold-train pop_count.
    d = d.assign(
        score_pop_norm=d["pop_count"] / max(float(d["pop_count"].max()), 1.0),
        score_pop_log=np.log1p(d["pop_count"].astype(float)),
        score_pop_sqrt=np.sqrt(d["pop_count"].astype(float)),
    ).copy()
    return d


def within_user_z(df: pd.DataFrame, col: str) -> np.ndarray:
    g = df.groupby("userID", sort=False)[col]
    mu = g.transform("mean").to_numpy(dtype=float)
    sd = g.transform(lambda s: float(s.std(ddof=0))).to_numpy(dtype=float)
    x = df[col].to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=float)
    m = sd > 1e-12
    out[m] = (x[m] - mu[m]) / sd[m]
    out[~np.isfinite(out)] = 0.0
    return out


def within_user_rank_high(df: pd.DataFrame, col: str) -> np.ndarray:
    ranks = np.zeros(len(df), dtype=float)
    values = df[col].to_numpy(dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        ranks[idx[np.argsort(values[idx], kind="mergesort")]] = np.arange(len(idx), dtype=float)
    return ranks


def predict_tophalf(df: pd.DataFrame, score_values: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    v = np.asarray(score_values, dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        k = len(idx) // 2
        pred[idx[np.argsort(v[idx], kind="mergesort")[::-1][:k]]] = 1
    return pred


def metric(df: pd.DataFrame, score: np.ndarray, base_pred: np.ndarray) -> dict[str, Any]:
    pred = predict_tophalf(df, score)
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
    }


def boundary_score(base: np.ndarray, variant: np.ndarray, df: pd.DataFrame, width_frac: float, cap: int) -> np.ndarray:
    out = np.zeros(len(df), dtype=float)
    base = np.asarray(base, dtype=float)
    variant = np.asarray(variant, dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        n = len(idx)
        h = n // 2
        w = min(cap, max(1, int(math.ceil(width_frac * h))))
        order = idx[np.argsort(base[idx], kind="mergesort")[::-1]]
        ranks = np.empty(n, dtype=int)
        ranks[np.argsort(base[idx], kind="mergesort")[::-1]] = np.arange(n)
        local = -ranks.astype(float) * 1000.0
        lo = max(0, h - w)
        hi = min(n, h + w)
        band = order[lo:hi]
        if len(band):
            vv = variant[band]
            vz = (vv - vv.mean()) / vv.std(ddof=0) if vv.std(ddof=0) > 1e-12 else np.zeros(len(band))
            pos = {rid: j for j, rid in enumerate(idx)}
            for rid, val in zip(band, vz, strict=True):
                local[pos[rid]] = -h * 1000.0 + val
        out[idx] = local
    return out


def build_split_frame(split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = load_lightgcn_ensemble(split)
    axes = load_readme_axes(split)
    before = len(base)
    df = base.merge(axes.drop(columns=["userID", "gameID", "Label"]), on="ID", how="inner", validate="one_to_one")
    if len(df) != before:
        raise RuntimeError(f"Row count changed while merging README axes for {split}: {len(df)} vs {before}")
    df = df.sort_values("ID", kind="mergesort").reset_index(drop=True).copy()
    if not (df.groupby("userID").size() % 2 == 0).all():
        raise RuntimeError(f"{split}: not all user candidate counts are even")

    readme_cols = [
        "score_pop_norm",
        "score_pop_log",
        "score_pop_sqrt",
        "score_bpr_f32_it100",
        "score_bpr_f32_it100_popa4",
        "score_bpr_f32_it100_popa8",
        "score_bpr_htr_f32_it100_popa4",
        "score_bpr_htr_f32_it100_popa8",
        "score_als_f32_it30_alpha20",
        "score_als_f32_it30_alpha20_popa4",
        "score_als_f32_it30_alpha20_popa8",
        "score_als_htr_f32_it30_alpha20_popa4",
        "score_als_htr_f32_it30_alpha20_popa8",
    ]
    readme_cols = [c for c in readme_cols if c in df.columns]
    for c in ["base_e128", *readme_cols]:
        df[f"z_{c}"] = within_user_z(df, c)
        df[f"r_{c}"] = within_user_rank_high(df, c)

    solo = {}
    base_pred = predict_tophalf(df, df["base_e128"].to_numpy(dtype=float))
    for c in readme_cols:
        solo[c] = metric(df, df[c].to_numpy(dtype=float), base_pred)
    return df, {"readme_cols": readme_cols, "solo_vs_base": solo}


def generate_variant_scores(df: pd.DataFrame, readme_cols: list[str]) -> dict[str, tuple[np.ndarray, str, dict[str, Any]]]:
    out: dict[str, tuple[np.ndarray, str, dict[str, Any]]] = {}
    weights = (0.025, 0.05, 0.1, 0.2, 0.5, 1.0)
    base_z = df["z_base_e128"].to_numpy(dtype=float)
    base_r = df["r_base_e128"].to_numpy(dtype=float)
    base_raw = df["base_e128"].to_numpy(dtype=float)
    for c in readme_cols:
        z = df[f"z_{c}"].to_numpy(dtype=float)
        r = df[f"r_{c}"].to_numpy(dtype=float)
        for w in weights:
            out[f"z_base_plus_{c}_w{w:g}"] = (base_z + w * z, "readme_weighted_z", {"axis": c, "weight": w})
            out[f"rank_base_plus_{c}_w{w:g}"] = (base_r + w * r, "readme_weighted_rank", {"axis": c, "weight": w})
        if c in {"score_pop_norm", "score_pop_log", "score_pop_sqrt"}:
            # Also test a raw additive popularity prior, matching the README's CF + alpha*pop recipe.
            norm = df[c].to_numpy(dtype=float)
            norm = (norm - np.nanmean(norm)) / (np.nanstd(norm) + 1e-12)
            for w in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02):
                out[f"raw_base_plus_{c}_w{w:g}"] = (base_raw + w * norm, "readme_raw_pop_prior", {"axis": c, "weight": w})

    # README-style ensemble idea, but applied as boundary-only rescue to avoid disturbing confident rows.
    boundary_axes = [
        "score_pop_log",
        "score_als_f32_it30_alpha20_popa4",
        "score_als_f32_it30_alpha20_popa8",
        "score_bpr_f32_it100_popa8",
        "score_als_htr_f32_it30_alpha20_popa8",
    ]
    for c in [x for x in boundary_axes if x in readme_cols]:
        z = df[f"z_{c}"].to_numpy(dtype=float)
        for frac in (0.05, 0.10, 0.20):
            for cap in (5, 10, 20):
                out[f"boundary_base_{c}_frac{frac:g}_cap{cap}"] = (
                    boundary_score(base_raw, z, df, frac, cap),
                    "readme_boundary_only",
                    {"axis": c, "width_frac": frac, "cap": cap},
                )
    return out


def clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        x = float(v)
        return None if (math.isnan(x) or math.isinf(x)) else x
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-json", default=str(OUT_JSON_DEFAULT))
    ap.add_argument("--report-md", default=str(OUT_MD_DEFAULT))
    args = ap.parse_args()

    all_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    split_notes: dict[str, Any] = {}
    for split in SPLITS:
        df, notes = build_split_frame(split)
        split_notes[split] = notes
        base_pred = predict_tophalf(df, df["base_e128"].to_numpy(dtype=float))
        y = df["Label"].to_numpy(dtype=np.int8)
        base_acc = float((base_pred == y).mean())
        if abs(base_acc - BASE_EXPECTED[split]) > 1e-9:
            raise RuntimeError(f"{split}: base acc mismatch {base_acc:.12f} != {BASE_EXPECTED[split]:.12f}")
        base_rows.append({"split": split, "base_acc": base_acc})
        for variant, (score, family, params) in generate_variant_scores(df, notes["readme_cols"]).items():
            m = metric(df, score, base_pred)
            all_rows.append({"split": split, "variant": variant, "family": family, **params, **m})

    raw = pd.DataFrame(all_rows)
    grouped_rows: list[dict[str, Any]] = []
    for variant, g in raw.groupby("variant", sort=False):
        fixes = int(g["fixes"].sum())
        breaks = int(g["breaks"].sum())
        row = {
            "variant": variant,
            "family": str(g["family"].iloc[0]),
            "axis": str(g["axis"].iloc[0]),
            "mean_delta": float(g["delta"].mean()),
            "min_delta": float(g["delta"].min()),
            "max_delta": float(g["delta"].max()),
            "positive_splits": int((g["delta"] > 0).sum()),
            "fixes": fixes,
            "breaks": breaks,
            "discordant": int(g["discordant"].sum()),
            "changed": int(g["changed"].sum()),
            "p_exact": exact_two_sided_binom_p(min(fixes, breaks), fixes + breaks),
        }
        # Preserve common parameter columns when present.
        for extra in ("weight", "width_frac", "cap"):
            if extra in g.columns and not pd.isna(g[extra].iloc[0]):
                val = g[extra].iloc[0]
                row[extra] = float(val) if isinstance(val, (float, np.floating)) else int(val) if isinstance(val, (int, np.integer)) else val
        row["strict_pass"] = bool(row["positive_splits"] == len(SPLITS) and row["mean_delta"] >= MDE and row["fixes"] > row["breaks"] and row["p_exact"] < 0.05)
        row["manual_risk_signal"] = bool(row["positive_splits"] == len(SPLITS) and row["mean_delta"] > 0 and row["fixes"] > row["breaks"] and row["p_exact"] < 0.05)
        grouped_rows.append(row)

    grouped = pd.DataFrame(grouped_rows).sort_values(
        ["strict_pass", "manual_risk_signal", "mean_delta", "fixes"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    strict_pass_count = int(grouped["strict_pass"].sum())
    manual_risk_count = int(grouped["manual_risk_signal"].sum())
    verdict = "README_STRICT_PASS" if strict_pass_count else "README_MANUAL_RISK_ONLY" if manual_risk_count else "README_NO_SIGNAL"

    out = {
        "note": "In-bundle README insight application probe. Validation-only; no candidate CSV; no Kaggle submission.",
        "readme_sha256": "039a986734b47097be4cf0eea03ad3a8ce2adc2eaa56c920cbd3016c52f36576",
        "splits": list(SPLITS),
        "base": "emb128_L4_reg1e-3_4seed_LightGCN",
        "base_rows": base_rows,
        "mde": MDE,
        "verdict": verdict,
        "variant_count": int(len(grouped)),
        "strict_pass_count": strict_pass_count,
        "manual_risk_signal_count": manual_risk_count,
        "top_variants": grouped.head(50).to_dict(orient="records"),
        "split_notes": split_notes,
    }
    out_path = Path(args.report_json); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(clean(out), ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# README insight application probe",
        "",
        f"- verdict: **{verdict}**",
        f"- source README SHA256: `039a986734b47097be4cf0eea03ad3a8ce2adc2eaa56c920cbd3016c52f36576`",
        f"- base: `emb128 L4 reg1e-3 4-seed LightGCN`",
        f"- splits: `{', '.join(SPLITS)}`",
        f"- variants: `{len(grouped)}`",
        f"- strict pass count: `{strict_pass_count}` (requires all 3 splits positive, mean Δ ≥ {MDE:.5f}, fixes>breaks, p<0.05)",
        f"- manual-risk signal count: `{manual_risk_count}`",
        "- safety: validation-only; no hidden/test labels; no candidate CSV; no Kaggle submit.",
        "",
        "## README hints applied / reconciled",
        "",
        "| README hint | Applied here | Outcome |",
        "|---|---|---|",
        "| Per-user positive:negative = 1:1 | Base and all variants decode with per-user top-half on each validation user | hard constraint preserved |",
        "| Popularity + CF hybrid / alpha tuning | Exact README-style BPR/ALS + positive popularity alphas evaluated, then re-added as z/rank/raw residuals on the final LightGCN base | see table below |",
        "| hours_transformed confidence | `bpr_htr` / `als_htr` axes included in the residual grid | see table below |",
        "| LightGCN | current strongest LightGCN ensemble used as the base to test whether README residuals still add signal | base verified on all splits |",
        "| Ensemble | evaluated fixed weighted z/rank and boundary-only ensembles against the base; no submission artifact written | see verdict |",
        "",
        "## Base verification",
        "",
        "| split | base row acc | expected |",
        "|---|---:|---:|",
    ]
    for b in base_rows:
        lines.append(f"| `{b['split']}` | {b['base_acc']:.6f} | {BASE_EXPECTED[b['split']]:.6f} |")
    lines += [
        "",
        "## Standalone README-style BPR/ALS/pop floors",
        "",
        "| split | best standalone README axis | row acc | Δ vs base |",
        "|---|---|---:|---:|",
    ]
    for split in SPLITS:
        solo_items = split_notes[split]["solo_vs_base"].items()
        best_name, best_metric = max(solo_items, key=lambda kv: kv[1]["accuracy"])
        lines.append(f"| `{split}` | `{best_name}` | {best_metric['accuracy']:.6f} | {best_metric['delta']:+.6f} |")
    lines += [
        "",
        "## Top README-derived residual variants",
        "",
        "| rank | variant | family | axis | mean Δ | min~max Δ | splits+ | fixes | breaks | p | strict |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in enumerate(grouped.head(25).to_dict(orient="records"), 1):
        lines.append(
            f"| {i} | `{row['variant']}` | {row['family']} | `{row['axis']}` | "
            f"{row['mean_delta']:+.6f} | {row['min_delta']:+.6f}~{row['max_delta']:+.6f} | "
            f"{row['positive_splits']}/3 | {row['fixes']} | {row['breaks']} | {row['p_exact']:.4g} | {row['strict_pass']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "README의 구조 힌트(유저별 top-half, LightGCN, popularity+CF, hours confidence)는 모두 실제 파이프라인에 적용되었고, 이 프로브는 특히 final backbone 위에 남은 README residual을 다시 얹어본 검증이다.",
        "Mean Δ가 MDE 0.00355를 넘는 strict pass가 없으면, README에서 얻은 신호는 이미 LightGCN backbone에 흡수됐거나 popularity sampler artifact로 남은 것으로 본다.",
    ]
    md_path = Path(args.report_md); md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_path} and {md_path}")
    print(f"[verdict] {verdict}; strict_pass={strict_pass_count}; manual_risk={manual_risk_count}; variants={len(grouped)}")
    if len(grouped):
        top = grouped.iloc[0]
        print(
            f"[top] {top['variant']} mean_delta={top['mean_delta']:+.6f} "
            f"splits={int(top['positive_splits'])}/3 fixes={int(top['fixes'])} breaks={int(top['breaks'])} p={top['p_exact']:.4g}"
        )


if __name__ == "__main__":
    main()
