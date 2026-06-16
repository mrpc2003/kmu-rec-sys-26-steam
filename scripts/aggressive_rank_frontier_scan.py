#!/usr/bin/env python3
"""Aggressive validation-only rank/z frontier scan for KMURecSys26 Steam.

This is a *risk-frontier* scan, not a strict-submit gate. It intentionally tries
more aggressive fixed combinations around the current LightGCN capacity family:
emb64, emb128, emb192 4-seed ensembles on the calibrated seed42 uniform split.

Safety:
- validation-only: reads `artifacts/.../val_random_uniform_seed42/lightgcn_scores.csv` only;
- no hidden labels, no real test pairs, no candidate/submission CSV materialization;
- uses a predeclared finite operator/weight grid and reports the multiplicity risk.

The goal is to identify whether any aggressive fixed rule is worth escalating to a
3-split panel or user-approved manual-risk candidate materialization.
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
SPLIT = "val_random_uniform_seed42"
SEEDS = (42, 123, 2024, 7)
BASE_REF_ACC = 0.7650530106021204
PUBLIC_BEST = 0.77825
MDE = 0.00355
OUT_DIR_DEFAULT = ROOT / "artifacts/aggressive_rank_frontier_seed42"
REPORT_JSON_DEFAULT = ROOT / "reports/20260602_aggressive_rank_frontier_seed42.json"
REPORT_MD_DEFAULT = ROOT / "reports/20260602_aggressive_rank_frontier_seed42.md"


def exact_two_sided_binom_p(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    kk = min(k, n - k)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0) for i in range(kk + 1)]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(v - m) for v in logs)
    return min(1.0, 2.0 * tail)


def score_col(df: pd.DataFrame) -> str:
    for c in ("score_layermix_uniform", "score_lightgcn", "score"):
        if c in df.columns:
            return c
    raise ValueError(f"No score column in {df.columns.tolist()}")


def path_for(axis: str, seed: int) -> Path:
    if axis == "e64":
        if seed == 42:
            return ROOT / "artifacts/lightgcn_ood_robustness" / SPLIT / "lightgcn_scores.csv"
        return ROOT / f"artifacts/lightgcn_uniform_eval/seed{seed}" / SPLIT / "lightgcn_scores.csv"
    if axis == "e128":
        if seed == 42:
            return ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / SPLIT / "lightgcn_scores.csv"
        return ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}" / SPLIT / "lightgcn_scores.csv"
    if axis == "e192":
        base = "emb192_L4_r3" if seed == 42 else f"emb192_L4_r3_seed{seed}"
        return ROOT / "artifacts/capacity_uniform" / base / SPLIT / "lightgcn_scores.csv"
    raise ValueError(axis)


def load_axis(axis: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    score_cols: list[str] = []
    for seed in SEEDS:
        p = path_for(axis, seed)
        if not p.exists():
            raise FileNotFoundError(f"Missing {axis} seed={seed}: {p}")
        d = pd.read_csv(p)
        sc = score_col(d)
        need = {"ID", "userID", "gameID", "Label", sc}
        if not need.issubset(d.columns):
            raise ValueError(f"Missing columns in {p}: {need - set(d.columns)}")
        col = f"{axis}_seed{seed}"
        part = d[["ID", "userID", "gameID", "Label", sc]].rename(columns={sc: col})
        if merged is None:
            merged = part
        else:
            before = len(merged)
            merged = merged.merge(part[["ID", col]], on="ID", how="inner", validate="one_to_one")
            if len(merged) != before:
                raise RuntimeError(f"Row alignment changed while merging {p}")
        score_cols.append(col)
    assert merged is not None
    merged[axis] = merged[score_cols].mean(axis=1)
    return merged[["ID", "userID", "gameID", "Label", axis]]


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
    # Larger rank = higher original score. This matches the previous rank-blend convention.
    ranks = np.zeros(len(df), dtype=float)
    values = df[col].to_numpy(dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        order_low_to_high = np.argsort(values[idx], kind="mergesort")
        ranks[idx[order_low_to_high]] = np.arange(len(idx), dtype=float)
    return ranks


def robust_rank01(df: pd.DataFrame, col: str) -> np.ndarray:
    r = within_user_rank_high(df, col)
    n = df.groupby("userID", sort=False)["ID"].transform("size").to_numpy(dtype=float)
    return np.divide(r, np.maximum(n - 1.0, 1.0))


def predict_tophalf(df: pd.DataFrame, score_values: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    score_values = np.asarray(score_values, dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        k = len(idx) // 2
        order = np.argsort(score_values[idx], kind="mergesort")[::-1]
        pred[idx[order[:k]]] = 1
    return pred


def metric(df: pd.DataFrame, score_values: np.ndarray, base_pred: np.ndarray) -> dict[str, Any]:
    pred = predict_tophalf(df, score_values)
    y = df["Label"].to_numpy(dtype=np.int8)
    ok = pred == y
    base_ok = base_pred == y
    fixes = int((~base_ok & ok).sum())
    breaks = int((base_ok & ~ok).sum())
    discord = fixes + breaks
    return {
        "accuracy": float(ok.mean()),
        "delta_vs_emb128": float(ok.mean() - base_ok.mean()),
        "fixes": fixes,
        "breaks": breaks,
        "discordant": discord,
        "p_exact_two_sided": exact_two_sided_binom_p(fixes, discord),
        "changed_rows_vs_emb128": int((pred != base_pred).sum()),
    }


def anchor_boundary_score(base_score: np.ndarray, variant_score: np.ndarray, df: pd.DataFrame, width_frac: float, cap: int) -> np.ndarray:
    """Use variant only in a fixed boundary band, keep outside rows anchored."""
    out = np.zeros(len(df), dtype=float)
    base = np.asarray(base_score, dtype=float)
    var = np.asarray(variant_score, dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        n = len(idx)
        h = n // 2
        w = min(cap, max(1, int(math.ceil(width_frac * h))))
        # base order high to low
        order = idx[np.argsort(base[idx], kind="mergesort")[::-1]]
        # Outside boundary gets large monotone anchors. Boundary reorders by variant.
        ranks = np.empty(n, dtype=int)
        ranks[np.argsort(base[idx], kind="mergesort")[::-1]] = np.arange(n)
        local_out = -ranks.astype(float) * 1000.0
        lo = max(0, h - w)
        hi = min(n, h + w)
        band_ids = order[lo:hi]
        # Variant z inside band, shifted into the anchored slot range.
        if len(band_ids):
            v = var[band_ids]
            if float(np.std(v)) > 1e-12:
                vz = (v - float(np.mean(v))) / float(np.std(v))
            else:
                vz = np.zeros(len(band_ids), dtype=float)
            # anchor around -h*1000; only reorders inside the band.
            local_positions = {rid: j for j, rid in enumerate(idx)}
            for rid, val in zip(band_ids, vz, strict=True):
                local_out[local_positions[rid]] = -h * 1000.0 + val
        out[idx] = local_out
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--report-json", default=str(REPORT_JSON_DEFAULT))
    ap.add_argument("--report-md", default=str(REPORT_MD_DEFAULT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_axis("e128")
    for axis in ("e64", "e192"):
        df = df.merge(load_axis(axis)[["ID", axis]], on="ID", how="inner", validate="one_to_one")
    df = df.sort_values("ID", kind="mergesort").reset_index(drop=True)

    # Canonical base: emb128 raw-mean top-half.
    base_pred = predict_tophalf(df, df["e128"].to_numpy(dtype=float))
    base_acc = float((base_pred == df["Label"].to_numpy(dtype=np.int8)).mean())
    if abs(base_acc - BASE_REF_ACC) > 1e-9:
        raise RuntimeError(f"emb128 base mismatch: {base_acc:.12f} != {BASE_REF_ACC:.12f}")

    # Precompute normalized axes.
    for axis in ("e64", "e128", "e192"):
        df[f"z_{axis}"] = within_user_z(df, axis)
        df[f"r_{axis}"] = within_user_rank_high(df, axis)
        df[f"r01_{axis}"] = robust_rank01(df, axis)

    rows: list[dict[str, Any]] = []

    def add(name: str, score: np.ndarray, family: str, params: dict[str, Any]) -> None:
        m = metric(df, score, base_pred)
        rows.append({"variant": name, "family": family, **params, **m})

    # Fixed weighted rank/z grids around the current public-best rank-blend story.
    weight_grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    for w192 in weight_grid:
        for w64 in weight_grid:
            if w192 == 0 and w64 == 0:
                continue
            # keep emb128 anchor weight fixed at 1; no negative weights.
            add(
                f"rank_w128_1_w192_{w192:g}_w64_{w64:g}",
                df["r_e128"].to_numpy() + w192 * df["r_e192"].to_numpy() + w64 * df["r_e64"].to_numpy(),
                "weighted_rank_sum",
                {"w128": 1.0, "w192": w192, "w64": w64},
            )
            add(
                f"z_w128_1_w192_{w192:g}_w64_{w64:g}",
                df["z_e128"].to_numpy() + w192 * df["z_e192"].to_numpy() + w64 * df["z_e64"].to_numpy(),
                "weighted_z_sum",
                {"w128": 1.0, "w192": w192, "w64": w64},
            )

    # RRF-like fixed operators: emphasize near-boundary rank but with a stable formula.
    for k in (5, 10, 20, 50):
        # Smaller denominator means stronger top emphasis; r01 high -> convert to low rank approx.
        n = df.groupby("userID", sort=False)["ID"].transform("size").to_numpy(dtype=float)
        lowrank128 = (n - 1.0) - df["r_e128"].to_numpy()
        lowrank192 = (n - 1.0) - df["r_e192"].to_numpy()
        lowrank64 = (n - 1.0) - df["r_e64"].to_numpy()
        add(f"rrf_128_192_k{k}", 1/(k+lowrank128) + 1/(k+lowrank192), "rrf", {"axes": "128+192", "k": k})
        add(f"rrf_128_192_64_k{k}", 1/(k+lowrank128) + 1/(k+lowrank192) + 1/(k+lowrank64), "rrf", {"axes": "128+192+64", "k": k})

    # Boundary-only variants: take an aggressive score but only near emb128 cutoff.
    candidate_scores = {
        "rank_128_192": df["r_e128"].to_numpy() + df["r_e192"].to_numpy(),
        "rank_128_192_64": df["r_e128"].to_numpy() + df["r_e192"].to_numpy() + df["r_e64"].to_numpy(),
        "z_128_192": df["z_e128"].to_numpy() + df["z_e192"].to_numpy(),
        "z_128_192_64": df["z_e128"].to_numpy() + df["z_e192"].to_numpy() + df["z_e64"].to_numpy(),
    }
    for cname, cscore in candidate_scores.items():
        for frac in (0.05, 0.10, 0.20):
            for cap in (5, 10, 20):
                add(
                    f"boundary_{cname}_frac{frac:g}_cap{cap}",
                    anchor_boundary_score(df["e128"].to_numpy(dtype=float), cscore, df, width_frac=frac, cap=cap),
                    "boundary_only",
                    {"inner_score": cname, "width_frac": frac, "cap": cap},
                )

    res = pd.DataFrame(rows).sort_values(["delta_vs_emb128", "fixes"], ascending=[False, False], kind="mergesort")
    res.to_csv(out_dir / "aggressive_rank_frontier_metrics.csv", index=False)

    top = res.head(30).to_dict(orient="records")
    payload = {
        "safety": {"validation_only": True, "hidden_test_read": False, "candidate_csv_written": False, "kaggle_submit_executed": False},
        "split": SPLIT,
        "base": {"variant": "emb128_4seed_raw_mean", "accuracy": base_acc, "public_reference": 0.77745},
        "current_public_best": PUBLIC_BEST,
        "mde": MDE,
        "variant_count": int(len(res)),
        "top_variants": top,
        "strict_gate_pass_count_seed42_only": int(((res["delta_vs_emb128"] >= MDE) & (res["fixes"] > res["breaks"]) & (res["p_exact_two_sided"] < 0.05)).sum()),
        "note": "Seed42-only aggressive frontier. Any candidate from this report is manual-risk and must be rechecked on independent splits if possible; no submission/candidate file was produced.",
        "output_csv": str(out_dir / "aggressive_rank_frontier_metrics.csv"),
    }
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Aggressive rank/z frontier scan — validation only\n\n",
        "## Verdict context\n",
        f"- split: `{SPLIT}`\n",
        f"- base emb128 acc: `{base_acc:.6f}`\n",
        f"- variants scanned: `{len(res)}`\n",
        f"- seed42-only strict gate pass count: `{payload['strict_gate_pass_count_seed42_only']}`\n",
        "- safety: no hidden/test read, no candidate CSV, no Kaggle submit.\n\n",
        "## Top variants\n",
        "| rank | variant | family | Δ vs emb128 | fixes | breaks | p | changed |\n",
        "|---:|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for i, r in enumerate(top[:20], 1):
        lines.append(
            f"| {i} | `{r['variant']}` | {r['family']} | {r['delta_vs_emb128']:+.6f} | {r['fixes']} | {r['breaks']} | {r['p_exact_two_sided']:.4g} | {r['changed_rows_vs_emb128']} |\n"
        )
    lines.append("\n## Interpretation\n")
    best = top[0]
    if best["delta_vs_emb128"] >= MDE and best["p_exact_two_sided"] < 0.05:
        lines.append("Seed42에서 MDE를 넘는 공격 후보가 발견됐다. 그러나 단일 split이므로 곧바로 제출하지 말고 가능한 독립 split/paired panel로 재검증해야 한다.\n")
    elif best["delta_vs_emb128"] > 0:
        lines.append("공격 후보는 양수지만 MDE 미만이다. rank-blend public 개선처럼 manual-risk 후보는 될 수 있으나, 통계적으로는 noise-chasing 범주다.\n")
    else:
        lines.append("공격 grid에서도 emb128을 넘는 후보가 없다. 추가 public-LB형 조합은 근거가 약하다.\n")
    Path(args.report_md).write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"report_json": args.report_json, "report_md": args.report_md, "best": top[0]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
