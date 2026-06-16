#!/usr/bin/env python3
"""BSVD boundary seed-vote decoder probe (validation-only).

This script evaluates one fixed post-hoc decoder variant over the calibrated
uniform validation panel. It never reads hidden test pairs and never writes a
submission/candidate artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import evaluate_tophalf, predict_tophalf  # noqa: E402

SPLITS = [
    "val_random_uniform_seed42",
    "val_random_uniform_seed7",
    "val_random_uniform_seed123",
]
MODEL_SEEDS = [42, 7, 123, 2024]
EXPECTED_BASE_ACCURACY = {
    "val_random_uniform_seed42": 0.7650530106021204,
    "val_random_uniform_seed7": 0.7609521904380876,
    "val_random_uniform_seed123": 0.7599519903980796,
}
BASE_SCORE_COL = "score_base_mean"
VAR_SCORE_COL = "score_bsvd_boundary_vote_w10cap20"
MDE = 0.00355


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def exact_two_sided_binom_p(k: int, n: int) -> float:
    """Exact two-sided binomial p-value for p=0.5 using log-combinations."""
    if n <= 0:
        return 1.0
    kk = min(k, n - k)
    # symmetric tails: 2 * P[X <= kk], capped at 1.0
    logs = [_log_choose(n, i) - n * math.log(2.0) for i in range(kk + 1)]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(x - m) for x in logs)
    return min(1.0, 2.0 * tail)


def source_path_for(split: str, seed: int) -> tuple[Path, str]:
    """Return validation-only source path and score column for a split/model seed."""
    if split == "val_random_uniform_seed42":
        if seed == 42:
            # Same canonical emb128/L4/reg=1e-3/seed42 checkpoint; layer-mix uniform
            # equals the ordinary LightGCN h0..h4 uniform average. This source keeps
            # the seed42 score aligned with the known 4-seed base acc 0.765053.
            return (
                ROOT / "artifacts/layermix_probe/emb128_L4_r3_seed42/layermix_validation_scores.csv",
                "score_layermix_uniform",
            )
        return (
            ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}/{split}/lightgcn_scores.csv",
            "score_lightgcn",
        )
    return (
        ROOT / f"artifacts/split_panel_emb128/{split}/seed{seed}/lightgcn_scores.csv",
        "score_lightgcn",
    )


def load_split_scores(split: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for seed in MODEL_SEEDS:
        path, col = source_path_for(split, seed)
        path_s = str(path)
        lowered = path_s.lower()
        if any(token in lowered for token in ["test_candidate", "submission"]):
            raise RuntimeError(f"Forbidden source path for validation probe: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Missing score source for {split} seed{seed}: {path}")
        df = pd.read_csv(path)
        if col not in df.columns:
            raise ValueError(f"Missing score column {col} in {path}")
        keep = ["ID", "userID", "gameID", "Label", col] if merged is None else ["ID", col]
        part = df[keep].copy().rename(columns={col: f"score_seed{seed}"})
        if merged is None:
            merged = part
        else:
            before = len(merged)
            merged = merged.merge(part, on="ID", how="inner", validate="one_to_one")
            if len(merged) != before:
                raise RuntimeError(f"Row alignment changed for {split} seed{seed}: {before} -> {len(merged)}")
    assert merged is not None
    required = {"ID", "userID", "gameID", "Label", *[f"score_seed{s}" for s in MODEL_SEEDS]}
    missing = required - set(merged.columns)
    if missing:
        raise ValueError(f"{split}: missing columns {sorted(missing)}")
    merged = merged.sort_values("ID", kind="mergesort").reset_index(drop=True)
    merged["ID"] = merged["ID"].astype(int)
    merged["Label"] = merged["Label"].astype(int)
    merged[BASE_SCORE_COL] = merged[[f"score_seed{s}" for s in MODEL_SEEDS]].mean(axis=1)
    return merged


def add_rank(df: pd.DataFrame, score_col: str, rank_col: str) -> pd.DataFrame:
    ordered = df.sort_values(["userID", score_col, "ID"], ascending=[True, False, True], kind="mergesort")
    ranks = ordered.groupby("userID", sort=False).cumcount() + 1
    out = df.copy()
    out.loc[ordered.index, rank_col] = ranks.to_numpy()
    out[rank_col] = out[rank_col].astype(int)
    return out


def check_validation_cardinality(df: pd.DataFrame, split: str) -> None:
    g = df.groupby("userID", sort=False).agg(n=("ID", "size"), positives=("Label", "sum"))
    bad_even = g[g["n"] % 2 != 0]
    bad_half = g[g["positives"] * 2 != g["n"]]
    if not bad_even.empty or not bad_half.empty:
        raise RuntimeError(
            f"{split}: invalid per-user 1:1 validation cardinality "
            f"odd_users={len(bad_even)} half_mismatch_users={len(bad_half)}"
        )


def add_bsvd_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seed_cols = [f"score_seed{s}" for s in MODEL_SEEDS]
    for seed, col in zip(MODEL_SEEDS, seed_cols, strict=True):
        out = add_rank(out, col, f"rank_seed{seed}")
    out = add_rank(out, BASE_SCORE_COL, "rank_base")

    out["n_user"] = out.groupby("userID", sort=False)["ID"].transform("size").astype(int)
    out["h_user"] = out["n_user"] // 2
    out["w_user"] = np.minimum(20, np.maximum(3, np.ceil(0.10 * out["h_user"]).astype(int)))

    votes = np.zeros(len(out), dtype=np.int16)
    for seed in MODEL_SEEDS:
        votes += (out[f"rank_seed{seed}"].to_numpy() <= out["h_user"].to_numpy()).astype(np.int16)
    out["seed_tophalf_vote"] = votes
    rank_cols = [f"rank_seed{s}" for s in MODEL_SEEDS]
    out["mean_seed_rank"] = out[rank_cols].mean(axis=1)

    grouped = out.groupby("userID", sort=False)[BASE_SCORE_COL]
    mean = grouped.transform("mean")
    std = grouped.transform(lambda s: float(s.std(ddof=0)))
    out["z_base_user"] = np.where(std.to_numpy() > 1e-12, (out[BASE_SCORE_COL] - mean) / std.replace(0, np.nan), 0.0)
    out["z_base_user"] = out["z_base_user"].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    lo = out["h_user"] - out["w_user"] + 1
    hi = out["h_user"] + out["w_user"]
    top_safe = out["rank_base"] < lo
    bottom_safe = out["rank_base"] > hi
    boundary = ~(top_safe | bottom_safe)
    out["is_bsvd_boundary"] = boundary.astype(np.int8)

    denom = (out["n_user"] - 1).replace(0, 1).astype(float)
    inverse_mean_rank = 1.0 - ((out["mean_seed_rank"] - 1.0) / denom)
    boundary_score = out["seed_tophalf_vote"].astype(float) + 1e-3 * inverse_mean_rank + 1e-6 * out["z_base_user"]

    final_score = np.empty(len(out), dtype=float)
    # Anchors intentionally make boundary-only swaps possible: all top-safe rows
    # remain above the band and all bottom-safe rows remain below it.
    final_score[top_safe.to_numpy()] = 10.0 + 1e-6 * (out.loc[top_safe, "n_user"] - out.loc[top_safe, "rank_base"])
    final_score[boundary.to_numpy()] = boundary_score.loc[boundary]
    final_score[bottom_safe.to_numpy()] = -10.0 + 1e-6 * (out.loc[bottom_safe, "n_user"] - out.loc[bottom_safe, "rank_base"])
    out[VAR_SCORE_COL] = final_score
    return out


def evaluate_pair(df: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    base_summary, base_pred = evaluate_tophalf(df, BASE_SCORE_COL, label_col="Label", user_col="userID", id_col="ID")
    var_summary, var_pred = evaluate_tophalf(df, VAR_SCORE_COL, label_col="Label", user_col="userID", id_col="ID")
    base = base_pred[["ID", "Pred", "Correct", "rank_in_user"]].rename(
        columns={"Pred": "pred_base", "Correct": "correct_base", "rank_in_user": "rank_base_decode"}
    )
    var = var_pred[["ID", "Pred", "Correct", "rank_in_user"]].rename(
        columns={"Pred": "pred_bsvd", "Correct": "correct_bsvd", "rank_in_user": "rank_bsvd_decode"}
    )
    pred = df.merge(base, on="ID", how="inner", validate="one_to_one").merge(var, on="ID", how="inner", validate="one_to_one")
    pred = pred.sort_values("ID", kind="mergesort")

    base_ok = pred["correct_base"].astype(bool).to_numpy()
    var_ok = pred["correct_bsvd"].astype(bool).to_numpy()
    fixes = int((~base_ok & var_ok).sum())
    breaks = int((base_ok & ~var_ok).sum())
    flips = fixes + breaks
    changed = int((pred["pred_base"].astype(int) != pred["pred_bsvd"].astype(int)).sum())
    metrics = {
        "base_accuracy": float(base_summary["row_accuracy"]),
        "variant_accuracy": float(var_summary["row_accuracy"]),
        "delta": float(var_summary["row_accuracy"] - base_summary["row_accuracy"]),
        "fixes": fixes,
        "breaks": breaks,
        "flips": flips,
        "changed_predictions": changed,
        "paired_exact_binom_p_two_sided": exact_two_sided_binom_p(fixes, flips),
        "direction_positive": bool(fixes > breaks),
        "base_predicted_positive_total": int(base_summary["predicted_positive_total"]),
        "variant_predicted_positive_total": int(var_summary["predicted_positive_total"]),
        "all_user_positive_counts_match_base": bool(base_summary["all_user_positive_counts_match"]),
        "all_user_positive_counts_match_variant": bool(var_summary["all_user_positive_counts_match"]),
    }
    return metrics, pred


def boundary_oracle_upper_bound(pred: pd.DataFrame) -> dict[str, Any]:
    total_gain = 0
    boundary_rows = 0
    boundary_users = 0
    for _, g in pred.groupby("userID", sort=False):
        b = g[g["is_bsvd_boundary"] == 1].copy()
        if b.empty:
            continue
        boundary_users += 1
        boundary_rows += len(b)
        k_boundary_pos = int((b["pred_base"].astype(int) == 1).sum())
        base_correct = int((b["pred_base"].astype(int) == b["Label"].astype(int)).sum())
        # Diagnostic only: among the fixed boundary rows, choose exactly as many
        # positives as baseline selected inside the band, using labels. This is
        # never used to tune or gate a variant, only to quantify remaining budget.
        oracle = b.sort_values(["Label", "rank_base", "ID"], ascending=[False, True, True], kind="mergesort")
        oracle_pos_ids = set(oracle.head(k_boundary_pos)["ID"].tolist())
        oracle_pred = b["ID"].isin(oracle_pos_ids).astype(int)
        oracle_correct = int((oracle_pred.to_numpy() == b["Label"].astype(int).to_numpy()).sum())
        total_gain += oracle_correct - base_correct
    return {
        "boundary_rows": int(boundary_rows),
        "boundary_users": int(boundary_users),
        "boundary_oracle_net_gain_rows": int(total_gain),
        "boundary_oracle_delta_upper_bound": float(total_gain / len(pred)) if len(pred) else 0.0,
    }


def run_split(split: str, out_root: Path, write_scores: bool = True) -> dict[str, Any]:
    df = load_split_scores(split)
    check_validation_cardinality(df, split)
    df = add_bsvd_score(df)
    metrics, pred = evaluate_pair(df)
    metrics.update(boundary_oracle_upper_bound(pred))
    metrics["split"] = split
    metrics["variant"] = VAR_SCORE_COL
    metrics["validation_only"] = True
    metrics["candidate_csv_written"] = False
    metrics["kaggle_submit_executed"] = False

    expected = EXPECTED_BASE_ACCURACY[split]
    if abs(metrics["base_accuracy"] - expected) > 1e-12:
        raise RuntimeError(f"{split}: base accuracy mismatch {metrics['base_accuracy']} != expected {expected}")

    # Hard anchor invariant: if a prediction changed, it must be a boundary row.
    changed_mask = pred["pred_base"].astype(int) != pred["pred_bsvd"].astype(int)
    non_boundary_changed = int(((pred["is_bsvd_boundary"].astype(int) == 0) & changed_mask).sum())
    metrics["non_boundary_changed_predictions"] = non_boundary_changed
    if non_boundary_changed != 0:
        raise RuntimeError(f"{split}: BSVD changed {non_boundary_changed} non-boundary predictions")

    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "metrics.json", metrics)
    if write_scores:
        keep = [
            "ID",
            "userID",
            "gameID",
            "Label",
            BASE_SCORE_COL,
            VAR_SCORE_COL,
            "pred_base",
            "pred_bsvd",
            "rank_base",
            "rank_base_decode",
            "rank_bsvd_decode",
            "seed_tophalf_vote",
            "mean_seed_rank",
            "h_user",
            "w_user",
            "is_bsvd_boundary",
        ] + [f"score_seed{s}" for s in MODEL_SEEDS]
        pred[keep].to_csv(out_dir / "bsvd_validation_scores.csv", index=False)
    return metrics


def aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    mean_delta = float(np.mean([m["delta"] for m in metrics]))
    pooled_fixes = int(sum(m["fixes"] for m in metrics))
    pooled_breaks = int(sum(m["breaks"] for m in metrics))
    pooled_flips = pooled_fixes + pooled_breaks
    positive_splits = int(sum(1 for m in metrics if m["fixes"] > m["breaks"]))
    p = exact_two_sided_binom_p(pooled_fixes, pooled_flips)
    gate_pass = bool(mean_delta >= MDE and pooled_fixes > pooled_breaks and p < 0.05 and positive_splits >= 2)
    return {
        "experiment": "BSVD boundary seed-vote decoder",
        "variant": VAR_SCORE_COL,
        "validation_only": True,
        "candidate_csv_written": False,
        "kaggle_submit_executed": False,
        "gate_policy": {
            "fixed_variant_only": True,
            "mean_delta_min": MDE,
            "pooled_fixes_gt_breaks": True,
            "paired_exact_p_lt": 0.05,
            "direction_splits_min": "2/3",
            "no_split_cherry_pick": True,
            "no_candidate_test_submission_csv": True,
        },
        "panel": {
            "mean_delta": mean_delta,
            "mean_base_accuracy": float(np.mean([m["base_accuracy"] for m in metrics])),
            "mean_variant_accuracy": float(np.mean([m["variant_accuracy"] for m in metrics])),
            "positive_direction_splits": positive_splits,
            "pooled_fixes": pooled_fixes,
            "pooled_breaks": pooled_breaks,
            "pooled_flips": pooled_flips,
            "pooled_exact_binom_p_two_sided": p,
            "gate_pass": gate_pass,
            "verdict": "PASS" if gate_pass else "REJECT",
        },
        "splits": metrics,
    }


def write_report(panel: dict[str, Any], md_path: Path, json_path: Path, table_path: Path) -> None:
    _write_json(json_path, panel)
    p = panel["panel"]
    lines: list[str] = []
    lines.append("# BSVD Boundary Seed-Vote Decoder — 3-Split Panel Aggregate")
    lines.append("")
    lines.append("## 결론")
    if p["gate_pass"]:
        lines.append("- **판정: PASS.** fixed BSVD variant가 predeclared panel gate를 통과했습니다. 단, 제출 전 별도 사용자 승인 필요.")
    else:
        lines.append("- **판정: REJECT / NO-SUBMIT.** fixed BSVD variant가 predeclared panel gate를 통과하지 못했습니다.")
    lines.append(
        f"- mean Δ={p['mean_delta']:+.6f}, fixes/breaks={p['pooled_fixes']}/{p['pooled_breaks']}, "
        f"p={p['pooled_exact_binom_p_two_sided']:.6g}, direction={p['positive_direction_splits']}/3."
    )
    lines.append("- 범위: validation-only. candidate/test/submission CSV 생성 및 Kaggle submit 없음.")
    lines.append("")
    lines.append("## Split Results")
    lines.append("| split | base acc | BSVD acc | Δ | fixes | breaks | p | changed | boundary oracle UB | verdict |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for m in panel["splits"]:
        verdict = "PASS" if (m["delta"] >= MDE and m["fixes"] > m["breaks"] and m["paired_exact_binom_p_two_sided"] < 0.05) else "REJECT"
        lines.append(
            f"| {m['split']} | {m['base_accuracy']:.6f} | {m['variant_accuracy']:.6f} | "
            f"{m['delta']:+.6f} | {m['fixes']} | {m['breaks']} | "
            f"{m['paired_exact_binom_p_two_sided']:.6g} | {m['changed_predictions']} | "
            f"{m['boundary_oracle_delta_upper_bound']:+.6f} | {verdict} |"
        )
    lines.append("")
    lines.append("## Gate Policy")
    lines.append(f"- fixed variant only: `{VAR_SCORE_COL}`")
    lines.append(f"- mean Δ ≥ {MDE:+.5f}")
    lines.append("- pooled fixes > breaks and exact paired/binomial p < 0.05")
    lines.append("- direction split ≥ 2/3")
    lines.append("- split별 best, width tuning, tie-break tuning 금지")
    lines.append("- no candidate/test/submission CSV; no public LB probing")
    lines.append("")
    lines.append("## 해석")
    if not p["gate_pass"]:
        lines.append("- BSVD는 seed-level top-half vote를 boundary band에만 적용하는 독립적인 decoder probe였지만, panel gate를 통과하지 못했다면 마지막 제출권을 쓸 근거가 없다.")
        lines.append("- boundary oracle UB는 라벨을 사용한 진단 상한일 뿐이며, variant 선택/튜닝에 사용하지 않았다.")
    else:
        lines.append("- 통과 시에도 이 산출물은 validation-only probe이며, full-test candidate 생성/제출은 사용자 승인 후 별도 preflight가 필요하다.")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    table_lines = [
        "| split | base acc | BSVD acc | Δ | fixes | breaks | p | oracle UB | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in panel["splits"]:
        table_lines.append(
            f"| {m['split'].replace('val_random_uniform_', '')} | {m['base_accuracy']:.6f} | "
            f"{m['variant_accuracy']:.6f} | {m['delta']:+.6f} | {m['fixes']} | {m['breaks']} | "
            f"{m['paired_exact_binom_p_two_sided']:.3g} | {m['boundary_oracle_delta_upper_bound']:+.6f} | REJECT |"
        )
    table_lines.append(
        f"| aggregate | {p['mean_base_accuracy']:.6f} | {p['mean_variant_accuracy']:.6f} | "
        f"{p['mean_delta']:+.6f} | {p['pooled_fixes']} | {p['pooled_breaks']} | "
        f"{p['pooled_exact_binom_p_two_sided']:.3g} | — | {p['verdict']} |"
    )
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default=str(ROOT / "artifacts/bsvd_boundary_vote"))
    ap.add_argument("--report-md", default=str(ROOT / "reports/20260601_bsvd_boundary_vote_panel_aggregate.md"))
    ap.add_argument("--report-json", default=str(ROOT / "reports/20260601_bsvd_boundary_vote_panel_aggregate.json"))
    ap.add_argument("--table-md", default=str(ROOT / "reports/20260601_bsvd_boundary_vote_panel_table.md"))
    ap.add_argument("--no-score-csv", action="store_true", help="Do not write validation score CSV artifacts")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    lowered = str(out_root).lower()
    if any(token in lowered for token in ["submission", "test_candidate"]):
        raise RuntimeError(f"Forbidden output root for validation probe: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    split_metrics = []
    for split in SPLITS:
        print(f"[BSVD] running {split}", flush=True)
        m = run_split(split, out_root, write_scores=not args.no_score_csv)
        split_metrics.append(m)
        print(
            f"  base={m['base_accuracy']:.6f} bsvd={m['variant_accuracy']:.6f} "
            f"delta={m['delta']:+.6f} fixes/breaks={m['fixes']}/{m['breaks']} "
            f"p={m['paired_exact_binom_p_two_sided']:.6g} oracle_ub={m['boundary_oracle_delta_upper_bound']:+.6f}",
            flush=True,
        )
    panel = aggregate(split_metrics)
    write_report(panel, Path(args.report_md), Path(args.report_json), Path(args.table_md))
    print(json.dumps(panel["panel"], indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
