#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recsys_played_utils import build_user_item_matrix, ensure_dir, evaluate_tophalf, load_pairs_csv, load_train_interactions, write_json

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
SPLIT = "val_random_uniform_seed42"
BASE_REF = 0.7650530106021204
SOLO_GATE = 0.735
DELTA_GATE = 0.0007
PASS_BLEND = BASE_REF + DELTA_GATE
NOISE_BAND = 0.0007
DESCRIPTION = "BM25 + Logistic Matrix Factorization seed42 validation-only smoke. No submit, no submissions writes, no full-test materialization."
SEED_SCORE_PATHS = {
    42: ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03/val_random_uniform_seed42/lightgcn_scores.csv",
    123: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed123/val_random_uniform_seed42/lightgcn_scores.csv",
    2024: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed2024/val_random_uniform_seed42/lightgcn_scores.csv",
    7: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed7/val_random_uniform_seed42/lightgcn_scores.csv",
}


def module(name: str) -> Any:
    return importlib.import_module(name)


def assert_not_submissions(path: Path) -> None:
    resolved = path.resolve()
    submissions = (ROOT / "submissions").resolve()
    if resolved == submissions or submissions in resolved.parents:
        raise ValueError(f"Refusing to write under submissions/: {path}")


def finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): finite_or_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_or_none(v) for v in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def within_user_z(df: Any, col: str, pd: Any, np: Any) -> Any:
    grouped = df.groupby("userID", sort=False)[col]
    std = grouped.transform("std").replace(0, np.nan).fillna(1.0)
    return ((df[col] - grouped.transform("mean")) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def within_user_pct_rank(df: Any, col: str, np: Any) -> Any:
    grouped = df.groupby("userID", sort=False)[col]
    denom = grouped.transform("size") - 1
    rank = grouped.rank(method="first", ascending=False)
    return np.where(denom > 0, (rank - 1) / denom, 0.0)


def load_base_scores(pd: Any, np: Any) -> Any:
    merged = None
    seed_cols: list[str] = []
    for seed, path in SEED_SCORE_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        col = f"score_emb128_seed{seed}"
        seed_cols.append(col)
        part = pd.read_csv(path)[["ID", "userID", "gameID", "Label", "score_lightgcn"]].rename(columns={"score_lightgcn": col})
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part[["ID", col]], on="ID", validate="one_to_one")
    if merged is None:
        raise RuntimeError("No base score files loaded")
    merged["score_emb128_4seed"] = merged[seed_cols].mean(axis=1).astype(np.float32)
    return merged[["ID", "score_emb128_4seed"]]


def candidate_scores(model: Any, candidates: Any, user_to_idx: dict[str, int], item_to_idx: dict[str, int], np: Any) -> Any:
    user_factors = np.asarray(model.user_factors)
    item_factors = np.asarray(model.item_factors)
    scores = np.full(len(candidates), -1e30, dtype=np.float32)
    for n, row in enumerate(candidates[["userID", "gameID"]].astype(str).itertuples(index=False)):
        ui = user_to_idx.get(row.userID)
        ii = item_to_idx.get(row.gameID)
        if ui is not None and ii is not None:
            scores[n] = float(user_factors[ui] @ item_factors[ii])
    return scores


def label_predictions(df: Any, score_col: str) -> Any:
    _, pred = evaluate_tophalf(df, score_col, label_col="Label", user_col="userID", id_col="ID")
    return pred[["ID", "Pred", "Correct", "rank_in_user"]].rename(
        columns={"Pred": f"pred_{score_col}", "Correct": f"correct_{score_col}", "rank_in_user": f"rank_{score_col}"}
    )


def prediction_delta(base_pred: Any, candidate_pred: Any, pd: Any) -> dict[str, Any]:
    merged = base_pred.merge(candidate_pred, on="ID", validate="one_to_one")
    base_correct = merged["correct_score_emb128_4seed"].astype(bool)
    cand_correct = merged["correct_score_bm25_lmf"].astype(bool)
    fixes = int((~base_correct & cand_correct).sum())
    breaks = int((base_correct & ~cand_correct).sum())
    changed = int((merged["pred_score_emb128_4seed"] != merged["pred_score_bm25_lmf"]).sum())
    return {"fixes": fixes, "breaks": breaks, "net_fixes": fixes - breaks, "changed_rows_vs_base": changed}


def load_failed_union(pd: Any) -> set[int]:
    calibration_path = ROOT / "reports/boundary_public_failure_calibration.csv"
    current_path = ROOT / "submissions/candidate_rank_blend_emb128_emb192.csv"
    if not calibration_path.exists() or not current_path.exists():
        return set()
    current = pd.read_csv(current_path)
    current_label = "Played" if "Played" in current.columns else "Label"
    current = current[["ID", current_label]].rename(columns={current_label: "current"})
    failed: set[int] = set()
    cal = pd.read_csv(calibration_path)
    for _, row in cal.iterrows():
        rel = str(row.get("file", ""))
        path = ROOT / rel
        if not path.exists():
            continue
        cand = pd.read_csv(path)
        label = "Played" if "Played" in cand.columns else "Label"
        diff = current.merge(cand[["ID", label]].rename(columns={label: "candidate"}), on="ID", validate="one_to_one")
        failed.update(diff.loc[diff["current"] != diff["candidate"], "ID"].astype(int).tolist())
    return failed


def item_degree_deciles(train_df: Any, pd: Any) -> Any:
    counts = train_df.groupby("gameID").size().rename("item_degree").reset_index()
    ranks = counts["item_degree"].rank(method="first")
    bins = pd.qcut(ranks, q=10, labels=False, duplicates="drop")
    counts["item_degree_decile"] = bins.astype(int)
    return counts


def degree_bucket_audit(scored: Any, changed_ids: set[int], pd: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    changed = scored[scored["ID"].astype(int).isin(changed_ids)].copy()
    if changed.empty:
        return rows
    changed["base_correct"] = changed["pred_score_emb128_4seed"] == changed["Label"]
    changed["lmf_correct"] = changed["pred_score_bm25_lmf"] == changed["Label"]
    for decile, group in changed.groupby("item_degree_decile", sort=True):
        fixes = int((~group["base_correct"] & group["lmf_correct"]).sum())
        breaks = int((group["base_correct"] & ~group["lmf_correct"]).sum())
        rows.append({"item_degree_decile": int(decile), "changed_rows": int(len(group)), "fixes": fixes, "breaks": breaks, "net_fixes": fixes - breaks})
    return rows


def head_only_lift(bucket_rows: list[dict[str, Any]]) -> bool:
    total_net = sum(max(0, int(row["net_fixes"])) for row in bucket_rows)
    head_net = sum(max(0, int(row["net_fixes"])) for row in bucket_rows if int(row["item_degree_decile"]) >= 8)
    return bool(total_net > 0 and head_net / total_net >= 0.8)


def metric_float(summary: dict[str, object], key: str) -> float:
    value = summary[key]
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# bm25_lmf_seed42_smoke",
        "",
        "- validation_only: true",
        "- no_kaggle_submit: true",
        "- candidate_csv_written: false",
        "- full_test_candidate_materialized: false",
        "- split: `val_random_uniform_seed42`",
        "",
        "## verdict",
        "",
        f"`{payload['tier']}`",
        "",
        payload["verdict"],
        "",
        "## metrics",
        "",
        "| metric | value | gate |",
        "|---|---:|---:|",
        f"| solo accuracy | {payload['solo_accuracy']:.6f} | >= {SOLO_GATE:.6f} |",
        f"| base emb128 4-seed accuracy | {payload['base_accuracy']:.6f} | ref {BASE_REF:.6f} |",
        f"| 50/50 z-blend accuracy | {payload['blend50_accuracy']:.6f} | >= {PASS_BLEND:.6f} |",
        f"| blend delta vs base | {payload['blend50_delta_vs_base']:+.6f} | > +{DELTA_GATE:.6f} |",
        f"| corr_z vs base | {payload['corr_z_vs_base']:.6f} | <= 0.950000 |",
        f"| rank corr vs base | {payload['rank_corr_vs_base']:.6f} | diagnostic |",
        f"| fixes / breaks | {payload['fixes']} / {payload['breaks']} | fixes > breaks |",
        f"| changed-row failed-union overlap | {payload['failed_union_overlap_frac']:.6f} | low |",
        "",
        "## item-degree bucket audit",
        "",
        "| item degree decile | changed rows | fixes | breaks | net |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["item_degree_bucket_rows"]:
        lines.append(f"| {row['item_degree_decile']} | {row['changed_rows']} | {row['fixes']} | {row['breaks']} | {row['net_fixes']} |")
    lines.extend(
        [
            "",
            "## outputs",
            "",
            f"- `{payload['score_csv']}`",
            f"- `{payload['report_json']}`",
            f"- `{payload['report_md']}`",
            "",
            "BM25_LMF_SEED42_SMOKE_DONE",
        ]
    )
    assert_not_submissions(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--regularization", type=float, default=0.6)
    parser.add_argument("--neg-prop", type=int, default=30)
    parser.add_argument("--bm25-k1", type=float, default=100.0)
    parser.add_argument("--bm25-b", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="artifacts/bm25_lmf_seed42_smoke")
    args = parser.parse_args()

    np = module("numpy")
    pd = module("pandas")
    bm25_weight = module("implicit.nearest_neighbours").bm25_weight
    LogisticMatrixFactorization = module("implicit.cpu.lmf").LogisticMatrixFactorization

    out_root = ROOT / args.out_dir
    report_json = ROOT / "reports/20260616T_bm25_lmf_seed42_smoke.json"
    report_md = ROOT / "reports/20260616T_bm25_lmf_seed42_smoke.md"
    score_dir = out_root / args.split
    score_csv = score_dir / "bm25_lmf_validation_scores.csv"
    for path in (out_root, score_dir, score_csv, report_json, report_md):
        assert_not_submissions(path)

    split_dir = ROOT / "artifacts/validation" / args.split
    train_df = load_train_interactions(split_dir / "train_interactions.csv")
    candidates = load_pairs_csv(split_dir / "candidates.csv")
    matrix, user_to_idx, item_to_idx, _, _ = build_user_item_matrix(train_df, binary=True)
    weighted = bm25_weight(matrix.tocsr(), K1=args.bm25_k1, B=args.bm25_b).tocsr()
    start = time.time()
    model = LogisticMatrixFactorization(
        factors=args.factors,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        iterations=args.iterations,
        neg_prop=args.neg_prop,
        random_state=args.seed,
    )
    model.fit(weighted, show_progress=False)
    train_seconds = round(time.time() - start, 2)

    scored = candidates.copy()
    scored["score_bm25_lmf"] = candidate_scores(model, scored, user_to_idx, item_to_idx, np)
    base = load_base_scores(pd, np)
    scored = scored.merge(base, on="ID", validate="one_to_one")
    degrees = item_degree_deciles(train_df, pd)
    scored = scored.merge(degrees, on="gameID", how="left", validate="many_to_one")
    scored["item_degree"] = scored["item_degree"].fillna(0).astype(int)
    scored["item_degree_decile"] = scored["item_degree_decile"].fillna(0).astype(int)
    scored["z_bm25_lmf"] = within_user_z(scored, "score_bm25_lmf", pd, np)
    scored["z_emb128_4seed"] = within_user_z(scored, "score_emb128_4seed", pd, np)
    scored["blend50_z_bm25_lmf_emb128"] = 0.5 * scored["z_bm25_lmf"] + 0.5 * scored["z_emb128_4seed"]
    scored["pct_rank_bm25_lmf"] = within_user_pct_rank(scored, "score_bm25_lmf", np)
    scored["pct_rank_emb128_4seed"] = within_user_pct_rank(scored, "score_emb128_4seed", np)

    solo_summary, _ = evaluate_tophalf(scored, "score_bm25_lmf", label_col="Label", user_col="userID", id_col="ID")
    base_summary, _ = evaluate_tophalf(scored, "score_emb128_4seed", label_col="Label", user_col="userID", id_col="ID")
    blend_summary, _ = evaluate_tophalf(scored, "blend50_z_bm25_lmf_emb128", label_col="Label", user_col="userID", id_col="ID")
    base_pred = label_predictions(scored, "score_emb128_4seed")
    lmf_pred = label_predictions(scored, "score_bm25_lmf")
    blend_pred = label_predictions(scored, "blend50_z_bm25_lmf_emb128")
    scored = scored.merge(base_pred, on="ID", validate="one_to_one").merge(lmf_pred, on="ID", validate="one_to_one").merge(blend_pred, on="ID", validate="one_to_one")
    deltas = prediction_delta(base_pred, lmf_pred, pd)
    changed_ids = set(scored.loc[scored["pred_score_emb128_4seed"] != scored["pred_score_bm25_lmf"], "ID"].astype(int).tolist())
    failed_union = load_failed_union(pd)
    failed_overlap = len(changed_ids & failed_union)
    failed_overlap_frac = 0.0 if not changed_ids else failed_overlap / len(changed_ids)
    bucket_rows = degree_bucket_audit(scored, changed_ids, pd)
    head_only = head_only_lift(bucket_rows)
    corr_z = float(np.corrcoef(scored["z_bm25_lmf"].to_numpy(dtype=float), scored["z_emb128_4seed"].to_numpy(dtype=float))[0, 1])
    rank_corr = float(np.corrcoef(scored["pct_rank_bm25_lmf"].astype(float), scored["pct_rank_emb128_4seed"].astype(float))[0, 1])
    solo_acc = metric_float(solo_summary, "row_accuracy")
    base_acc = metric_float(base_summary, "row_accuracy")
    blend_acc = metric_float(blend_summary, "row_accuracy")
    blend_delta = blend_acc - base_acc
    pass_gate = bool(solo_acc >= SOLO_GATE and blend_delta > DELTA_GATE and deltas["fixes"] > deltas["breaks"] and corr_z <= 0.95 and failed_overlap_frac < 0.75 and not head_only)
    if pass_gate:
        tier = "PASS_SEED42_ESCALATE_TO_PANEL"
        verdict = "BM25+LMF cleared the seed42 one-shot gate; next safe action is a fixed 3-split validation panel, still no submit."
    elif solo_acc < SOLO_GATE:
        tier = "KILL_WEAK_SOLO"
        verdict = "BM25+LMF solo accuracy missed the 0.735 floor, so this one-shot is underpowered despite competition precedent."
    elif blend_delta <= DELTA_GATE:
        tier = "KILL_TINY_OR_NEGATIVE_BLEND"
        verdict = "BM25+LMF did not improve the emb128 4-seed base by more than the +0.0007 noise band."
    else:
        tier = "KILL_GUARD_OR_BUCKET_FAIL"
        verdict = "BM25+LMF had some local signal but failed fixes/breaks, correlation, failed-row overlap, or head-only guard checks."

    score_dir.mkdir(parents=True, exist_ok=True)
    scored[
        [
            "ID",
            "userID",
            "gameID",
            "Label",
            "score_bm25_lmf",
            "score_emb128_4seed",
            "z_bm25_lmf",
            "z_emb128_4seed",
            "blend50_z_bm25_lmf_emb128",
            "pred_score_bm25_lmf",
            "pred_score_emb128_4seed",
            "pred_blend50_z_bm25_lmf_emb128",
            "item_degree",
            "item_degree_decile",
        ]
    ].to_csv(score_csv, index=False)
    payload = {
        "artifact": "bm25_lmf_seed42_smoke",
        "validation_only": True,
        "no_kaggle_submit": True,
        "candidate_csv_written": False,
        "full_test_candidate_materialized": False,
        "split": args.split,
        "params": {
            "factors": args.factors,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "regularization": args.regularization,
            "neg_prop": args.neg_prop,
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
            "seed": args.seed,
        },
        "train_seconds": train_seconds,
        "rows": int(len(scored)),
        "users": int(scored["userID"].nunique()),
        "items": int(train_df["gameID"].nunique()),
        "solo_accuracy": solo_acc,
        "base_accuracy": base_acc,
        "base_ref": BASE_REF,
        "base_accuracy_delta_vs_ref": base_acc - BASE_REF,
        "blend50_accuracy": blend_acc,
        "blend50_delta_vs_base": blend_delta,
        "solo_gate": SOLO_GATE,
        "blend_gate_accuracy": PASS_BLEND,
        "delta_gate": DELTA_GATE,
        "corr_z_vs_base": corr_z,
        "rank_corr_vs_base": rank_corr,
        **deltas,
        "failed_union_rows": len(failed_union),
        "failed_union_overlap_rows": failed_overlap,
        "failed_union_overlap_frac": failed_overlap_frac,
        "head_only_lift": head_only,
        "item_degree_bucket_rows": bucket_rows,
        "tier": tier,
        "verdict": verdict,
        "score_csv": str(score_csv.relative_to(ROOT)),
        "report_json": str(report_json.relative_to(ROOT)),
        "report_md": str(report_md.relative_to(ROOT)),
    }
    write_json(report_json, finite_or_none(payload))
    write_md(report_md, payload)
    print(json.dumps(finite_or_none({k: payload[k] for k in ["artifact", "tier", "solo_accuracy", "base_accuracy", "blend50_accuracy", "blend50_delta_vs_base", "corr_z_vs_base", "fixes", "breaks", "failed_union_overlap_frac", "score_csv"]}), indent=2))


if __name__ == "__main__":
    main()
