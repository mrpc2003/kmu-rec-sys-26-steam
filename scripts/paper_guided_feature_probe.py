#!/usr/bin/env python3
"""Paper-guided feature feasibility probes for KMU RecSys 26 Steam.

This script is intentionally validation-only.  It translates recent recommender
paper families into lightweight, reproducible probes:

- graph/global-structure CF: truncated sparse SVD scores, a cheap proxy for
  LightGCL/graph-contrastive global collaborative structure before implementing
  full contrastive training;
- temporal filtration / recency-aware positive construction: time-decayed SVD
  and user-item time-affinity scores;
- implicit-feedback negative-sampling papers: robust validation across
  uniform/sqrt-pop/pop-bin/recent candidate splits rather than a single split;
- review/text-enhanced recommendation: text length and review-count coverage
  summaries as a preflight before costly LLM/TF-IDF review encoders;
- hours/intensity modeling: hours-weighted SVD and user-item hours affinity.

No Kaggle submission is performed and no hidden labels are touched.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

# Allow running from repo root without package installation.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recsys_played_utils import (  # noqa: E402
    DEFAULT_DATA_DIR,
    ensure_dir,
    evaluate_tophalf,
    load_pairs_csv,
    load_train_interactions,
    normalize_within_user,
    percentile_summary,
    write_json,
)


def _day_number(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series)
    return (dt.astype("int64") // 86_400_000_000_000).to_numpy(dtype=np.float64)


def _safe_z(values: pd.Series) -> pd.Series:
    arr = values.astype(float)
    std = float(arr.std(ddof=0))
    if not math.isfinite(std) or std < 1e-12:
        return pd.Series(np.zeros(len(arr), dtype=np.float32), index=values.index)
    return ((arr - float(arr.mean())) / std).astype(np.float32)


def build_maps(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], list[str], list[str]]:
    users = sorted(train_df["userID"].astype(str).unique().tolist())
    items = sorted(train_df["gameID"].astype(str).unique().tolist())
    return {u: i for i, u in enumerate(users)}, {g: i for i, g in enumerate(items)}, users, items


def build_matrix(
    train_df: pd.DataFrame,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    value: str,
    recency_half_life_days: float = 365.0,
) -> sp.csr_matrix:
    row = train_df["userID"].astype(str).map(user_to_idx).to_numpy(dtype=np.int32)
    col = train_df["gameID"].astype(str).map(item_to_idx).to_numpy(dtype=np.int32)
    if value == "binary":
        data = np.ones(len(train_df), dtype=np.float32)
    elif value == "hours":
        data = np.log1p(train_df["hours_transformed"].fillna(0).to_numpy(dtype=np.float32)).astype(np.float32)
        data = np.maximum(data, 1e-3)
    elif value == "recency":
        days = _day_number(train_df["date"])
        max_day = float(np.nanmax(days))
        age = np.maximum(0.0, max_day - days)
        data = np.power(0.5, age / recency_half_life_days).astype(np.float32)
    else:
        raise ValueError(f"unknown matrix value: {value}")
    mat = sp.csr_matrix((data, (row, col)), shape=(len(user_to_idx), len(item_to_idx)), dtype=np.float32)
    mat.sum_duplicates()
    return mat


def add_svd_score(
    candidates: pd.DataFrame,
    train_df: pd.DataFrame,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    value: str,
    dim: int,
) -> tuple[pd.Series, dict[str, object]]:
    mat = build_matrix(train_df, user_to_idx, item_to_idx, value=value)
    max_rank = max(1, min(mat.shape) - 1)
    k = int(min(dim, max_rank))
    # svds is deterministic enough for a feasibility probe when v0 is fixed.
    rng = np.random.default_rng(20260530)
    v0 = rng.normal(size=min(mat.shape)).astype(np.float32)
    u, s, vt = svds(mat.astype(np.float32), k=k, v0=v0, return_singular_vectors=True)
    order = np.argsort(s)[::-1]
    s = s[order]
    u = u[:, order]
    vt = vt[order, :]
    user_emb = u * np.sqrt(s)[None, :]
    item_emb = vt.T * np.sqrt(s)[None, :]

    uidx = candidates["userID"].astype(str).map(user_to_idx)
    iidx = candidates["gameID"].astype(str).map(item_to_idx)
    known = (~uidx.isna()) & (~iidx.isna())
    scores = np.full(len(candidates), -1e9, dtype=np.float32)
    rows = uidx[known].to_numpy(dtype=np.int64)
    cols = iidx[known].to_numpy(dtype=np.int64)
    scores[known.to_numpy()] = np.einsum("ij,ij->i", user_emb[rows], item_emb[cols]).astype(np.float32)
    meta = {
        "value": value,
        "dim_requested": dim,
        "dim_used": k,
        "matrix_shape": [int(mat.shape[0]), int(mat.shape[1])],
        "matrix_nnz": int(mat.nnz),
        "candidate_known_user_item_rate": float(known.mean()),
        "top_singular_values": [float(x) for x in s[:10]],
    }
    return pd.Series(scores, index=candidates.index), meta


def add_statistical_scores(train_df: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    df = candidates.copy()
    train = train_df.copy()
    train["day_num"] = _day_number(train["date"])
    max_day = float(train["day_num"].max())
    train["age_days"] = np.maximum(0.0, max_day - train["day_num"])
    train["recency_weight_365"] = np.power(0.5, train["age_days"] / 365.0)
    train["log_hours_t"] = np.log1p(train["hours_transformed"].fillna(0).astype(float))

    item_stats = train.groupby("gameID").agg(
        item_count=("gameID", "size"),
        item_mean_day=("day_num", "mean"),
        item_last_day=("day_num", "max"),
        item_recency_sum=("recency_weight_365", "sum"),
        item_mean_hours_t=("hours_transformed", "mean"),
        item_mean_log_hours_t=("log_hours_t", "mean"),
        item_mean_text_len=("text_len", "mean"),
        item_text_count=("text_len", lambda s: int((s > 0).sum())),
        item_early_rate=("early_access", "mean"),
    )
    user_stats = train.groupby("userID").agg(
        user_count=("userID", "size"),
        user_mean_day=("day_num", "mean"),
        user_last_day=("day_num", "max"),
        user_mean_hours_t=("hours_transformed", "mean"),
        user_mean_log_hours_t=("log_hours_t", "mean"),
        user_mean_text_len=("text_len", "mean"),
        user_text_count=("text_len", lambda s: int((s > 0).sum())),
        user_early_rate=("early_access", "mean"),
    )
    df = df.merge(item_stats, left_on="gameID", right_index=True, how="left")
    df = df.merge(user_stats, left_on="userID", right_index=True, how="left")
    fill_cols = [c for c in df.columns if c.startswith("item_") or c.startswith("user_")]
    for col in fill_cols:
        df[col] = df[col].fillna(0.0)

    df["score_item_log_pop"] = np.log1p(df["item_count"].astype(float))
    df["score_item_recency_log_pop"] = np.log1p(df["item_recency_sum"].astype(float))
    df["score_time_affinity"] = -np.abs(df["user_mean_day"].astype(float) - df["item_mean_day"].astype(float)) / 365.0
    df["score_last_time_affinity"] = -np.abs(df["user_last_day"].astype(float) - df["item_last_day"].astype(float)) / 365.0
    df["score_hours_affinity"] = -np.abs(df["user_mean_log_hours_t"].astype(float) - df["item_mean_log_hours_t"].astype(float))
    df["score_text_len_affinity"] = -np.abs(np.log1p(df["user_mean_text_len"].astype(float)) - np.log1p(df["item_mean_text_len"].astype(float)))
    df["score_text_presence_pop"] = np.log1p(df["item_text_count"].astype(float))

    meta = {
        "train_date_min": str(pd.to_datetime(train["date"]).min().date()),
        "train_date_max": str(pd.to_datetime(train["date"]).max().date()),
        "train_text_len": percentile_summary(train["text_len"].astype(float)),
        "user_history_len": percentile_summary(user_stats["user_count"].astype(float)),
        "item_history_len": percentile_summary(item_stats["item_count"].astype(float)),
        "item_count_coverage_in_candidates": float(df["item_count"].gt(0).mean()),
        "user_count_coverage_in_candidates": float(df["user_count"].gt(0).mean()),
    }
    return df, meta


def evaluate_split(split_dir: Path, out_dir: Path, dim: int) -> dict[str, object]:
    train_df = load_train_interactions(split_dir / "train_interactions.csv")
    candidates = load_pairs_csv(split_dir / "candidates.csv")
    user_to_idx, item_to_idx, _, _ = build_maps(train_df)
    scored, stat_meta = add_statistical_scores(train_df, candidates)

    svd_meta = {}
    for value in ["binary", "hours", "recency"]:
        col = f"score_svd_{value}_k{dim}"
        scored[col], svd_meta[value] = add_svd_score(scored, train_df, user_to_idx, item_to_idx, value=value, dim=dim)

    # Within-user z-blends preserve the competition's per-user top-half decision structure.
    blend_cols = [
        "score_item_log_pop",
        "score_item_recency_log_pop",
        "score_time_affinity",
        "score_hours_affinity",
        f"score_svd_binary_k{dim}",
        f"score_svd_hours_k{dim}",
        f"score_svd_recency_k{dim}",
    ]
    zdf = normalize_within_user(scored, blend_cols, user_col="userID")
    scored = zdf
    scored["score_blend_graph_time_hours"] = (
        0.45 * scored[f"z_score_svd_recency_k{dim}"]
        + 0.25 * scored[f"z_score_svd_binary_k{dim}"]
        + 0.15 * scored["z_score_item_recency_log_pop"]
        + 0.10 * scored["z_score_time_affinity"]
        + 0.05 * scored["z_score_hours_affinity"]
    ).astype(np.float32)
    scored["score_blend_svd_family_mean"] = scored[
        [f"z_score_svd_binary_k{dim}", f"z_score_svd_hours_k{dim}", f"z_score_svd_recency_k{dim}"]
    ].mean(axis=1).astype(np.float32)

    score_cols = [
        "score_item_log_pop",
        "score_item_recency_log_pop",
        "score_time_affinity",
        "score_last_time_affinity",
        "score_hours_affinity",
        "score_text_len_affinity",
        "score_text_presence_pop",
        f"score_svd_binary_k{dim}",
        f"score_svd_hours_k{dim}",
        f"score_svd_recency_k{dim}",
        "score_blend_svd_family_mean",
        "score_blend_graph_time_hours",
    ]
    summaries = []
    for col in score_cols:
        summary, _ = evaluate_tophalf(scored, col, label_col="Label", user_col="userID", id_col="ID")
        summaries.append(summary)
    summaries = sorted(summaries, key=lambda x: (x["row_accuracy"], x["per_user_mean_accuracy"]), reverse=True)

    split_out = ensure_dir(out_dir / split_dir.name)
    scored[["ID", "userID", "gameID", "Label"] + score_cols].to_csv(split_out / "paper_guided_scores.csv", index=False)
    write_json(split_out / "summary.json", {"split": split_dir.name, "stat_meta": stat_meta, "svd_meta": svd_meta, "summaries": summaries})

    md_lines = [
        f"# Paper-guided feature probe — {split_dir.name}",
        "",
        "Validation-only; no Kaggle submission.",
        "",
        "| rank | score_col | row acc | per-user mean acc | pred pos ok |",
        "|---:|---|---:|---:|---|",
    ]
    for i, s in enumerate(summaries, 1):
        md_lines.append(
            f"| {i} | `{s['score_col']}` | {s['row_accuracy']:.6f} | {s['per_user_mean_accuracy']:.6f} | {s['all_user_positive_counts_match']} |"
        )
    md_lines.extend([
        "",
        "## Feasibility metadata",
        "",
        f"- Train date range: {stat_meta['train_date_min']} .. {stat_meta['train_date_max']}",
        f"- Candidate known user coverage: {stat_meta['user_count_coverage_in_candidates']:.6f}",
        f"- Candidate known item coverage: {stat_meta['item_count_coverage_in_candidates']:.6f}",
        f"- SVD matrix shape: {svd_meta['binary']['matrix_shape']}, nnz={svd_meta['binary']['matrix_nnz']}",
    ])
    (split_out / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"split": split_dir.name, "best": summaries[0], "summaries": summaries, "stat_meta": stat_meta, "svd_meta": svd_meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", default="artifacts/validation")
    parser.add_argument("--splits", nargs="*", default=[
        "val_random_sqrtpop_seed42",
        "val_recent_sqrtpop_seed42",
        "val_random_popbin_seed42",
        "val_random_uniform_seed42",
    ])
    parser.add_argument("--out-dir", default="artifacts/paper_guided_feature_probe_20260530")
    parser.add_argument("--report-json", default="reports/20260530_paper_guided_feature_probe.json")
    parser.add_argument("--report-md", default="reports/20260530_paper_guided_feature_probe.md")
    parser.add_argument("--svd-dim", type=int, default=64)
    args = parser.parse_args()

    validation_root = Path(args.validation_root)
    out_dir = ensure_dir(args.out_dir)
    results = []
    for split in args.splits:
        split_dir = validation_root / split
        if not split_dir.exists():
            raise FileNotFoundError(split_dir)
        results.append(evaluate_split(split_dir, out_dir, dim=args.svd_dim))

    overall = {
        "note": "Validation-only paper-guided feasibility probe; no Kaggle submission.",
        "svd_dim": args.svd_dim,
        "results": results,
    }
    write_json(args.report_json, overall)

    lines = [
        "# KMU RecSys 26 Steam — paper-guided feature feasibility probe",
        "",
        "이 리포트는 최신 추천시스템 논문 계열을 현재 Steam played prediction 과제에 적용하기 전, 로컬 validation에서 가볍게 검증한 탐색 결과다. Kaggle 제출은 수행하지 않았다.",
        "",
        "## Split별 best probe",
        "",
        "| split | best score | row acc | per-user mean acc |",
        "|---|---|---:|---:|",
    ]
    for r in results:
        b = r["best"]
        lines.append(f"| `{r['split']}` | `{b['score_col']}` | {b['row_accuracy']:.6f} | {b['per_user_mean_accuracy']:.6f} |")
    lines.extend([
        "",
        "## 논문 계열별 해석",
        "",
        "- Graph/global collaborative signal: `score_svd_*`는 LightGCL류의 SVD/global-structure augmentation을 구현하기 전 cheap proxy다. 기존 ItemKNN/EASE/ALS보다 강하면 full graph-contrastive 구현 가치가 있다.",
        "- Temporal filtration: `score_svd_recency_*`, `score_time_affinity`, `score_item_recency_log_pop`은 TFPS/시간 필터링 논문의 핵심인 시간 가중 positive 신호의 1차 proxy다.",
        "- Negative-sampling/debiasing: 동일 feature를 sqrt-pop/recent/pop-bin/uniform split 모두에서 비교해 false-negative와 surrogate mismatch 위험을 본다.",
        "- Review/text-enhanced recommendation: 현재는 text length/count proxy만 포함했다. 이 값이 약하면 LLM/TF-IDF review embedding은 단독 scorer보다 blend/regularizer로 시작한다.",
        "- Hours/intensity: `score_svd_hours_*`, `score_hours_affinity`는 플레이 시간 강도 정보를 반영한다. Accuracy 과제에서는 intensity가 preference와 noise를 동시에 담을 수 있어 split 간 안정성이 중요하다.",
        "",
        "## 전체 score table",
        "",
    ])
    for r in results:
        lines.extend([
            f"### {r['split']}",
            "",
            "| rank | score_col | row acc | per-user mean acc |",
            "|---:|---|---:|---:|",
        ])
        for i, s in enumerate(r["summaries"], 1):
            lines.append(f"| {i} | `{s['score_col']}` | {s['row_accuracy']:.6f} | {s['per_user_mean_accuracy']:.6f} |")
        lines.append("")
    Path(args.report_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report_json": args.report_json, "report_md": args.report_md, "out_dir": str(out_dir), "splits": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
