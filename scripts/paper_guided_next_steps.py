#!/usr/bin/env python3
"""Run the next paper-guided RecSys exploration steps end-to-end.

This is a validation-only pipeline for KMU RecSys 26 Steam.  It implements the
next recommended actions from the paper-guided exploration report:

1. ICPNS-style community-aware validation negatives.
2. Correct/Weight or PU-inspired weighted implicit scorer.
3. TFPS-style time-decay graph scores for ItemKNN/EASE.
4. Train-only review pseudo-category features.

The script never calls the Kaggle API and never writes a submission.  All heavy
CSV outputs go under artifacts/; human and machine-readable summaries go under
reports/.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recsys_played_utils import (  # noqa: E402
    DEFAULT_DATA_DIR,
    ensure_dir,
    evaluate_tophalf,
    item_popularity,
    load_pairs_csv,
    load_train_interactions,
    load_train_json,
    normalize_within_user,
    percentile_summary,
    user_histories,
    write_json,
)


@dataclass(frozen=True)
class CommunitySplitConfig:
    holdout: str
    seed: int
    name: str


def choose_positives(user_df: pd.DataFrame, k: int, holdout: str, rng: np.random.Generator) -> pd.DataFrame:
    if holdout == "recent":
        return user_df.sort_values(["date", "gameID"], ascending=[False, True]).head(k)
    if holdout == "random":
        idx = rng.choice(user_df.index.to_numpy(), size=k, replace=False)
        return user_df.loc[idx]
    raise ValueError(f"unknown holdout mode: {holdout}")


def day_number(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series)
    return (dt.astype("int64") // 86_400_000_000_000).to_numpy(dtype=np.float64)


def build_maps(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], list[str], list[str]]:
    users = sorted(train_df["userID"].astype(str).unique().tolist())
    items = sorted(train_df["gameID"].astype(str).unique().tolist())
    return {u: i for i, u in enumerate(users)}, {g: i for i, g in enumerate(items)}, users, items


def build_sparse_matrix(
    train_df: pd.DataFrame,
    user_to_idx: dict[str, int],
    item_to_idx: dict[str, int],
    values: np.ndarray | None = None,
) -> sp.csr_matrix:
    row = train_df["userID"].astype(str).map(user_to_idx).to_numpy(dtype=np.int32)
    col = train_df["gameID"].astype(str).map(item_to_idx).to_numpy(dtype=np.int32)
    data = np.ones(len(train_df), dtype=np.float32) if values is None else values.astype(np.float32)
    mat = sp.csr_matrix((data, (row, col)), shape=(len(user_to_idx), len(item_to_idx)), dtype=np.float32)
    mat.sum_duplicates()
    return mat


def recency_weights(train_df: pd.DataFrame, half_life_days: float) -> np.ndarray:
    days = day_number(train_df["date"])
    max_day = float(np.nanmax(days))
    age = np.maximum(0.0, max_day - days)
    return np.power(0.5, age / float(half_life_days)).astype(np.float32)


def fit_user_communities(
    train_df: pd.DataFrame,
    n_communities: int,
    svd_dim: int,
    seed: int,
) -> tuple[dict[str, int], dict[tuple[int, str], float], dict[int, float], dict[str, float], dict[str, object]]:
    user_to_idx, item_to_idx, users, _ = build_maps(train_df)
    X = build_sparse_matrix(train_df, user_to_idx, item_to_idx)
    dim = int(max(2, min(svd_dim, min(X.shape) - 1)))
    emb = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(X)
    k = int(max(2, min(n_communities, len(users))))
    labels = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=2048, n_init=5).fit_predict(emb)
    user_comm = {u: int(labels[i]) for u, i in user_to_idx.items()}

    tmp = train_df[["userID", "gameID"]].copy()
    tmp["community"] = tmp["userID"].map(user_comm).astype(int)
    comm_item = tmp.groupby(["community", "gameID"]).size().astype(float).to_dict()
    comm_total = tmp.groupby("community").size().astype(float).to_dict()
    global_pop = item_popularity(train_df).to_dict()
    meta = {
        "n_users": int(X.shape[0]),
        "n_items": int(X.shape[1]),
        "n_communities": k,
        "svd_dim": dim,
        "community_size_summary": percentile_summary(pd.Series(labels).value_counts().to_numpy(dtype=float)),
    }
    return user_comm, comm_item, comm_total, global_pop, meta


def quantile_bins(pop: dict[str, float], n_bins: int = 10) -> dict[str, int]:
    if not pop:
        return {}
    s = pd.Series(pop, dtype=float)
    ranks = s.rank(method="first", ascending=True)
    bins = pd.qcut(ranks, q=min(n_bins, len(s)), labels=False, duplicates="drop")
    return {str(k): int(v) for k, v in bins.items()}


def weighted_choice_without_replacement(
    pool: list[str], weights: np.ndarray, rng: np.random.Generator, size: int,
) -> list[str]:
    if not pool or size <= 0:
        return []
    size = min(size, len(pool))
    weights = np.asarray(weights, dtype=np.float64)
    if (not np.isfinite(weights).all()) or weights.sum() <= 0:
        return [str(x) for x in rng.choice(np.array(pool, dtype=object), size=size, replace=False)]
    weights = weights / weights.sum()
    return [str(x) for x in rng.choice(np.array(pool, dtype=object), size=size, replace=False, p=weights)]


def sample_community_negatives(
    user_id: str,
    positive_games: list[str],
    all_items: list[str],
    full_hist: dict[str, set[str]],
    user_comm: dict[str, int],
    comm_item: dict[tuple[int, str], float],
    global_pop: dict[str, float],
    pop_bins: dict[str, int],
    rng: np.random.Generator,
) -> list[str]:
    seen = full_hist.get(user_id, set())
    base_pool = [g for g in all_items if g not in seen]
    if len(base_pool) < len(positive_games):
        raise ValueError(f"not enough community negatives for {user_id}: pool={len(base_pool)}")
    comm = user_comm.get(user_id, -1)
    chosen: list[str] = []
    remaining = list(base_pool)
    for pos_gid in positive_games:
        target_bin = pop_bins.get(pos_gid)
        candidate_pool = [g for g in remaining if target_bin is None or abs(pop_bins.get(g, -999) - target_bin) <= 1]
        if not candidate_pool:
            candidate_pool = remaining
        weights = np.array(
            [
                (comm_item.get((comm, g), 0.0) + 0.25) ** 1.0
                * (global_pop.get(g, 0.0) + 0.25) ** 0.25
                for g in candidate_pool
            ],
            dtype=float,
        )
        pick = weighted_choice_without_replacement(candidate_pool, weights, rng, 1)[0]
        chosen.append(pick)
        remaining = [g for g in remaining if g != pick]
    return chosen


def build_community_split(
    train_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    config: CommunitySplitConfig,
    out_root: Path,
    n_communities: int,
    svd_dim: int,
) -> dict[str, object]:
    started = time.time()
    rng = np.random.default_rng(config.seed)
    out_dir = ensure_dir(out_root / config.name)
    pair_user_counts = pairs_df.groupby("userID").size().astype(int)
    full_hist = user_histories(train_df)
    train_by_user = {uid: grp for uid, grp in train_df.groupby("userID", sort=False)}

    heldout_idx: list[int] = []
    requested_k: dict[str, int] = {}
    actual_k: dict[str, int] = {}
    adjusted_users: list[dict[str, object]] = []
    skipped_users: list[str] = []
    for user_id, cand_count in pair_user_counts.items():
        user_rows = train_by_user.get(user_id)
        if user_rows is None or user_rows.empty:
            skipped_users.append(user_id)
            continue
        requested = int(cand_count // 2)
        k = requested
        if len(user_rows) <= k:
            k = max(0, len(user_rows) - 1)
            adjusted_users.append({"userID": user_id, "requested_k": requested, "actual_k": k, "train_n": int(len(user_rows))})
        if k < 1:
            skipped_users.append(user_id)
            continue
        pos = choose_positives(user_rows, k, config.holdout, rng)
        heldout_idx.extend(pos["row_idx"].astype(int).tolist())
        requested_k[user_id] = requested
        actual_k[user_id] = k

    heldout_idx_set = set(heldout_idx)
    fold_train = train_df[~train_df["row_idx"].isin(heldout_idx_set)].copy()
    heldout = train_df[train_df["row_idx"].isin(heldout_idx_set)].copy()

    user_comm, comm_item, comm_total, global_pop, comm_meta = fit_user_communities(fold_train, n_communities, svd_dim, config.seed)
    pop_bins = quantile_bins(global_pop, n_bins=10)
    all_items = sorted(fold_train["gameID"].astype(str).unique().tolist())

    rows: list[dict[str, object]] = []
    for user_id, pos_df in heldout.groupby("userID", sort=False):
        k = actual_k[user_id]
        pos_games = [str(x) for x in pos_df["gameID"].tolist()]
        neg_games = sample_community_negatives(
            user_id=str(user_id),
            positive_games=pos_games,
            all_items=all_items,
            full_hist=full_hist,
            user_comm=user_comm,
            comm_item=comm_item,
            global_pop=global_pop,
            pop_bins=pop_bins,
            rng=rng,
        )
        for _, r in pos_df.iterrows():
            rows.append(
                {
                    "userID": user_id,
                    "gameID": r["gameID"],
                    "Label": 1,
                    "source": "heldout_positive",
                    "heldout_row_idx": int(r["row_idx"]),
                    "heldout_date": str(pd.Timestamp(r["date"]).date()),
                    "requested_pos_k": requested_k[user_id],
                    "actual_pos_k": k,
                }
            )
        for gid in neg_games:
            rows.append(
                {
                    "userID": user_id,
                    "gameID": gid,
                    "Label": 0,
                    "source": "negative_communitypop",
                    "heldout_row_idx": -1,
                    "heldout_date": "",
                    "requested_pos_k": requested_k[user_id],
                    "actual_pos_k": k,
                }
            )

    candidates = pd.DataFrame(rows).sample(frac=1.0, random_state=config.seed).reset_index(drop=True)
    candidates.insert(0, "ID", np.arange(len(candidates), dtype=int))

    # Safety checks.
    train_pairs = set(zip(fold_train["userID"], fold_train["gameID"]))
    overlap = sum((u, g) in train_pairs for u, g in candidates[["userID", "gameID"]].itertuples(index=False))
    if overlap:
        raise RuntimeError(f"community split overlaps fold train: {overlap}")
    pos_counts = candidates.groupby("userID")["Label"].sum().astype(int)
    cand_counts = candidates.groupby("userID").size().astype(int)
    if not (cand_counts == 2 * pos_counts).all():
        raise RuntimeError("community split is not per-user 1:1")

    fold_train.to_csv(out_dir / "train_interactions.csv", index=False)
    candidates.to_csv(out_dir / "candidates.csv", index=False)
    summary = {
        "name": config.name,
        "holdout": config.holdout,
        "negative": "communitypop",
        "seed": config.seed,
        "fold_train_rows": int(len(fold_train)),
        "candidate_rows": int(len(candidates)),
        "users": int(candidates["userID"].nunique()),
        "positives": int(candidates["Label"].sum()),
        "negatives": int((1 - candidates["Label"]).sum()),
        "adjusted_users": adjusted_users[:50],
        "adjusted_user_count": int(len(adjusted_users)),
        "skipped_user_count": int(len(skipped_users)),
        "community_meta": comm_meta,
        "duration_sec": round(time.time() - started, 3),
    }
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(
        "\n".join(
            [
                f"# {config.name}",
                "",
                "Validation-only ICPNS-style community-popularity negative split.",
                "",
                f"- rows: {summary['candidate_rows']}",
                f"- users: {summary['users']}",
                f"- positives: {summary['positives']}",
                f"- negatives: {summary['negatives']}",
                f"- communities: {comm_meta['n_communities']}",
                f"- duration_sec: {summary['duration_sec']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def load_text_by_row(raw_train_json: Path, row_indices: set[int]) -> dict[int, str]:
    if not row_indices:
        return {}
    out: dict[int, str] = {}
    with raw_train_json.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx not in row_indices:
                continue
            d = ast.literal_eval(line)
            out[idx] = str(d.get("text") or "")
            if len(out) == len(row_indices):
                break
    return out


def fit_review_pseudocats(
    train_df: pd.DataFrame,
    raw_train_json: Path,
    n_clusters: int,
    svd_dim: int,
    seed: int,
    max_features: int,
) -> tuple[dict[str, int], dict[str, float], dict[tuple[str, int], float], dict[str, object]]:
    rows = set(train_df["row_idx"].astype(int).tolist())
    text_by_row = load_text_by_row(raw_train_json, rows)
    tmp = train_df[["row_idx", "userID", "gameID"]].copy()
    tmp["text"] = tmp["row_idx"].astype(int).map(text_by_row).fillna("")
    item_docs = tmp.groupby("gameID")["text"].apply(lambda s: "\n".join(x for x in s if x)[:60000])
    items = item_docs.index.astype(str).tolist()
    docs = item_docs.tolist()
    nonempty = sum(bool(x.strip()) for x in docs)
    if nonempty < 2:
        return {}, {}, {}, {"enabled": False, "reason": "not enough non-empty item docs"}
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]{1,}\b",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=3,
        norm="l2",
        dtype=np.float32,
    )
    X = vec.fit_transform(docs)
    dim = int(max(2, min(svd_dim, min(X.shape) - 1)))
    emb = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(X)
    k = int(max(2, min(n_clusters, len(items))))
    labels = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=1024, n_init=5).fit_predict(emb)
    item_cat = {g: int(labels[i]) for i, g in enumerate(items)}
    tmp["cat"] = tmp["gameID"].astype(str).map(item_cat).fillna(-1).astype(int)
    item_cat_pop = tmp.groupby("cat").size().astype(float).to_dict()
    user_cat_counts = tmp.groupby(["userID", "cat"]).size().astype(float).to_dict()
    meta = {
        "enabled": True,
        "items": len(items),
        "nonempty_item_docs": int(nonempty),
        "vocab_size": int(len(vec.vocabulary_)),
        "svd_dim": dim,
        "n_clusters": k,
        "cluster_size_summary": percentile_summary(pd.Series(labels).value_counts().to_numpy(dtype=float)),
    }
    return item_cat, item_cat_pop, user_cat_counts, meta


def add_basic_stats(train_df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    tmp = train_df.copy()
    tmp["day_num"] = day_number(tmp["date"])
    tmp["log_hours_t"] = np.log1p(tmp["hours_transformed"].astype(float).fillna(0.0))
    max_day = float(tmp["day_num"].max())
    tmp["recency_w365"] = np.power(0.5, np.maximum(0.0, max_day - tmp["day_num"]) / 365.0)
    item_stats = tmp.groupby("gameID").agg(
        item_count=("gameID", "size"),
        item_recency_sum365=("recency_w365", "sum"),
        item_mean_day=("day_num", "mean"),
        item_last_day=("day_num", "max"),
        item_mean_log_hours=("log_hours_t", "mean"),
        item_mean_text_len=("text_len", "mean"),
    )
    user_stats = tmp.groupby("userID").agg(
        user_count=("userID", "size"),
        user_mean_day=("day_num", "mean"),
        user_last_day=("day_num", "max"),
        user_mean_log_hours=("log_hours_t", "mean"),
        user_mean_text_len=("text_len", "mean"),
    )
    out = out.merge(item_stats, left_on="gameID", right_index=True, how="left")
    out = out.merge(user_stats, left_on="userID", right_index=True, how="left")
    for col in out.columns:
        if col.startswith("item_") or col.startswith("user_"):
            out[col] = out[col].fillna(0.0).astype(float)
    out["pop_count"] = out["item_count"].astype(float)
    out["score_item_log_pop"] = np.log1p(out["item_count"].astype(float))
    out["score_item_recency_log_pop365"] = np.log1p(out["item_recency_sum365"].astype(float))
    out["score_time_affinity_mean"] = -np.abs(out["user_mean_day"] - out["item_mean_day"]) / 365.0
    out["score_time_affinity_last"] = -np.abs(out["user_last_day"] - out["item_last_day"]) / 365.0
    out["score_hours_affinity"] = -np.abs(out["user_mean_log_hours"] - out["item_mean_log_hours"])
    out["score_text_len_affinity"] = -np.abs(np.log1p(out["user_mean_text_len"]) - np.log1p(out["item_mean_text_len"]))
    return out


def add_community_scores(train_df: pd.DataFrame, scored: pd.DataFrame, seed: int, n_communities: int, svd_dim: int) -> tuple[pd.DataFrame, dict[str, object]]:
    user_comm, comm_item, comm_total, global_pop, meta = fit_user_communities(train_df, n_communities, svd_dim, seed)
    out = scored.copy()
    vals = []
    rates = []
    for uid, gid in out[["userID", "gameID"]].itertuples(index=False):
        comm = user_comm.get(str(uid), -1)
        cnt = float(comm_item.get((comm, str(gid)), 0.0))
        total = float(comm_total.get(comm, 0.0))
        vals.append(cnt)
        rates.append(cnt / total if total > 0 else 0.0)
    out["score_icpns_comm_log_pop"] = np.log1p(np.asarray(vals, dtype=float))
    out["score_icpns_comm_rate"] = np.asarray(rates, dtype=float)
    out["score_icpns_comm_global_blend"] = out["score_icpns_comm_log_pop"] + 0.25 * out["score_item_log_pop"]
    return out, meta


def cosine_itemknn_scores(X: sp.csr_matrix, candidates: pd.DataFrame, user_to_idx: dict[str, int], item_to_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    X = X.astype(np.float32).tocsr()
    norms = np.sqrt(np.asarray(X.power(2).sum(axis=0)).ravel()).astype(np.float32)
    gram = (X.T @ X).astype(np.float32).toarray()
    denom = np.maximum(norms[:, None] * norms[None, :], 1e-12)
    sim = gram / denom
    np.fill_diagonal(sim, 0.0)
    score_sum = np.zeros(len(candidates), dtype=np.float32)
    score_top3 = np.zeros(len(candidates), dtype=np.float32)
    for n, (uid, gid) in enumerate(candidates[["userID", "gameID"]].itertuples(index=False)):
        ui = user_to_idx.get(str(uid))
        gi = item_to_idx.get(str(gid))
        if ui is None or gi is None:
            continue
        hist = X.indices[X.indptr[ui] : X.indptr[ui + 1]]
        if hist.size == 0:
            continue
        vals = sim[gi, hist]
        score_sum[n] = float(vals.sum())
        k = min(3, vals.size)
        score_top3[n] = float(np.partition(vals, -k)[-k:].mean())
    return score_sum, score_top3


def ease_scores(X: sp.csr_matrix, candidates: pd.DataFrame, user_to_idx: dict[str, int], item_to_idx: dict[str, int], lam: float) -> np.ndarray:
    X64 = X.astype(np.float64).tocsr()
    G = (X64.T @ X64).toarray().astype(np.float64)
    diag = np.diag_indices(G.shape[0])
    G[diag] += lam
    P = np.linalg.inv(G)
    B = P / np.maximum(-np.diag(P), 1e-12)[None, :]
    B[diag] = 0.0
    user_scores = X64 @ B
    out = np.zeros(len(candidates), dtype=np.float32)
    for n, (uid, gid) in enumerate(candidates[["userID", "gameID"]].itertuples(index=False)):
        ui = user_to_idx.get(str(uid))
        gi = item_to_idx.get(str(gid))
        if ui is not None and gi is not None:
            out[n] = float(user_scores[ui, gi])
    return out


def add_time_decay_graph_scores(train_df: pd.DataFrame, scored: pd.DataFrame, half_lives: list[int], ease_half_life: int, ease_lambda: float) -> tuple[pd.DataFrame, dict[str, object]]:
    out = scored.copy()
    user_to_idx, item_to_idx, _, _ = build_maps(train_df)
    meta: dict[str, object] = {"half_lives": half_lives, "ease_half_life": ease_half_life, "ease_lambda": ease_lambda}
    for hl in half_lives:
        weights = recency_weights(train_df, hl)
        X = build_sparse_matrix(train_df, user_to_idx, item_to_idx, values=weights)
        ssum, stop3 = cosine_itemknn_scores(X, out, user_to_idx, item_to_idx)
        out[f"score_time_itemknn_hl{hl}_sum"] = ssum
        out[f"score_time_itemknn_hl{hl}_top3"] = stop3
        if hl == ease_half_life:
            out[f"score_time_ease_hl{hl}_lambda{ease_lambda:g}"] = ease_scores(X, out, user_to_idx, item_to_idx, ease_lambda)
        meta[f"hl{hl}_matrix_nnz"] = int(X.nnz)
    return out, meta


def add_svd_scores(train_df: pd.DataFrame, scored: pd.DataFrame, dim: int, seed: int) -> tuple[pd.DataFrame, dict[str, object]]:
    out = scored.copy()
    user_to_idx, item_to_idx, _, _ = build_maps(train_df)
    X = build_sparse_matrix(train_df, user_to_idx, item_to_idx)
    k = int(max(2, min(dim, min(X.shape) - 1)))
    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=min(X.shape)).astype(np.float32)
    u, s, vt = svds(X.astype(np.float32), k=k, v0=v0)
    order = np.argsort(s)[::-1]
    s = s[order]
    u = u[:, order]
    vt = vt[order, :]
    U = u * np.sqrt(s)[None, :]
    V = vt.T * np.sqrt(s)[None, :]
    vals = np.zeros(len(out), dtype=np.float32)
    for n, (uid, gid) in enumerate(out[["userID", "gameID"]].itertuples(index=False)):
        ui = user_to_idx.get(str(uid))
        gi = item_to_idx.get(str(gid))
        if ui is not None and gi is not None:
            vals[n] = float(U[ui] @ V[gi])
    out[f"score_graph_svd_k{k}"] = vals
    return out, {"svd_dim": k, "top_singular_values": [float(x) for x in s[:10]]}


def add_review_pseudocat_scores(train_df: pd.DataFrame, scored: pd.DataFrame, raw_train_json: Path, seed: int, clusters: int, svd_dim: int, max_features: int) -> tuple[pd.DataFrame, dict[str, object]]:
    item_cat, item_cat_pop, user_cat_counts, meta = fit_review_pseudocats(train_df, raw_train_json, clusters, svd_dim, seed, max_features)
    out = scored.copy()
    if not meta.get("enabled"):
        out["score_review_pseudocat_affinity"] = 0.0
        out["score_review_pseudocat_pop"] = 0.0
        return out, meta
    user_total = train_df.groupby("userID").size().astype(float).to_dict()
    aff = []
    pop = []
    for uid, gid in out[["userID", "gameID"]].itertuples(index=False):
        cat = item_cat.get(str(gid), -1)
        cnt = float(user_cat_counts.get((str(uid), cat), 0.0))
        total = float(user_total.get(str(uid), 0.0))
        aff.append(cnt / total if total > 0 else 0.0)
        pop.append(float(item_cat_pop.get(cat, 0.0)))
    out["score_review_pseudocat_affinity"] = np.asarray(aff, dtype=np.float32)
    out["score_review_pseudocat_log_pop"] = np.log1p(np.asarray(pop, dtype=np.float32))
    out["score_review_pseudocat_blend"] = out["score_review_pseudocat_affinity"] + 0.05 * out["score_review_pseudocat_log_pop"]
    return out, meta


def sample_weighted_training_pairs(
    train_df: pd.DataFrame,
    seed: int,
    max_users: int | None = None,
    max_pos_per_user: int = 20,
) -> pd.DataFrame:
    """Create a compact PU/CW-style train set from fold-train only.

    Sampling every observed positive and then scanning the item universe for a
    community negative is unnecessary for the lightweight scorer and can turn a
    validation sweep into a slow Python loop.  We therefore cap positives per
    user while keeping broad user/item coverage.
    """
    rng = np.random.default_rng(seed)
    full_hist = user_histories(train_df)
    all_items = sorted(train_df["gameID"].astype(str).unique().tolist())
    user_comm, comm_item, _, global_pop, _ = fit_user_communities(train_df, n_communities=24, svd_dim=32, seed=seed)
    pop_bins = quantile_bins(global_pop, n_bins=10)
    users = train_df["userID"].astype(str).unique().tolist()
    if max_users is not None:
        users = users[:max_users]
    user_set = set(users)

    train_with_age = train_df.copy()
    train_with_age["_age_w365"] = recency_weights(train_with_age, 365)

    rows: list[dict[str, object]] = []
    for uid, grp in train_with_age.groupby("userID", sort=False):
        if str(uid) not in user_set:
            continue
        if len(grp) > max_pos_per_user:
            grp = grp.sample(n=max_pos_per_user, random_state=int(rng.integers(0, 2**31 - 1)))
        pos_games = [str(g) for g in grp["gameID"].tolist()]
        neg_games = sample_community_negatives(str(uid), pos_games, all_items, full_hist, user_comm, comm_item, global_pop, pop_bins, rng)
        for _, r in grp.iterrows():
            age_w = float(r.get("_age_w365", 1.0))
            rows.append({"userID": uid, "gameID": r["gameID"], "Label": 1, "sample_weight": 1.0 + 0.15 * float(r.get("hours_transformed", 0.0)) + 0.25 * age_w})
        comm = user_comm.get(str(uid), -1)
        for gid in neg_games:
            reliability = math.log1p(comm_item.get((comm, gid), 0.0)) + 0.25 * math.log1p(global_pop.get(gid, 0.0))
            rows.append({"userID": uid, "gameID": gid, "Label": 0, "sample_weight": 0.5 + min(2.0, reliability / 4.0)})
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    allow = [
        "score_item_log_pop",
        "score_item_recency_log_pop365",
        "score_time_affinity_mean",
        "score_time_affinity_last",
        "score_hours_affinity",
        "score_text_len_affinity",
        "score_icpns_comm_log_pop",
        "score_icpns_comm_rate",
        "score_icpns_comm_global_blend",
        "score_review_pseudocat_affinity",
        "score_review_pseudocat_log_pop",
        "score_review_pseudocat_blend",
        "score_graph_svd_k64",
    ]
    allow += [c for c in df.columns if c.startswith("score_time_itemknn_") or c.startswith("score_time_ease_")]
    return [c for c in allow if c in df.columns]


def add_weighted_implicit_logit(train_df: pd.DataFrame, scored: pd.DataFrame, raw_train_json: Path, seed: int, feature_cols: list[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    out = scored.copy()
    train_pairs = sample_weighted_training_pairs(train_df, seed=seed)
    train_scored = add_basic_stats(train_df, train_pairs)
    train_scored, _ = add_community_scores(train_df, train_scored, seed=seed, n_communities=24, svd_dim=32)
    # The expensive graph/text features are omitted for training examples; fill missing with 0 after selecting columns.
    for col in feature_cols:
        if col not in train_scored.columns:
            train_scored[col] = 0.0
    X = train_scored[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=np.float32)
    y = train_scored["Label"].astype(int).to_numpy()
    w = train_scored["sample_weight"].astype(float).to_numpy()
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=250, C=0.7, solver="lbfgs", random_state=seed),
    )
    clf.fit(X, y, logisticregression__sample_weight=w)
    Xc = out[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=np.float32)
    out["score_cw_weighted_implicit_logit"] = clf.decision_function(Xc).astype(np.float32)
    meta = {
        "train_pairs": int(len(train_pairs)),
        "positive_train_pairs": int(train_pairs["Label"].sum()),
        "negative_train_pairs": int((1 - train_pairs["Label"]).sum()),
        "feature_columns": feature_cols,
    }
    return out, meta


def score_split(split_dir: Path, raw_train_json: Path, out_root: Path, seed: int) -> dict[str, object]:
    started = time.time()
    train_df = load_train_interactions(split_dir / "train_interactions.csv")
    candidates = load_pairs_csv(split_dir / "candidates.csv")
    scored = add_basic_stats(train_df, candidates)
    scored, comm_meta = add_community_scores(train_df, scored, seed=seed, n_communities=24, svd_dim=32)
    scored, graph_meta = add_time_decay_graph_scores(train_df, scored, half_lives=[90, 365, 730], ease_half_life=365, ease_lambda=1000.0)
    scored, svd_meta = add_svd_scores(train_df, scored, dim=64, seed=seed)
    scored, text_meta = add_review_pseudocat_scores(train_df, scored, raw_train_json, seed=seed, clusters=32, svd_dim=32, max_features=50000)
    fcols = feature_columns(scored)
    scored, cw_meta = add_weighted_implicit_logit(train_df, scored, raw_train_json, seed=seed, feature_cols=fcols)

    # Blends: all normalized within user to respect per-user top-half ranking.
    base_cols = feature_columns(scored) + ["score_cw_weighted_implicit_logit"]
    scored = normalize_within_user(scored, base_cols)
    zcols = [f"z_{c}" for c in base_cols if f"z_{c}" in scored.columns]
    scored["score_next_blend_mean_z"] = scored[zcols].mean(axis=1).astype(np.float32)
    priority = [
        "z_score_cw_weighted_implicit_logit",
        "z_score_icpns_comm_global_blend",
        "z_score_time_ease_hl365_lambda1000",
        "z_score_time_itemknn_hl365_top3",
        "z_score_review_pseudocat_blend",
        "z_score_graph_svd_k64",
    ]
    use_priority = [c for c in priority if c in scored.columns]
    scored["score_next_blend_priority_z"] = scored[use_priority].mean(axis=1).astype(np.float32) if use_priority else scored["score_next_blend_mean_z"]

    eval_cols = base_cols + ["score_next_blend_mean_z", "score_next_blend_priority_z"]
    summaries = []
    split_out = ensure_dir(out_root / split_dir.name)
    for col in eval_cols:
        summary, _ = evaluate_tophalf(scored, col, label_col="Label", user_col="userID", id_col="ID", tie_cols=[("pop_count", True), ("gameID", False)])
        summaries.append(summary)
    summaries = sorted(summaries, key=lambda s: (s["row_accuracy"], s["per_user_mean_accuracy"]), reverse=True)
    keep_cols = ["ID", "userID", "gameID", "Label", "pop_count"] + eval_cols
    scored[keep_cols].to_csv(split_out / "next_step_scores.csv", index=False)
    result = {
        "split": split_dir.name,
        "rows": int(len(scored)),
        "train_rows": int(len(train_df)),
        "duration_sec": round(time.time() - started, 3),
        "summaries": summaries,
        "community_meta": comm_meta,
        "graph_meta": graph_meta,
        "svd_meta": svd_meta,
        "text_meta": text_meta,
        "cw_meta": cw_meta,
    }
    write_json(split_out / "summary.json", result)
    return result


def write_run_reports(results: list[dict[str, object]], community_summaries: list[dict[str, object]], report_md: Path, report_json: Path) -> None:
    payload = {
        "note": "Validation-only paper-guided next-step run. No Kaggle submission.",
        "community_splits": community_summaries,
        "results": results,
    }
    write_json(report_json, payload)
    lines = [
        "# KMU RecSys 26 Steam — paper-guided next-step run",
        "",
        "Validation-only run covering ICPNS-style community negatives, CW/PU-inspired weighted implicit scorer, time-decay graph scores, and train-only review pseudo-categories. No Kaggle submission was performed.",
        "",
        "## Community-aware validation splits built",
        "",
        "| split | rows | users | positives | negatives | communities |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in community_summaries:
        cm = s.get("community_meta", {})
        lines.append(f"| `{s['name']}` | {s['candidate_rows']} | {s['users']} | {s['positives']} | {s['negatives']} | {cm.get('n_communities')} |")
    lines += ["", "## Best score by split", "", "| split | best score | row acc | per-user mean acc | duration sec |", "|---|---|---:|---:|---:|"]
    for r in results:
        b = r["summaries"][0]
        lines.append(f"| `{r['split']}` | `{b['score_col']}` | {b['row_accuracy']:.6f} | {b['per_user_mean_accuracy']:.6f} | {r['duration_sec']} |")
    lines += ["", "## Full score tables", ""]
    for r in results:
        lines += [f"### {r['split']}", "", "| rank | score | row acc | per-user mean acc |", "|---:|---|---:|---:|"]
        for i, s in enumerate(r["summaries"], 1):
            lines.append(f"| {i} | `{s['score_col']}` | {s['row_accuracy']:.6f} | {s['per_user_mean_accuracy']:.6f} |")
        lines.append("")
    lines += [
        "## Promotion interpretation",
        "",
        "- `score_cw_weighted_implicit_logit` is the low-cost PURL/CW proxy: it trains a weighted implicit classifier on fold-train positives and community-reliable sampled negatives.",
        "- `score_icpns_*` and the `val_*_communitypop_seed42` splits are the ICPNS-style exposure/community validation work.",
        "- `score_time_itemknn_*` and `score_time_ease_*` are TFPS-style time-decay graph probes.",
        "- `score_review_pseudocat_*` uses only train reviews to create pseudo semantic categories; it avoids external Steam metadata.",
        "- Any future submission candidate must still beat the existing Stage2 gates and must be approved explicitly by 우현 before Kaggle submission.",
        "",
    ]
    report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--validation-root", default="artifacts/validation")
    ap.add_argument("--out-root", default="artifacts/paper_guided_next_steps_20260530")
    ap.add_argument("--report-md", default="reports/20260530_paper_guided_next_steps.md")
    ap.add_argument("--report-json", default="reports/20260530_paper_guided_next_steps.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score-splits", nargs="*", default=[
        "val_random_sqrtpop_seed42",
        "val_recent_sqrtpop_seed42",
        "val_random_popbin_seed42",
        "val_random_communitypop_seed42",
        "val_recent_communitypop_seed42",
    ])
    ap.add_argument("--skip-build-community-splits", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    validation_root = ensure_dir(args.validation_root)
    out_root = ensure_dir(args.out_root)
    train_df = load_train_json(data_dir / "train.json")
    pairs_df = load_pairs_csv(data_dir / "pairs.csv")

    community_summaries: list[dict[str, object]] = []
    if not args.skip_build_community_splits:
        for cfg in [
            CommunitySplitConfig("random", args.seed, f"val_random_communitypop_seed{args.seed}"),
            CommunitySplitConfig("recent", args.seed, f"val_recent_communitypop_seed{args.seed}"),
        ]:
            community_summaries.append(build_community_split(train_df, pairs_df, cfg, validation_root, n_communities=24, svd_dim=32))

    results = []
    raw_train_json = data_dir / "train.json"
    for split in args.score_splits:
        split_dir = validation_root / split
        if not split_dir.exists():
            raise FileNotFoundError(split_dir)
        print(f"[score] {split}", flush=True)
        results.append(score_split(split_dir, raw_train_json, out_root, seed=args.seed))

    write_run_reports(results, community_summaries, Path(args.report_md), Path(args.report_json))
    print(json.dumps({"report_md": args.report_md, "report_json": args.report_json, "splits": len(results), "community_splits": len(community_summaries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
