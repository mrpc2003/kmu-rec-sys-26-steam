#!/usr/bin/env python3
"""Deep data-signature EDA for KMU RecSys 26 Steam.

Validation-only / analysis-only:
- reads public competition train/pairs and local validation score artifacts;
- writes JSON + Markdown EDA reports;
- never writes a Kaggle submission CSV and never calls Kaggle APIs.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from recsys_played_utils import load_pairs_csv, load_train_interactions, predict_tophalf, write_json

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "public" / "data"
OUT_JSON = ROOT / "reports" / "20260601_deep_data_signature_eda.json"
OUT_MD = ROOT / "reports" / "20260601_deep_data_signature_eda.md"


def q(values: Iterable[float], ps=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)) -> dict[str, float | int | None]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, **{f"p{p:g}": None for p in ps}}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        **{f"p{p:g}": float(np.percentile(arr, p)) for p in ps},
    }


def gini(values: Iterable[float]) -> float:
    arr = np.sort(np.asarray(list(values), dtype=float))
    arr = arr[arr >= 0]
    if arr.size == 0 or arr.sum() == 0:
        return 0.0
    n = arr.size
    return float((2 * np.sum((np.arange(n) + 1) * arr) / (n * arr.sum())) - (n + 1) / n)


def ks_stat(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.sort(np.asarray(list(a), dtype=float))
    b = np.sort(np.asarray(list(b), dtype=float))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.sort(np.unique(np.concatenate([a, b])))
    fa = np.searchsorted(a, grid, side="right") / a.size
    fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def corr(a: Iterable[float], b: Iterable[float], log: bool = False) -> float | None:
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if log:
        x = np.log1p(np.maximum(x, 0))
        y = np.log1p(np.maximum(y, 0))
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def auc_binary(y: Iterable[int], score: Iterable[float]) -> float | None:
    y_arr = np.asarray(list(y), dtype=int)
    s = pd.Series(list(score), dtype=float)
    mask = np.isfinite(s.to_numpy())
    y_arr = y_arr[mask]
    s = s[mask]
    n_pos = int((y_arr == 1).sum())
    n_neg = int((y_arr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = s.rank(method="average").to_numpy()
    sum_pos = float(ranks[y_arr == 1].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def residualize(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    mask = np.isfinite(x) & np.isfinite(z)
    out = np.zeros_like(x, dtype=float)
    if mask.sum() < 3 or np.std(z[mask]) == 0:
        out[:] = x - np.nanmean(x)
        return out
    X = np.column_stack([np.ones(mask.sum()), z[mask]])
    beta, *_ = np.linalg.lstsq(X, x[mask], rcond=None)
    out[mask] = x[mask] - X @ beta
    out[~mask] = 0.0
    return out


def md_table(rows: list[dict], columns: list[str], digits: int = 4, max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for r in rows:
        vals = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float):
                if math.isnan(v):
                    vals.append("nan")
                elif abs(v) >= 100:
                    vals.append(f"{v:.1f}")
                else:
                    vals.append(f"{v:.{digits}f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def base_aggregates(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train.copy()
    tr["date_ordinal"] = pd.to_datetime(tr["date"]).map(lambda x: x.toordinal())
    user = tr.groupby("userID").agg(
        user_n=("gameID", "size"),
        user_hours_mean=("hours", "mean"),
        user_htr_mean=("hours_transformed", "mean"),
        user_text_mean=("text_len", "mean"),
        user_early_rate=("early_access", "mean"),
        user_first_date=("date", "min"),
        user_last_date=("date", "max"),
        user_date_mean_ord=("date_ordinal", "mean"),
    )
    user["user_active_days"] = (user["user_last_date"] - user["user_first_date"]).dt.days.astype(int)
    item = tr.groupby("gameID").agg(
        item_n=("userID", "size"),
        item_hours_mean=("hours", "mean"),
        item_htr_mean=("hours_transformed", "mean"),
        item_text_mean=("text_len", "mean"),
        item_early_rate=("early_access", "mean"),
        item_first_date=("date", "min"),
        item_last_date=("date", "max"),
        item_date_mean_ord=("date_ordinal", "mean"),
    )
    item["item_active_days"] = (item["item_last_date"] - item["item_first_date"]).dt.days.astype(int)
    item["item_log_n"] = np.log1p(item["item_n"])
    return user, item


def matrix_and_feature_builder(train: pd.DataFrame):
    users = sorted(train["userID"].unique())
    items = sorted(train["gameID"].unique())
    u2i = {u: i for i, u in enumerate(users)}
    g2i = {g: i for i, g in enumerate(items)}
    rows = train["userID"].map(u2i).to_numpy(np.int32)
    cols = train["gameID"].map(g2i).to_numpy(np.int32)
    R = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (rows, cols)), shape=(len(users), len(items)))
    R.data[:] = 1.0
    item_pop = np.asarray(R.sum(axis=0)).ravel().astype(np.float32)
    cooc = (R.T @ R).astype(np.float32).toarray()
    np.fill_diagonal(cooc, 0.0)
    denom = np.sqrt(np.maximum(item_pop[:, None] * item_pop[None, :], 1.0))
    cosine = cooc / denom
    np.fill_diagonal(cosine, 0.0)

    def add_features(candidates: pd.DataFrame) -> pd.DataFrame:
        user_agg, item_agg = base_aggregates(train)
        p = candidates.copy()
        p = p.join(user_agg, on="userID")
        p = p.join(item_agg, on="gameID")
        p["candidate_count"] = p.groupby("userID")["gameID"].transform("size")
        p["known_k"] = p["candidate_count"] // 2
        hist_cos_max = []
        hist_cos_top3 = []
        hist_cos_sum = []
        hist_cooc_sum = []
        hist_cooc_max = []
        hist_pop_mean = []
        hist_pop_std = []
        pop_z = []
        for u, g, item_n in p[["userID", "gameID", "item_n"]].itertuples(index=False):
            if u not in u2i or g not in g2i:
                vals = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            else:
                ui = u2i[u]
                gi = g2i[g]
                hist = R[ui].indices
                sims = cosine[gi, hist]
                co = cooc[gi, hist]
                hp = item_pop[hist]
                if sims.size == 0:
                    vals = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    topn = np.sort(sims)[-min(3, sims.size):]
                    hp_mean = float(hp.mean())
                    hp_std = float(hp.std())
                    vals = (
                        float(sims.max()),
                        float(topn.mean()),
                        float(sims.sum()),
                        float(co.sum()),
                        float(co.max()),
                        hp_mean,
                        hp_std,
                        float((float(item_n) - hp_mean) / max(hp_std, 1.0)),
                    )
            a, b, c, d, e, f, h, z = vals
            hist_cos_max.append(a)
            hist_cos_top3.append(b)
            hist_cos_sum.append(c)
            hist_cooc_sum.append(d)
            hist_cooc_max.append(e)
            hist_pop_mean.append(f)
            hist_pop_std.append(h)
            pop_z.append(z)
        p["hist_cos_max"] = hist_cos_max
        p["hist_cos_top3"] = hist_cos_top3
        p["hist_cos_sum"] = hist_cos_sum
        p["hist_cooc_sum"] = hist_cooc_sum
        p["hist_cooc_max"] = hist_cooc_max
        p["hist_pop_mean"] = hist_pop_mean
        p["hist_pop_std"] = hist_pop_std
        p["candidate_pop_z_vs_user_hist"] = pop_z
        return p

    return R, u2i, g2i, item_pop, add_features


def graph_stats(train: pd.DataFrame) -> dict:
    users = sorted(train["userID"].unique())
    items = sorted(train["gameID"].unique())
    u2i = {u: i for i, u in enumerate(users)}
    g2i = {g: i for i, g in enumerate(items)}
    row = train["userID"].map(u2i).to_numpy(np.int32)
    col = train["gameID"].map(g2i).to_numpy(np.int32)
    R = sp.csr_matrix((np.ones(len(train), dtype=np.int8), (row, col)), shape=(len(users), len(items)))
    A = sp.bmat([[None, R], [R.T, None]], format="csr")
    n_comp, labels = connected_components(A, directed=False)
    sizes = Counter(labels)
    sorted_sizes = sorted(sizes.values(), reverse=True)
    return {
        "matrix_shape_users_items": [len(users), len(items)],
        "interactions": int(len(train)),
        "density": float(len(train) / (len(users) * len(items))),
        "connected_components": int(n_comp),
        "largest_component_nodes": int(sorted_sizes[0]),
        "largest_component_node_share": float(sorted_sizes[0] / (len(users) + len(items))),
        "component_size_top10": [int(x) for x in sorted_sizes[:10]],
    }


def candidate_structure(train: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    train_pairs = set(zip(train["userID"], train["gameID"]))
    user_agg, item_agg = base_aggregates(train)
    cand_n = pairs.groupby("userID").size().rename("candidate_count")
    user_diag = user_agg.join(cand_n, how="inner")
    user_diag["known_k"] = (user_diag["candidate_count"] // 2).astype(int)
    user_diag["test_frac_vs_train"] = user_diag["known_k"] / user_diag["user_n"]
    item_cand = pairs.groupby("gameID").size().rename("candidate_item_n")
    item_diag = item_agg.join(item_cand, how="left").fillna({"candidate_item_n": 0})
    item_diag["candidate_item_n"] = item_diag["candidate_item_n"].astype(int)

    k_counts = user_diag["known_k"].value_counts().sort_index()
    k_rows = [
        {
            "K": int(k),
            "users": int(v),
            "user_share": float(v / len(user_diag)),
            "median_train_n": float(user_diag.loc[user_diag["known_k"] == k, "user_n"].median()),
            "mean_test_frac": float(user_diag.loc[user_diag["known_k"] == k, "test_frac_vs_train"].mean()),
        }
        for k, v in k_counts.items()
    ]
    bins = [0, 10, 20, 30, 40, 60, 80, 120, 200, 10_000]
    user_diag["train_n_bin"] = pd.cut(user_diag["user_n"], bins=bins, right=True).astype(str)
    degree_rows = []
    for b, gp in user_diag.groupby("train_n_bin", observed=True):
        degree_rows.append({
            "train_n_bin": str(b),
            "users": int(len(gp)),
            "median_K": float(gp["known_k"].median()),
            "mean_K": float(gp["known_k"].mean()),
            "median_test_frac": float(gp["test_frac_vs_train"].median()),
        })

    return {
        "pairs_rows": int(len(pairs)),
        "pair_users": int(pairs["userID"].nunique()),
        "pair_games": int(pairs["gameID"].nunique()),
        "cold_users": int((~pairs["userID"].isin(set(train["userID"]))).sum()),
        "cold_games": int((~pairs["gameID"].isin(set(train["gameID"]))).sum()),
        "train_pair_overlap_rows": int(sum((u, g) in train_pairs for u, g in pairs[["userID", "gameID"]].itertuples(index=False))),
        "all_candidate_counts_even": bool((cand_n % 2 == 0).all()),
        "known_positive_total_from_half": int((cand_n // 2).sum()),
        "known_k_stats": q(user_diag["known_k"]),
        "candidate_count_distribution": {str(int(k)): int(v) for k, v in cand_n.value_counts().sort_index().items()},
        "k_distribution_rows": k_rows,
        "degree_bin_rows": degree_rows,
        "corr_known_k_train_n": corr(user_diag["known_k"], user_diag["user_n"]),
        "corr_candidate_count_train_n": corr(user_diag["candidate_count"], user_diag["user_n"]),
        "test_frac_stats": q(user_diag["test_frac_vs_train"]),
        "pair_user_train_n_stats": q(user_diag["user_n"]),
        "nonpair_user_train_n_stats": q(user_agg.loc[~user_agg.index.isin(set(pairs["userID"])), "user_n"]),
        "candidate_item_n_stats": q(pairs.join(item_agg, on="gameID")["item_n"]),
        "pair_game_train_n_stats": q(item_agg.loc[item_agg.index.isin(set(pairs["gameID"])), "item_n"]),
        "nonpair_game_train_n_stats": q(item_agg.loc[~item_agg.index.isin(set(pairs["gameID"])), "item_n"]),
        "item_candidate_count_stats": q(item_diag["candidate_item_n"]),
        "corr_item_candidate_count_item_pop": corr(item_diag["candidate_item_n"], item_diag["item_n"]),
        "corr_log_item_candidate_count_log_item_pop": corr(item_diag["candidate_item_n"], item_diag["item_n"], log=True),
        "top_candidate_items": [
            {"gameID": idx, "candidate_item_n": int(row["candidate_item_n"]), "train_item_n": int(row["item_n"])}
            for idx, row in item_diag.sort_values("candidate_item_n", ascending=False).head(12).iterrows()
        ],
    }


def simulate_candidate_mixture(train: pd.DataFrame, pairs: pd.DataFrame, repeats: int = 5) -> dict:
    user_agg, item_agg = base_aggregates(train)
    item_ids = np.asarray(item_agg.index.tolist(), dtype=object)
    pop = item_agg["item_n"].to_numpy(dtype=float)
    id_to_idx = {g: i for i, g in enumerate(item_ids)}
    hist_by_user = {
        u: np.asarray([id_to_idx[g] for g in gp["gameID"].tolist() if g in id_to_idx], dtype=np.int32)
        for u, gp in train.groupby("userID", sort=False)
    }
    recent_hist_by_user = {}
    popular_hist_by_user = {}
    for u, gp in train.sort_values("date").groupby("userID", sort=False):
        idx = np.asarray([id_to_idx[g] for g in gp["gameID"].tolist() if g in id_to_idx], dtype=np.int32)
        recent_hist_by_user[u] = idx[::-1]
        popular_hist_by_user[u] = idx[np.argsort(pop[idx])[::-1]]
    cand_n = pairs.groupby("userID").size()
    actual_pair_item_pop = pairs.join(item_agg, on="gameID")["item_n"].to_numpy(dtype=float)
    actual_user_mean_pop = pairs.join(item_agg, on="gameID").groupby("userID")["item_n"].mean().reindex(cand_n.index).to_numpy(dtype=float)
    actual_user_max_pop = pairs.join(item_agg, on="gameID").groupby("userID")["item_n"].max().reindex(cand_n.index).to_numpy(dtype=float)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    modes = ["random_positive", "recent_positive", "popular_history_positive"]
    rows = []
    neg_only_rows = []
    for alpha in alphas:
        base_w = np.power(pop, alpha)
        for mode in modes:
            ks_all = []
            ks_user_mean = []
            ks_user_max = []
            cand_item_corrs = []
            for rep in range(repeats):
                rng = np.random.default_rng(10_000 + rep * 97 + int(alpha * 1000))
                all_idx = []
                user_means = []
                user_maxs = []
                item_counts = np.zeros(len(item_ids), dtype=np.int32)
                for u, c in cand_n.items():
                    k = int(c // 2)
                    if mode == "recent_positive":
                        hist = recent_hist_by_user[u]
                        pos = hist[:k]
                    elif mode == "popular_history_positive":
                        hist = popular_hist_by_user[u]
                        pos = hist[:k]
                    else:
                        hist = hist_by_user[u]
                        pos = rng.choice(hist, size=min(k, len(hist)), replace=False)
                    mask = np.ones(len(item_ids), dtype=bool)
                    mask[hist_by_user[u]] = False
                    unseen = np.flatnonzero(mask)
                    w = base_w[unseen]
                    w = w / w.sum()
                    neg = rng.choice(unseen, size=k, replace=False, p=w)
                    idx = np.concatenate([pos, neg])
                    all_idx.extend(idx.tolist())
                    item_counts[idx] += 1
                    vals = pop[idx]
                    user_means.append(float(vals.mean()))
                    user_maxs.append(float(vals.max()))
                vals = pop[np.asarray(all_idx, dtype=np.int32)]
                ks_all.append(ks_stat(actual_pair_item_pop, vals))
                ks_user_mean.append(ks_stat(actual_user_mean_pop, user_means))
                ks_user_max.append(ks_stat(actual_user_max_pop, user_maxs))
                cand_item_corrs.append(corr(item_counts, pop, log=True) or float("nan"))
            rows.append({
                "mode": mode,
                "negative_alpha": alpha,
                "ks_item_pop_all_rows": float(np.mean(ks_all)),
                "ks_user_mean_item_pop": float(np.mean(ks_user_mean)),
                "ks_user_max_item_pop": float(np.mean(ks_user_max)),
                "mean_three_ks": float(np.mean([np.mean(ks_all), np.mean(ks_user_mean), np.mean(ks_user_max)])),
                "corr_log_candidate_item_count_log_pop": float(np.nanmean(cand_item_corrs)),
            })

        # diagnostic: the old, flawed fit that treats all candidate rows as negatives.
        neg_ks = []
        for rep in range(repeats):
            rng = np.random.default_rng(20_000 + rep * 101 + int(alpha * 1000))
            all_idx = []
            for u, c in cand_n.items():
                hist = hist_by_user[u]
                mask = np.ones(len(item_ids), dtype=bool)
                mask[hist] = False
                unseen = np.flatnonzero(mask)
                w = base_w[unseen]
                w = w / w.sum()
                idx = rng.choice(unseen, size=int(c), replace=False, p=w)
                all_idx.extend(idx.tolist())
            neg_ks.append(ks_stat(actual_pair_item_pop, pop[np.asarray(all_idx, dtype=np.int32)]))
        neg_only_rows.append({"negative_alpha": alpha, "ks_if_all_rows_were_negative": float(np.mean(neg_ks))})

    rows_sorted = sorted(rows, key=lambda r: r["mean_three_ks"])
    return {
        "mixture_fit_rows_sorted": rows_sorted,
        "all_negative_fit_rows_sorted": sorted(neg_only_rows, key=lambda r: r["ks_if_all_rows_were_negative"]),
        "interpretation": "Candidate all-row popularity can look sqrt-pop even when negatives are uniform, because half the rows are held-out positives whose item distribution is already popularity-skewed.",
    }


def validation_diagnostics(train: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    score_path = ROOT / "artifacts" / "last_slot_rankagg" / "rankagg_seed42_scores.csv"
    if not score_path.exists():
        return {"available": False, "reason": f"missing {score_path}"}
    scores = pd.read_csv(score_path)
    base_col = "base_emb128_raw_mean"
    pred = predict_tophalf(scores, base_col, label_col="Label")
    pred["correct"] = (pred["Pred"] == pred["Label"]).astype(int)
    val_train_path = ROOT / "artifacts" / "validation" / "val_random_uniform_seed42" / "train_interactions.csv"
    val_train = load_train_interactions(val_train_path)
    _, _, _, _, add_val_features = matrix_and_feature_builder(val_train)
    vf = add_val_features(pred)

    # user-level score margin at the K/K+1 boundary.
    margins = {}
    for u, gp in vf.sort_values(["userID", base_col], ascending=[True, False]).groupby("userID", sort=False):
        k = int(gp["Label"].sum())
        if k <= 0 or k >= len(gp):
            margins[u] = np.nan
        else:
            vals = gp[base_col].to_numpy(dtype=float)
            margins[u] = float(vals[k - 1] - vals[k])
    vf["user_boundary_margin"] = vf["userID"].map(margins)

    def bucket_accuracy(col: str, bins: int = 4) -> list[dict]:
        tmp = vf[[col, "correct"]].dropna().copy()
        if tmp[col].nunique() <= bins:
            groups = tmp.groupby(col, observed=True)
            return [{"bucket": str(k), "rows": int(len(g)), "accuracy": float(g["correct"].mean())} for k, g in groups]
        tmp["bucket"] = pd.qcut(tmp[col], q=bins, duplicates="drop")
        return [{"bucket": str(k), "rows": int(len(g)), "accuracy": float(g["correct"].mean())} for k, g in tmp.groupby("bucket", observed=True)]

    feature_cols = [
        "item_n", "item_log_n", "hist_cos_top3", "hist_cooc_sum",
        "candidate_pop_z_vs_user_hist", "item_htr_mean", "item_text_mean", "item_early_rate",
    ]
    feature_auc_rows = []
    for col in feature_cols:
        if col in vf.columns:
            feature_auc_rows.append({
                "feature": col,
                "row_auc_label": auc_binary(vf["Label"], vf[col]),
                "corr_with_base_score": corr(vf[col], vf[base_col]),
                "feature_tophalf_acc": float(predict_tophalf(vf[["ID", "userID", "gameID", "Label", col]].rename(columns={col: "score"}), "score", label_col="Label")["Pred"].eq(vf["Label"]).mean()),
            })

    # K/K+1 boundary pair diagnostics.
    boundary_rows = []
    for u, gp in vf.sort_values(["userID", base_col], ascending=[True, False]).groupby("userID", sort=False):
        k = int(gp["Label"].sum())
        if k <= 0 or k >= len(gp):
            continue
        top = gp.iloc[k - 1]
        bot = gp.iloc[k]
        if int(top["Label"]) == int(bot["Label"]):
            continue
        target_top_is_positive = int(top["Label"] == 1)
        row = {"top_is_positive": target_top_is_positive, "d_base_score": float(top[base_col] - bot[base_col])}
        for col in feature_cols:
            if col in vf.columns:
                row[f"d_{col}"] = float(top[col] - bot[col])
        boundary_rows.append(row)
    bdf = pd.DataFrame(boundary_rows)
    boundary_auc_rows = []
    if not bdf.empty:
        d_logpop = bdf.get("d_item_log_n", pd.Series(np.zeros(len(bdf)))).to_numpy(dtype=float)
        for col in [c for c in bdf.columns if c.startswith("d_") and c != "d_base_score"]:
            raw_auc = auc_binary(bdf["top_is_positive"], bdf[col])
            resid_auc = auc_binary(bdf["top_is_positive"], residualize(bdf[col].to_numpy(dtype=float), d_logpop)) if col != "d_item_log_n" else None
            boundary_auc_rows.append({
                "boundary_delta_feature": col,
                "raw_auc_top_candidate_is_true_positive": raw_auc,
                "logpop_residual_auc": resid_auc,
            })

    # Compare actual candidate feature distribution to the calibrated uniform validation split.
    _, _, _, _, add_full_features = matrix_and_feature_builder(train)
    actual_feat = add_full_features(pairs)
    val_feat = vf
    ks_rows = []
    for col in ["candidate_count", "user_n", "item_n", "hist_cos_top3", "hist_cooc_sum", "candidate_pop_z_vs_user_hist"]:
        if col in actual_feat.columns and col in val_feat.columns:
            ks_rows.append({
                "feature": col,
                "ks_actual_pairs_vs_uniform_val": ks_stat(actual_feat[col], val_feat[col]),
                "actual_median": float(np.nanmedian(actual_feat[col])),
                "val_median": float(np.nanmedian(val_feat[col])),
            })

    return {
        "available": True,
        "score_path": str(score_path.relative_to(ROOT)),
        "base_col": base_col,
        "row_accuracy": float(vf["correct"].mean()),
        "per_user_mean_accuracy": float(vf.groupby("userID")["correct"].mean().mean()),
        "accuracy_by_known_k": bucket_accuracy("known_k"),
        "accuracy_by_user_n_quartile": bucket_accuracy("user_n"),
        "accuracy_by_item_n_quartile": bucket_accuracy("item_n"),
        "accuracy_by_boundary_margin_quartile": bucket_accuracy("user_boundary_margin"),
        "feature_auc_rows": feature_auc_rows,
        "boundary_pairs_count": int(len(bdf)),
        "boundary_delta_auc_rows": boundary_auc_rows,
        "actual_vs_uniform_validation_feature_ks": ks_rows,
    }


def train_schema_stats(train: pd.DataFrame) -> dict:
    user_n = train.groupby("userID").size()
    item_n = train.groupby("gameID").size()
    total_possible = train["userID"].nunique() * train["gameID"].nunique()
    top_cover = {}
    counts = item_n.sort_values(ascending=False).to_numpy()
    csum = np.cumsum(counts)
    for frac in [0.25, 0.5, 0.75, 0.9, 0.95]:
        n_items = int(np.searchsorted(csum, counts.sum() * frac, side="left") + 1)
        top_cover[str(frac)] = {"n_games": n_items, "game_share": float(n_items / len(counts))}
    return {
        "train_rows": int(len(train)),
        "users": int(train["userID"].nunique()),
        "games": int(train["gameID"].nunique()),
        "duplicate_user_game_rows": int(len(train) - train.drop_duplicates(["userID", "gameID"]).shape[0]),
        "density": float(len(train) / total_possible),
        "user_degree_stats": q(user_n),
        "item_degree_stats": q(item_n),
        "user_degree_gini": gini(user_n),
        "item_degree_gini": gini(item_n),
        "top_item_coverage": top_cover,
        "date_min": str(train["date"].min().date()),
        "date_max": str(train["date"].max().date()),
        "year_counts": {str(int(k)): int(v) for k, v in train["date"].dt.year.value_counts().sort_index().items()},
        "hours_stats": q(train["hours"]),
        "hours_transformed_stats": q(train["hours_transformed"]),
        "text_len_stats": q(train["text_len"]),
        "blank_text_rows": int((train["text_len"] == 0).sum()),
        "early_access_rate": float(train["early_access"].mean()),
        "found_funny_present_rate": float(train.get("has_found_funny", pd.Series(False, index=train.index)).mean()),
        "compensation_present_rate": float(train.get("has_compensation", pd.Series(False, index=train.index)).mean()),
    }


def build_report(out: dict) -> str:
    cs = out["candidate_structure"]
    sim = out["candidate_generation_mixture"]
    val = out["validation_diagnostics"]
    graph = out["graph_stats"]
    train = out["train_schema"]

    best_mix = sim["mixture_fit_rows_sorted"][:8]
    all_neg = sim["all_negative_fit_rows_sorted"][:6]
    lines = []
    lines.append("# 2026-06-01 Deep Data-Signature EDA — KMU RecSys 26 Steam")
    lines.append("")
    lines.append("분석 범위: 제공된 `train.json`, `pairs.csv`, 로컬 validation/score artifact만 사용했다. Kaggle 제출, hidden label 추정/외부 수집, submission CSV 생성은 수행하지 않았다.")
    lines.append("")
    lines.append("## 1. 핵심 결론")
    lines.append("")
    lines.append("1. 이 데이터는 **cold-start가 전혀 없는, 작은 item universe의 known-user/known-item constrained ranking**이다. bipartite graph의 largest component share가 "
                 f"{graph['largest_component_node_share']:.4f}이고 matrix density가 {train['density']:.4%}라서 LightGCN/EASE류가 같은 공기(co-occurrence)를 거의 다 빨아먹기 쉽다.")
    lines.append("2. `pairs.csv`의 모든 user candidate 수가 짝수이고 hidden positive 총량은 구조적으로 "
                 f"{cs['known_positive_total_from_half']:,}개다. K=1 또는 2인 유저가 대부분이라, 긴 list 추천보다 **K/K+1 boundary** 문제가 본질이다.")
    lines.append("3. 초반 EDA의 `candidate item popularity ≈ sqrt-pop` 결론은 절반만 맞다. 모든 candidate row를 negative처럼 맞추면 alpha≈0.5가 나오지만, **candidate는 positive 50% + negative 50% 혼합**이다. train-like held-out positive + uniform negative 혼합도 actual pair marginal을 강하게 설명한다. 따라서 public이 uniform surrogate를 따라간 기존 관찰과 모순되지 않는다.")
    lines.append("4. pair-level side feature(`cooc`, `cos`, `hours`, `text`, `early_access`)는 대부분 item popularity와 confound되어 있다. raw AUC/상관이 있어 보여도 log-pop residual boundary AUC를 봐야 한다.")
    lines.append("5. 남은 탐색은 새 encoder가 아니라 **K-aware objective / boundary-only fine-tune / residualized feature gate**처럼 데이터 구조를 직접 겨냥해야 한다.")
    lines.append("")

    lines.append("## 2. Train/test 구조")
    lines.append("")
    lines.append(f"- train rows/users/games: {train['train_rows']:,} / {train['users']:,} / {train['games']:,}")
    lines.append(f"- duplicate user-game rows: {train['duplicate_user_game_rows']:,}")
    lines.append(f"- user×item density: {train['density']:.4%}")
    lines.append(f"- user degree Gini / item degree Gini: {train['user_degree_gini']:.4f} / {train['item_degree_gini']:.4f}")
    lines.append(f"- graph components: {graph['connected_components']}개, largest component node share: {graph['largest_component_node_share']:.4f}")
    lines.append(f"- pairs rows/users/games: {cs['pairs_rows']:,} / {cs['pair_users']:,} / {cs['pair_games']:,}")
    lines.append(f"- cold users/games rows: {cs['cold_users']:,} / {cs['cold_games']:,}; train-pair overlap rows: {cs['train_pair_overlap_rows']:,}")
    lines.append(f"- candidate item count vs train item popularity corr(log-log): {cs['corr_log_item_candidate_count_log_item_pop']:.4f}")
    lines.append("")
    lines.append("### K 분포")
    lines.append("")
    lines.append(md_table(cs["k_distribution_rows"], ["K", "users", "user_share", "median_train_n", "mean_test_frac"], max_rows=12))
    lines.append("")
    lines.append("해석: K=1/2가 지배적이므로, `NDCG@large K`식 방법보다 rank-`K`와 rank-`K+1` 사이를 직접 움직이는 방법이 더 적합하다.")
    lines.append("")

    lines.append("## 3. Candidate generation signature")
    lines.append("")
    lines.append("아래는 실제 `pairs.csv`의 item-pop/user-mean-pop/user-max-pop 분포를, `train history에서 K개 positive holdout + unseen negative K개` 시뮬레이션으로 맞춘 결과다. 낮을수록 actual pairs와 가깝다.")
    lines.append("")
    lines.append(md_table(best_mix, ["mode", "negative_alpha", "ks_item_pop_all_rows", "ks_user_mean_item_pop", "ks_user_max_item_pop", "mean_three_ks", "corr_log_candidate_item_count_log_pop"], digits=4))
    lines.append("")
    lines.append("비교용으로 **모든 candidate row를 negative라고 잘못 가정**하면 다음처럼 alpha≈0.5가 best처럼 보인다.")
    lines.append("")
    lines.append(md_table(all_neg, ["negative_alpha", "ks_if_all_rows_were_negative"], digits=4))
    lines.append("")
    lines.append("해석: actual pair marginal이 sqrt-pop처럼 보였던 것은 positive half의 popularity skew가 섞인 효과다. 실제 negative half가 uniform-like일 가능성을 배제하지 않으며, 이미 public LB가 uniform split과 가장 가까웠던 경험적 사실을 유지해야 한다.")
    lines.append("")

    if val.get("available"):
        lines.append("## 4. Current-best validation error anatomy")
        lines.append("")
        lines.append(f"분석 score: `{val['score_path']}` / `{val['base_col']}`")
        lines.append(f"- row accuracy: {val['row_accuracy']:.5f}")
        lines.append(f"- per-user mean accuracy: {val['per_user_mean_accuracy']:.5f}")
        lines.append("")
        lines.append("### Accuracy by K")
        lines.append("")
        lines.append(md_table(val["accuracy_by_known_k"], ["bucket", "rows", "accuracy"], digits=4, max_rows=20))
        lines.append("")
        lines.append("### Actual pairs vs calibrated uniform validation feature KS")
        lines.append("")
        lines.append(md_table(val["actual_vs_uniform_validation_feature_ks"], ["feature", "ks_actual_pairs_vs_uniform_val", "actual_median", "val_median"], digits=4))
        lines.append("")
        lines.append("### Feature-only signal and base-score correlation")
        lines.append("")
        lines.append(md_table(val["feature_auc_rows"], ["feature", "row_auc_label", "corr_with_base_score", "feature_tophalf_acc"], digits=4))
        lines.append("")
        lines.append("### K/K+1 boundary delta AUC")
        lines.append("")
        lines.append(md_table(val["boundary_delta_auc_rows"], ["boundary_delta_feature", "raw_auc_top_candidate_is_true_positive", "logpop_residual_auc"], digits=4))
        lines.append("")
        lines.append("해석: base score 자체가 이미 popularity/co-occurrence 계열을 흡수했다. feature-only top-half accuracy가 높더라도 base와 상관이 높으면 새 축이 아니다. boundary에서 log-pop residual AUC가 0.55 부근 이하인 feature는 마지막 슬롯 실험으로 승격하지 않는다.")
        lines.append("")

    lines.append("## 5. 이 EDA가 탐색 전략을 어떻게 바꾸는가")
    lines.append("")
    lines.append("### 유지해야 할 것")
    lines.append("- 1차 gate는 계속 `val_random_uniform_seed42` 및 3-split uniform panel이다. 전체 candidate marginal만 보고 sqrt-pop을 primary로 되돌리면 안 된다.")
    lines.append("- 모든 검증은 user별 `K_u = candidate_count_u / 2`를 고정한 top-half decoding으로 해야 한다.")
    lines.append("- popularity/co-occurrence/text/hour feature는 raw gain이 아니라 **base 대비 residual + paired McNemar**로만 판단한다.")
    lines.append("")
    lines.append("### 새로 구체화된 probe")
    lines.append("1. **SL@K / TopKGAT-lite K-aware boundary fine-tune**: K가 작고 구조적으로 알려져 있으므로 가장 데이터 구조에 직접 맞는다. 단, old-loss continuation(A) vs new-loss fine-tune(B)로 분기해 `B-A`만 gate한다.")
    lines.append("2. **Ambiguity-only objective**: 모든 pair를 다시 학습하지 말고 rank `K/K+1` 근방, boundary margin 하위 분위 유저만 loss에 크게 반영한다. DNS와 달리 hard-negative pool이 아니라 실제 metric boundary에 조건을 건다.")
    lines.append("3. **Mixture-faithful validation refresh**: 새 probe마다 random-positive+uniform-negative split뿐 아니라 recent-positive+uniform-negative stress도 같이 본다. positive holdout mode가 바뀌어도 sign-stable한 축만 남긴다.")
    lines.append("4. **Residualized multi-interest cheap probe**: multi-interest를 하더라도 raw cooc/cos 말고 log-pop residualized history-similarity로 segment를 나눠야 한다. residual boundary AUC가 낮으면 구현하지 않는다.")
    lines.append("")
    lines.append("### 더 이상 우선하지 않을 것")
    lines.append("- 전체 candidate item-pop 분포를 맞추기 위한 pop-bias 추가/감산: positive+negative mixture confound 때문에 public-transfer trap이다.")
    lines.append("- 긴 sequence/transformer/diffusion 모델: K=1/2 유저가 대부분이고 no-cold small graph라, 새 encoder는 base score와 상관만 올라갈 가능성이 크다.")
    lines.append("- item-level global quota: candidate item marginal은 label marginal이 아니고, positive half와 negative half가 섞여 있어 per-user decision에 균일 적용하면 회귀하기 쉽다.")
    lines.append("")
    lines.append("## 6. Safety")
    lines.append("")
    lines.append("- `kaggle competitions submit` 호출 없음")
    lines.append("- submission CSV 생성 없음")
    lines.append("- hidden label 외부 수집/추론 없음")
    lines.append("- 산출물: JSON/Markdown EDA report only")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    train = load_train_interactions(DATA_DIR / "train.json")
    pairs = load_pairs_csv(DATA_DIR / "pairs.csv")
    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "safety": {
            "validation_only": True,
            "kaggle_submit_executed": False,
            "submission_csv_written": False,
            "uses_hidden_labels": False,
        },
        "train_schema": train_schema_stats(train),
        "graph_stats": graph_stats(train),
        "candidate_structure": candidate_structure(train, pairs),
        "candidate_generation_mixture": simulate_candidate_mixture(train, pairs),
        "validation_diagnostics": validation_diagnostics(train, pairs),
    }
    write_json(OUT_JSON, out)
    OUT_MD.write_text(build_report(out), encoding="utf-8")
    print(json.dumps({
        "json": str(OUT_JSON.relative_to(ROOT)),
        "markdown": str(OUT_MD.relative_to(ROOT)),
        "safety": out["safety"],
        "best_mixture_fit": out["candidate_generation_mixture"]["mixture_fit_rows_sorted"][:3],
        "validation_available": out["validation_diagnostics"].get("available"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
