#!/usr/bin/env python3
"""Deep EDA for KMU RecSys 26 Steam played prediction.

No submission generation. Uses only downloaded competition files.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "public" / "data"
REPORT = ROOT / "reports" / "deep_eda"
REPORT.mkdir(parents=True, exist_ok=True)

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_']{2,}")
STOP = {
    "the", "and", "for", "you", "this", "that", "with", "have", "are", "but", "not",
    "game", "games", "play", "played", "it's", "its", "was", "were", "from", "your",
    "all", "can", "just", "like", "good", "get", "will", "one", "out", "has", "had",
    "more", "very", "really", "than", "there", "what", "when", "into", "about", "they",
    "them", "would", "only", "time", "much", "even", "because", "some", "also", "their",
}


def q(values, ps=(0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)):
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {f"p{p:02g}": None for p in ps} | {"count": 0, "mean": None, "std": None}
    out = {f"p{p:02g}": float(np.percentile(arr, p)) for p in ps}
    out.update(count=int(arr.size), mean=float(arr.mean()), std=float(arr.std(ddof=0)))
    return out


def top_dict(counter: Counter, n=20):
    return [{"key": k, "count": int(v)} for k, v in counter.most_common(n)]


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def gini(values):
    vals = np.sort(np.asarray(list(values), dtype=float))
    vals = vals[vals >= 0]
    n = vals.size
    s = vals.sum()
    if n == 0 or s == 0:
        return 0.0
    return float((2 * np.sum((np.arange(n) + 1) * vals) / (n * s)) - (n + 1) / n)


def load_train():
    rows = []
    field_counts = Counter()
    with (DATA / "train.json").open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            d = ast.literal_eval(line)
            field_counts.update(d.keys())
            rows.append({
                "userID": d["userID"],
                "gameID": d["gameID"],
                "hours": float(d.get("hours", 0.0)),
                "hours_transformed": float(d.get("hours_transformed", 0.0)),
                "early_access": bool(d.get("early_access", False)),
                "date": d.get("date"),
                "text_len": len(d.get("text") or ""),
                "found_funny": float(d.get("found_funny", 0.0) or 0.0),
                "has_found_funny": "found_funny" in d,
                "has_user_id": "user_id" in d,
                "has_compensation": "compensation" in d,
                "text": d.get("text") or "",
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df, field_counts


def load_pairs():
    return pd.read_csv(DATA / "pairs.csv")


def aggregate_train(df):
    user = df.groupby("userID").agg(
        user_n=("gameID", "size"),
        user_unique_games=("gameID", "nunique"),
        user_hours_sum=("hours", "sum"),
        user_hours_mean=("hours", "mean"),
        user_hours_median=("hours", "median"),
        user_htr_mean=("hours_transformed", "mean"),
        user_early_rate=("early_access", "mean"),
        user_text_mean=("text_len", "mean"),
        user_funny_sum=("found_funny", "sum"),
        user_funny_present_rate=("has_found_funny", "mean"),
        user_first_date=("date", "min"),
        user_last_date=("date", "max"),
        user_active_days=("date", lambda s: int((s.max() - s.min()).days)),
    )
    item = df.groupby("gameID").agg(
        item_n=("userID", "size"),
        item_unique_users=("userID", "nunique"),
        item_hours_sum=("hours", "sum"),
        item_hours_mean=("hours", "mean"),
        item_hours_median=("hours", "median"),
        item_htr_mean=("hours_transformed", "mean"),
        item_early_rate=("early_access", "mean"),
        item_text_mean=("text_len", "mean"),
        item_funny_sum=("found_funny", "sum"),
        item_funny_present_rate=("has_found_funny", "mean"),
        item_first_date=("date", "min"),
        item_last_date=("date", "max"),
        item_active_days=("date", lambda s: int((s.max() - s.min()).days)),
    )
    # add popularity rank, percentiles
    item = item.sort_values("item_n", ascending=False)
    item["item_pop_rank"] = np.arange(1, len(item) + 1)
    item["item_pop_pct_from_top"] = item["item_pop_rank"] / len(item)
    item["item_log_n"] = np.log1p(item["item_n"])
    user["user_log_n"] = np.log1p(user["user_n"])
    return user, item


def make_interaction_arrays(df):
    users = sorted(df["userID"].unique())
    games = sorted(df["gameID"].unique())
    u2i = {u: i for i, u in enumerate(users)}
    g2i = {g: i for i, g in enumerate(games)}
    X = np.zeros((len(users), len(games)), dtype=np.int16)
    hours = np.zeros((len(users), len(games)), dtype=np.float32)
    for u, g, h in df[["userID", "gameID", "hours_transformed"]].itertuples(index=False):
        ui = u2i[u]
        gi = g2i[g]
        X[ui, gi] = 1
        hours[ui, gi] = float(h)
    deg = X.sum(axis=0).astype(np.float32)
    cooc = (X.T @ X).astype(np.float32)
    denom = np.sqrt(np.maximum(deg[:, None] * deg[None, :], 1.0))
    cosine = cooc / denom
    np.fill_diagonal(cosine, 0.0)
    np.fill_diagonal(cooc, 0.0)
    return users, games, u2i, g2i, X, hours, deg, cooc, cosine


def add_pair_features(pairs, df, user_agg, item_agg, u2i, g2i, X, hours, deg, cooc, cosine):
    train_pairs = set(zip(df["userID"], df["gameID"]))
    p = pairs.copy()
    p["in_train_pair"] = [(u, g) in train_pairs for u, g in p[["userID", "gameID"]].itertuples(index=False)]
    p = p.join(user_agg, on="userID")
    p = p.join(item_agg, on="gameID")
    # user candidate stats
    cand_user_counts = p.groupby("userID").size().rename("candidate_count")
    p = p.join(cand_user_counts, on="userID")
    # history-item similarity features
    max_cos = []
    mean_cos = []
    top3_cos = []
    sum_cos = []
    max_cooc = []
    sum_cooc = []
    mean_hist_pop = []
    max_hist_pop = []
    candidate_pop_gt_user_mean = []
    candidate_pop_z_user = []
    hist_htr_to_sim_items = []
    for row in p[["userID", "gameID", "item_n"]].itertuples(index=False):
        u, g, item_n = row
        ui = u2i[u]
        gi = g2i[g]
        hist_idx = np.flatnonzero(X[ui] > 0)
        sims = cosine[gi, hist_idx]
        co = cooc[gi, hist_idx]
        hist_deg = deg[hist_idx]
        if sims.size:
            max_cos.append(float(sims.max()))
            mean_cos.append(float(sims.mean()))
            top3_cos.append(float(np.sort(sims)[-min(3, sims.size):].mean()))
            sum_cos.append(float(sims.sum()))
            max_cooc.append(float(co.max()))
            sum_cooc.append(float(co.sum()))
            mean_hist_pop.append(float(hist_deg.mean()))
            max_hist_pop.append(float(hist_deg.max()))
            candidate_pop_gt_user_mean.append(bool(item_n > hist_deg.mean()))
            # Robust within-user popularity contrast.  A true z-score explodes for
            # users whose history has near-constant item popularity, so floor the
            # denominator at 1 interaction to keep the EDA statistic interpretable.
            candidate_pop_z_user.append(float((item_n - hist_deg.mean()) / max(float(hist_deg.std()), 1.0)))
            # hours-weighted similarity to user history
            hist_h = hours[ui, hist_idx]
            hist_htr_to_sim_items.append(float(np.sum(sims * hist_h) / (np.sum(hist_h) + 1e-6)))
        else:
            max_cos.append(0.0); mean_cos.append(0.0); top3_cos.append(0.0); sum_cos.append(0.0)
            max_cooc.append(0.0); sum_cooc.append(0.0); mean_hist_pop.append(0.0); max_hist_pop.append(0.0)
            candidate_pop_gt_user_mean.append(False); candidate_pop_z_user.append(0.0); hist_htr_to_sim_items.append(0.0)
    p["hist_item_cos_max"] = max_cos
    p["hist_item_cos_mean"] = mean_cos
    p["hist_item_cos_top3_mean"] = top3_cos
    p["hist_item_cos_sum"] = sum_cos
    p["hist_item_cooc_max"] = max_cooc
    p["hist_item_cooc_sum"] = sum_cooc
    p["user_hist_item_pop_mean"] = mean_hist_pop
    p["user_hist_item_pop_max"] = max_hist_pop
    p["candidate_pop_gt_user_hist_mean"] = candidate_pop_gt_user_mean
    p["candidate_pop_z_vs_user_hist"] = candidate_pop_z_user
    p["hist_htr_weighted_cos"] = hist_htr_to_sim_items
    # within-user candidate ranks for unsupervised diagnostic
    for col in ["item_n", "item_log_n", "hist_item_cos_max", "hist_item_cos_top3_mean", "hist_item_cooc_sum", "candidate_pop_z_vs_user_hist", "hist_htr_weighted_cos"]:
        p[f"rank_desc_{col}"] = p.groupby("userID")[col].rank(method="first", ascending=False)
        p[f"pct_rank_desc_{col}"] = (p[f"rank_desc_{col}"] - 1) / (p["candidate_count"] - 1).replace(0, np.nan)
    return p


def analyze_candidate_distribution(pairs_feat, user_agg, item_agg, df):
    pair_users = set(pairs_feat["userID"])
    pair_games = set(pairs_feat["gameID"])
    non_pair_users = user_agg.loc[~user_agg.index.isin(pair_users)]
    pair_user_agg = user_agg.loc[user_agg.index.isin(pair_users)]
    non_pair_games = item_agg.loc[~item_agg.index.isin(pair_games)]
    pair_game_agg = item_agg.loc[item_agg.index.isin(pair_games)]
    out = {
        "pair_user_vs_nonpair_user": {
            col: {"pair_users": q(pair_user_agg[col]), "nonpair_users": q(non_pair_users[col])}
            for col in ["user_n", "user_hours_sum", "user_hours_mean", "user_htr_mean", "user_early_rate", "user_active_days", "user_text_mean"]
        },
        "pair_game_vs_nonpair_game": {
            col: {"pair_games": q(pair_game_agg[col]), "nonpair_games": q(non_pair_games[col])}
            for col in ["item_n", "item_hours_sum", "item_hours_mean", "item_htr_mean", "item_early_rate", "item_active_days", "item_text_mean", "item_pop_rank"]
        },
        "candidate_pair_feature_stats": {
            col: q(pairs_feat[col]) for col in [
                "candidate_count", "user_n", "item_n", "item_pop_rank", "hist_item_cos_max",
                "hist_item_cos_top3_mean", "hist_item_cooc_sum", "candidate_pop_z_vs_user_hist",
                "hist_htr_weighted_cos", "user_hist_item_pop_mean",
            ]
        },
        "candidate_pair_feature_correlations": {},
    }
    corr_cols = ["item_n", "item_pop_rank", "hist_item_cos_max", "hist_item_cos_top3_mean", "hist_item_cooc_sum", "candidate_pop_z_vs_user_hist", "hist_htr_weighted_cos", "user_n", "candidate_count"]
    for i, a in enumerate(corr_cols):
        for b in corr_cols[i+1:]:
            out["candidate_pair_feature_correlations"][f"{a}__{b}"] = corr(pairs_feat[a], pairs_feat[b])
    # Test generator diagnostics: pair candidate item popularity vs all non-played sampled distributions
    rng = np.random.default_rng(42)
    all_games = np.array(item_agg.index.tolist(), dtype=object)
    item_pop = item_agg["item_n"].to_dict()
    user_hist = df.groupby("userID")["gameID"].apply(set).to_dict()
    uniform_pops = []
    pop_weighted_pops = []
    weights = np.array([item_pop[g] for g in all_games], dtype=float)
    weights = weights / weights.sum()
    sample_n = len(pairs_feat)
    # draw user-conditioned negatives but avoid slow rejection by using candidate pool per sampled user
    sampled_users = rng.choice(pairs_feat["userID"].unique(), size=min(sample_n, 20000), replace=True)
    for u in sampled_users:
        unseen = np.array([g for g in all_games if g not in user_hist[u]], dtype=object)
        ug = rng.choice(unseen)
        uniform_pops.append(item_pop[ug])
        uw = np.array([item_pop[g] for g in unseen], dtype=float); uw = uw / uw.sum()
        pg = rng.choice(unseen, p=uw)
        pop_weighted_pops.append(item_pop[pg])
    out["negative_sampler_popularity_reference"] = {
        "actual_pairs_item_n": q(pairs_feat["item_n"]),
        "uniform_unseen_negative_item_n_sample": q(uniform_pops),
        "pop_weighted_unseen_negative_item_n_sample": q(pop_weighted_pops),
    }
    return out


def temporal_eda(df, pairs, user_agg, item_agg):
    pair_users = set(pairs["userID"])
    pair_games = set(pairs["gameID"])
    user_tmp = user_agg.copy()
    item_tmp = item_agg.copy()
    user_tmp["is_pair_user"] = user_tmp.index.isin(pair_users)
    item_tmp["is_pair_game"] = item_tmp.index.isin(pair_games)
    # convert datetimes to days since min for corr
    min_date = df["date"].min()
    user_tmp["user_last_days_since_min"] = (user_tmp["user_last_date"] - min_date).dt.days
    item_tmp["item_last_days_since_min"] = (item_tmp["item_last_date"] - min_date).dt.days
    out = {
        "year_counts": {str(k): int(v) for k, v in df["year"].value_counts().sort_index().items()},
        "month_counts_tail": {str(k): int(v) for k, v in df["month"].value_counts().sort_index().tail(18).items()},
        "pair_users_last_date_stats": q(user_tmp.loc[user_tmp.is_pair_user, "user_last_days_since_min"]),
        "nonpair_users_last_date_stats": q(user_tmp.loc[~user_tmp.is_pair_user, "user_last_days_since_min"]),
        "pair_games_last_date_stats": q(item_tmp.loc[item_tmp.is_pair_game, "item_last_days_since_min"]),
        "nonpair_games_last_date_stats": q(item_tmp.loc[~item_tmp.is_pair_game, "item_last_days_since_min"]),
        "corr_user_degree_last_date_days": corr(user_tmp["user_n"], user_tmp["user_last_days_since_min"]),
        "corr_item_degree_last_date_days": corr(item_tmp["item_n"], item_tmp["item_last_days_since_min"]),
    }
    return out


def text_eda(df, pairs, user_agg, item_agg):
    pair_users = set(pairs["userID"])
    pair_games = set(pairs["gameID"])
    token_user_counter = Counter()
    token_item_counter = Counter()
    # sample enough but deterministic: all rows, token top counts
    global_tokens = Counter()
    pair_user_tokens = Counter()
    pair_game_tokens = Counter()
    for u, g, text in df[["userID", "gameID", "text"]].itertuples(index=False):
        toks = [t.lower() for t in TOKEN_RE.findall(text.lower()) if t.lower() not in STOP and len(t) >= 3]
        if not toks:
            continue
        c = Counter(toks)
        global_tokens.update(c)
        if u in pair_users:
            pair_user_tokens.update(c)
        if g in pair_games:
            pair_game_tokens.update(c)
    return {
        "global_top_tokens": top_dict(global_tokens, 50),
        "pair_user_history_top_tokens": top_dict(pair_user_tokens, 30),
        "pair_game_history_top_tokens": top_dict(pair_game_tokens, 30),
        "text_feature_feasibility": {
            "train_rows_with_blank_text": int((df["text_len"] == 0).sum()),
            "train_text_len_stats": q(df["text_len"]),
            "pair_user_text_mean_stats": q(user_agg.loc[user_agg.index.isin(pair_users), "user_text_mean"]),
            "pair_game_text_mean_stats": q(item_agg.loc[item_agg.index.isin(pair_games), "item_text_mean"]),
        }
    }


def make_plots(df, pairs_feat, user_agg, item_agg):
    if plt is None:
        return []
    paths = []
    plt.style.use("seaborn-v0_8-whitegrid")

    def save(fig, name):
        path = REPORT / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    df["year"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#4C78A8")
    ax.set_title("Train interactions by review year")
    ax.set_xlabel("year"); ax.set_ylabel("rows")
    save(fig, "train_interactions_by_year.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(np.log1p(item_agg["item_n"]), bins=50, alpha=0.75, label="all train games")
    ax.hist(np.log1p(pairs_feat.drop_duplicates("gameID")["item_n"]), bins=50, alpha=0.65, label="games appearing in pairs")
    ax.set_title("Item popularity distribution: all vs candidate games")
    ax.set_xlabel("log1p(train item count)"); ax.set_ylabel("games")
    ax.legend()
    save(fig, "item_pop_all_vs_pair_games.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(user_agg["user_n"], bins=50, alpha=0.75, label="all train users")
    ax.hist(pairs_feat.drop_duplicates("userID")["user_n"], bins=50, alpha=0.65, label="users appearing in pairs")
    ax.set_title("User activity distribution: all vs pair users")
    ax.set_xlabel("train interactions per user"); ax.set_ylabel("users")
    ax.legend()
    save(fig, "user_degree_all_vs_pair_users.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    pairs_feat["candidate_count"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#F58518")
    ax.set_title("Number of candidate pairs per test user")
    ax.set_xlabel("candidate count"); ax.set_ylabel("users")
    save(fig, "candidate_count_per_user.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(np.log1p(pairs_feat["item_n"]), pairs_feat["hist_item_cos_max"], s=6, alpha=0.25)
    ax.set_title("Candidate pair: item popularity vs max history-item cosine")
    ax.set_xlabel("log1p(item train count)"); ax.set_ylabel("max cosine with user's history")
    save(fig, "candidate_pop_vs_history_similarity.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    top = pairs_feat.groupby("userID").apply(lambda x: x.nlargest(len(x)//2, "hist_item_cos_top3_mean")["item_n"].mean(), include_groups=False)
    bot = pairs_feat.groupby("userID").apply(lambda x: x.nsmallest(len(x)//2, "hist_item_cos_top3_mean")["item_n"].mean(), include_groups=False)
    ax.hist(top - bot, bins=60, color="#54A24B", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Within user: popularity gap between high-sim and low-sim candidate halves")
    ax.set_xlabel("mean item_n(top similarity half) - mean item_n(bottom half)")
    ax.set_ylabel("users")
    save(fig, "within_user_similarity_half_pop_gap.png")

    return paths


def build_report(summary, plot_paths):
    md = []
    md.append("# KMU RecSys 26 Steam — Deep EDA Report\n\n")
    md.append("이 리포트는 제공 데이터만 사용했으며 Kaggle 제출/외부 Steam 정답 수집은 수행하지 않았다.\n\n")
    md.append("## 1. Candidate/test 구조\n")
    p = summary["pairs"]
    md.append(f"- pairs rows: {p['rows']:,}, users: {p['unique_users']:,}, games: {p['unique_games']:,}\n")
    md.append(f"- cold users/games: {p['cold_users']} / {p['cold_games']}; train에 이미 있는 user-game pair: {p['in_train_pair_rows']}\n")
    md.append(f"- 유저별 후보 수는 모두 짝수이며 top-half label=1 총합이 {p['floor_top_half_positive_total']:,}개로 정확히 50%다.\n")
    md.append(f"- 후보 수 분포: {p['candidate_count_distribution']}\n")
    md.append("\n## 2. Train 분포의 강한 신호\n")
    tr = summary["train"]
    md.append(f"- train rows/users/games: {tr['rows']:,} / {tr['users']:,} / {tr['games']:,}\n")
    md.append(f"- item popularity Gini: {tr['item_pop_gini']:.4f}; user activity Gini: {tr['user_degree_gini']:.4f}\n")
    md.append(f"- 상위 게임 coverage: {tr['popularity_coverage']}\n")
    md.append("- 결론: popularity는 단일 feature가 아니라 모든 CF/graph/text score에 들어갈 calibration 축이다.\n")
    md.append("\n## 3. Pair users/games selection shift\n")
    shift = summary["selection_shift"]
    md.append(f"- pair users median train degree: {shift['pair_user_degree_median']:.1f}, non-pair users median: {shift['nonpair_user_degree_median']:.1f}\n")
    md.append(f"- pair games median train popularity: {shift['pair_game_pop_median']:.1f}, non-pair games median: {shift['nonpair_game_pop_median']:.1f}\n")
    md.append(f"- actual candidate pair item_n median: {shift['actual_pair_item_n_median']:.1f}\n")
    md.append(f"- user-conditioned uniform unseen negative item_n median: {shift['uniform_negative_item_n_median']:.1f}\n")
    md.append(f"- popularity-weighted unseen negative item_n median: {shift['pop_weighted_negative_item_n_median']:.1f}\n")
    md.append("- 결론: validation negative를 uniform만 쓰면 실제 pairs보다 너무 쉬울 가능성이 높다. popularity-matched 또는 mixed sampler가 필요하다.\n")
    md.append("\n## 4. Candidate pair score feature 관찰\n")
    fs = summary["pair_feature_stats"]
    for k in ["item_n", "hist_item_cos_max", "hist_item_cos_top3_mean", "hist_item_cooc_sum", "candidate_pop_z_vs_user_hist", "hist_htr_weighted_cos"]:
        v = fs[k]
        md.append(f"- `{k}` median/mean/p95: {v['p50']:.4g} / {v['mean']:.4g} / {v['p95']:.4g}\n")
    md.append("- 결론: 후보 pair마다 item popularity, user-history item similarity, hours-weighted similarity를 모두 만들 수 있다. 이들은 BPR/LightGCN 외의 strong rank features로 쓸 수 있다.\n")
    md.append("\n## 5. Temporal/text feature 관찰\n")
    tmp = summary["temporal"]
    md.append(f"- train date range: {tr['date_min']} ~ {tr['date_max']}\n")
    md.append(f"- year counts: {tmp['year_counts']}\n")
    md.append(f"- user degree vs last-date correlation: {tmp['corr_user_degree_last_date_days']:.4f}\n")
    md.append(f"- item degree vs last-date correlation: {tmp['corr_item_degree_last_date_days']:.4f}\n")
    md.append("- 결론: test에 date가 없어도 user/item aggregate recency, active_days, recent-popularity weighting은 validation에서 실험 가치가 있다.\n")
    text = summary["text"]
    md.append(f"- blank review rows: {text['blank_text_rows']:,}; text length median/mean/p95: {text['text_len_stats']['p50']:.1f} / {text['text_len_stats']['mean']:.1f} / {text['text_len_stats']['p95']:.1f}\n")
    md.append("- 텍스트는 pair별 직접 텍스트가 없으므로 user profile/item profile aggregate embedding 또는 topic vector 방식으로만 쓰는 것이 안전하다.\n")
    md.append("\n## 6. 검증 설계 권고\n")
    md.append("1. 기본 LOO: 유저별 마지막 또는 무작위 positive 1개 holdout + 같은 유저의 unseen negative 1개.\n")
    md.append("2. 실제 pairs 난이도 반영: negative sampler는 uniform, popularity-weighted, popularity-bin matched, item-similarity hard negative를 모두 만들고 split별 rank correlation을 본다.\n")
    md.append("3. scoring은 반드시 per-user top-half accuracy로 통일한다. 후보 수가 모두 짝수이므로 validation도 유저별 짝수 후보 구조로 만든다.\n")
    md.append("4. public/private mismatch 방지를 위해 seed 5개 이상의 repeated LOO와 user-group split을 병행한다.\n")
    md.append("\n## 7. 실험 우선순위\n")
    md.append("- Tier 0: baseline 재현, filename mismatch 수정, deterministic pipeline/sha 기록.\n")
    md.append("- Tier 1: popularity variants + per-user rank, item co-occurrence/cosine/KNN, EASE/SLIM/ALS/BPR 튜닝.\n")
    md.append("- Tier 2: LightGCN/NGCF, recency-weighted graph, hours_transformed confidence.\n")
    md.append("- Tier 3: FM/GBM ranker with user/item/pair features, text/user-profile embeddings, rank aggregation ensemble.\n")
    md.append("- 제출은 후보 CSV 생성 후에도 우현 승인 전까지 금지.\n")
    if plot_paths:
        md.append("\n## 8. Generated plots\n")
        for pth in plot_paths:
            md.append(f"- `{pth}`\n")
    return "".join(md)


def main():
    df, field_counts = load_train()
    pairs = load_pairs()
    user_agg, item_agg = aggregate_train(df)
    users, games, u2i, g2i, X, hours, deg, cooc, cosine = make_interaction_arrays(df)
    pairs_feat = add_pair_features(pairs, df, user_agg, item_agg, u2i, g2i, X, hours, deg, cooc, cosine)

    train_pairs = set(zip(df["userID"], df["gameID"]))
    cand_counts = pairs_feat.groupby("userID").size()
    pop_counts_sorted = np.sort(item_agg["item_n"].values)[::-1]
    coverage = {}
    total = pop_counts_sorted.sum()
    csum = np.cumsum(pop_counts_sorted)
    for target in [0.25, 0.5, 0.75, 0.9, 0.95]:
        idx = int(np.searchsorted(csum, total * target)) + 1
        coverage[f"{target:.0%}"] = {"n_games": idx, "pct_games": idx / len(pop_counts_sorted)}

    dist_analysis = analyze_candidate_distribution(pairs_feat, user_agg, item_agg, df)
    temporal = temporal_eda(df, pairs, user_agg, item_agg)
    text = text_eda(df, pairs, user_agg, item_agg)

    # save feature snippets for next modeling stage
    feature_cols = [
        "ID", "userID", "gameID", "candidate_count", "user_n", "item_n", "item_pop_rank",
        "hist_item_cos_max", "hist_item_cos_mean", "hist_item_cos_top3_mean", "hist_item_cos_sum",
        "hist_item_cooc_max", "hist_item_cooc_sum", "candidate_pop_z_vs_user_hist",
        "hist_htr_weighted_cos", "rank_desc_item_n", "rank_desc_hist_item_cos_top3_mean",
    ]
    pairs_feat[feature_cols].to_csv(REPORT / "candidate_pair_engineered_features_preview.csv", index=False)
    user_agg.reset_index().to_csv(REPORT / "user_aggregate_features.csv", index=False)
    item_agg.reset_index().to_csv(REPORT / "item_aggregate_features.csv", index=False)

    plot_paths = make_plots(df, pairs_feat, user_agg, item_agg)

    pair_users = set(pairs["userID"]); pair_games = set(pairs["gameID"])
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "root": str(ROOT),
            "report_dir": str(REPORT),
        },
        "train": {
            "rows": int(len(df)),
            "users": int(df["userID"].nunique()),
            "games": int(df["gameID"].nunique()),
            "field_counts": {k: int(v) for k, v in field_counts.items()},
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "hours": q(df["hours"]),
            "hours_transformed": q(df["hours_transformed"]),
            "text_len": q(df["text_len"]),
            "user_degree": q(user_agg["user_n"]),
            "item_popularity": q(item_agg["item_n"]),
            "user_degree_gini": gini(user_agg["user_n"]),
            "item_pop_gini": gini(item_agg["item_n"]),
            "popularity_coverage": coverage,
            "top_games": item_agg.sort_values("item_n", ascending=False).head(20)[["item_n", "item_hours_mean", "item_htr_mean", "item_last_date"]].assign(item_last_date=lambda x: x.item_last_date.astype(str)).to_dict("index"),
        },
        "pairs": {
            "rows": int(len(pairs)),
            "unique_users": int(pairs["userID"].nunique()),
            "unique_games": int(pairs["gameID"].nunique()),
            "cold_users": int(len(set(pairs["userID"]) - set(df["userID"]))),
            "cold_games": int(len(set(pairs["gameID"]) - set(df["gameID"]))),
            "in_train_pair_rows": int(sum((u, g) in train_pairs for u, g in pairs[["userID", "gameID"]].itertuples(index=False))),
            "candidate_count_stats": q(cand_counts),
            "candidate_count_distribution": {str(k): int(v) for k, v in cand_counts.value_counts().sort_index().items()},
            "all_candidate_counts_even": bool((cand_counts % 2 == 0).all()),
            "floor_top_half_positive_total": int((cand_counts // 2).sum()),
        },
        "selection_shift": {
            "pair_user_degree_median": float(user_agg.loc[user_agg.index.isin(pair_users), "user_n"].median()),
            "nonpair_user_degree_median": float(user_agg.loc[~user_agg.index.isin(pair_users), "user_n"].median()),
            "pair_game_pop_median": float(item_agg.loc[item_agg.index.isin(pair_games), "item_n"].median()),
            "nonpair_game_pop_median": float(item_agg.loc[~item_agg.index.isin(pair_games), "item_n"].median()),
            "actual_pair_item_n_median": float(np.median(pairs_feat["item_n"])),
            "uniform_negative_item_n_median": float(dist_analysis["negative_sampler_popularity_reference"]["uniform_unseen_negative_item_n_sample"]["p50"]),
            "pop_weighted_negative_item_n_median": float(dist_analysis["negative_sampler_popularity_reference"]["pop_weighted_unseen_negative_item_n_sample"]["p50"]),
        },
        "distribution_analysis": dist_analysis,
        "pair_feature_stats": {col: q(pairs_feat[col]) for col in [
            "item_n", "item_pop_rank", "hist_item_cos_max", "hist_item_cos_mean", "hist_item_cos_top3_mean",
            "hist_item_cos_sum", "hist_item_cooc_max", "hist_item_cooc_sum", "candidate_pop_z_vs_user_hist",
            "hist_htr_weighted_cos", "user_hist_item_pop_mean",
        ]},
        "pair_feature_correlations_selected": {
            "item_n_vs_hist_item_cos_top3_mean": corr(pairs_feat["item_n"], pairs_feat["hist_item_cos_top3_mean"]),
            "item_n_vs_hist_item_cooc_sum": corr(pairs_feat["item_n"], pairs_feat["hist_item_cooc_sum"]),
            "hist_item_cos_max_vs_hist_item_cos_top3_mean": corr(pairs_feat["hist_item_cos_max"], pairs_feat["hist_item_cos_top3_mean"]),
            "user_n_vs_candidate_count": corr(pairs_feat.drop_duplicates("userID")["user_n"], pairs_feat.drop_duplicates("userID")["candidate_count"]),
        },
        "temporal": temporal,
        "text": {
            "blank_text_rows": int((df["text_len"] == 0).sum()),
            "text_len_stats": q(df["text_len"]),
            **text,
        },
        "plot_paths": plot_paths,
    }
    (REPORT / "deep_eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md = build_report(summary, plot_paths)
    (REPORT / "deep_eda_report.md").write_text(md, encoding="utf-8")
    print(json.dumps({
        "summary_json": str(REPORT / "deep_eda_summary.json"),
        "report_md": str(REPORT / "deep_eda_report.md"),
        "feature_preview_csv": str(REPORT / "candidate_pair_engineered_features_preview.csv"),
        "plots": plot_paths,
        "key": {
            "actual_pair_item_n_median": summary["selection_shift"]["actual_pair_item_n_median"],
            "uniform_negative_item_n_median": summary["selection_shift"]["uniform_negative_item_n_median"],
            "pop_weighted_negative_item_n_median": summary["selection_shift"]["pop_weighted_negative_item_n_median"],
            "item_n_vs_hist_item_cos_top3_corr": summary["pair_feature_correlations_selected"]["item_n_vs_hist_item_cos_top3_mean"],
        }
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
