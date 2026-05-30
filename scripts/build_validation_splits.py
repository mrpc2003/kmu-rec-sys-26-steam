#!/usr/bin/env python3
"""Build validation candidate sets for KMU RecSys 26 Steam played prediction.

The validation files mimic the real test structure:
- known users/items only,
- no candidate pair remains in the fold-train interactions,
- per-user positive:negative is 1:1,
- user candidate counts are anchored to the actual pairs.csv candidate counts.

No Kaggle submission is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from recsys_played_utils import (
    DEFAULT_DATA_DIR,
    ensure_dir,
    item_popularity,
    load_pairs_csv,
    load_train_json,
    percentile_summary,
    user_histories,
    write_json,
)


@dataclass(frozen=True)
class SplitConfig:
    holdout: str
    negative: str
    seed: int

    @property
    def name(self) -> str:
        return f"val_{self.holdout}_{self.negative}_seed{self.seed}"


def parse_configs(raw: str) -> list[SplitConfig]:
    configs: list[SplitConfig] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        # accepted forms: random:sqrtpop:42 or random_sqrtpop_seed42
        if ":" in token:
            holdout, negative, seed = token.split(":")
            configs.append(SplitConfig(holdout=holdout, negative=negative, seed=int(seed)))
            continue
        if token.startswith("val_"):
            token = token[4:]
        parts = token.split("_")
        if len(parts) < 3 or not parts[-1].startswith("seed"):
            raise ValueError(f"Bad split config: {token}")
        configs.append(SplitConfig(holdout=parts[0], negative="_".join(parts[1:-1]), seed=int(parts[-1][4:])))
    return configs


def choose_positives(user_df: pd.DataFrame, k: int, holdout: str, rng: np.random.Generator) -> pd.DataFrame:
    if holdout == "recent":
        return user_df.sort_values(["date", "gameID"], ascending=[False, True]).head(k)
    if holdout == "random":
        idx = rng.choice(user_df.index.to_numpy(), size=k, replace=False)
        return user_df.loc[idx]
    raise ValueError(f"unknown holdout mode: {holdout}")


def quantile_bins(pop: pd.Series, n_bins: int = 10) -> dict[str, int]:
    ranks = pop.rank(method="first", ascending=True)
    # qcut can fail with repeated values; rank first gives a stable total ordering.
    bins = pd.qcut(ranks, q=min(n_bins, len(pop)), labels=False, duplicates="drop")
    return {gid: int(b) for gid, b in bins.items()}


def sample_one_from_pool(
    pool: np.ndarray,
    item_weights: dict[str, float],
    rng: np.random.Generator,
) -> str:
    if len(pool) == 0:
        raise ValueError("empty negative pool")
    weights = np.array([item_weights.get(g, 0.0) for g in pool], dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        return str(rng.choice(pool))
    weights = weights / weights.sum()
    return str(rng.choice(pool, p=weights))


def sample_negatives_for_user(
    user_id: str,
    k: int,
    positive_games: list[str],
    all_items: np.ndarray,
    full_history: dict[str, set[str]],
    train_pop: pd.Series,
    negative: str,
    pop_bins: dict[str, int],
    rng: np.random.Generator,
) -> list[str]:
    seen = full_history.get(user_id, set())
    base_pool = np.array([g for g in all_items if g not in seen], dtype=object)
    if len(base_pool) < k:
        raise ValueError(f"not enough negatives for {user_id}: pool={len(base_pool)}, k={k}")

    if negative == "uniform":
        return [str(x) for x in rng.choice(base_pool, size=k, replace=False)]

    if negative == "sqrtpop":
        weights = {gid: float(train_pop.get(gid, 0.0) ** 0.5) for gid in base_pool}
        chosen: list[str] = []
        remaining = base_pool.copy()
        for _ in range(k):
            pick = sample_one_from_pool(remaining, weights, rng)
            chosen.append(pick)
            remaining = remaining[remaining != pick]
        return chosen

    if negative == "popbin":
        chosen = []
        remaining = base_pool.copy()
        fallback_weights = {gid: float(train_pop.get(gid, 0.0) ** 0.5) for gid in base_pool}
        for pos_gid in positive_games:
            target_bin = pop_bins.get(pos_gid)
            if target_bin is None:
                candidate_pool = remaining
            else:
                candidate_pool = np.array([g for g in remaining if pop_bins.get(g) == target_bin], dtype=object)
                if len(candidate_pool) == 0:
                    candidate_pool = remaining
            pick = sample_one_from_pool(candidate_pool, fallback_weights, rng)
            chosen.append(pick)
            remaining = remaining[remaining != pick]
        if len(chosen) < k:
            extra = rng.choice(remaining, size=k - len(chosen), replace=False)
            chosen.extend([str(x) for x in extra])
        return chosen[:k]

    raise ValueError(f"unknown negative sampler: {negative}")


def build_split(train_df: pd.DataFrame, pairs_df: pd.DataFrame, config: SplitConfig, out_root: Path) -> dict[str, object]:
    rng = np.random.default_rng(config.seed)
    out_dir = ensure_dir(out_root / config.name)

    pair_user_counts = pairs_df.groupby("userID").size().astype(int)
    full_hist = user_histories(train_df)
    # Avoid repeatedly scanning the 175k-row frame for each test user.
    train_by_user = {uid: grp for uid, grp in train_df.groupby("userID", sort=False)}
    all_items = np.array(sorted(train_df["gameID"].unique()), dtype=object)

    # First select positives.  Keep at least one train interaction per user so
    # fold-time scoring remains known-user.
    heldout_idx: list[int] = []
    requested_k: dict[str, int] = {}
    actual_k: dict[str, int] = {}
    skipped_users: list[str] = []
    adjusted_users: list[dict[str, object]] = []

    for user_id, cand_count in pair_user_counts.items():
        user_rows = train_by_user.get(user_id)
        if user_rows is None or user_rows.empty:
            skipped_users.append(user_id)
            continue
        requested = int(cand_count // 2)
        if requested < 1:
            skipped_users.append(user_id)
            continue
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

    train_pop = item_popularity(fold_train)
    pop_bins = quantile_bins(train_pop, n_bins=10)
    all_train_items = np.array(sorted(fold_train["gameID"].unique()), dtype=object)

    rows: list[dict[str, object]] = []
    by_user_pos = heldout.groupby("userID")
    for user_id, pos_df in by_user_pos:
        k = actual_k[user_id]
        pos_games = pos_df["gameID"].tolist()
        neg_games = sample_negatives_for_user(
            user_id=user_id,
            k=k,
            positive_games=pos_games,
            all_items=all_train_items,
            full_history=full_hist,
            train_pop=train_pop,
            negative=config.negative,
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
                    "source": f"negative_{config.negative}",
                    "heldout_row_idx": -1,
                    "heldout_date": "",
                    "requested_pos_k": requested_k[user_id],
                    "actual_pos_k": k,
                }
            )

    candidates = pd.DataFrame(rows)
    candidates = candidates.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)
    candidates.insert(0, "ID", np.arange(len(candidates), dtype=int))

    # Safety checks.
    train_pairs = set(zip(fold_train["userID"], fold_train["gameID"]))
    train_users = set(fold_train["userID"])
    train_items = set(fold_train["gameID"])
    overlap = sum((u, g) in train_pairs for u, g in candidates[["userID", "gameID"]].itertuples(index=False))
    missing_user_rows = int((~candidates["userID"].isin(train_users)).sum())
    missing_item_rows = int((~candidates["gameID"].isin(train_items)).sum())
    if overlap:
        raise RuntimeError(f"validation candidates overlap fold train: {overlap}")
    if missing_user_rows or missing_item_rows:
        raise RuntimeError(
            f"validation candidates are not known-user/known-item in fold train: "
            f"missing_user_rows={missing_user_rows}, missing_item_rows={missing_item_rows}"
        )
    pos_counts = candidates.groupby("userID")["Label"].sum().astype(int)
    cand_counts = candidates.groupby("userID").size().astype(int)
    if not ((cand_counts == 2 * pos_counts).all()):
        raise RuntimeError("validation split is not per-user 1:1")

    fold_train.to_csv(out_dir / "train_interactions.csv", index=False)
    candidates.to_csv(out_dir / "candidates.csv", index=False)

    item_n = train_pop.to_dict()
    actual_pair_item_pop = pairs_df["gameID"].map(item_n).fillna(0).to_numpy()
    val_neg_item_pop = candidates.loc[candidates["Label"] == 0, "gameID"].map(item_n).fillna(0).to_numpy()
    val_pos_item_pop = candidates.loc[candidates["Label"] == 1, "gameID"].map(item_n).fillna(0).to_numpy()

    summary = {
        "name": config.name,
        "holdout": config.holdout,
        "negative": config.negative,
        "seed": config.seed,
        "out_dir": str(out_dir),
        "fold_train_rows": int(len(fold_train)),
        "heldout_positive_rows": int(len(heldout)),
        "candidate_rows": int(len(candidates)),
        "candidate_users": int(candidates["userID"].nunique()),
        "skipped_users": int(len(skipped_users)),
        "adjusted_users": int(len(adjusted_users)),
        "adjusted_user_examples": adjusted_users[:20],
        "per_user_candidate_count": percentile_summary(cand_counts),
        "per_user_positive_count": percentile_summary(pos_counts),
        "actual_pairs_item_popularity_in_fold_train": percentile_summary(actual_pair_item_pop),
        "validation_negative_item_popularity": percentile_summary(val_neg_item_pop),
        "validation_positive_item_popularity": percentile_summary(val_pos_item_pop),
        "overlap_with_fold_train": int(overlap),
        "missing_user_rows_vs_fold_train": missing_user_rows,
        "missing_item_rows_vs_fold_train": missing_item_rows,
        "all_candidate_counts_even": bool((cand_counts % 2 == 0).all()),
    }
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(render_summary_md(summary), encoding="utf-8")
    return summary


def render_summary_md(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            f"# Validation split: {summary['name']}",
            "",
            f"- holdout: `{summary['holdout']}`",
            f"- negative: `{summary['negative']}`",
            f"- seed: `{summary['seed']}`",
            f"- fold train rows: {summary['fold_train_rows']:,}",
            f"- heldout positives: {summary['heldout_positive_rows']:,}",
            f"- candidate rows/users: {summary['candidate_rows']:,} / {summary['candidate_users']:,}",
            f"- skipped users: {summary['skipped_users']}",
            f"- adjusted users: {summary['adjusted_users']}",
            f"- overlap with fold train: {summary['overlap_with_fold_train']}",
            f"- missing user rows vs fold train: {summary['missing_user_rows_vs_fold_train']}",
            f"- missing item rows vs fold train: {summary['missing_item_rows_vs_fold_train']}",
            f"- all candidate counts even: {summary['all_candidate_counts_even']}",
            "",
            "## Item popularity diagnostics",
            "",
            f"- actual pairs item popularity in fold train: `{summary['actual_pairs_item_popularity_in_fold_train']}`",
            f"- validation negative item popularity: `{summary['validation_negative_item_popularity']}`",
            f"- validation positive item popularity: `{summary['validation_positive_item_popularity']}`",
            "",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out-root", default="artifacts/validation")
    ap.add_argument(
        "--configs",
        default="random:sqrtpop:42,random:uniform:42,random:popbin:42,recent:sqrtpop:42",
        help="Comma list: holdout:negative:seed. negative in {uniform,sqrtpop,popbin}",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_df = load_train_json(data_dir / "train.json")
    pairs_df = load_pairs_csv(data_dir / "pairs.csv")
    out_root = ensure_dir(args.out_root)

    summaries = []
    for cfg in parse_configs(args.configs):
        print(f"[build] {cfg.name}", flush=True)
        summaries.append(build_split(train_df, pairs_df, cfg, out_root))
    write_json(out_root / "validation_splits_summary.json", summaries)
    print(f"Wrote {len(summaries)} validation split(s) under {out_root}")


if __name__ == "__main__":
    main()
