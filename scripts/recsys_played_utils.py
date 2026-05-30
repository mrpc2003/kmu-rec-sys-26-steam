#!/usr/bin/env python3
"""Utilities for KMU RecSys 26 Steam played-prediction experiments.

No Kaggle submission is performed here.  The helpers intentionally keep the
competition's key constraint explicit: predictions are made by ranking each
user's candidate list and marking exactly the top half (or the known validation
positive count) as played.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp


DEFAULT_DATA_DIR = Path("data/raw/public/data")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: object) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_train_json(path: str | Path) -> pd.DataFrame:
    """Load the competition train.json safely with ast.literal_eval."""
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if not line.strip():
                continue
            d = ast.literal_eval(line)
            rows.append(
                {
                    "row_idx": row_idx,
                    "userID": d["userID"],
                    "gameID": d["gameID"],
                    "hours": float(d.get("hours", 0.0)),
                    "hours_transformed": float(d.get("hours_transformed", 0.0)),
                    "early_access": bool(d.get("early_access", False)),
                    "date": d.get("date", ""),
                    "text_len": len(d.get("text") or ""),
                    "has_found_funny": "found_funny" in d,
                    "found_funny": float(d.get("found_funny", 0.0) or 0.0),
                    "has_compensation": "compensation" in d,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_train_interactions(path: str | Path) -> pd.DataFrame:
    """Load either train_interactions.csv from a split or raw train.json."""
    path = Path(path)
    if path.suffix == ".json":
        return load_train_json(path)
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if "row_idx" not in df.columns:
        df["row_idx"] = np.arange(len(df), dtype=np.int64)
    return df


def load_pairs_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"ID", "userID", "gameID"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"pairs/candidates file is missing columns: {sorted(missing)}")
    df["ID"] = df["ID"].astype(int)
    return df


def build_id_maps(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], list[str], list[str]]:
    users = sorted(train_df["userID"].unique().tolist())
    items = sorted(train_df["gameID"].unique().tolist())
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {g: i for i, g in enumerate(items)}
    return user_to_idx, item_to_idx, users, items


def build_user_item_matrix(
    train_df: pd.DataFrame,
    user_to_idx: dict[str, int] | None = None,
    item_to_idx: dict[str, int] | None = None,
    value_col: str | None = None,
    binary: bool = True,
) -> tuple[sp.csr_matrix, dict[str, int], dict[str, int], list[str], list[str]]:
    """Create a CSR user-item matrix.

    When binary=True duplicate rows are collapsed by the CSR sum and clipped to 1.
    The competition train currently has no duplicate user-game rows, but clipping
    keeps validation split variants safe.
    """
    if user_to_idx is None or item_to_idx is None:
        user_to_idx, item_to_idx, users, items = build_id_maps(train_df)
    else:
        users = [None] * len(user_to_idx)  # type: ignore[list-item]
        items = [None] * len(item_to_idx)  # type: ignore[list-item]
        for u, i in user_to_idx.items():
            users[i] = u  # type: ignore[index]
        for g, i in item_to_idx.items():
            items[i] = g  # type: ignore[index]

    row = train_df["userID"].map(user_to_idx).to_numpy()
    col = train_df["gameID"].map(item_to_idx).to_numpy()
    mask = (~pd.isna(row)) & (~pd.isna(col))
    row = row[mask].astype(np.int32)
    col = col[mask].astype(np.int32)
    if value_col is None or binary:
        data = np.ones(len(row), dtype=np.float32)
    else:
        data = train_df.loc[mask, value_col].to_numpy(dtype=np.float32)
    mat = sp.csr_matrix((data, (row, col)), shape=(len(user_to_idx), len(item_to_idx)), dtype=np.float32)
    if binary:
        mat.data[:] = 1.0
        mat.eliminate_zeros()
    return mat, user_to_idx, item_to_idx, users, items  # type: ignore[return-value]


def item_popularity(train_df: pd.DataFrame) -> pd.Series:
    return train_df.groupby("gameID").size().astype(float).sort_values(ascending=False)


def user_histories(train_df: pd.DataFrame) -> dict[str, set[str]]:
    return train_df.groupby("userID")["gameID"].apply(lambda s: set(s)).to_dict()


def percentile_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "p25": None, "p50": None, "p75": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def normalize_within_user(df: pd.DataFrame, cols: list[str], user_col: str = "userID") -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby(user_col, sort=False)
    for col in cols:
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std").fillna(0.0)
        out[f"z_{col}"] = np.where(std.to_numpy() > 1e-12, (out[col] - mean) / std.replace(0, np.nan), 0.0)
        out[f"rank_{col}"] = grouped[col].rank(method="first", ascending=False)
        denom = grouped[col].transform("size") - 1
        out[f"pct_rank_{col}"] = np.where(denom > 0, (out[f"rank_{col}"] - 1) / denom, 0.0)
    return out


def predict_tophalf(
    candidates: pd.DataFrame,
    score_col: str,
    label_col: str | None = "Label",
    user_col: str = "userID",
    id_col: str = "ID",
    tie_cols: list[tuple[str, bool]] | None = None,
) -> pd.DataFrame:
    """Return candidates with Pred labels from per-user top-half ranking.

    If label_col exists, the number of predicted positives per user is exactly the
    number of validation positives for that user.  Otherwise it is floor(n/2),
    which is correct for the verified test pairs where every user has an even
    candidate count.

    tie_cols entries are (column, descending).  `score_col` is always the primary
    descending key, and `id_col` is the final deterministic ascending tie-breaker.
    """
    if score_col not in candidates.columns:
        raise ValueError(f"score column not found: {score_col}")
    df = candidates.copy()
    df[score_col] = df[score_col].replace([np.inf, -np.inf], np.nan).fillna(-1e30)
    if tie_cols is None:
        tie_cols = []

    sort_cols = [user_col, score_col]
    ascending = [True, False]
    for col, desc in tie_cols:
        if col in df.columns and col not in sort_cols:
            sort_cols.append(col)
            ascending.append(not desc)
    if id_col in df.columns and id_col not in sort_cols:
        sort_cols.append(id_col)
        ascending.append(True)
    df = df.sort_values(sort_cols, ascending=ascending, kind="mergesort").copy()

    if label_col and label_col in df.columns:
        k_by_user = df.groupby(user_col)[label_col].sum().astype(int).to_dict()
    else:
        k_by_user = (df.groupby(user_col).size() // 2).astype(int).to_dict()

    df["rank_in_user"] = df.groupby(user_col).cumcount() + 1
    df["target_pos_in_user"] = df[user_col].map(k_by_user).astype(int)
    df["Pred"] = (df["rank_in_user"] <= df["target_pos_in_user"]).astype(int)
    return df.sort_values(id_col, kind="mergesort") if id_col in df.columns else df


def evaluate_tophalf(
    candidates: pd.DataFrame,
    score_col: str,
    label_col: str = "Label",
    user_col: str = "userID",
    id_col: str = "ID",
    tie_cols: list[tuple[str, bool]] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    if label_col not in candidates.columns:
        raise ValueError(f"cannot evaluate without label column: {label_col}")
    pred_df = predict_tophalf(candidates, score_col, label_col, user_col, id_col, tie_cols)
    correct = (pred_df["Pred"].astype(int) == pred_df[label_col].astype(int)).astype(float)
    pred_df["Correct"] = correct
    per_user = pred_df.groupby(user_col)["Correct"].mean()
    pos_match = pred_df.groupby(user_col).agg(pred_pos=("Pred", "sum"), true_pos=(label_col, "sum"))
    summary = {
        "score_col": score_col,
        "rows": int(len(pred_df)),
        "users": int(pred_df[user_col].nunique()),
        "row_accuracy": float(correct.mean()),
        "per_user_mean_accuracy": float(per_user.mean()),
        "per_user_min_accuracy": float(per_user.min()),
        "per_user_p05_accuracy": float(np.percentile(per_user.to_numpy(), 5)),
        "per_user_p50_accuracy": float(np.percentile(per_user.to_numpy(), 50)),
        "per_user_p95_accuracy": float(np.percentile(per_user.to_numpy(), 95)),
        "predicted_positive_total": int(pred_df["Pred"].sum()),
        "true_positive_total": int(pred_df[label_col].sum()),
        "all_user_positive_counts_match": bool((pos_match["pred_pos"] == pos_match["true_pos"]).all()),
    }
    return summary, pred_df


def write_submission_like(pred_df: pd.DataFrame, out_path: str | Path, pred_col: str = "Pred") -> None:
    out = pred_df[["ID", pred_col]].rename(columns={pred_col: "Label"}).sort_values("ID")
    ensure_dir(Path(out_path).parent)
    out.to_csv(out_path, index=False)


def readme_popularity_labels(train_df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Replicate README baseline: games covering >50% of train plays => 1.

    The original baseline uses `collections.Counter(...).most_common()`.  For
    tied play counts, Counter preserves first-observed insertion order, so we
    sort by count descending and first row index ascending rather than by gameID.
    This avoids small but real reproduction drift around the cumulative 50%
    cutoff.
    """
    stats = train_df.groupby("gameID", sort=False).agg(count=("gameID", "size"), first_row=("row_idx", "min"))
    stats = stats.sort_values(["count", "first_row"], ascending=[False, True], kind="mergesort")
    total = int(stats["count"].sum())
    running = 0
    selected: set[str] = set()
    for gid, row in stats.iterrows():
        selected.add(str(gid))
        running += int(row["count"])
        if running > total / 2:
            break
    out = candidates[["ID", "userID", "gameID"]].copy()
    out["score_readme_popularity"] = out["gameID"].isin(selected).astype(float)
    out["Pred"] = out["score_readme_popularity"].astype(int)
    return out
