#!/usr/bin/env python3
"""Score KMU RecSys Steam candidates with implicit BPR/ALS models.

This script is validation/test-score only.  It never calls the Kaggle API and
never submits predictions.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from recsys_played_utils import (
    DEFAULT_DATA_DIR,
    build_user_item_matrix,
    ensure_dir,
    evaluate_tophalf,
    item_popularity,
    load_pairs_csv,
    load_train_interactions,
    normalize_within_user,
    predict_tophalf,
    write_json,
    write_submission_like,
)


def parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def load_inputs(split_dir: str | None, data_dir: str, candidates_path: str | None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if split_dir:
        d = Path(split_dir)
        train_df = load_train_interactions(d / "train_interactions.csv")
        candidates = pd.read_csv(d / "candidates.csv")
        return train_df, candidates, d.name
    data = Path(data_dir)
    train_df = load_train_interactions(data / "train.json")
    candidates = load_pairs_csv(candidates_path or data / "pairs.csv")
    return train_df, candidates, "test_pairs_full_train"


def matrix_for_model(train_df: pd.DataFrame, kind: str):
    if kind.endswith("_htr"):
        X, user_to_idx, item_to_idx, _, _ = build_user_item_matrix(train_df, value_col="hours_transformed", binary=False)
        # Give every observed interaction at least unit confidence/preference.
        X = X.copy().astype(np.float32)
        X.data = np.maximum(X.data, 0.0) + 1.0
    else:
        X, user_to_idx, item_to_idx, _, _ = build_user_item_matrix(train_df, binary=True)
        X = X.astype(np.float32)
    return X.tocsr(), user_to_idx, item_to_idx


def latent_scores(model, candidates: pd.DataFrame, user_to_idx: dict[str, int], item_to_idx: dict[str, int]) -> np.ndarray:
    user_factors = np.asarray(model.user_factors)
    item_factors = np.asarray(model.item_factors)
    scores = np.zeros(len(candidates), dtype=np.float32)
    for n, (uid, gid) in enumerate(candidates[["userID", "gameID"]].itertuples(index=False)):
        ui = user_to_idx.get(uid)
        gi = item_to_idx.get(gid)
        if ui is None or gi is None:
            continue
        scores[n] = float(user_factors[ui] @ item_factors[gi])
    return scores


def add_pop_hybrids(scores: pd.DataFrame, base_col: str, pop_alphas: list[float], train_df: pd.DataFrame) -> list[str]:
    pop = item_popularity(train_df).to_dict()
    if "pop_count" not in scores.columns:
        scores["pop_count"] = scores["gameID"].map(pop).fillna(0.0).astype(float)
    max_pop = max(pop.values()) if pop else 1.0
    norm_pop = scores["pop_count"].to_numpy(dtype=np.float32) / float(max_pop)
    new_cols: list[str] = []
    for alpha in pop_alphas:
        col = f"{base_col}_popa{alpha:g}"
        scores[col] = scores[base_col].to_numpy(dtype=np.float32) + alpha * norm_pop
        new_cols.append(col)
    return new_cols


def fit_bpr_scores(
    scores: pd.DataFrame,
    train_df: pd.DataFrame,
    kind: str,
    factors_list: list[int],
    iterations: int,
    learning_rate: float,
    regularization: float,
    seed: int,
    pop_alphas: list[float],
) -> list[str]:
    from implicit.bpr import BayesianPersonalizedRanking

    score_cols: list[str] = []
    X, user_to_idx, item_to_idx = matrix_for_model(train_df, kind)
    for factors in factors_list:
        started = time.time()
        model = BayesianPersonalizedRanking(
            factors=factors,
            iterations=iterations,
            learning_rate=learning_rate,
            regularization=regularization,
            random_state=seed,
        )
        model.fit(X, show_progress=False)
        col = f"score_{kind}_f{factors}_it{iterations}"
        scores[col] = latent_scores(model, scores, user_to_idx, item_to_idx)
        score_cols.append(col)
        score_cols.extend(add_pop_hybrids(scores, col, pop_alphas, train_df))
        print(f"[bpr] {col} done in {time.time() - started:.1f}s", flush=True)
    return score_cols


def fit_als_scores(
    scores: pd.DataFrame,
    train_df: pd.DataFrame,
    kind: str,
    factors_list: list[int],
    iterations: int,
    regularization: float,
    alpha: float,
    seed: int,
    pop_alphas: list[float],
) -> list[str]:
    from implicit.als import AlternatingLeastSquares

    score_cols: list[str] = []
    X, user_to_idx, item_to_idx = matrix_for_model(train_df, kind)
    X = X.copy().astype(np.float32)
    X.data *= alpha
    for factors in factors_list:
        started = time.time()
        model = AlternatingLeastSquares(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            random_state=seed,
        )
        model.fit(X, show_progress=False)
        col = f"score_{kind}_f{factors}_it{iterations}_alpha{alpha:g}"
        scores[col] = latent_scores(model, scores, user_to_idx, item_to_idx)
        score_cols.append(col)
        score_cols.extend(add_pop_hybrids(scores, col, pop_alphas, train_df))
        print(f"[als] {col} done in {time.time() - started:.1f}s", flush=True)
    return score_cols


def evaluate_score_columns(scores: pd.DataFrame, score_cols: list[str], out_dir: Path) -> list[dict[str, object]]:
    if "Label" not in scores.columns:
        return []
    summaries: list[dict[str, object]] = []
    for col in score_cols:
        summary, pred_df = evaluate_tophalf(scores, col, tie_cols=[("pop_count", True), ("gameID", False)])
        summaries.append(summary)
        pred_df[["ID", "userID", "gameID", "Label", col, "Pred", "Correct", "rank_in_user"]].to_csv(out_dir / f"pred_eval_{col}.csv", index=False)
    return sorted(summaries, key=lambda x: x["row_accuracy"], reverse=True)


def write_eval_md(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = ["# CF model evaluation summary", "", "| rank | score_col | row_acc | user_acc |", "|---:|---|---:|---:|"]
    for i, s in enumerate(summaries, 1):
        lines.append(f"| {i} | `{s['score_col']}` | {s['row_accuracy']:.6f} | {s['per_user_mean_accuracy']:.6f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--candidates")
    ap.add_argument("--out-dir")
    ap.add_argument("--models", default="bpr,als", help="Comma list from bpr,bpr_htr,als,als_htr")
    ap.add_argument("--factors", default="32")
    ap.add_argument("--bpr-iterations", type=int, default=100)
    ap.add_argument("--bpr-learning-rate", type=float, default=0.01)
    ap.add_argument("--bpr-regularization", type=float, default=0.1)
    ap.add_argument("--als-iterations", type=int, default=30)
    ap.add_argument("--als-regularization", type=float, default=0.05)
    ap.add_argument("--als-alpha", type=float, default=20.0)
    ap.add_argument("--pop-alphas", default="0,1,2,4,8")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write-predictions", action="store_true")
    args = ap.parse_args()

    train_df, candidates, default_name = load_inputs(args.split_dir, args.data_dir, args.candidates)
    out_dir = ensure_dir(args.out_dir or Path("artifacts/scores") / f"{default_name}_cf")
    models = {x.strip() for x in args.models.split(",") if x.strip()}
    factors = parse_int_list(args.factors)
    pop_alphas = parse_float_list(args.pop_alphas)

    scores = candidates.copy()
    # Always add pop_count for tie-breaking and hybrid construction.
    pop = item_popularity(train_df).to_dict()
    scores["pop_count"] = scores["gameID"].map(pop).fillna(0.0).astype(float)
    score_cols: list[str] = []

    if "bpr" in models:
        score_cols.extend(fit_bpr_scores(scores, train_df, "bpr", factors, args.bpr_iterations, args.bpr_learning_rate, args.bpr_regularization, args.seed, pop_alphas))
    if "bpr_htr" in models:
        score_cols.extend(fit_bpr_scores(scores, train_df, "bpr_htr", factors, args.bpr_iterations, args.bpr_learning_rate, args.bpr_regularization, args.seed, pop_alphas))
    if "als" in models:
        score_cols.extend(fit_als_scores(scores, train_df, "als", factors, args.als_iterations, args.als_regularization, args.als_alpha, args.seed, pop_alphas))
    if "als_htr" in models:
        score_cols.extend(fit_als_scores(scores, train_df, "als_htr", factors, args.als_iterations, args.als_regularization, args.als_alpha, args.seed, pop_alphas))

    # Within-user normalized columns are useful for later blending.
    scores = normalize_within_user(scores, score_cols)
    scores.to_csv(out_dir / "candidate_scores.csv", index=False)
    summaries = evaluate_score_columns(scores, score_cols, out_dir)
    write_json(out_dir / "evaluation_summary.json", {"scores": summaries, "score_columns": score_cols})
    write_eval_md(out_dir / "evaluation_summary.md", summaries)

    if args.write_predictions:
        pred_dir = ensure_dir(out_dir / "prediction_csv")
        for col in score_cols:
            pred_df = predict_tophalf(scores, col, label_col=None, tie_cols=[("pop_count", True), ("gameID", False)])
            write_submission_like(pred_df, pred_dir / f"candidate_{col}.csv")

    if summaries:
        print("[done] best CF validation scores:")
        for s in summaries[:15]:
            print(f"  {s['score_col']}: row_acc={s['row_accuracy']:.6f}, user_acc={s['per_user_mean_accuracy']:.6f}")
    else:
        print(f"[done] wrote CF scores to {out_dir}")


if __name__ == "__main__":
    main()
