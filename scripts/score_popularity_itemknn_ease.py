#!/usr/bin/env python3
"""Score validation/test candidates with popularity, item-item, and EASE prototypes.

This script writes score files and validation metrics only.  It never calls the
Kaggle API and never submits predictions.

Stage-2 additions:
- BM25 / TF-IDF weighted itemKNN
- BM25 / TF-IDF weighted EASE
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

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
    readme_popularity_labels,
    write_json,
    write_submission_like,
)


def parse_float_list(raw: str) -> list[float]:
    if not raw:
        return []
    return [float(x) for x in raw.split(",") if x.strip()]


def add_popularity_scores(candidates: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    pop = item_popularity(train_df).to_dict()
    out["pop_count"] = out["gameID"].map(pop).fillna(0.0).astype(float)
    out["score_pop_log"] = np.log1p(out["pop_count"].to_numpy())
    out["score_pop_sqrt"] = np.sqrt(out["pop_count"].to_numpy())
    # User-calibrated popularity: candidate popularity relative to user's history.
    user_hist_pop = (
        train_df.assign(_pop=train_df["gameID"].map(pop).fillna(0.0).astype(float))
        .groupby("userID")["_pop"]
        .agg(["mean", "median", "std"])
        .rename(columns={"mean": "user_hist_pop_mean", "median": "user_hist_pop_median", "std": "user_hist_pop_std"})
    )
    out = out.join(user_hist_pop, on="userID")
    out["user_hist_pop_mean"] = out["user_hist_pop_mean"].fillna(0.0)
    out["user_hist_pop_median"] = out["user_hist_pop_median"].fillna(0.0)
    out["user_hist_pop_std"] = out["user_hist_pop_std"].fillna(0.0)
    denom = out["user_hist_pop_std"].where(out["user_hist_pop_std"] > 1.0, 1.0)
    out["score_pop_vs_user_mean"] = (out["pop_count"] - out["user_hist_pop_mean"]) / denom
    out["score_pop_mismatch_neg"] = -np.abs(out["pop_count"] - out["user_hist_pop_median"]) / denom
    return out


def tfidf_weight(X: sp.csr_matrix) -> sp.csr_matrix:
    """Column-IDF weighting for a user-item matrix."""
    X = X.astype(np.float32).tocsr(copy=True)
    n_users = X.shape[0]
    df = np.diff(X.tocsc().indptr).astype(np.float32)
    idf = np.log((n_users + 1.0) / (df + 1.0)) + 1.0
    return X @ sp.diags(idf.astype(np.float32), format="csr")


def bm25_weight(X: sp.csr_matrix, k1: float = 1.2, b: float = 0.75) -> sp.csr_matrix:
    """BM25 weighting adapted to sparse user-item implicit matrices.

    Rows are users/documents and columns are items/terms.  Negative IDF values
    for extremely frequent items are clipped to zero to avoid inverting the
    recommendation signal.
    """
    X = X.astype(np.float32).tocsr(copy=True)
    n_users = X.shape[0]
    df = np.diff(X.tocsc().indptr).astype(np.float32)
    idf = np.log((n_users - df + 0.5) / (df + 0.5))
    idf = np.maximum(idf, 0.0).astype(np.float32)
    row_sums = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(row_sums.mean()) if row_sums.size else 1.0
    if avgdl <= 0:
        avgdl = 1.0
    coo = X.tocoo(copy=True)
    length_norm = k1 * (1.0 - b + b * row_sums[coo.row] / avgdl)
    coo.data = coo.data * (k1 + 1.0) / (coo.data + length_norm + 1e-12) * idf[coo.col]
    return coo.tocsr()


def apply_weighting(X: sp.csr_matrix, weighting: str) -> sp.csr_matrix:
    if weighting == "binary" or weighting == "raw":
        return X.astype(np.float32).tocsr(copy=True)
    if weighting == "tfidf":
        return tfidf_weight(X)
    if weighting == "bm25":
        return bm25_weight(X)
    raise ValueError(f"unknown weighting: {weighting}")


def compute_item_similarity(user_item: sp.csr_matrix) -> np.ndarray:
    """Cosine item-item similarity from any weighted user-item matrix."""
    X = user_item.astype(np.float32).tocsr()
    norms = np.sqrt(np.asarray(X.power(2).sum(axis=0)).ravel()).astype(np.float32)
    gram = (X.T @ X).astype(np.float32).toarray()
    denom = np.maximum(norms[:, None] * norms[None, :], 1e-12).astype(np.float32)
    sim = gram / denom
    np.fill_diagonal(sim, 0.0)
    return sim


def add_itemknn_variant_scores(
    candidates: pd.DataFrame,
    train_df: pd.DataFrame,
    weighting: str,
    prefix: str,
) -> pd.DataFrame:
    out = candidates.copy()
    X_binary, user_to_idx, item_to_idx, _, _ = build_user_item_matrix(train_df, binary=True)
    X_weighted = apply_weighting(X_binary, weighting)
    sim = compute_item_similarity(X_weighted)
    htr_matrix, _, _, _, _ = build_user_item_matrix(train_df, user_to_idx, item_to_idx, value_col="hours_transformed", binary=False)

    score_sum = np.zeros(len(out), dtype=np.float32)
    score_max = np.zeros(len(out), dtype=np.float32)
    score_top3 = np.zeros(len(out), dtype=np.float32)
    score_htr_weighted = np.zeros(len(out), dtype=np.float32)

    for n, (uid, gid) in enumerate(out[["userID", "gameID"]].itertuples(index=False)):
        ui = user_to_idx.get(uid)
        gi = item_to_idx.get(gid)
        if ui is None or gi is None:
            continue
        start, end = X_binary.indptr[ui], X_binary.indptr[ui + 1]
        hist_idx = X_binary.indices[start:end]
        if hist_idx.size == 0:
            continue
        sims = sim[gi, hist_idx]
        if sims.size == 0:
            continue
        score_sum[n] = float(sims.sum())
        score_max[n] = float(sims.max())
        k = min(3, sims.size)
        score_top3[n] = float(np.partition(sims, -k)[-k:].mean())
        h_start, h_end = htr_matrix.indptr[ui], htr_matrix.indptr[ui + 1]
        h_idx = htr_matrix.indices[h_start:h_end]
        h_val = htr_matrix.data[h_start:h_end]
        if h_idx.size:
            h_sims = sim[gi, h_idx]
            denom = float(h_val.sum()) + 1e-6
            score_htr_weighted[n] = float(np.dot(h_sims, h_val) / denom)

    out[f"score_{prefix}_sum"] = score_sum
    out[f"score_{prefix}_max"] = score_max
    out[f"score_{prefix}_top3"] = score_top3
    out[f"score_{prefix}_htr_weighted"] = score_htr_weighted
    return out


def add_itemknn_scores(candidates: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    return add_itemknn_variant_scores(candidates, train_df, weighting="binary", prefix="itemknn")


def fit_ease_scores(
    train_df: pd.DataFrame,
    candidates: pd.DataFrame,
    lambdas: list[float],
    value_col: str | None,
    weighting: str = "raw",
    prefix: str | None = None,
) -> pd.DataFrame:
    out = candidates.copy()
    X, user_to_idx, item_to_idx, _, _ = build_user_item_matrix(train_df, value_col=value_col, binary=(value_col is None))
    X = apply_weighting(X, weighting).astype(np.float64).tocsr()
    G = (X.T @ X).toarray().astype(np.float64)
    diag_idx = np.diag_indices(G.shape[0])

    cand_user_idx = candidates["userID"].map(user_to_idx).fillna(-1).astype(int).to_numpy()
    cand_item_idx = candidates["gameID"].map(item_to_idx).fillna(-1).astype(int).to_numpy()

    if prefix is None:
        if value_col:
            prefix = "ease_htr"
        elif weighting in {"bm25", "tfidf"}:
            prefix = f"ease_{weighting}"
        else:
            prefix = "ease"

    for lam in lambdas:
        started = time.time()
        A = G.copy()
        A[diag_idx] += lam
        P = np.linalg.inv(A)
        denom = -np.diag(P)
        B = P / denom[None, :]
        B[diag_idx] = 0.0
        user_scores = X @ B
        col = f"score_{prefix}_lambda{lam:g}"
        scores = np.zeros(len(candidates), dtype=np.float64)
        valid = (cand_user_idx >= 0) & (cand_item_idx >= 0)
        scores[valid] = user_scores[cand_user_idx[valid], cand_item_idx[valid]]
        out[col] = scores.astype(np.float32)
        print(f"[ease] {col} done in {time.time() - started:.1f}s", flush=True)
    return out


def add_rrf(out: pd.DataFrame, name: str, cols: list[str], k: float = 60.0) -> str | None:
    rank_cols = [f"rank_{c}" for c in cols if f"rank_{c}" in out.columns]
    if len(rank_cols) < 2:
        return None
    out[name] = sum(1.0 / (k + out[rc]) for rc in rank_cols)
    return name


def add_blends(scores: pd.DataFrame, score_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = normalize_within_user(scores, score_cols)
    blend_cols: list[str] = []

    def blend(name: str, cols: list[str]) -> None:
        zcols = [f"z_{c}" for c in cols if f"z_{c}" in out.columns]
        if len(zcols) < 2:
            return
        out[name] = out[zcols].mean(axis=1)
        blend_cols.append(name)

    # Stable small blends.  More exhaustive blends are handled by blend_score_files.py.
    if "score_pop_log" in score_cols and "score_itemknn_top3" in score_cols:
        blend("score_blend_pop_itemknn", ["score_pop_log", "score_itemknn_top3"])
    ease_cols = sorted([c for c in score_cols if c.startswith("score_ease_") and "htr" not in c])
    if ease_cols and "score_pop_log" in score_cols:
        blend("score_blend_pop_ease", ["score_pop_log", ease_cols[-1]])
    if ease_cols and "score_itemknn_top3" in score_cols and "score_pop_log" in score_cols:
        blend("score_blend_pop_itemknn_ease", ["score_pop_log", "score_itemknn_top3", ease_cols[-1]])
    if "score_itemknn_bm25_top3" in score_cols and ease_cols and "score_pop_log" in score_cols:
        blend("score_blend_pop_bm25knn_ease", ["score_pop_log", "score_itemknn_bm25_top3", ease_cols[-1]])
    rrf = add_rrf(out, "score_rrf_pop_itemknn_ease", [c for c in ["score_pop_log", "score_itemknn_top3", ease_cols[-1] if ease_cols else ""] if c])
    if rrf:
        blend_cols.append(rrf)
    return out, blend_cols


def evaluate_score_columns(scores: pd.DataFrame, score_cols: list[str], out_dir: Path) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    if "Label" not in scores.columns:
        return summaries
    tie_cols = [("pop_count", True), ("gameID", False)]
    for col in score_cols:
        summary, pred_df = evaluate_tophalf(scores, col, tie_cols=tie_cols)
        summaries.append(summary)
        pred_df[["ID", "userID", "gameID", "Label", col, "Pred", "Correct", "rank_in_user"]].to_csv(
            out_dir / f"pred_eval_{col}.csv", index=False
        )
    summaries = sorted(summaries, key=lambda x: x["row_accuracy"], reverse=True)
    return summaries


def write_eval_md(path: Path, summaries: list[dict[str, object]], readme_eval: dict[str, object] | None) -> None:
    lines = ["# Score evaluation summary", ""]
    if readme_eval:
        lines.extend(
            [
                "## README raw popularity-label baseline",
                "",
                f"- raw row accuracy: {readme_eval['raw_row_accuracy']:.6f}",
                f"- predicted positives: {readme_eval['predicted_positive_total']:,}",
                f"- true positives: {readme_eval['true_positive_total']:,}",
                "",
            ]
        )
    lines.extend(["## Per-user top-half scores", "", "| rank | score_col | row_acc | per_user_acc | pred_pos | true_pos |", "|---:|---|---:|---:|---:|---:|"])
    for i, s in enumerate(summaries, 1):
        lines.append(
            f"| {i} | `{s['score_col']}` | {s['row_accuracy']:.6f} | {s['per_user_mean_accuracy']:.6f} | {s['predicted_positive_total']} | {s['true_positive_total']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", help="Validation split directory containing train_interactions.csv and candidates.csv")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Used when --split-dir is omitted")
    ap.add_argument("--candidates", help="Candidates CSV when --split-dir is omitted; defaults to data-dir/pairs.csv")
    ap.add_argument("--out-dir", help="Output directory; default artifacts/scores/<split-or-test>")
    ap.add_argument(
        "--methods",
        default="popularity,itemknn,ease",
        help="Comma list from popularity,itemknn,itemknn_bm25,itemknn_tfidf,ease,ease_htr,ease_bm25,ease_tfidf",
    )
    ap.add_argument("--ease-lambdas", default="300", help="Comma-separated EASE lambdas")
    ap.add_argument("--write-predictions", action="store_true", help="Write top-half prediction CSVs for unlabeled/test candidates; no submission is made")
    ap.add_argument("--wandb", action="store_true", help="Log this scoring/evaluation run to W&B after local files are written")
    ap.add_argument("--wandb-project", default=None, help="W&B project; default from WANDB_PROJECT or kmu-rec-sys-26-steam")
    ap.add_argument("--wandb-entity", default=None, help="W&B entity/team; default from WANDB_ENTITY or mrpc2003-kookmin-university")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-tags", default="score-prototype", help="Comma-separated W&B tags")
    ap.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb-artifact-mode", default="summary", choices=["none", "summary", "full"], help="summary uploads JSON/Markdown summaries; full also uploads large CSVs")
    ap.add_argument("--no-wandb-artifacts", action="store_true", help="Shortcut for --wandb-artifact-mode none")
    args = ap.parse_args()

    methods = {m.strip() for m in args.methods.split(",") if m.strip()}
    lambdas = parse_float_list(args.ease_lambdas)

    if args.split_dir:
        split_dir = Path(args.split_dir)
        train_df = load_train_interactions(split_dir / "train_interactions.csv")
        candidates = load_pairs_csv(split_dir / "candidates.csv")
        if "Label" in pd.read_csv(split_dir / "candidates.csv", nrows=1).columns:
            candidates = pd.read_csv(split_dir / "candidates.csv")
        default_name = split_dir.name
    else:
        data_dir = Path(args.data_dir)
        train_df = load_train_interactions(data_dir / "train.json")
        candidates = load_pairs_csv(args.candidates or data_dir / "pairs.csv")
        default_name = "test_pairs_full_train"

    out_dir = ensure_dir(args.out_dir or Path("artifacts/scores") / default_name)
    scores = candidates.copy()

    if "popularity" in methods:
        print("[score] popularity", flush=True)
        scores = add_popularity_scores(scores, train_df)
    if "itemknn" in methods:
        print("[score] itemknn", flush=True)
        scores = add_itemknn_scores(scores, train_df)
    if "itemknn_bm25" in methods:
        print("[score] itemknn_bm25", flush=True)
        scores = add_itemknn_variant_scores(scores, train_df, weighting="bm25", prefix="itemknn_bm25")
    if "itemknn_tfidf" in methods:
        print("[score] itemknn_tfidf", flush=True)
        scores = add_itemknn_variant_scores(scores, train_df, weighting="tfidf", prefix="itemknn_tfidf")
    if "ease" in methods and lambdas:
        print(f"[score] EASE binary lambdas={lambdas}", flush=True)
        scores = fit_ease_scores(train_df, scores, lambdas, value_col=None, weighting="raw", prefix="ease")
    if "ease_htr" in methods and lambdas:
        print(f"[score] EASE htr lambdas={lambdas}", flush=True)
        scores = fit_ease_scores(train_df, scores, lambdas, value_col="hours_transformed", weighting="raw", prefix="ease_htr")
    if "ease_bm25" in methods and lambdas:
        print(f"[score] EASE bm25 lambdas={lambdas}", flush=True)
        scores = fit_ease_scores(train_df, scores, lambdas, value_col=None, weighting="bm25", prefix="ease_bm25")
    if "ease_tfidf" in methods and lambdas:
        print(f"[score] EASE tfidf lambdas={lambdas}", flush=True)
        scores = fit_ease_scores(train_df, scores, lambdas, value_col=None, weighting="tfidf", prefix="ease_tfidf")

    score_cols = [c for c in scores.columns if c.startswith("score_")]
    scores, blend_cols = add_blends(scores, score_cols)
    score_cols = [c for c in scores.columns if c.startswith("score_") and not c.startswith("score_readme")]

    scores.to_csv(out_dir / "candidate_scores.csv", index=False)

    readme_eval = None
    if "Label" in scores.columns:
        readme_pred = readme_popularity_labels(train_df, scores)
        tmp = scores[["ID", "Label"]].merge(readme_pred[["ID", "Pred"]], on="ID")
        raw_correct = (tmp["Label"].astype(int) == tmp["Pred"].astype(int)).mean()
        readme_eval = {
            "raw_row_accuracy": float(raw_correct),
            "predicted_positive_total": int(tmp["Pred"].sum()),
            "true_positive_total": int(tmp["Label"].sum()),
        }

    summaries = evaluate_score_columns(scores, score_cols, out_dir)
    run_metadata = {
        "split_dir": args.split_dir,
        "data_dir": args.data_dir,
        "candidates": args.candidates,
        "out_dir": str(out_dir),
        "methods": sorted(methods),
        "ease_lambdas": lambdas,
        "write_predictions": bool(args.write_predictions),
        "train_rows": int(len(train_df)),
        "candidate_rows": int(len(scores)),
        "candidate_users": int(scores["userID"].nunique()) if "userID" in scores.columns else None,
        "has_labels": "Label" in scores.columns,
        "no_kaggle_submission": True,
    }
    write_json(
        out_dir / "evaluation_summary.json",
        {
            "metadata": run_metadata,
            "scores": summaries,
            "readme_raw_baseline": readme_eval,
            "score_columns": score_cols,
            "blend_columns": blend_cols,
        },
    )
    write_eval_md(out_dir / "evaluation_summary.md", summaries, readme_eval)

    if args.write_predictions:
        pred_dir = ensure_dir(out_dir / "prediction_csv")
        for col in score_cols:
            pred_df = predict_tophalf(scores, col, label_col=None, tie_cols=[("pop_count", True), ("gameID", False)])
            write_submission_like(pred_df, pred_dir / f"candidate_{col}.csv")
        # README raw baseline reproduction candidate file.
        readme_pred = readme_popularity_labels(train_df, scores)
        write_submission_like(readme_pred, pred_dir / "candidate_readme_popularity_raw.csv")

    if args.wandb:
        try:
            from wandb_recsys_utils import log_score_dir, parse_tags

            run = log_score_dir(
                out_dir,
                project=args.wandb_project,
                entity=args.wandb_entity,
                run_name=args.wandb_run_name or out_dir.name,
                job_type="score-eval" if summaries else "score-candidates",
                tags=parse_tags(args.wandb_tags),
                notes="KMU RecSys 26 Steam scoring run. No Kaggle submission performed.",
                mode=args.wandb_mode,
                extra_config={"score_script_metadata": run_metadata},
                log_artifacts=args.wandb_artifact_mode != "none" and not args.no_wandb_artifacts,
                artifact_mode=args.wandb_artifact_mode,
            )
            print(f"[wandb] run logged: {run.url}", flush=True)
            run.finish()
        except Exception as exc:
            raise RuntimeError(f"W&B logging failed after local outputs were written: {exc}") from exc

    if summaries:
        print("[done] best validation scores:")
        for s in summaries[:15]:
            print(f"  {s['score_col']}: row_acc={s['row_accuracy']:.6f}, user_acc={s['per_user_mean_accuracy']:.6f}")
    else:
        print(f"[done] wrote unlabeled candidate scores to {out_dir}")


if __name__ == "__main__":
    main()
