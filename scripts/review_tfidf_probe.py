#!/usr/bin/env python3
"""Validation-only review TF-IDF probe for KMU RecSys 26 Steam.

Recent review-enhanced/LLM recommendation papers motivate using review text, but
this competition's test rows do not include text.  The safe adaptation is to
build user and item textual profiles only from train interactions and score a
candidate by profile similarity.  This script tests that idea on validation
splits without touching Kaggle submissions.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recsys_played_utils import ensure_dir, evaluate_tophalf, load_pairs_csv, load_train_interactions, write_json  # noqa: E402


def load_text_by_row(raw_train_json: Path, needed_rows: set[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    with raw_train_json.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx not in needed_rows:
                continue
            d = ast.literal_eval(line)
            text = str(d.get("text") or "")
            out[idx] = text
            if len(out) == len(needed_rows):
                break
    return out


def truncate_join(parts: list[str], max_chars: int) -> str:
    if not parts:
        return ""
    s = "\n".join(p for p in parts if p)
    if len(s) <= max_chars:
        return s
    return s[:max_chars]


def run_split(split_dir: Path, raw_train_json: Path, out_dir: Path, max_features: int, min_df: int, max_chars_per_profile: int) -> dict[str, object]:
    train_df = load_train_interactions(split_dir / "train_interactions.csv")
    candidates = load_pairs_csv(split_dir / "candidates.csv")
    needed = set(train_df["row_idx"].astype(int).tolist())
    text_by_row = load_text_by_row(raw_train_json, needed)
    train_df = train_df.copy()
    train_df["text"] = train_df["row_idx"].astype(int).map(text_by_row).fillna("")

    user_parts: dict[str, list[str]] = defaultdict(list)
    item_parts: dict[str, list[str]] = defaultdict(list)
    for row in train_df[["userID", "gameID", "text"]].itertuples(index=False):
        txt = row.text
        if txt:
            user_parts[str(row.userID)].append(txt)
            item_parts[str(row.gameID)].append(txt)

    users = sorted(train_df["userID"].astype(str).unique().tolist())
    items = sorted(train_df["gameID"].astype(str).unique().tolist())
    user_corpus = [truncate_join(user_parts.get(u, []), max_chars_per_profile) for u in users]
    item_corpus = [truncate_join(item_parts.get(g, []), max_chars_per_profile) for g in items]
    corpus = user_corpus + item_corpus

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]{1,}\b",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=min_df,
        norm="l2",
        dtype=np.float32,
    )
    mat = vectorizer.fit_transform(corpus)
    user_mat = mat[: len(users)]
    item_mat = mat[len(users) :]
    u_map = {u: i for i, u in enumerate(users)}
    i_map = {g: i for i, g in enumerate(items)}

    uidx = candidates["userID"].astype(str).map(u_map)
    iidx = candidates["gameID"].astype(str).map(i_map)
    known = (~uidx.isna()) & (~iidx.isna())
    scores = np.full(len(candidates), -1e9, dtype=np.float32)
    rows = uidx[known].to_numpy(dtype=np.int64)
    cols = iidx[known].to_numpy(dtype=np.int64)
    # Row-wise cosine because TF-IDF rows are L2 normalized.
    scores[known.to_numpy()] = np.asarray(user_mat[rows].multiply(item_mat[cols]).sum(axis=1)).ravel().astype(np.float32)
    scored = candidates.copy()
    scored["score_review_tfidf_user_item_cosine"] = scores
    scored["score_item_review_count"] = scored["gameID"].astype(str).map({g: len(item_parts.get(g, [])) for g in items}).fillna(0).astype(float)
    scored["score_user_review_count"] = scored["userID"].astype(str).map({u: len(user_parts.get(u, [])) for u in users}).fillna(0).astype(float)

    summaries = []
    for col in ["score_review_tfidf_user_item_cosine", "score_item_review_count", "score_user_review_count"]:
        summary, _ = evaluate_tophalf(scored, col, label_col="Label", user_col="userID", id_col="ID")
        summaries.append(summary)
    summaries = sorted(summaries, key=lambda s: (s["row_accuracy"], s["per_user_mean_accuracy"]), reverse=True)

    split_out = ensure_dir(out_dir / split_dir.name)
    scored[["ID", "userID", "gameID", "Label", "score_review_tfidf_user_item_cosine", "score_item_review_count", "score_user_review_count"]].to_csv(split_out / "review_tfidf_scores.csv", index=False)
    result = {
        "split": split_dir.name,
        "rows": int(len(scored)),
        "train_rows": int(len(train_df)),
        "users": int(len(users)),
        "items": int(len(items)),
        "known_candidate_rate": float(known.mean()),
        "vocab_size": int(len(vectorizer.vocabulary_)),
        "max_features": max_features,
        "min_df": min_df,
        "max_chars_per_profile": max_chars_per_profile,
        "nonempty_user_profiles": int(sum(bool(x) for x in user_corpus)),
        "nonempty_item_profiles": int(sum(bool(x) for x in item_corpus)),
        "summaries": summaries,
    }
    write_json(split_out / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-train-json", default="data/raw/public/data/train.json")
    parser.add_argument("--validation-root", default="artifacts/validation")
    parser.add_argument("--splits", nargs="*", default=["val_random_sqrtpop_seed42", "val_recent_sqrtpop_seed42", "val_random_popbin_seed42", "val_random_uniform_seed42"])
    parser.add_argument("--out-dir", default="artifacts/review_tfidf_probe_20260530")
    parser.add_argument("--report-json", default="reports/20260530_review_tfidf_probe.json")
    parser.add_argument("--report-md", default="reports/20260530_review_tfidf_probe.md")
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--min-df", type=int, default=3)
    parser.add_argument("--max-chars-per-profile", type=int, default=24000)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    results = []
    for split in args.splits:
        results.append(run_split(Path(args.validation_root) / split, Path(args.raw_train_json), out_dir, args.max_features, args.min_df, args.max_chars_per_profile))
    payload = {"note": "Validation-only; no Kaggle submission.", "results": results}
    write_json(args.report_json, payload)

    lines = [
        "# KMU RecSys 26 Steam — review TF-IDF validation probe",
        "",
        "Review-enhanced/LLM recommendation papers suggest using text, but test pairs have no text. This probe uses only train-review user/item profiles and scores candidate cosine similarity. No Kaggle submission was performed.",
        "",
        "| split | best score | row acc | per-user mean acc | vocab |",
        "|---|---|---:|---:|---:|",
    ]
    for r in results:
        b = r["summaries"][0]
        lines.append(f"| `{r['split']}` | `{b['score_col']}` | {b['row_accuracy']:.6f} | {b['per_user_mean_accuracy']:.6f} | {r['vocab_size']} |")
    lines.extend(["", "## Full table", ""])
    for r in results:
        lines.extend([f"### {r['split']}", "", "| score | row acc | per-user mean acc |", "|---|---:|---:|"])
        for s in r["summaries"]:
            lines.append(f"| `{s['score_col']}` | {s['row_accuracy']:.6f} | {s['per_user_mean_accuracy']:.6f} |")
        lines.append("")
    Path(args.report_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report_json": args.report_json, "report_md": args.report_md, "out_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
