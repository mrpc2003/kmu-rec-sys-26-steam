#!/usr/bin/env python3
"""Validation-only pseudo-label transduction probe for KMURecSys26 Steam.

Goal
----
Test whether unlabeled candidate-group structure can improve the current emb128 LightGCN
backbone without using hidden labels, external Steam metadata, or Kaggle submissions.

For one validation split and one student seed:
  1. Load the existing emb128 L4 reg1e-3 4-seed teacher scores for that split.
  2. Select only the top-N teacher-ranked candidates per user as pseudo-positive edges.
     Selection uses scores/ranks only. Validation labels are used only afterward for diagnostics.
  3. Append those pseudo-positive edges to the split's training interactions.
  4. Train the same emb128 LightGCN student on the augmented graph.
  5. Evaluate on the validation candidates and write score/summary artifacts.

Safety contract
---------------
- validation_only: true
- no Kaggle submission
- no submissions/*.csv writes
- no full-test candidate writes
- no external Steam metadata or scraping
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.io.formats.csvs  # noqa: F401  # eager import avoids parallel to_csv lazy-import race

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
sys.path.insert(0, str(ROOT / "scripts"))

from recsys_played_utils import (  # noqa: E402
    build_user_item_matrix,
    ensure_dir,
    evaluate_tophalf,
    load_pairs_csv,
    load_train_interactions,
    write_json,
)
from lightgcn_train import score_candidates, train_lightgcn  # noqa: E402

TEACHER_SEEDS = [42, 123, 2024, 7]
BASELINE_4SEED = {
    "val_random_uniform_seed42": 0.7650530106021204,
    "val_random_uniform_seed7": 0.7609521904380876,
    "val_random_uniform_seed123": 0.7599519903980796,
}


def teacher_seed_path(split: str, seed: int) -> Path:
    if split == "val_random_uniform_seed42":
        if seed == 42:
            return ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03/val_random_uniform_seed42/lightgcn_scores.csv"
        return ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}/val_random_uniform_seed42/lightgcn_scores.csv"
    return ROOT / f"artifacts/split_panel_emb128/{split}/seed{seed}/lightgcn_scores.csv"


def load_teacher(split: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    score_cols: list[str] = []
    for seed in TEACHER_SEEDS:
        p = teacher_seed_path(split, seed)
        if not p.exists():
            raise FileNotFoundError(f"missing teacher seed score: split={split} seed={seed} path={p}")
        df = pd.read_csv(p)
        score_col = "score_lightgcn" if "score_lightgcn" in df.columns else "score"
        keep = ["ID", score_col]
        base_cols = ["ID", "userID", "gameID", "Label"]
        if merged is None:
            merged = df[base_cols + [score_col]].rename(columns={score_col: f"score_seed{seed}"})
        else:
            merged = merged.merge(df[keep].rename(columns={score_col: f"score_seed{seed}"}), on="ID", how="inner")
        score_cols.append(f"score_seed{seed}")
    assert merged is not None
    merged["teacher_score"] = merged[score_cols].mean(axis=1)
    return merged


def select_pseudo_edges(teacher: pd.DataFrame, top_n: int, min_margin: float) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    details = []
    for user, g in teacher.groupby("userID", sort=False):
        gg = g.sort_values(["teacher_score", "ID"], ascending=[False, True], kind="mergesort").copy()
        n = len(gg)
        k = n // 2
        take = min(top_n, k)
        if take <= 0:
            continue
        # Boundary is the last predicted positive under exact top-half. Margin is conservative:
        # selected item's teacher score minus the score at the positive/negative boundary.
        boundary_idx = min(k, n - 1)
        boundary_score = float(gg.iloc[boundary_idx]["teacher_score"]) if boundary_idx < n else float(gg.iloc[-1]["teacher_score"])
        selected = gg.head(take).copy()
        selected["pseudo_margin_to_boundary"] = selected["teacher_score"] - boundary_score
        if min_margin > 0:
            selected = selected[selected["pseudo_margin_to_boundary"] >= min_margin]
        if len(selected):
            rows.append(selected[["userID", "gameID", "ID", "teacher_score", "pseudo_margin_to_boundary", "Label"]])
            details.append({"userID": user, "n": int(n), "k": int(k), "selected": int(len(selected))})
    if rows:
        pseudo = pd.concat(rows, ignore_index=True)
    else:
        pseudo = pd.DataFrame(columns=["userID", "gameID", "ID", "teacher_score", "pseudo_margin_to_boundary", "Label"])
    precision = float(pseudo["Label"].mean()) if len(pseudo) else None
    meta = {
        "top_n": int(top_n),
        "min_margin": float(min_margin),
        "pseudo_edges": int(len(pseudo)),
        "pseudo_users": int(pseudo["userID"].nunique()) if len(pseudo) else 0,
        "pseudo_label_precision_diagnostic": precision,
        "mean_margin": float(pseudo["pseudo_margin_to_boundary"].mean()) if len(pseudo) else None,
        "median_margin": float(pseudo["pseudo_margin_to_boundary"].median()) if len(pseudo) else None,
        "selection_uses_labels": False,
        "labels_used_only_for_diagnostic_precision": True,
    }
    return pseudo, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=sorted(BASELINE_4SEED))
    ap.add_argument("--top-n", type=int, default=1)
    ap.add_argument("--min-margin", type=float, default=0.0)
    ap.add_argument("--student-seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--reg", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out-root", default="artifacts/pseudolabel_transduction_20260612T2312KST")
    args = ap.parse_args()

    split_dir = ROOT / "artifacts/validation" / args.split
    train_df = load_train_interactions(split_dir / "train_interactions.csv")
    candidates = load_pairs_csv(split_dir / "candidates.csv")
    teacher = load_teacher(args.split)
    teacher_summary, teacher_pred = evaluate_tophalf(teacher, "teacher_score", label_col="Label", user_col="userID", id_col="ID")

    pseudo, pseudo_meta = select_pseudo_edges(teacher, top_n=args.top_n, min_margin=args.min_margin)
    pseudo_train = pseudo[["userID", "gameID"]].copy()
    pseudo_train["row_idx"] = np.arange(len(train_df), len(train_df) + len(pseudo_train), dtype=np.int64)
    pseudo_train["date"] = pd.NaT
    pseudo_train["hours"] = 0.0
    pseudo_train["hours_transformed"] = 0.0
    aug_train = pd.concat([train_df, pseudo_train], ignore_index=True, sort=False)
    before = len(aug_train)
    aug_train = aug_train.drop_duplicates(["userID", "gameID"], keep="first").reset_index(drop=True)
    pseudo_duplicates = before - len(aug_train)

    mat, u2i, i2i, users, items = build_user_item_matrix(aug_train, binary=True)
    tag = f"split={args.split} topn={args.top_n} margin={args.min_margin} seed={args.student_seed}"
    print(f"[pseudo] {tag}: train={len(train_df)} pseudo={len(pseudo)} dup={pseudo_duplicates} aug={len(aug_train)} users={len(users)} items={len(items)} nnz={mat.nnz}", flush=True)

    ue, ie, train_meta = train_lightgcn(
        mat,
        len(users),
        len(items),
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        lr=args.lr,
        reg=args.reg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.student_seed,
    )
    scored = candidates.copy()
    scored["score_lightgcn"] = score_candidates(scored, ue, ie, u2i, i2i)
    student_summary, student_pred = evaluate_tophalf(scored, "score_lightgcn", label_col="Label", user_col="userID", id_col="ID")
    student_acc = float(student_summary["row_accuracy"])
    teacher_acc = float(teacher_summary["row_accuracy"])
    baseline = BASELINE_4SEED[args.split]

    out = ensure_dir(Path(args.out_root) / args.split / f"topn{args.top_n}_margin{args.min_margin:g}" / f"student_seed{args.student_seed}")
    scored[["ID", "userID", "gameID", "Label", "score_lightgcn"]].to_csv(out / "lightgcn_scores.csv", index=False)
    pseudo.to_csv(out / "pseudo_edges_diagnostic.csv", index=False)

    summary = {
        "validation_only": True,
        "no_kaggle_submit": True,
        "candidate_csv_written": False,
        "full_test_scored": False,
        "external_metadata_used": False,
        "hidden_labels_used_for_selection": False,
        "split": args.split,
        "config": vars(args),
        "teacher_4seed_acc_recomputed": teacher_acc,
        "teacher_4seed_baseline_recorded": baseline,
        "student_acc": student_acc,
        "delta_vs_teacher_recomputed": student_acc - teacher_acc,
        "delta_vs_recorded_baseline": student_acc - baseline,
        "pseudo_meta": pseudo_meta | {"duplicates_removed": int(pseudo_duplicates), "aug_train_rows": int(len(aug_train))},
        "train_meta": train_meta,
        "out_dir": str(out),
    }
    tier = "PSEUDO_GAIN" if summary["delta_vs_teacher_recomputed"] >= 0.004 else ("PSEUDO_WEAK_POSITIVE" if summary["delta_vs_teacher_recomputed"] > 0 else "PSEUDO_NO_GAIN")
    summary["tier"] = tier
    write_json(out / "summary.json", summary)
    print("RESULT_JSON " + json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
