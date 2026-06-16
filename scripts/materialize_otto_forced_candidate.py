#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Materialize a user-approved forced/manual-risk OTTO residual full-test candidate.

This script intentionally writes an uploadable CSV only because the user explicitly
asked to submit the pre-registered OTTO residual to verify the rejection decision.
It uses public train.json + official pairs.csv and existing full-train LightGCN
score artifacts. It never reads hidden/private labels or external Steam data and
never calls the Kaggle submit API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from otto_source_covisit_smoke import FEATURE_COLS, build_indices, build_sources, score_candidates  # noqa: E402
from recsys_played_utils import (  # noqa: E402
    DEFAULT_DATA_DIR,
    ensure_dir,
    load_pairs_csv,
    load_train_json,
    normalize_within_user,
    predict_tophalf,
    write_json,
)

MODEL_SEEDS = [42, 123, 2024, 7]
FEATURE_TERMS = [("score_coplay_top5_mean", 0.090), ("score_reverse_recent", 0.040)]
CANDIDATE_NAME = "otto_coplay_top5_reverse_recent_w0090_w0040_forced"


def label_col(df: pd.DataFrame) -> str:
    for c in ("Played", "Label", "Pred"):
        if c in df.columns:
            return c
    raise ValueError(f"no label column in {df.columns.tolist()}")


def load_base_scores() -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    cols: list[str] = []
    for seed in MODEL_SEEDS:
        path = ROOT / f"artifacts/lightgcn_emb128L4r3_fulltest/seed{seed}/test.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        need = {"ID", "userID", "gameID", "score_lightgcn"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"missing {missing} in {path}")
        col = f"base_seed{seed}"
        part = df[["ID", "userID", "gameID", "score_lightgcn"]].rename(columns={"score_lightgcn": col})
        merged = part if merged is None else merged.merge(part[["ID", col]], on="ID", validate="one_to_one")
        cols.append(col)
    assert merged is not None
    merged["score_base"] = merged[cols].mean(axis=1)
    return merged[["ID", "userID", "gameID", "score_base"]].sort_values("ID", kind="mergesort").reset_index(drop=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submission_signature(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    lc = label_col(df)
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": sha256_file(path),
        "rows": int(len(df)),
        "label_col": lc,
        "ones": int(df[lc].sum()),
        "zeros": int((1 - df[lc].astype(int)).sum()) if df[lc].dropna().isin([0, 1]).all() else None,
    }


def rowdiff(a: pd.DataFrame, b_path: Path, b_name: str) -> dict[str, Any]:
    if not b_path.exists():
        return {"name": b_name, "path": str(b_path), "exists": False}
    b = pd.read_csv(b_path)
    bc = label_col(b)
    cmp = a[["ID", "Played"]].merge(b[["ID", bc]].rename(columns={bc: "Other"}), on="ID", validate="one_to_one")
    diff = cmp["Played"].astype(int) != cmp["Other"].astype(int)
    promoted = (cmp["Played"].astype(int) == 1) & (cmp["Other"].astype(int) == 0)
    demoted = (cmp["Played"].astype(int) == 0) & (cmp["Other"].astype(int) == 1)
    return {
        "name": b_name,
        "path": str(b_path.relative_to(ROOT) if b_path.is_relative_to(ROOT) else b_path),
        "exists": True,
        "rowdiff": int(diff.sum()),
        "rowdiff_frac": float(diff.mean()),
        "promoted_vs_other": int(promoted.sum()),
        "demoted_vs_other": int(demoted.sum()),
        "other_sha256": sha256_file(b_path),
    }


def user_tophalf_check(sub: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, Any]:
    merged = pairs.merge(sub, on="ID", validate="one_to_one")
    g = merged.groupby("userID", sort=False).agg(n=("ID", "size"), pred_pos=("Played", "sum"))
    g["expected"] = g["n"] // 2
    bad = g[g["pred_pos"] != g["expected"]]
    return {
        "users": int(len(g)),
        "bad_users": int(len(bad)),
        "bad_user_examples": bad.head(10).reset_index().to_dict(orient="records"),
        "all_candidate_counts_even": bool((g["n"] % 2 == 0).all()),
        "predicted_positive_total": int(g["pred_pos"].sum()),
        "expected_positive_total": int(g["expected"].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ts", required=True)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--artifact-dir", default=None)
    args = ap.parse_args()

    out_csv = ROOT / (args.out_csv or f"submissions/candidate_{CANDIDATE_NAME}_{args.run_ts}.csv")
    out_json = ROOT / (args.out_json or f"reports/{args.run_ts}_{CANDIDATE_NAME}_preflight.json")
    out_md = ROOT / (args.out_md or f"reports/{args.run_ts}_{CANDIDATE_NAME}_preflight.md")
    artifact_dir = ROOT / (args.artifact_dir or f"artifacts/{CANDIDATE_NAME}_{args.run_ts}")
    ensure_dir(out_csv.parent)
    ensure_dir(out_json.parent)
    ensure_dir(artifact_dir)

    data_dir = ROOT / DEFAULT_DATA_DIR
    pairs = load_pairs_csv(data_dir / "pairs.csv").sort_values("ID", kind="mergesort").reset_index(drop=True)
    train = load_train_json(data_dir / "train.json")
    base = load_base_scores()
    if len(base) != len(pairs):
        raise RuntimeError(f"base rows {len(base)} != pairs rows {len(pairs)}")
    if not base[["ID", "userID", "gameID"]].equals(pairs[["ID", "userID", "gameID"]]):
        raise RuntimeError("base score rows do not match pairs.csv order/keys")

    _, item_to_idx, _ = build_indices(train)
    sources = build_sources(train, item_to_idx)
    scored = score_candidates(train, base.copy(), item_to_idx, sources)
    scored = normalize_within_user(scored, ["score_base"], user_col="userID")
    score = scored["z_score_base"].to_numpy(np.float64).copy()
    for feature, weight in FEATURE_TERMS:
        col = f"z_{feature}"
        if col not in scored.columns:
            raise ValueError(f"missing feature z column {col}")
        score += weight * scored[col].to_numpy(np.float64)
    scored["score_otto_forced"] = score
    pred = predict_tophalf(scored, "score_otto_forced", label_col=None, user_col="userID", id_col="ID")
    sub = pred[["ID", "Pred"]].rename(columns={"Pred": "Played"}).sort_values("ID", kind="mergesort").reset_index(drop=True)
    sub["Played"] = sub["Played"].astype(int)
    sub_upload = sub.rename(columns={"Played": "Label"})

    # Strict structural preflight before writing final CSV.
    if not sub["ID"].equals(pairs["ID"]):
        raise RuntimeError("submission ID order mismatch against pairs.csv")
    if sub["ID"].duplicated().any():
        raise RuntimeError("duplicate IDs in candidate")
    labels = set(sub["Played"].unique().tolist())
    if labels - {0, 1}:
        raise RuntimeError(f"non-binary labels: {labels}")
    topcheck = user_tophalf_check(sub, pairs)
    if topcheck["bad_users"]:
        raise RuntimeError(f"top-half check failed: {topcheck['bad_users']} bad users")

    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    sub_upload.to_csv(tmp, index=False)
    tmp.replace(out_csv)
    score_artifact = artifact_dir / "fulltest_otto_source_scores.csv"
    keep_cols = ["ID", "userID", "gameID", "score_base"] + FEATURE_COLS + [f"z_{c}" for c in FEATURE_COLS if c != "score_source_mean_z"] + ["z_score_base", "score_otto_forced"]
    scored[keep_cols].to_csv(score_artifact, index=False)

    local_sigs = []
    for p in sorted((ROOT / "submissions").glob("*.csv")):
        try:
            local_sigs.append(submission_signature(p))
        except Exception as e:
            local_sigs.append({"path": str(p.relative_to(ROOT)), "error": str(e)})
    candidate_sha = sha256_file(out_csv)
    duplicate_local = [s for s in local_sigs if s.get("path") != str(out_csv.relative_to(ROOT)) and s.get("sha256") == candidate_sha]

    refs = [
        rowdiff(sub, ROOT / "artifacts/lightgcn_emb128L4r3_fulltest/test_candidate/candidate_lightgcn_emb128L4r3_seed_ens.csv", "emb128_L4_reg1e-3_4seed_public0.77745"),
        rowdiff(sub, ROOT / "submissions/candidate_rank_blend_emb128_emb192.csv", "current_live_best_candidate_rank_blend_public0.77825"),
        rowdiff(sub, ROOT / "submissions/candidate_autorun_rankblend_z_plus_score_als_htr_f32_it30_alpha20_popa4_w0.1.csv", "rankblend_als_htr_public0.77805"),
    ]

    payload: dict[str, Any] = {
        "timestamp_kst": args.run_ts,
        "candidate_name": CANDIDATE_NAME,
        "formula": {
            "base": "emb128_L4_reg1e-3 full-train 4-seed raw score mean, within-user z-normalized",
            "terms": FEATURE_TERMS,
            "score": "z_score_base + 0.090*z_score_coplay_top5_mean + 0.040*z_score_reverse_recent",
            "decoder": "per-user top-half over official pairs.csv",
        },
        "submission_file": str(out_csv.relative_to(ROOT)),
        "score_artifact": str(score_artifact.relative_to(ROOT)),
        "sha256": candidate_sha,
        "rows": int(len(sub)),
        "columns": sub_upload.columns.tolist(),
        "label_counts": {str(k): int(v) for k, v in sub["Played"].value_counts().sort_index().items()},
        "id_min": int(sub["ID"].min()),
        "id_max": int(sub["ID"].max()),
        "id_unique": int(sub["ID"].nunique()),
        "pairs_rows": int(len(pairs)),
        "tophalf_check": topcheck,
        "rowdiffs": refs,
        "duplicate_local_sha_matches": duplicate_local,
        "safety_flags": {
            "user_explicit_submit_request": True,
            "candidate_csv_written": True,
            "kaggle_submit_executed_by_this_script": False,
            "hidden_labels_used": False,
            "private_answers_used": False,
            "external_steam_scraping_used": False,
            "credentials_or_tokens_printed": False,
            "git_stage_commit_push_executed": False,
        },
        "validation_evidence": {
            "independent_verdict": "INDEPENDENT_DIAGNOSTIC_ONLY_POSITIVE_STRICT_FAIL",
            "independent_mean_delta_vs_emb128": 0.0006668000266719654,
            "independent_min_delta_vs_emb128": -0.000600120024004891,
            "independent_positive_splits": "2/3",
            "risk_label": "FORCED manual-risk verification probe; not strict-ready",
        },
    }
    write_json(out_json, payload)
    lines = [
        "# Forced OTTO residual candidate preflight",
        "",
        f"- Timestamp: `{args.run_ts}`",
        f"- Candidate: `{CANDIDATE_NAME}`",
        f"- File: `{payload['submission_file']}`",
        f"- SHA256: `{candidate_sha}`",
        f"- Rows: `{len(sub)}`",
        f"- Label counts: `{payload['label_counts']}`",
        f"- Top-half bad users: `{topcheck['bad_users']}`",
        "- Risk label: `FORCED manual-risk verification probe; not strict-ready`",
        "",
        "## Row diffs",
        "",
    ]
    for r in refs:
        lines.append(f"- {r['name']}: `{r}`")
    lines += ["", "## Validation evidence", "", "Independent strict confirmation failed: mean Δ +0.0006668, min Δ -0.0006001, positive 2/3, p 0.1700."]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"submission_file": payload["submission_file"], "sha256": candidate_sha, "rows": len(sub), "tophalf_bad_users": topcheck["bad_users"], "duplicate_local_sha_matches": len(duplicate_local), "rowdiffs": refs, "out_json": str(out_json.relative_to(ROOT)), "out_md": str(out_md.relative_to(ROOT))}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
