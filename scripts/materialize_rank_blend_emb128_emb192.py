#!/usr/bin/env python3
"""Materialize a forced/manual-risk rank-blend candidate for KMURecSys26 Steam.

Candidate: rank(emb128 4-seed full-test raw-mean + emb192 4-seed full-test raw-mean),
per-user top-half decode -> `ID,Played` Kaggle CSV.

Rationale: among already computed, unsubmitted end-game options this had the strongest
validation-only surrogate accuracy signal (all 3 uniform splits positive, mean Δ≈+0.00083),
while failing paired McNemar significance. Use only after explicit user request to spend a
last/manual-risk submission. This script does NOT submit to Kaggle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data/raw/public/data/pairs.csv"
OUT = ROOT / "submissions/candidate_rank_blend_emb128_emb192.csv"
REPORT = ROOT / "reports/20260602_rank_blend_emb128_emb192_preflight.json"

EMB128_FILES = [ROOT / f"artifacts/lightgcn_emb128L4r3_fulltest/seed{s}/test.csv" for s in (42, 123, 2024, 7)]
EMB192_FILES = [ROOT / f"artifacts/lightgcn_emb192L4r3_fulltest/seed{s}/test.csv" for s in (42, 123, 2024, 7)]
EMB128_CAND = ROOT / "artifacts/lightgcn_emb128L4r3_fulltest/test_candidate/candidate_lightgcn_emb128L4r3_seed_ens.csv"
EMB192_CAND = ROOT / "artifacts/lightgcn_emb192L4r3_fulltest/test_candidate/candidate_lightgcn_emb192L4r3_seed_ens.csv"
ZBLEND_CAND = ROOT / "submissions/candidate_emb128_emb64_zblend.csv"
N_EXPECTED = 19998


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_col(df: pd.DataFrame) -> str:
    for col in ("score_lightgcn", "score"):
        if col in df.columns:
            return col
    raise ValueError(f"No score column in {df.columns.tolist()}")


def ensemble(files: list[Path], name: str) -> pd.DataFrame:
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {name} score files: {missing}")
    first = pd.read_csv(files[0])
    sc = score_col(first)
    required = {"ID", "userID", "gameID", sc}
    if not required.issubset(first.columns):
        raise ValueError(f"Missing columns in {files[0]}: {required - set(first.columns)}")
    out = first[["ID", "userID", "gameID"]].copy()
    out[name] = first[sc].astype(float).to_numpy()
    for path in files[1:]:
        d = pd.read_csv(path)
        sc2 = score_col(d)
        part = d[["ID", sc2]].copy().rename(columns={sc2: "_score"})
        before = len(out)
        out = out.merge(part, on="ID", how="inner", validate="one_to_one")
        if len(out) != before:
            raise RuntimeError(f"Row alignment changed while merging {path}: {before}->{len(out)}")
        out[name] += out.pop("_score").astype(float).to_numpy()
    out[name] /= len(files)
    return out


def user_rank_high_is_good(df: pd.DataFrame, col: str) -> np.ndarray:
    # Match validation gate implementation: lowest score rank 0, highest rank n-1.
    ranks = np.zeros(len(df), dtype=np.float64)
    values = df[col].to_numpy(dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        ranks[idx[np.argsort(values[idx])]] = np.arange(len(idx), dtype=np.float64)
    return ranks


def top_half_decode(df: pd.DataFrame, score_col_name: str) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    values = df[score_col_name].to_numpy(dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        k = len(idx) // 2
        order = np.argsort(values[idx])[::-1]
        pred[idx[order[:k]]] = 1
    return pred


def read_candidate(path: Path, col_hint: str = "Played") -> pd.DataFrame | None:
    if not path.exists():
        return None
    d = pd.read_csv(path)
    label_cols = [c for c in ("Played", "Label") if c in d.columns]
    if not label_cols:
        raise ValueError(f"No Played/Label in {path}")
    return d[["ID", label_cols[0]]].rename(columns={label_cols[0]: col_hint})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args()

    out_path = Path(args.out)
    report_path = Path(args.report)
    pairs = pd.read_csv(PAIRS)
    if list(pairs.columns) != ["ID", "userID", "gameID"]:
        raise ValueError(f"Unexpected pairs columns: {pairs.columns.tolist()}")
    if len(pairs) != N_EXPECTED:
        raise ValueError(f"Unexpected pairs rows: {len(pairs)} != {N_EXPECTED}")
    if not pairs["ID"].is_unique or not (pairs["ID"].to_numpy() == np.arange(len(pairs))).all():
        raise ValueError("pairs ID is not unique contiguous 0..N-1")

    a = ensemble(EMB128_FILES, "score_emb128")
    b = ensemble(EMB192_FILES, "score_emb192")[["ID", "score_emb192"]]
    df = pairs.merge(a, on=["ID", "userID", "gameID"], how="left", validate="one_to_one").merge(
        b, on="ID", how="left", validate="one_to_one"
    )
    if len(df) != len(pairs) or df[["score_emb128", "score_emb192"]].isna().any().any():
        raise RuntimeError("Score merge failed or produced NaN")

    df["rank_emb128"] = user_rank_high_is_good(df, "score_emb128")
    df["rank_emb192"] = user_rank_high_is_good(df, "score_emb192")
    df["score_rank_blend"] = df["rank_emb128"] + df["rank_emb192"]
    df["Played"] = top_half_decode(df, "score_rank_blend").astype(int)

    out = df[["ID", "Played"]].sort_values("ID", kind="mergesort").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    sha = sha256_file(out_path)

    user_chk = pairs[["ID", "userID"]].merge(out, on="ID", validate="one_to_one")
    per_user = user_chk.groupby("userID", sort=False)["Played"].agg(["sum", "count"])
    bad_users = int(((per_user["count"] % 2 != 0) | (per_user["sum"] != per_user["count"] // 2)).sum())

    comparisons: dict[str, object] = {}
    for label, path in {
        "emb128": EMB128_CAND,
        "emb192": EMB192_CAND,
        "zblend_public_best": ZBLEND_CAND,
    }.items():
        c = read_candidate(path, col_hint=label)
        if c is None:
            comparisons[label] = {"path": str(path), "exists": False}
            continue
        m = out.merge(c, on="ID", validate="one_to_one")
        diff = int((m["Played"].astype(int) != m[label].astype(int)).sum())
        comparisons[label] = {"path": str(path), "exists": True, "row_diff": diff, "row_diff_frac": diff / len(out)}

    payload = {
        "candidate": "rank_blend_emb128_emb192",
        "role": "forced/manual-risk last-slot candidate; strongest unsubmitted surrogate signal but McNemar-rejected",
        "file": str(out_path),
        "sha256": sha,
        "validation_evidence": {
            "uniform_3split_mean_delta_vs_emb128": 0.00083,
            "split_deltas": [0.00170, 0.00030, 0.00050],
            "mcnemar_p_by_split": [0.0727, 0.7785, 0.5972],
            "fisher_combined_p": 0.3421,
            "strict_gate": False,
        },
        "preflight": {
            "rows": int(len(out)),
            "expected_rows": N_EXPECTED,
            "id_unique": bool(out["ID"].is_unique),
            "id_contiguous": bool((out["ID"].to_numpy() == np.arange(len(out))).all()),
            "columns": out.columns.tolist(),
            "labels_binary": bool(set(out["Played"].astype(int).unique()).issubset({0, 1})),
            "label_1": int(out["Played"].sum()),
            "label_0": int(len(out) - out["Played"].sum()),
            "bad_users_tophalf": bad_users,
        },
        "comparisons": comparisons,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)

    pf = payload["preflight"]
    ok = (
        pf["rows"] == N_EXPECTED
        and pf["id_unique"]
        and pf["id_contiguous"]
        and pf["labels_binary"]
        and pf["label_1"] == pf["label_0"] == N_EXPECTED // 2
        and pf["bad_users_tophalf"] == 0
    )
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
