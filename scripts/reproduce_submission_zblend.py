#!/usr/bin/env python3
"""eCampus one-file reproduction + verification for the QUOTA-BURN z-blend candidate.

Candidate: emb128 4-seed ensemble (+) emb64 4-seed ensemble, WITHIN-USER z-score each,
summed, per-user top-half decode -> Played 0/1.
  public 0.77815 (LB #1 at submit time) | uniform-surrogate 0.76295 (-0.0021 vs emb128;
  public gain is noise) | SHA256 99adb88a...

NOTE on final-2 status: This candidate's public top-ranking is NOT a real signal (uniform
surrogate is WORSE than emb128). Its only legitimate role is as a #2 hedge: it DOMINATES emb64
on both uniform (+0.0015) and public (+0.0069). emb128 remains the uniform-best #1.

--verify-existing (default): re-aggregate from the per-seed full-train test scores already in
    artifacts/, rebuild the candidate, assert SHA256 matches. No GPU. Standalone (portable ROOT).
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KMU_ROOT", str(Path(__file__).resolve().parent.parent)))

EMB128 = [str(ROOT / f"artifacts/lightgcn_emb128L4r3_fulltest/seed{s}/test.csv") for s in (42, 123, 2024, 7)]
EMB64 = [
    str(ROOT / "artifacts/lightgcn_20260530/test_full_train/lightgcn_test_raw_scores_emb64_L3_reg1e-04.csv"),
    str(ROOT / "artifacts/lightgcn_seed_ensemble/seed123/test.csv"),
    str(ROOT / "artifacts/lightgcn_seed_ensemble/seed2024/test.csv"),
    str(ROOT / "artifacts/lightgcn_seed_ensemble/seed7/test.csv"),
]
PAIRS = ROOT / "data/raw/public/data/pairs.csv"
OUT = ROOT / "submissions/candidate_emb128_emb64_zblend.csv"
EXPECTED_SHA = "99adb88a90dd99c01ea7f50c56b49cef034a24ebbf0fa0b485d0209da2bafb01"
N_EXPECTED = 19998


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _scol(df):
    for c in ("score_lightgcn", "score"):
        if c in df.columns:
            return c
    raise SystemExit(f"no score col in {df.columns.tolist()}")


def _ens(files):
    base = pd.read_csv(files[0]); sc = _scol(base)
    acc = base[["ID"]].copy(); acc["s"] = base[sc].astype(float).values
    for f in files[1:]:
        d = pd.read_csv(f)
        acc = acc.merge(d[["ID"]].assign(s2=d[_scol(d)].astype(float).values), on="ID")
        acc["s"] = acc["s"] + acc.pop("s2")
    acc["s"] = acc["s"] / len(files)
    return acc[["ID", "s"]]


def _wz(df, col):
    g = df.groupby("userID")[col]
    mu = g.transform("mean"); sd = g.transform("std").replace(0, 1.0)
    return (df[col] - mu) / sd


def build() -> tuple[Path, str, dict]:
    a = _ens(EMB128).rename(columns={"s": "a"})
    b = _ens(EMB64).rename(columns={"s": "b"})
    pairs = pd.read_csv(PAIRS)[["ID", "userID"]]
    df = pairs.merge(a, on="ID").merge(b, on="ID")
    assert len(df) == len(pairs), f"merge loss {len(df)} vs {len(pairs)}"
    df["score"] = _wz(df, "a") + _wz(df, "b")
    pred = np.zeros(len(df), dtype=int); sc = df["score"].values
    for uid, idx in df.groupby("userID").indices.items():
        idx = np.asarray(idx); k = len(idx) // 2
        pred[idx[np.argsort(sc[idx])[::-1][:k]]] = 1
    df["Played"] = pred
    out = df[["ID", "Played"]].sort_values("ID").reset_index(drop=True)
    n = len(out); pos = int(out["Played"].sum()); neg = n - pos
    chk = df.groupby("userID")["Played"].agg(["sum", "count"])
    bad = int(((chk["count"] % 2 != 0) | (chk["sum"] != chk["count"] // 2)).sum())
    preflight = {"rows": n, "pos": pos, "neg": neg, "balanced": pos == neg,
                 "bad_users": bad, "n_ok": n == N_EXPECTED}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    return OUT, sha256_file(OUT), preflight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-existing", action="store_true", default=True)
    ap.parse_args()
    path, sha, pf = build()
    match = sha == EXPECTED_SHA
    print("=" * 72)
    print(f"candidate : {path}")
    print(f"sha256    : {sha}")
    print(f"expected  : {EXPECTED_SHA}")
    print(f"SHA MATCH : {'YES ✅ reproducible' if match else 'NO ❌'}")
    print(f"preflight : rows={pf['rows']} pos/neg={pf['pos']}/{pf['neg']} "
          f"balanced={pf['balanced']} bad_users={pf['bad_users']}")
    print("=" * 72)
    rep = ROOT / "reports/20260601_repro_zblend_verification.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps({
        "submission": "emb128 4-seed (+) emb64 4-seed within-user z-blend, per-user top-half",
        "public_score": 0.77815, "uniform_surrogate": 0.76295,
        "uniform_delta_vs_emb128": -0.0021, "uniform_delta_vs_emb64": +0.0015,
        "role": "final-2 #2 hedge candidate (dominates emb64; public top-rank is noise vs emb128)",
        "candidate_sha256": sha, "expected_sha256": EXPECTED_SHA, "sha_match": match,
        "preflight": pf,
    }, indent=2))
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
