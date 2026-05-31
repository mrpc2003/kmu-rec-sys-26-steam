#!/usr/bin/env python3
"""eCampus one-file reproduction + verification for the FINAL #2 submission.

Final #2: emb64_L3_reg1e-4 LightGCN 4-seed ensemble (seeds 42/123/2024/7),
raw-score mean, per-user top-half decode.
  public 0.77125 | SHA256 dcc578de...

Mirrors scripts/lightgcn_seed_ensemble_aggregate.py test-candidate block byte-for-byte.
Note the asymmetric inputs (the #2 ensemble was built incrementally):
  - seed42 : artifacts/lightgcn_20260530/test_full_train/lightgcn_test_raw_scores_emb64_L3_reg1e-04.csv  (col score_lightgcn)
  - seed123/2024/7 : artifacts/lightgcn_seed_ensemble/seed{S}/test.csv  (col score)

--verify-existing (default): re-aggregate from the existing per-seed raw scores and assert
    the resulting CSV SHA256 matches dcc578de.... CPU only.
NO Kaggle submission is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import predict_tophalf, ensure_dir  # noqa: E402

NEW_SEEDS = [123, 2024, 7]
SEED42_TEST = ROOT / "artifacts/lightgcn_20260530/test_full_train/lightgcn_test_raw_scores_emb64_L3_reg1e-04.csv"
OUT_DIR = ROOT / "artifacts/lightgcn_seed_ensemble/test_candidate"
CANDIDATE_NAME = "candidate_lightgcn_seed_ens.csv"
EXPECTED_SHA = "dcc578de495f98133d1cbccfcb53a156f7d1f46973f571164b6ec90605d937f7"
DATA = ROOT / "data/raw/public/data"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def aggregate() -> tuple[Path, str, dict]:
    """EXACT mirror of lightgcn_seed_ensemble_aggregate.py test-candidate block."""
    test_files = {42: SEED42_TEST}
    for s in NEW_SEEDS:
        test_files[s] = ROOT / f"artifacts/lightgcn_seed_ensemble/seed{s}/test.csv"
    missing = [s for s, p in test_files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"per-seed test scores not ready: {missing}")

    t42 = pd.read_csv(test_files[42])[["ID", "userID", "gameID", "score_lightgcn"]].rename(
        columns={"score_lightgcn": "score_seed42"})
    mt = t42
    cols = ["score_seed42"]
    for s in NEW_SEEDS:
        d = pd.read_csv(test_files[s])[["ID", "score"]].rename(columns={"score": f"score_seed{s}"})
        mt = mt.merge(d, on="ID", how="inner")
        cols.append(f"score_seed{s}")

    mt["score_ens"] = mt[cols].mean(axis=1)
    pred = predict_tophalf(mt, "score_ens", label_col=None, user_col="userID", id_col="ID")
    sub = pred[["ID", "Pred"]].rename(columns={"Pred": "Label"}).sort_values("ID")

    out_dir = ensure_dir(OUT_DIR)
    csv_path = out_dir / CANDIDATE_NAME
    sub.to_csv(csv_path, index=False)
    sha = sha256_file(csv_path)

    pairs = pd.read_csv(DATA / "pairs.csv")
    g = sub.merge(pairs, on="ID").groupby("userID").agg(n=("ID", "size"), p=("Label", "sum"))
    preflight = {
        "rows": int(len(sub)),
        "label_1": int(sub.Label.sum()),
        "label_0": int((1 - sub.Label).sum()),
        "users": int(len(g)),
        "bad_users_p_ne_half": int((g.p != g.n // 2).sum()),
        "id_unique": bool(sub.ID.is_unique),
        "id_matches_pairs": bool(set(sub.ID) == set(pairs.ID)),
    }
    return csv_path, sha, preflight


def git_state() -> dict:
    def _run(args):
        try:
            return subprocess.run(["git"] + args, cwd=str(ROOT), capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None
    return {"commit": _run(["rev-parse", "HEAD"]),
            "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_run(["status", "--porcelain"]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify-existing", action="store_true",
                    help="Explicit no-op for the default verify-from-existing mode (canonical eCampus command).")
    ap.parse_args()

    csv_path, sha, preflight = aggregate()
    match = (sha == EXPECTED_SHA)
    data_fp = {"train.json": sha256_file(DATA / "train.json"), "pairs.csv": sha256_file(DATA / "pairs.csv")}
    provenance = {
        "submission": "emb64_L3_reg1e-4 LightGCN 4-seed ensemble (raw-mean, per-user top-half)",
        "rank": "final #2",
        "public_score": 0.77125,
        "candidate_file": str(csv_path),
        "candidate_sha256": sha,
        "expected_sha256": EXPECTED_SHA,
        "sha_match": match,
        "config": {"emb_dim": 64, "n_layers": 3, "reg": 1e-4, "lr": 1e-3,
                   "epochs": 200, "batch_size": 4096, "seeds": [42] + NEW_SEEDS},
        "input_paths": {"seed42": str(SEED42_TEST),
                        "seed123/2024/7": "artifacts/lightgcn_seed_ensemble/seed{S}/test.csv"},
        "verify_command": "uv run --with numpy --with pandas --with scipy python3 scripts/reproduce_submission_emb64.py --verify-existing",
        "data_fingerprint_sha256": data_fp,
        "environment": {"python": sys.version.split()[0], "pandas": pd.__version__},
        "git_state": git_state(),
        "preflight": preflight,
    }
    report = ROOT / "reports/20260601_ecampus_repro_emb64_verification.json"
    report.write_text(json.dumps(provenance, indent=2, ensure_ascii=False))

    print("\n" + "=" * 72)
    print(f"candidate : {csv_path}")
    print(f"sha256    : {sha}")
    print(f"expected  : {EXPECTED_SHA}")
    print(f"SHA MATCH : {'YES ✅ reproducible' if match else 'NO ❌ MISMATCH'}")
    print(f"preflight : rows={preflight['rows']} 1/0={preflight['label_1']}/{preflight['label_0']} "
          f"bad_users={preflight['bad_users_p_ne_half']} id_ok={preflight['id_matches_pairs']}")
    print(f"report    : {report}")
    print("=" * 72)
    if not match:
        sys.exit(1)


if __name__ == "__main__":
    main()
