#!/usr/bin/env python3
"""eCampus one-file reproduction + verification for the FINAL #1 submission.

Final submission: emb128_L4_reg1e-3 LightGCN 4-seed ensemble (seeds 42/123/2024/7),
raw-score mean, per-user top-half decode.
  uniform-surrogate gate 0.76505 | public 0.77745 | SHA256 7e3191de...

This script reproduces / verifies that exact candidate CSV and records all eCampus
provenance (SHA, command, seeds, data fingerprints, environment, git state) in one place.

Two modes
---------
--verify-existing (default): re-run the EXACT aggregate from the 4 deterministic per-seed
    full-train test scores (artifacts/lightgcn_emb128L4r3_fulltest/seed{S}/test.csv), then
    assert the resulting CSV SHA256 matches the recorded value. No GPU needed. This proves
    the recorded submission is byte-identically reproducible from the seed scores.
--from-scratch: first regenerate each per-seed test.csv by training the canonical LightGCN
    (scripts/lightgcn_fulltest_param.py, GPU), then aggregate + verify. Full end-to-end.

Aggregate logic mirrors scripts/emb128L4r3_test_aggregate.py byte-for-byte.
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

SEEDS = [42, 123, 2024, 7]
EMB_DIM, N_LAYERS, REG, LR, EPOCHS, BATCH = 128, 4, 1e-3, 1e-3, 200, 4096
FT_DIR = ROOT / "artifacts/lightgcn_emb128L4r3_fulltest"
OUT_DIR = FT_DIR / "test_candidate"
CANDIDATE_NAME = "candidate_lightgcn_emb128L4r3_seed_ens.csv"
EXPECTED_SHA = "7e3191dead2d85637dbf01f7ad50aeb61394c6b709959f2c2850bc53c430c195"
DATA = ROOT / "data/raw/public/data"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def train_one_seed(seed: int, device: str) -> None:
    """Regenerate artifacts/lightgcn_emb128L4r3_fulltest/seed{seed}/test.csv (GPU)."""
    out = ensure_dir(FT_DIR / f"seed{seed}")
    cmd = [
        "uv", "run", "--python", "3.13", "--with", "torch==2.10.0",
        "--with", "numpy", "--with", "pandas", "--with", "scipy",
        "python3", "scripts/lightgcn_fulltest_param.py",
        "--emb-dim", str(EMB_DIM), "--n-layers", str(N_LAYERS), "--lr", str(LR),
        "--reg", str(REG), "--epochs", str(EPOCHS), "--batch-size", str(BATCH),
        "--seed", str(seed), "--device", device, "--out-dir", str(out),
    ]
    print(f"[from-scratch] training seed {seed} -> {out}", flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def aggregate() -> tuple[Path, str, dict]:
    """EXACT mirror of emb128L4r3_test_aggregate.py."""
    test_files = {s: FT_DIR / f"seed{s}" / "test.csv" for s in SEEDS}
    missing = [s for s, p in test_files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"per-seed test.csv not ready: {missing}")

    mt = None
    cols = []
    for s in SEEDS:
        d = pd.read_csv(test_files[s])[["ID", "userID", "gameID", "score_lightgcn"]].rename(
            columns={"score_lightgcn": f"score_seed{s}"})
        mt = d if mt is None else mt.merge(d[["ID", f"score_seed{s}"]], on="ID", how="inner")
        cols.append(f"score_seed{s}")

    mt["score_ens"] = mt[cols].mean(axis=1)
    pred = predict_tophalf(mt, "score_ens", label_col=None, user_col="userID", id_col="ID")
    sub = pred[["ID", "Pred"]].rename(columns={"Pred": "Label"}).sort_values("ID")

    out_dir = ensure_dir(OUT_DIR)
    csv_path = out_dir / CANDIDATE_NAME
    sub.to_csv(csv_path, index=False)
    sha = sha256_file(csv_path)

    # preflight: every user exactly half positive
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


def env_state() -> dict:
    info = {"python": sys.version.split()[0]}
    try:
        import torch  # noqa
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
    except Exception:
        info["torch"] = "not-imported-in-verify-mode"
    info["pandas"] = pd.__version__
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-scratch", action="store_true",
                    help="Retrain all 4 seeds (GPU) before aggregating. Default: verify from existing seed scores.")
    ap.add_argument("--verify-existing", action="store_true",
                    help="Explicit no-op for the default verify-from-existing mode (canonical eCampus command).")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.from_scratch and args.verify_existing:
        ap.error("--from-scratch and --verify-existing are mutually exclusive")

    if args.from_scratch:
        for s in SEEDS:
            if not (FT_DIR / f"seed{s}" / "test.csv").exists():
                train_one_seed(s, args.device)
            else:
                print(f"[from-scratch] seed {s} test.csv exists, skipping retrain", flush=True)

    csv_path, sha, preflight = aggregate()
    match = (sha == EXPECTED_SHA)

    data_fp = {
        "train.json": sha256_file(DATA / "train.json"),
        "pairs.csv": sha256_file(DATA / "pairs.csv"),
    }
    provenance = {
        "submission": "emb128_L4_reg1e-3 LightGCN 4-seed ensemble (raw-mean, per-user top-half)",
        "public_score": 0.77745,
        "uniform_surrogate": 0.76505,
        "candidate_file": str(csv_path),
        "candidate_sha256": sha,
        "expected_sha256": EXPECTED_SHA,
        "sha_match": match,
        "config": {"emb_dim": EMB_DIM, "n_layers": N_LAYERS, "reg": REG, "lr": LR,
                   "epochs": EPOCHS, "batch_size": BATCH, "seeds": SEEDS},
        "generation_command": (
            "for s in 42 123 2024 7: python3 scripts/lightgcn_fulltest_param.py "
            "--emb-dim 128 --n-layers 4 --lr 1e-3 --reg 1e-3 --epochs 200 --batch-size 4096 "
            "--seed $s --device cuda:0 --out-dir artifacts/lightgcn_emb128L4r3_fulltest/seed$s ; "
            "then aggregate (raw-mean -> per-user top-half) -> candidate_lightgcn_emb128L4r3_seed_ens.csv"),
        "verify_command": "uv run --with numpy --with pandas --with scipy python3 scripts/reproduce_submission_emb128.py --verify-existing",
        "data_fingerprint_sha256": data_fp,
        "environment": env_state(),
        "git_state": git_state(),
        "preflight": preflight,
        "mode": "from-scratch" if args.from_scratch else "verify-existing",
    }
    report = ROOT / "reports/20260601_ecampus_repro_emb128_verification.json"
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
