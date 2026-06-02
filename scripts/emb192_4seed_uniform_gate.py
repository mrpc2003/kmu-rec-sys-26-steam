#!/usr/bin/env python3
"""emb192 L4 reg1e-3 — 4-seed ENSEMBLE uniform gate vs emb128 ensemble.

Capacity probe found emb192 single-seed42 uniform 0.76665, already +0.0016 over the emb128
4-seed ensemble (0.76505) and +0.0046 over the emb128 single seed (6.5x the 0.0007 noise
band). This gate confirms the finding is not a single-seed fluke: it averages emb192 raw
scores across seeds {42,123,2024,7} on the uniform public-surrogate split and checks whether
the emb192 4-seed ensemble beats the emb128 4-seed ensemble by > noise 0.0007.

Seed paths (uniform eval, col=score_lightgcn, all from lightgcn_capacity_uniform_probe.py):
  42  : artifacts/capacity_uniform/emb192_L4_r3/val_random_uniform_seed42/lightgcn_scores.csv
  123 : artifacts/capacity_uniform/emb192_L4_r3_seed123/...
  2024: artifacts/capacity_uniform/emb192_L4_r3_seed2024/...
  7   : artifacts/capacity_uniform/emb192_L4_r3_seed7/...

No Kaggle submission inside this script. Report-only; the autonomous runner handles any
Kaggle submission after safety gates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import evaluate_tophalf  # noqa: E402

SPLIT = "val_random_uniform_seed42"
EMB128_ENS_REF = 0.76505
EMB128_SINGLE = 0.76205
NOISE = 0.0007
SEED_PATHS = {
    42:   ROOT / "artifacts/capacity_uniform/emb192_L4_r3" / SPLIT / "lightgcn_scores.csv",
    123:  ROOT / "artifacts/capacity_uniform/emb192_L4_r3_seed123" / SPLIT / "lightgcn_scores.csv",
    2024: ROOT / "artifacts/capacity_uniform/emb192_L4_r3_seed2024" / SPLIT / "lightgcn_scores.csv",
    7:    ROOT / "artifacts/capacity_uniform/emb192_L4_r3_seed7" / SPLIT / "lightgcn_scores.csv",
}
OUT_JSON = ROOT / "reports/20260531_emb192_4seed_uniform_gate.json"
OUT_MD = ROOT / "reports/20260531_emb192_4seed_uniform_gate.md"


def main() -> None:
    base = None
    cols, present, missing = [], [], []
    for s, p in SEED_PATHS.items():
        if not p.exists():
            missing.append(s); continue
        d = pd.read_csv(p)[["ID", "userID", "gameID", "Label", "score_lightgcn"]].rename(
            columns={"score_lightgcn": f"s{s}"})
        if base is None:
            base = d
        else:
            base = base.merge(d[["ID", f"s{s}"]], on="ID", how="inner")
        cols.append(f"s{s}"); present.append(s)

    if base is None:
        print(f"[wait] no seeds ready yet (missing {missing})"); return
    if missing:
        print(f"[wait] seeds not ready: {missing} (present={present})", flush=True)

    per_seed = {}
    for c in cols:
        summ, _ = evaluate_tophalf(base, c, label_col="Label", user_col="userID", id_col="ID")
        per_seed[int(c[1:])] = round(float(summ["row_accuracy"]), 5)

    base["ens"] = base[cols].mean(axis=1)
    es, _ = evaluate_tophalf(base, "ens", label_col="Label", user_col="userID", id_col="ID")
    ens_acc = round(float(es["row_accuracy"]), 5)
    vs_ref = round(ens_acc - EMB128_ENS_REF, 5)

    if len(present) < 4:
        tier = "PARTIAL"
        verdict = f"only {present} ready; partial ensemble {ens_acc}. Wait for all 4."
    elif vs_ref > NOISE:
        tier = "UPGRADE_MATERIALIZE"
        verdict = (f"emb192 4-seed ensemble {ens_acc} beats emb128 ensemble {EMB128_ENS_REF} by "
                   f"{vs_ref:+.5f} > noise {NOISE} -> GENUINE backbone upgrade. Materialize the "
                   f"emb192 4-seed full-test candidate for autonomous-runner safety checks.")
    elif vs_ref >= -NOISE:
        tier = "TIED"
        verdict = (f"emb192 4-seed ensemble {ens_acc} vs emb128 ensemble {EMB128_ENS_REF} = "
                   f"{vs_ref:+.5f}, within noise {NOISE}. The single-seed42 edge averaged out; "
                   f"emb192 ensemble is interchangeable with emb128 ensemble, not a clear upgrade.")
    else:
        tier = "WORSE"
        verdict = (f"emb192 4-seed ensemble {ens_acc} LOSES to emb128 ensemble {EMB128_ENS_REF} "
                   f"by {vs_ref:+.5f}. seed42 was a lucky high; keep emb128 4-seed (public 0.77745).")

    summary = {"note": "emb192 L4 reg1e-3 4-seed ensemble uniform gate vs emb128 ensemble. No submission.",
               "split": SPLIT, "seeds_present": present, "seeds_missing": missing,
               "rows": int(len(base)), "per_seed_uniform": per_seed,
               "emb192_ens_uniform": ens_acc, "emb128_ens_ref": EMB128_ENS_REF,
               "emb128_single": EMB128_SINGLE, "vs_emb128_ens": vs_ref, "noise": NOISE,
               "tier": tier, "verdict": verdict}
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md = ["# emb192 L4 reg1e-3 — 4-seed Ensemble UNIFORM gate (public surrogate)\n",
          f"- split `{SPLIT}` (rows={len(base)}) | seeds {present}" + (f" | missing {missing}" if missing else ""),
          f"- emb128 4-seed ensemble ref (public 0.77745): **{EMB128_ENS_REF}** | emb128 single {EMB128_SINGLE} | noise ±{NOISE}",
          f"- **emb192 4-seed ensemble: {ens_acc}**  (vs emb128 ens {vs_ref:+.5f})",
          f"- **tier: {tier}** — {verdict}\n",
          "| seed | emb192 uniform acc |", "|---|---:|"]
    for s in present:
        md.append(f"| {s} | {per_seed[s]} |")
    md.append(f"| **4-seed ens** | **{ens_acc}** |")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\n[EMB192 4-SEED GATE] ens={ens_acc} vs emb128_ens {vs_ref:+.5f} tier={tier}", flush=True)
    print(f"per-seed: {per_seed}", flush=True)
    print(f"saved: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
