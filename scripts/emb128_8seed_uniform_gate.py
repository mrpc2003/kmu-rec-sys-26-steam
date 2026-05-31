"""8-SEED uniform gate + candidate materializer for emb128_L4_reg1e-3.

SGL and DirectAU failed the strong+orthogonal bet, so the honest remaining move is pure
variance reduction: expand the validated emb128_L4_reg1e-3 ensemble from 4 to 8 seeds.
Raw-score averaging of the SAME config learns nothing from validation labels, so it cannot
overfit the negative sampler the way the logreg stacker did (public 0.76245->0.75355).

Gate logic
----------
- 4-seed ref (already submitted, public 0.77745): uniform 0.76505
- Compute 8-seed uniform row_acc; only materialize the 8-seed test candidate if it beats
  the 4-seed ensemble by MORE than the single-seed noise band (~0.0007). A within-noise
  result means 8-seed is statistically tied -> keep 4-seed as primary, no new submission.

Seed paths (uniform eval, col=score_lightgcn):
  42       : artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03/{SPLIT}/lightgcn_scores.csv
  others   : artifacts/lightgcn_emb128L4r3_ens/seed{S}/{SPLIT}/lightgcn_scores.csv
Full-test paths (col=score_lightgcn):
  all seeds: artifacts/lightgcn_emb128L4r3_fulltest/seed{S}/test.csv

No Kaggle submission. Report + candidate-file only (submission stays gated on user approval).
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import evaluate_tophalf, predict_tophalf, ensure_dir  # noqa: E402

SPLIT = "val_random_uniform_seed42"
BASE_SEEDS = [42, 123, 2024, 7]
NEW_SEEDS = [11, 99, 2025, 314]
EMB128_4SEED_UNIFORM = 0.76505   # current primary (public 0.77745)
NOISE_BAND = 0.0007
OUT_JSON = ROOT / "reports/20260531_emb128_8seed_uniform_gate.json"
OUT_MD = ROOT / "reports/20260531_emb128_8seed_uniform_gate.md"


def uni_path(seed: int) -> Path:
    if seed == 42:
        return ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / SPLIT / "lightgcn_scores.csv"
    return ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}" / SPLIT / "lightgcn_scores.csv"


def test_path(seed: int) -> Path:
    return ROOT / f"artifacts/lightgcn_emb128L4r3_fulltest/seed{seed}/test.csv"


def main() -> None:
    # ---- uniform gate ----
    base = pd.read_csv(uni_path(42))[["ID", "userID", "gameID", "Label", "score_lightgcn"]]
    base = base.rename(columns={"score_lightgcn": "s42"})
    m = base
    cols, present, missing = ["s42"], [42], []
    for s in BASE_SEEDS[1:] + NEW_SEEDS:
        p = uni_path(s)
        if not p.exists():
            missing.append(s); continue
        d = pd.read_csv(p)[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": f"s{s}"})
        m = m.merge(d, on="ID", how="inner")
        cols.append(f"s{s}"); present.append(s)

    if missing:
        print(f"[wait] uniform seeds not ready: {missing} (present={present})", flush=True)

    per_seed = {}
    for c in cols:
        summ, _ = evaluate_tophalf(m, c, label_col="Label", user_col="userID", id_col="ID")
        per_seed[int(c[1:])] = round(float(summ["row_accuracy"]), 5)

    def ens_acc(seed_cols):
        mm = m.copy(); mm["e"] = mm[seed_cols].mean(axis=1)
        summ, _ = evaluate_tophalf(mm, "e", label_col="Label", user_col="userID", id_col="ID")
        return round(float(summ["row_accuracy"]), 5)

    cols4 = [f"s{s}" for s in BASE_SEEDS if f"s{s}" in cols]
    acc4 = ens_acc(cols4) if len(cols4) == 4 else None
    acc8 = ens_acc(cols) if len(cols) == 8 else None

    vs_ref = round(acc8 - EMB128_4SEED_UNIFORM, 5) if acc8 is not None else None
    if acc8 is None:
        tier, verdict = "INCOMPLETE", f"only {len(present)}/8 seeds ready: {present}"
    elif vs_ref > NOISE_BAND:
        tier = "UPGRADE"
        verdict = (f"8-seed ({acc8}) beats 4-seed ({EMB128_4SEED_UNIFORM}) by {vs_ref:+.5f} "
                   f"> noise {NOISE_BAND} -> materialize candidate, consider as new primary.")
    elif vs_ref < -NOISE_BAND:
        tier = "WORSE"
        verdict = (f"8-seed ({acc8}) LOSES to 4-seed by {vs_ref:+.5f} -> keep 4-seed (public 0.77745).")
    else:
        tier = "TIED"
        verdict = (f"8-seed ({acc8}) vs 4-seed ({EMB128_4SEED_UNIFORM}) = {vs_ref:+.5f}, within "
                   f"noise {NOISE_BAND} -> statistically tied; 4-seed stays primary. 8-seed only "
                   f"as a marginally lower-variance alternate, not a clear upgrade.")

    summary = {
        "note": "emb128_L4_reg1e-3 4->8 seed expansion uniform gate. No submission.",
        "split": SPLIT, "seeds_present": present, "seeds_missing": missing,
        "rows": int(len(m)), "per_seed_uniform": per_seed,
        "ens4_uniform": acc4, "ens4_ref": EMB128_4SEED_UNIFORM, "ens8_uniform": acc8,
        "ens8_vs_4seed_ref": vs_ref, "noise_band": NOISE_BAND, "tier": tier, "verdict": verdict,
    }

    # ---- materialize 8-seed candidate if UPGRADE and all test.csv ready ----
    test_ready = all(test_path(s).exists() for s in present)
    if tier == "UPGRADE" and len(present) == 8 and test_ready:
        t = pd.read_csv(test_path(42))[["ID", "userID", "gameID", "score_lightgcn"]].rename(
            columns={"score_lightgcn": "s42"})
        tcols = ["s42"]
        for s in BASE_SEEDS[1:] + NEW_SEEDS:
            d = pd.read_csv(test_path(s))[["ID", "score_lightgcn"]].rename(
                columns={"score_lightgcn": f"s{s}"})
            t = t.merge(d, on="ID", how="inner"); tcols.append(f"s{s}")
        t["e"] = t[tcols].mean(axis=1)
        pred = predict_tophalf(t, "e", label_col=None, user_col="userID", id_col="ID")
        sub = pred[["ID", "Pred"]].rename(columns={"Pred": "Label"}).sort_values("ID")
        out_dir = ensure_dir(ROOT / "artifacts/lightgcn_emb128L4r3_fulltest/test_candidate_8seed")
        csv_path = out_dir / "candidate_lightgcn_emb128L4r3_8seed_ens.csv"
        sub.to_csv(csv_path, index=False)
        sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        pairs = pd.read_csv(ROOT / "data/raw/public/data/pairs.csv")
        g = sub.merge(pairs, on="ID").groupby("userID").agg(n=("ID", "size"), p=("Label", "sum"))
        bad_users = int((g.p != g.n // 2).sum())
        ref4 = pd.read_csv(ROOT / "artifacts/lightgcn_emb128L4r3_fulltest/test_candidate/candidate_lightgcn_emb128L4r3_seed_ens.csv").rename(columns={"Label": "L4"})
        cmp = sub.rename(columns={"Label": "L8"}).merge(ref4, on="ID")
        rowdiff = int((cmp.L8 != cmp.L4).sum())
        summary["test_candidate"] = {
            "file": str(csv_path), "sha256": sha, "rows": int(len(sub)),
            "label_1": int(sub.Label.sum()), "label_0": int((1 - sub.Label).sum()),
            "bad_users": bad_users, "rowdiff_vs_4seed": rowdiff,
            "rowdiff_frac": round(rowdiff / len(cmp), 4),
        }
        print(f"\n[8-seed candidate] {csv_path}\n  sha={sha}\n  rows={len(sub)} "
              f"bad_users={bad_users} rowdiff_vs_4seed={rowdiff} ({100*rowdiff/len(cmp):.2f}%)", flush=True)
    else:
        summary["test_candidate"] = {"materialized": False,
                                     "reason": f"tier={tier} present={len(present)} test_ready={test_ready}"}

    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md = ["# emb128_L4_reg1e-3 — 4->8 Seed Expansion (UNIFORM gate)\n",
          f"- split: `{SPLIT}` (rows={len(m)})",
          f"- seeds present: {present}" + (f" | missing: {missing}" if missing else ""),
          f"- 4-seed ref (public 0.77745): **{EMB128_4SEED_UNIFORM}**",
          f"- 4-seed (recomputed here): {acc4}",
          f"- **8-seed uniform: {acc8}**",
          (f"- **Δ vs 4-seed ref: {vs_ref:+.5f}**" if isinstance(vs_ref, float)
           else "- Δ vs 4-seed ref: n/a"),
          f"- **tier: {tier}** — {verdict}\n",
          "| seed | uniform acc |", "|---|---:|"]
    for s in present:
        md.append(f"| {s} | {per_seed[s]} |")
    if acc4 is not None:
        md.append(f"| **4-seed ens** | **{acc4}** |")
    if acc8 is not None:
        md.append(f"| **8-seed ens** | **{acc8}** |")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\n[8-SEED GATE] 4seed={acc4} 8seed={acc8} ref={EMB128_4SEED_UNIFORM} "
          f"Δ={vs_ref} tier={tier}", flush=True)
    print(f"per-seed uniform: {per_seed}", flush=True)
    print(f"saved: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
