"""Aggregate LightGCN seed ensemble: average raw scores, evaluate, materialize test candidate.

Robustness rationale: averaging raw scores of the SAME verified config across seeds is a
variance-reduction step that learns NOTHING from validation labels/negatives — so unlike
the failed logreg stacker (which overfit the validation negative sampler and lost on
public), this cannot exploit split-specific artifacts. Expected effect: small, consistent
gain or flat; never a calibration reversal.

Seeds:
  - seed42: reuse existing assets
      val:  artifacts/lightgcn_20260530/{split}/lightgcn_scores.csv   (col: score_lightgcn)
      test: artifacts/lightgcn_20260530/test_full_train/lightgcn_test_raw_scores_emb64_L3_reg1e-04.csv (col: score_lightgcn)
  - seed123/2024/7: artifacts/lightgcn_seed_ensemble/seed{S}/val_{split}.csv & test.csv (col: score)

Outputs: reports/20260530_seed_ensemble.{json,md} and, if the ensemble holds up,
artifacts/lightgcn_seed_ensemble/test_candidate/candidate_lightgcn_seed_ens.csv (+preflight).
NO Kaggle submission.
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
from recsys_played_utils import evaluate_tophalf, predict_tophalf, ensure_dir

SPLITS = ["val_random_sqrtpop_seed42", "val_recent_sqrtpop_seed42", "val_random_popbin_seed42"]
NEW_SEEDS = [123, 2024, 7]
LGCN_SINGLE = {  # seed42 single-model reference (from prior runs)
    "val_random_sqrtpop_seed42": 0.67483,
    "val_recent_sqrtpop_seed42": 0.63963,
    "val_random_popbin_seed42":  0.60202,
}
OUT_JSON = ROOT / "reports/20260530_seed_ensemble.json"
OUT_MD = ROOT / "reports/20260530_seed_ensemble.md"


def load_seed42_val(split):
    p = ROOT / f"artifacts/lightgcn_20260530/{split}/lightgcn_scores.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)[["ID", "userID", "gameID", "Label", "score_lightgcn"]]
    return df.rename(columns={"score_lightgcn": "score_seed42"})


def load_new_seed_val(split, seed):
    p = ROOT / f"artifacts/lightgcn_seed_ensemble/seed{seed}/val_{split}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)[["ID", "score"]]
    return df.rename(columns={"score": f"score_seed{seed}"})


def main():
    results = {}
    avail_seeds_overall = None
    for split in SPLITS:
        base = load_seed42_val(split)
        if base is None:
            print(f"[skip] {split}: seed42 missing", flush=True)
            continue
        m = base
        seed_cols = ["score_seed42"]
        present = [42]
        for s in NEW_SEEDS:
            d = load_new_seed_val(split, s)
            if d is not None:
                m = m.merge(d, on="ID", how="inner")
                seed_cols.append(f"score_seed{s}")
                present.append(s)
        # ensemble = mean of available seed raw scores
        m["score_ens"] = m[seed_cols].mean(axis=1)

        # per-seed acc + ensemble acc
        seed_accs = {}
        for c in seed_cols:
            summ, _ = evaluate_tophalf(m, c, label_col="Label", user_col="userID", id_col="ID")
            seed_accs[c] = round(float(summ["row_accuracy"]), 5)
        ens_summ, _ = evaluate_tophalf(m, "score_ens", label_col="Label", user_col="userID", id_col="ID")
        ens_acc = round(float(ens_summ["row_accuracy"]), 5)

        results[split] = {
            "seeds_present": present,
            "per_seed_acc": seed_accs,
            "ensemble_acc": ens_acc,
            "single_seed42_ref": LGCN_SINGLE[split],
            "ens_vs_seed42": round(ens_acc - LGCN_SINGLE[split], 5),
        }
        print(f"[{split}] seeds={present} ens={ens_acc} "
              f"(seed42={LGCN_SINGLE[split]}, Δ={ens_acc-LGCN_SINGLE[split]:+.5f})", flush=True)
        avail_seeds_overall = present

    if not results:
        print("No splits ready yet.")
        return

    mean_ens = round(float(np.mean([results[s]["ensemble_acc"] for s in results])), 5)
    mean_single = round(float(np.mean([LGCN_SINGLE[s] for s in results])), 5)
    summary = {
        "seeds_used": avail_seeds_overall,
        "mean_ensemble": mean_ens,
        "mean_single_seed42": mean_single,
        "mean_gain": round(mean_ens - mean_single, 5),
        "splits": results,
    }

    # Materialize test candidate ONLY if all 3 splits ready, all seeds present, and gain >= 0
    all_ready = len(results) == len(SPLITS)
    test_files = {42: ROOT / "artifacts/lightgcn_20260530/test_full_train/lightgcn_test_raw_scores_emb64_L3_reg1e-04.csv"}
    for s in NEW_SEEDS:
        test_files[s] = ROOT / f"artifacts/lightgcn_seed_ensemble/seed{s}/test.csv"
    test_ready = all(p.exists() for p in test_files.values())

    if all_ready and test_ready and summary["mean_gain"] >= 0:
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
        out_dir = ensure_dir(ROOT / "artifacts/lightgcn_seed_ensemble/test_candidate")
        csv_path = out_dir / "candidate_lightgcn_seed_ens.csv"
        sub.to_csv(csv_path, index=False)
        sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        pairs = pd.read_csv(ROOT / "data/raw/public/data/pairs.csv")
        g = sub.merge(pairs, on="ID").groupby("userID").agg(n=("ID", "size"), p=("Label", "sum"))
        bad_users = int((g.p != g.n // 2).sum())
        # diff vs submitted single-seed LightGCN
        lg = pd.read_csv(ROOT / "artifacts/lightgcn_20260530/test_full_train/candidate_lightgcn_full_train.csv").rename(columns={"Label": "L_lg"})
        cmp = sub.rename(columns={"Label": "L_e"}).merge(lg, on="ID")
        row_diff = int((cmp.L_e != cmp.L_lg).sum())
        summary["test_candidate"] = {
            "file": str(csv_path), "sha256": sha, "rows": int(len(sub)),
            "label_1": int(sub.Label.sum()), "label_0": int((1 - sub.Label).sum()),
            "bad_users": bad_users, "rowdiff_vs_single_lgcn": row_diff,
            "rowdiff_frac": round(row_diff / len(cmp), 4),
        }
        print(f"\n[test candidate] {csv_path}\n  sha={sha}\n  rows={len(sub)} "
              f"bad_users={bad_users} rowdiff_vs_single={row_diff} ({100*row_diff/len(cmp):.2f}%)", flush=True)
    else:
        summary["test_candidate"] = {"materialized": False,
                                     "reason": f"all_ready={all_ready} test_ready={test_ready} mean_gain={summary['mean_gain']}"}

    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md = ["# LightGCN Seed Ensemble (raw-score averaging)\n"]
    md.append(f"- seeds: {avail_seeds_overall}")
    md.append(f"- mean single (seed42): {mean_single}")
    md.append(f"- mean ensemble: {mean_ens}")
    md.append(f"- **mean gain: {summary['mean_gain']:+.5f}**\n")
    md.append("| split | seed42 | ensemble | Δ |")
    md.append("|---|---:|---:|---:|")
    for s in results:
        r = results[s]
        md.append(f"| {s.replace('val_','').replace('_seed42','')} | {r['single_seed42_ref']} "
                  f"| {r['ensemble_acc']} | {r['ens_vs_seed42']:+.5f} |")
    md.append("\n## Note\n")
    md.append("Raw-score averaging of the same verified config; no validation-label learning, "
              "so it cannot overfit the negative sampler the way the logreg stacker did "
              "(which failed public 0.76245→0.75355). Expect small/flat gain, low downside.")
    OUT_MD.write_text("\n".join(md))
    print(f"\nsaved: {OUT_JSON}\nMEAN ensemble={mean_ens} single={mean_single} gain={summary['mean_gain']:+.5f}")


if __name__ == "__main__":
    main()
