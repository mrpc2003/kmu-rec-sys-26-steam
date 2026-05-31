#!/usr/bin/env python3
"""Aggregate the 3-split emb128 panel and validate the gate-floor BLUNT verdict empirically.

For each independent uniform split (seeds 42/7/123, differing ONLY by the random seed that
draws heldout positives + uniform negatives), build the canonical emb128 4-seed raw-mean
ensemble and measure uniform row accuracy. The BETWEEN-split spread is an INDEPENDENT estimate
of the same sampling variance the within-split bootstrap approximated.

Decision:
  between-split std ≈ bootstrap absolute-SE (0.00367)  -> BLUNT verdict CONFIRMED (bootstrap valid)
  between-split std << 0.00367                          -> bootstrap OVERESTIMATED; gate sharper than feared

seed42 reuses existing per-model-seed scores; seed7/seed123 read the freshly trained panel cells.
Validation-only. No Kaggle submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import evaluate_tophalf, ensure_dir, write_json  # noqa: E402

MODEL_SEEDS = [42, 123, 2024, 7]
BOOTSTRAP_ABS_SE = 0.00367   # from gate_floor PART A (single-split user bootstrap)
PANEL_DIR = ROOT / "artifacts/split_panel_emb128"

# seed42 split: canonical existing per-model-seed score files
SEED42_PATHS = {
    42:   ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03/val_random_uniform_seed42/lightgcn_scores.csv",
    123:  ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed123/val_random_uniform_seed42/lightgcn_scores.csv",
    2024: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed2024/val_random_uniform_seed42/lightgcn_scores.csv",
    7:    ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed7/val_random_uniform_seed42/lightgcn_scores.csv",
}


def panel_path(split, ms):
    return PANEL_DIR / split / f"seed{ms}" / "lightgcn_scores.csv"


def ensemble_acc(paths: dict[int, Path]) -> tuple[float, list[float]]:
    base = None
    cols = []
    per_seed = []
    for ms, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(p)
        d = pd.read_csv(p)[["ID", "userID", "gameID", "Label", "score_lightgcn"]]
        c = f"s{ms}"
        d = d.rename(columns={"score_lightgcn": c})
        base = d if base is None else base.merge(d[["ID", c]], on="ID")
        cols.append(c)
    for c in cols:
        a = float(evaluate_tophalf(base, c, label_col="Label", user_col="userID", id_col="ID")[0]["row_accuracy"])
        per_seed.append(round(a, 5))
    base["s"] = base[cols].mean(axis=1)
    ens = float(evaluate_tophalf(base, "s", label_col="Label", user_col="userID", id_col="ID")[0]["row_accuracy"])
    return round(ens, 5), per_seed


def main():
    splits = {
        "val_random_uniform_seed42": SEED42_PATHS,
        "val_random_uniform_seed7": {ms: panel_path("val_random_uniform_seed7", ms) for ms in MODEL_SEEDS},
        "val_random_uniform_seed123": {ms: panel_path("val_random_uniform_seed123", ms) for ms in MODEL_SEEDS},
    }
    results = {}
    ens_accs = []
    for split, paths in splits.items():
        ens, per_seed = ensemble_acc(paths)
        results[split] = {"ensemble_acc": ens, "per_model_seed_acc": per_seed,
                          "per_seed_spread": round(max(per_seed) - min(per_seed), 5)}
        ens_accs.append(ens)
        print(f"[panel] {split}: ensemble={ens} per-seed={per_seed}", flush=True)

    ens_arr = np.array(ens_accs)
    between_mean = float(ens_arr.mean())
    between_std = float(ens_arr.std(ddof=1))
    between_range = float(ens_arr.max() - ens_arr.min())

    # The gate threshold is 0.003. Compare the REAL between-split spread to it and to the bootstrap SE.
    if between_std >= 0.8 * BOOTSTRAP_ABS_SE:
        verdict = (f"BLUNT_CONFIRMED: between-split ensemble std={between_std:.5f} ≈ bootstrap abs-SE "
                   f"{BOOTSTRAP_ABS_SE} -> single-split gating IS blunt; a 3-split panel is justified.")
    elif between_std <= 0.4 * BOOTSTRAP_ABS_SE:
        verdict = (f"BOOTSTRAP_OVERESTIMATED: between-split ensemble std={between_std:.5f} << bootstrap abs-SE "
                   f"{BOOTSTRAP_ABS_SE} -> ensemble averaging tightens the real gate; single-split sharper than feared.")
    else:
        verdict = (f"INTERMEDIATE: between-split ensemble std={between_std:.5f} vs bootstrap abs-SE "
                   f"{BOOTSTRAP_ABS_SE} -> partial variance reduction from ensembling.")

    payload = {
        "model_seeds": MODEL_SEEDS, "bootstrap_abs_SE_ref": BOOTSTRAP_ABS_SE,
        "per_split": results,
        "between_split_ensemble": {"accs": [round(a, 5) for a in ens_accs],
                                   "mean": round(between_mean, 5), "std": round(between_std, 5),
                                   "range": round(between_range, 5)},
        "verdict": verdict,
        "note": "Independent 3-split emb128 4-seed ensemble panel. Validation-only. No Kaggle submission.",
    }
    out = ensure_dir(PANEL_DIR)
    write_json(out / "panel_summary.json", payload)
    print("\n" + "=" * 80, flush=True)
    print(f"[panel] between-split ensemble accs={[round(a,5) for a in ens_accs]} "
          f"mean={between_mean:.5f} std={between_std:.5f} range={between_range:.5f}", flush=True)
    print(f"[panel] {verdict}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
