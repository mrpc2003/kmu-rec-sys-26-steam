#!/usr/bin/env python3
"""Log the GPT-5.5 Pro structural-lever round to W&B (no-submit negative-result record).

Four orthogonal levers gated on the uniform surrogate (public LB proxy). All CLOSED:
  1. exact-K subset loss        -> NO_GAIN  (loss-geometry Δ=+0.00000, McNemar p=0.934)
  2. temporal compatibility     -> REGRESS  (corr -0.04 orthogonal yet non-predictive)
  3. candidate-marginal residual-> REGRESS  (estimator works r=0.96 but item-prior trap)
  4. hours confidence-weighting -> (logged from artifacts if present)

Tolerant of missing files. Usage:
  env -u VIRTUAL_ENV UV_NO_ACTIVE_VENV=1 HOME=/opt/data/home \
    uv run --with wandb --with pandas python scripts/log_wandb_structural_levers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
sys.path.insert(0, str(ROOT / "scripts"))
from wandb_recsys_utils import init_run, flatten_numeric  # noqa: E402

SPLIT = "val_random_uniform_seed42"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def log_run(name, job_type, tier, notes, config, metrics):
    run = init_run(
        name=name, job_type=job_type,
        tags=["kaggle", "recsys", "steam", "no-submit", "gpt55pro-structural-levers",
              "negative-result", tier],
        notes=str(notes)[:500], config=config,
    )
    if metrics:
        run.log(metrics)
    run.summary["tier"] = tier
    run.summary["verdict"] = notes
    url = run.url
    run.finish()
    print(f"[logged] {name:34s} tier={tier:24s} {url}", flush=True)
    return {"name": name, "tier": tier, "url": url}


def main():
    records = []

    # 1. exact-K subset loss (flat schema)
    d = load(ROOT / "artifacts/exactk_subset" / SPLIT / "summary.json")
    if d:
        records.append(log_run(
            "lever-exactk-subset-loss", "exactk-subset-gate", str(d.get("tier")),
            d.get("verdict"),
            {"split": SPLIT, "acc_pretrained": d.get("acc_pretrained"),
             "acc_bpr_ft": d.get("acc_bpr_ft"), "acc_subset_ft": d.get("acc_subset_ft"),
             "delta_isolated": d.get("delta_subset_vs_bprft_ISOLATED")},
            flatten_numeric(d)))
    else:
        print("[skip] exactk subset not ready", flush=True)

    # 2. temporal compatibility (nested results; report best combiner)
    d = load(ROOT / "artifacts/temporal_compat" / SPLIT / "summary.json")
    if d:
        res = d.get("results", {})
        best = max(res.items(), key=lambda kv: kv[1].get("delta_vs_base", -9)) if res else (None, {})
        metrics = {"base_acc": d.get("base_acc"), "corr_T_base": d.get("corr_T_base"),
                   "corr_T_logpop": d.get("corr_T_logpop")}
        for cname, cv in res.items():
            metrics[f"{cname}_acc"] = cv.get("acc")
            metrics[f"{cname}_delta"] = cv.get("delta_vs_base")
        records.append(log_run(
            "lever-temporal-compat", "temporal-compat-gate", str(best[1].get("tier", "REGRESS")),
            f"corr(T,base)={d.get('corr_T_base')} orthogonal but best combiner {best[0]} "
            f"Δ={best[1].get('delta_vs_base')} -> all REGRESS",
            {"split": SPLIT, "beta": d.get("beta"), "best_combiner": best[0]},
            metrics))
    else:
        print("[skip] temporal not ready", flush=True)

    # 3. candidate-marginal residual (nested lambda sweep; primary = lambda 1.0)
    d = load(ROOT / "artifacts/candidate_marginal" / SPLIT / "summary.json")
    if d:
        pr = d.get("primary_result", {})
        metrics = {"base_acc": d.get("base_acc"),
                   "sanity_corr_residual_truepos": d.get("sanity_corr_residual_truepos"),
                   "sanity_corr_ncand_truepos": d.get("sanity_corr_ncand_truepos"),
                   "primary_acc": pr.get("acc"), "primary_delta": pr.get("delta_vs_base")}
        records.append(log_run(
            "lever-candidate-marginal", "candidate-marginal-gate", str(pr.get("tier", "REGRESS")),
            f"residual estimator works r={d.get('sanity_corr_residual_truepos')} but λ=1.0 "
            f"Δ={pr.get('delta_vs_base')} (item-prior popularity trap); validation-first moots rule risk",
            {"split": SPLIT, "primary_gate_lambda": 1.0},
            metrics))
    else:
        print("[skip] candidate-marginal not ready", flush=True)

    # 4. hours confidence-weighted (4 modes)
    for mode in ["user_quantile", "item_quantile", "balanced", "binary_control"]:
        d = load(ROOT / "artifacts/hours_confidence" / mode / SPLIT / "summary.json")
        if d:
            records.append(log_run(
                f"lever-hours-{mode}", "hours-confidence-gate", str(d.get("tier")),
                f"hours-as-edge-confidence {mode}: uniform {d.get('acc')} vs binary "
                f"{d.get('binary_single_seed_ref')} Δ={d.get('delta_vs_binary')}",
                {"split": SPLIT, "confidence_mode": mode, "acc": d.get("acc"),
                 "delta_vs_binary": d.get("delta_vs_binary")},
                flatten_numeric(d)))
        else:
            print(f"[skip] hours {mode} not ready", flush=True)

    out = ROOT / "reports/20260601_wandb_structural_levers_runs.json"
    out.write_text(json.dumps({"runs": records}, indent=2, ensure_ascii=False))
    print(f"\nsaved run index: {out} ({len(records)} runs)", flush=True)


if __name__ == "__main__":
    main()
