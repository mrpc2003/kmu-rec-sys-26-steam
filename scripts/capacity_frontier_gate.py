#!/usr/bin/env python3
"""Collect LightGCN capacity-frontier uniform results -> single verdict + escalation decision.

Reads each capacity config's summary.json (uniform row_accuracy) and compares against the
emb128 single-seed (0.76205) and 4-seed ensemble (0.76505) references. Decides whether any
capacity point warrants escalation to a full seed ensemble + full-test candidate (which would
still be gated on 우현's explicit one-file approval before any Kaggle submission).

No Kaggle submission. Report-only.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
EMB128_SINGLE = 0.76205
EMB128_ENS = 0.76505
NOISE = 0.0007
SPLIT = "val_random_uniform_seed42"
CONFIGS = [
    ("emb192_L4_r3", "artifacts/capacity_uniform/emb192_L4_r3"),
    ("emb256_L4_r3", "artifacts/capacity_uniform/emb256_L4_r3"),
    ("emb256_L4_r2", "artifacts/capacity_uniform/emb256_L4_r2"),
    ("emb320_L4_r3", "artifacts/capacity_uniform/emb320_L4_r3"),
]
OUT_JSON = ROOT / "reports/20260531_capacity_frontier_uniform_gate.json"
OUT_MD = ROOT / "reports/20260531_capacity_frontier_uniform_gate.md"


def main() -> None:
    rows, missing = [], []
    for label, d in CONFIGS:
        p = ROOT / d / SPLIT / "summary.json"
        if not p.exists():
            missing.append(label); continue
        s = json.loads(p.read_text())
        rows.append((label, round(float(s["row_accuracy"]), 5),
                     s.get("meta", {}).get("train_seconds"), s.get("tier")))
    if missing:
        print(f"[wait] not ready: {missing}", flush=True)
    if not rows:
        print("no results yet"); return

    rows.sort(key=lambda r: r[1], reverse=True)
    best_label, best_acc, _, _ = rows[0]
    vs_single = round(best_acc - EMB128_SINGLE, 5)
    vs_ens = round(best_acc - EMB128_ENS, 5)

    # Escalation logic: a higher-capacity SINGLE seed must clearly beat the emb128 single seed
    # to justify spending GPU on a 4-seed ensemble of that config. Matching emb128 single is
    # NOT enough (we already have the emb128 ensemble at 0.76505).
    if vs_single > NOISE:
        tier = "ESCALATE_SEED_ENSEMBLE"
        verdict = (f"best {best_label} single {best_acc} beats emb128 single-seed {EMB128_SINGLE} "
                   f"by {vs_single:+.5f} > noise {NOISE} -> capacity frontier still rising. "
                   f"Worth a 4-seed ensemble of {best_label}; its ensemble would likely exceed "
                   f"the emb128 ensemble {EMB128_ENS}. Then gate the candidate to 우현.")
    elif vs_single >= -NOISE:
        tier = "PLATEAU_CLOSE"
        verdict = (f"best {best_label} single {best_acc} ~ emb128 single-seed {EMB128_SINGLE} "
                   f"({vs_single:+.5f}, within noise). Capacity has PLATEAUED at emb128; more "
                   f"dims give no uniform gain. Backbone-capacity axis closed; keep emb128 4-seed.")
    else:
        tier = "OVERFIT_CLOSE"
        verdict = (f"best {best_label} single {best_acc} < emb128 single-seed {EMB128_SINGLE} "
                   f"by {vs_single:+.5f}. Higher capacity OVERFITS this small dense graph "
                   f"(165k interactions). Frontier closed below emb128; keep emb128 4-seed.")

    summary = {"note": "LightGCN capacity-frontier uniform gate (vs emb128). No submission.",
               "split": SPLIT, "emb128_single_seed": EMB128_SINGLE, "emb128_ens": EMB128_ENS,
               "noise": NOISE, "missing": missing,
               "results": [{"config": l, "uniform_acc": a, "train_s": t, "self_tier": st}
                           for l, a, t, st in rows],
               "best": {"config": best_label, "uniform_acc": best_acc,
                        "vs_single_seed": vs_single, "vs_ens": vs_ens},
               "tier": tier, "verdict": verdict}
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md = ["# LightGCN Capacity Frontier — UNIFORM gate (public surrogate)\n",
          f"- split `{SPLIT}` | emb128 single-seed **{EMB128_SINGLE}** | emb128 4-seed ens "
          f"**{EMB128_ENS}** (public 0.77745) | noise ±{NOISE}",
          f"- **best: {best_label} = {best_acc}  (vs emb128 single {vs_single:+.5f}, vs ens {vs_ens:+.5f})**",
          f"- **tier: {tier}** — {verdict}\n",
          "| config | uniform acc | vs emb128 single | train s |", "|---|---:|---:|---:|"]
    for l, a, t, st in rows:
        md.append(f"| {l} | {a} | {round(a-EMB128_SINGLE,5):+.5f} | {t} |")
    md.append("\n## Why this gate was needed\n")
    md.append("The original hparam sweep DID try emb256 but evaluated it ONLY on the hard "
              "samplers (sqrtpop/recent/popbin). We later proved the public LB tracks the "
              "UNIFORM split, so those emb256 numbers were measured on the wrong distribution. "
              "This probe gates capacity on the actual public surrogate for the first time.")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\n[CAPACITY GATE] best={best_label}({best_acc}) vs emb128_single {vs_single:+.5f} "
          f"vs ens {vs_ens:+.5f} tier={tier}", flush=True)
    for l, a, t, st in rows:
        print(f"  {l}: {a} ({t}s)", flush=True)
    print(f"saved: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
