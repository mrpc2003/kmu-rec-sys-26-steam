#!/usr/bin/env python3
"""Collect DNS (Dynamic Negative Sampling) pool-sweep uniform results -> verdict.

DNS strengthens the backbone (hard negatives) rather than adding a weak orthogonal axis.
The control (pool=1) is plain random-negative BPR and should land near the single-seed
backbone (~0.762); the question is whether any pool size pushes the SINGLE model above the
emb128 4-seed ENSEMBLE 0.76505 on the uniform public-surrogate split. A single model beating
the 4-seed ensemble on uniform would be a genuine backbone upgrade (and could then be
ensembled itself). Bar: > noise 0.0007 over 0.76505. No validation-label learning -> not a
stacker-trap, but watch the popularity-skew risk (hard negatives can favor popular-item
discrimination, which may not transfer to the uniform test).

No Kaggle submission. Report-only.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
EMB128_ENS_REF = 0.76505
SINGLE_SEED_REF = 0.762
NOISE = 0.0007
POOLS = [1, 8, 16, 32]
SPLIT = "val_random_uniform_seed42"
OUT_JSON = ROOT / "reports/20260531_dns_uniform_gate.json"
OUT_MD = ROOT / "reports/20260531_dns_uniform_gate.md"


def main() -> None:
    rows, missing = [], []
    for m in POOLS:
        p = ROOT / f"artifacts/dns_uniform/pool{m}" / SPLIT / "summary.json"
        if not p.exists():
            missing.append(m); continue
        s = json.loads(p.read_text())
        rows.append((m, round(float(s["row_accuracy"]), 5),
                     s.get("meta", {}).get("train_seconds")))
    if missing:
        print(f"[wait] not ready: pools {missing}", flush=True)
    if not rows:
        print("no results yet"); return

    rows.sort(key=lambda r: r[1], reverse=True)
    best_pool, best_acc, _ = rows[0]
    ctrl = next((a for m, a, _ in rows if m == 1), None)
    vs_ens = round(best_acc - EMB128_ENS_REF, 5)

    if vs_ens > NOISE:
        tier = "UPGRADE_CANDIDATE"
        verdict = (f"DNS pool={best_pool} single model {best_acc} beats emb128 4-seed ENSEMBLE "
                   f"{EMB128_ENS_REF} by {vs_ens:+.5f} > noise {NOISE} on uniform -> genuine "
                   f"backbone upgrade; build a seed ensemble of this config and gate to user.")
    elif best_acc >= EMB128_ENS_REF - NOISE:
        tier = "TIED_ENSEMBLE_LEVEL"
        verdict = (f"DNS pool={best_pool} single model {best_acc} ~ emb128 4-seed ensemble "
                   f"{EMB128_ENS_REF} (Δ{vs_ens:+.5f}); a SINGLE DNS model matching a 4-seed "
                   f"ensemble is promising -> a DNS seed ensemble may exceed it; consider escalation.")
    elif ctrl is not None and best_acc > ctrl + NOISE:
        tier = "BACKBONE_GAIN_SUBENSEMBLE"
        verdict = (f"DNS best {best_acc} (pool {best_pool}) beats plain-BPR control {ctrl} by "
                   f"{best_acc-ctrl:+.5f} -> hard negatives help the single model, but still "
                   f"below the 4-seed ensemble {EMB128_ENS_REF}. Worth a DNS seed ensemble check.")
    else:
        tier = "NO_GAIN"
        verdict = (f"DNS best {best_acc} (pool {best_pool}) vs control {ctrl} and ensemble "
                   f"{EMB128_ENS_REF}: no uniform gain. Hard-negative training does not help "
                   f"on the uniform public surrogate (likely popularity-skew mismatch).")

    summary = {"note": "DNS pool-sweep uniform gate (emb128 L4 reg1e-3). No submission.",
               "split": SPLIT, "emb128_ens_ref": EMB128_ENS_REF, "single_seed_ref": SINGLE_SEED_REF,
               "noise": NOISE, "control_pool1": ctrl,
               "results": [{"pool": m, "uniform_acc": a, "train_s": t} for m, a, t in rows],
               "best": {"pool": best_pool, "uniform_acc": best_acc, "vs_ens_ref": vs_ens},
               "missing": missing, "tier": tier, "verdict": verdict}
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md = ["# DNS (Dynamic Negative Sampling) pool-sweep — UNIFORM gate\n",
          f"- backbone emb128 L4 reg1e-3 | split `{SPLIT}`",
          f"- emb128 4-seed ensemble ref (public 0.77745): **{EMB128_ENS_REF}** | "
          f"single-seed plain-BPR ~{SINGLE_SEED_REF} | noise ±{NOISE}",
          f"- control (pool=1, plain random neg): **{ctrl}**",
          f"- **best: pool={best_pool} = {best_acc}  (vs ensemble {vs_ens:+.5f})**",
          f"- **tier: {tier}** — {verdict}\n",
          "| DNS pool M | uniform acc | train s |", "|---|---:|---:|"]
    for m, a, t in rows:
        tagc = " (control)" if m == 1 else ""
        md.append(f"| {m}{tagc} | {a} | {t} |")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\n[DNS GATE] control={ctrl} best=pool{best_pool}({best_acc}) ens_ref={EMB128_ENS_REF} "
          f"vs_ens={vs_ens:+.5f} tier={tier}", flush=True)
    for m, a, t in rows:
        print(f"  pool{m}: {a} ({t}s)", flush=True)
    print(f"saved: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
