#!/usr/bin/env python3
"""Log the 2026-SOTA-exploration gate verdicts to W&B (no-submit experiment record).

This round tested structurally-distinct families against best (emb128 4-seed, public 0.77745):
  - XSimGCL   (TKDE'23 noise-CL)        -> TIED_OR_WEAK   (contrastive-CF family closed)
  - Turbo-CF  (SIGIR'24 graph filtering) -> REDUNDANT      (item-item linear, like EASE)
  - AlphaRec  (ICLR'25 LM-rep CF)        -> (logged if ready)
  - DNS       (hard negative sampling)   -> (logged if ready)
Logged as no-submit runs so the experiment record is complete. Tolerant of missing files.

Usage:
  env -u VIRTUAL_ENV UV_NO_ACTIVE_VENV=1 HOME=/opt/data/home \
    uv run --with wandb --with pandas python scripts/log_wandb_sota_exploration_gates.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from wandb_recsys_utils import init_run, flatten_numeric  # noqa: E402

GATES = [
    ("gate-xsimgcl-noise-cl", "reports/20260531_xsimgcl_uniform_gate.json", "contrastive-cf-gate"),
    ("gate-turbocf-graph-filter", "reports/20260531_turbocf_uniform_gate.json", "graph-filter-cf-gate"),
    ("gate-alpharec-lm-rep", "reports/20260531_alpharec_core_uniform_gate.json", "lm-rep-cf-gate"),
    ("gate-dns-hard-negative", "reports/20260531_dns_uniform_gate.json", "hard-negative-gate"),
]


def main() -> None:
    records = []
    for name, rel, job_type in GATES:
        p = ROOT / rel
        if not p.exists():
            print(f"[skip] not ready: {rel}", flush=True)
            continue
        data = json.loads(p.read_text())
        metrics = flatten_numeric(data)
        tier = str(data.get("tier", ""))
        run = init_run(
            name=name, job_type=job_type,
            tags=["kaggle", "recsys", "steam", "no-submit", "sota-exploration-2026",
                  "negative-result", tier],
            notes=str(data.get("verdict", ""))[:500],
            config={"report_json": rel, "split": data.get("split"),
                    "verdict": data.get("verdict")},
        )
        if metrics:
            run.log(metrics)
        run.summary["tier"] = tier
        run.summary["verdict"] = data.get("verdict")
        records.append({"name": run.name, "id": run.id, "url": run.url, "tier": tier})
        print(f"[logged] {name} tier={tier} -> {run.url}", flush=True)
        run.finish()

    out = ROOT / "reports/20260531_wandb_sota_exploration_runs.json"
    out.write_text(json.dumps({"runs": records}, indent=2, ensure_ascii=False))
    print(f"\nsaved run index: {out}")
    for r in records:
        print(f"  {r['tier']:20s} {r['name']}  {r['url']}")


if __name__ == "__main__":
    main()
