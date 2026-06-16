#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Follow-up grid over validation-only OTTO source co-visitation artifacts.

This script reads only validation feature-score CSVs produced by
`scripts/otto_source_covisit_smoke.py`. It does not read full-test pairs, does
not write `ID,Label` candidate/submission files, and does not call Kaggle.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
SPLITS = ["val_random_uniform_seed42", "val_random_uniform_seed7", "val_random_uniform_seed123"]
FEATURES = [
    "score_coplay_sum",
    "score_coplay_max",
    "score_coplay_top5_mean",
    "score_hours_sum",
    "score_hours_max",
    "score_forward_recent",
    "score_reverse_recent",
    "score_last5_coplay",
    "score_last5_forward",
    "score_source_mean_z",
]


def exact_binom_two_sided(fixes: int, breaks: int) -> float | None:
    n = fixes + breaks
    if n <= 0:
        return None
    k = min(fixes, breaks)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * cdf))


def z_within_user(df: pd.DataFrame, col: str) -> pd.Series:
    grp = df.groupby("userID")[col]
    mean = grp.transform("mean")
    std = grp.transform("std").fillna(0.0)
    return pd.Series(
        np.where(std.to_numpy() > 1e-12, (df[col] - mean) / std.replace(0, np.nan), 0.0),
        index=df.index,
    ).fillna(0.0)


class SplitCache:
    def __init__(self, split: str, path: Path) -> None:
        self.split = split
        self.df = pd.read_csv(path)
        if "z_score_base" not in self.df.columns:
            self.df["z_score_base"] = z_within_user(self.df, "score_base")
        self.y = self.df["Label"].to_numpy(np.int8)
        self.group_indices: list[np.ndarray] = []
        self.group_k: list[int] = []
        for _, idx in self.df.groupby("userID", sort=False).indices.items():
            arr = np.asarray(idx, dtype=np.int64)
            self.group_indices.append(arr)
            self.group_k.append(int(self.y[arr].sum()))
        self.base_pred = self.predict(self.df["score_base"].to_numpy(np.float64))
        self.base_acc = float((self.base_pred == self.y).mean())
        self.z_base = self.df["z_score_base"].to_numpy(np.float64)
        self.features: dict[str, np.ndarray] = {}
        for feat in FEATURES:
            col = feat if feat == "score_source_mean_z" else f"z_{feat}"
            if col not in self.df.columns:
                raise ValueError(f"missing feature column {col} in {path}")
            self.features[feat] = self.df[col].to_numpy(np.float64)

    def predict(self, score: np.ndarray) -> np.ndarray:
        pred = np.zeros(len(score), dtype=np.int8)
        for idx, k in zip(self.group_indices, self.group_k, strict=True):
            if k <= 0:
                continue
            # stable deterministic tie-break by original row order after score descending.
            order = np.lexsort((idx, -score[idx]))
            pred[idx[order[:k]]] = 1
        return pred


def build_variants() -> list[tuple[str, list[tuple[str, float]]]]:
    variants: list[tuple[str, list[tuple[str, float]]]] = []
    # Fine grid around the initial winning singleton and a modest singleton scan.
    singleton_weights = [round(float(x), 3) for x in np.arange(0.02, 0.401, 0.01)]
    for feat in FEATURES:
        for w in singleton_weights:
            variants.append((f"base_plus_{feat}_w{w:g}", [(feat, w)]))

    # Pair scan: keep primary small and add one auxiliary source with conservative weights.
    primary = "score_coplay_top5_mean"
    primary_weights = [round(float(x), 3) for x in np.arange(0.12, 0.301, 0.02)]
    aux_weights = [0.03, 0.06, 0.09, 0.12]
    for feat in FEATURES:
        if feat == primary:
            continue
        for w1 in primary_weights:
            for w2 in aux_weights:
                variants.append((f"base_plus_{primary}_w{w1:g}_plus_{feat}_w{w2:g}", [(primary, w1), (feat, w2)]))
    return variants


def score_variant(caches: list[SplitCache], name: str, terms: list[tuple[str, float]]) -> dict[str, Any]:
    deltas: dict[str, float] = {}
    accs: dict[str, float] = {}
    fixes = 0
    breaks = 0
    for cache in caches:
        score = cache.z_base.copy()
        for feat, weight in terms:
            score += weight * cache.features[feat]
        pred = cache.predict(score)
        acc = float((pred == cache.y).mean())
        accs[cache.split] = acc
        deltas[cache.split] = acc - cache.base_acc
        fixes += int(((pred == cache.y) & (cache.base_pred != cache.y)).sum())
        breaks += int(((pred != cache.y) & (cache.base_pred == cache.y)).sum())
    vals = list(deltas.values())
    return {
        "variant": name,
        "terms": terms,
        "mean_delta_vs_base": float(np.mean(vals)),
        "min_delta_vs_base": float(np.min(vals)),
        "max_delta_vs_base": float(np.max(vals)),
        "positive_splits": int(sum(v > 0 for v in vals)),
        "fixes": fixes,
        "breaks": breaks,
        "pooled_p_exact": exact_binom_two_sided(fixes, breaks),
        "split_deltas": deltas,
        "split_accs": accs,
    }


def write_markdown(payload: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# OTTO source co-visitation follow-up grid",
        "",
        f"- Timestamp: {payload['timestamp_kst']}",
        "- Safety: validation-only; reused existing validation feature artifacts; no candidate/submission CSV; no Kaggle submit.",
        f"- Variants scanned: {payload['num_variants_scanned']}",
        f"- Verdict: `{payload['verdict']}`",
        f"- Strict pass count: `{payload['strict_pass_count']}`",
        "",
        "## Top 10 by mean delta",
        "",
    ]
    for i, row in enumerate(payload["top10"], 1):
        lines.append(
            f"{i}. `{row['variant']}` "
            f"meanΔ={row['mean_delta_vs_base']:+.10f}, "
            f"minΔ={row['min_delta_vs_base']:+.10f}, "
            f"pos={row['positive_splits']}/3, "
            f"fixes/breaks={row['fixes']}/{row['breaks']}, "
            f"p={row['pooled_p_exact']}, "
            f"deltas={row['split_deltas']}"
        )
    if payload["strict_top"]:
        lines += ["", "## Strict diagnostic rows", ""]
        for i, row in enumerate(payload["strict_top"], 1):
            lines.append(
                f"{i}. `{row['variant']}` "
                f"meanΔ={row['mean_delta_vs_base']:+.10f}, "
                f"minΔ={row['min_delta_vs_base']:+.10f}, "
                f"fixes/breaks={row['fixes']}/{row['breaks']}, "
                f"p={row['pooled_p_exact']}, "
                f"deltas={row['split_deltas']}"
            )
    else:
        lines += ["", "No row passed the strict gate; strongest row remains below mean Δ +0.0015 or another strict condition."]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ts", required=True)
    ap.add_argument("--source-artifact-dir", default="artifacts/opencode_axis_loop_20260607T090941KST/otto_source_covisit")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    source_dir = ROOT / args.source_artifact_dir
    caches = [SplitCache(split, source_dir / split / "validation_otto_source_scores.csv") for split in SPLITS]
    rows = [score_variant(caches, name, terms) for name, terms in build_variants()]
    rows.sort(key=lambda r: (r["mean_delta_vs_base"], r["min_delta_vs_base"], r["fixes"] - r["breaks"]), reverse=True)
    strict = [
        r for r in rows
        if r["mean_delta_vs_base"] >= 0.0015
        and r["min_delta_vs_base"] >= 0
        and r["positive_splits"] == 3
        and r["fixes"] > r["breaks"]
        and r["pooled_p_exact"] is not None
        and r["pooled_p_exact"] < 0.05
    ]
    payload: dict[str, Any] = {
        "timestamp_kst": args.run_ts,
        "source_artifact_dir": args.source_artifact_dir,
        "safety_flags": {
            "validation_only": True,
            "candidate_csv_written": False,
            "full_test_candidate_or_submission_csv_created": False,
            "kaggle_submit_executed": False,
            "hidden_labels_used": False,
            "private_answers_used": False,
            "external_steam_scraping_used": False,
            "credentials_or_tokens_printed": False,
            "git_stage_commit_push_executed": False,
        },
        "note": "Follow-up diagnostic grid over already-created validation source-score artifacts only; no test/full-test rows, no submission CSV.",
        "base_by_split": {cache.split: cache.base_acc for cache in caches},
        "num_variants_scanned": len(rows),
        "top10": rows[:10],
        "strict_pass_count": len(strict),
        "strict_top": strict[:10],
        "verdict": "STRICT_PASS_DIAGNOSTIC_NEEDS_INDEPENDENT_CONFIRMATION" if strict else ("WEAK_SIGNAL" if rows and rows[0]["mean_delta_vs_base"] > 0 else "REJECT"),
    }
    out_json = ROOT / args.out_json
    out_md = ROOT / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, out_md)
    print(json.dumps({"out_json": args.out_json, "out_md": args.out_md, "verdict": payload["verdict"], "strict_pass_count": len(strict), "top": rows[0]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
