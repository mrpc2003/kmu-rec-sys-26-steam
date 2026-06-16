#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""OTTO-style source-separated co-visitation smoke for KMU RecSys26 Steam.

Validation-only: reads fold train/candidates for uniform validation splits and existing
validation LightGCN score artifacts. It does not read full test pairs, does not write
candidate/submission CSVs, and does not call Kaggle.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import ensure_dir, evaluate_tophalf, load_pairs_csv, load_train_interactions, normalize_within_user, write_json  # noqa: E402

RUN_TS = "20260607T090941KST"
ARTIFACT_DIR = ROOT / "artifacts" / f"opencode_axis_loop_{RUN_TS}"
REPORT_JSON = ROOT / "reports" / f"{RUN_TS}_opencode_improvement_axis_loop.json"
REPORT_MD = ROOT / "reports" / f"{RUN_TS}_opencode_improvement_axis_loop.md"
SPLITS = ["val_random_uniform_seed42", "val_random_uniform_seed7", "val_random_uniform_seed123"]
MODEL_SEEDS = [42, 123, 2024, 7]
FEATURE_COLS = [
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
WEIGHTS = [0.02, 0.05, 0.1, 0.2, 0.35]


def emb128_files(split: str) -> list[Path]:
    if split == "val_random_uniform_seed42":
        return [
            ROOT / f"artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03/{split}/lightgcn_scores.csv",
            ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed123/{split}/lightgcn_scores.csv",
            ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed2024/{split}/lightgcn_scores.csv",
            ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed7/{split}/lightgcn_scores.csv",
        ]
    return [ROOT / f"artifacts/split_panel_emb128/{split}/seed{s}/lightgcn_scores.csv" for s in MODEL_SEEDS]


def load_base(split: str) -> pd.DataFrame:
    base = None
    cols = []
    for seed, path in zip(MODEL_SEEDS, emb128_files(split), strict=True):
        if not path.exists():
            raise FileNotFoundError(path)
        d = pd.read_csv(path)[["ID", "userID", "gameID", "Label", "score_lightgcn"]]
        col = f"base_seed{seed}"
        d = d.rename(columns={"score_lightgcn": col})
        base = d if base is None else base.merge(d[["ID", col]], on="ID", validate="one_to_one")
        cols.append(col)
    assert base is not None
    base["score_base"] = base[cols].mean(axis=1)
    return base[["ID", "userID", "gameID", "Label", "score_base"]]


def build_indices(train: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], list[str]]:
    users = sorted(train["userID"].astype(str).unique())
    items = sorted(train["gameID"].astype(str).unique())
    return {u: i for i, u in enumerate(users)}, {g: i for i, g in enumerate(items)}, items


def build_sources(train: pd.DataFrame, item_to_idx: dict[str, int]) -> dict[str, np.ndarray]:
    n_items = len(item_to_idx)
    rows = train["userID"].astype(str).map({u: i for i, u in enumerate(sorted(train["userID"].astype(str).unique()))}).to_numpy(np.int32)
    cols = train["gameID"].astype(str).map(item_to_idx).to_numpy(np.int32)
    X = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (rows, cols)), shape=(rows.max() + 1, n_items), dtype=np.float32)
    pop = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
    co = (X.T @ X).astype(np.float32).toarray()
    np.fill_diagonal(co, 0.0)
    denom = np.sqrt(np.maximum(pop[:, None] * pop[None, :], 1.0))
    co_norm = co / denom

    hval = np.log1p(train["hours_transformed"].fillna(0.0).to_numpy(np.float32))
    Xh = sp.csr_matrix((hval, (rows, cols)), shape=X.shape, dtype=np.float32)
    hrs = (Xh.T @ X).astype(np.float32).toarray() + (X.T @ Xh).astype(np.float32).toarray()
    np.fill_diagonal(hrs, 0.0)
    hrs_norm = hrs / np.maximum(co + 1.0, 1.0)

    fwd = np.zeros((n_items, n_items), dtype=np.float32)
    rev = np.zeros((n_items, n_items), dtype=np.float32)
    sorted_train = train.sort_values(["userID", "date", "gameID"], kind="mergesort")
    for _, grp in sorted_train.groupby("userID", sort=False):
        item_ids = [item_to_idx.get(str(g)) for g in grp["gameID"].tolist()]
        item_ids = [i for i in item_ids if i is not None]
        days = pd.to_datetime(grp["date"]).astype("int64").to_numpy(dtype=np.float64) / 86_400_000_000_000.0
        for a in range(len(item_ids)):
            ia = item_ids[a]
            upper = min(len(item_ids), a + 21)
            for b in range(a + 1, upper):
                ib = item_ids[b]
                gap = max(0.0, float(days[b] - days[a])) if b < len(days) else 0.0
                w = float((0.75 ** (b - a - 1)) * (0.5 ** (gap / 730.0)))
                fwd[ia, ib] += w
                rev[ib, ia] += w
    fwd /= np.sqrt(np.maximum(pop[:, None] * pop[None, :], 1.0))
    rev /= np.sqrt(np.maximum(pop[:, None] * pop[None, :], 1.0))
    return {"co": co_norm, "hrs": hrs_norm, "fwd": fwd, "rev": rev}


def score_candidates(train: pd.DataFrame, candidates: pd.DataFrame, item_to_idx: dict[str, int], sources: dict[str, np.ndarray]) -> pd.DataFrame:
    co = sources["co"]
    hrs = sources["hrs"]
    fwd = sources["fwd"]
    rev = sources["rev"]
    histories: dict[str, list[int]] = {}
    sorted_train = train.sort_values(["userID", "date", "gameID"], kind="mergesort")
    for uid, grp in sorted_train.groupby("userID", sort=False):
        histories[str(uid)] = [item_to_idx[str(g)] for g in grp["gameID"].astype(str).tolist() if str(g) in item_to_idx]
    out = candidates.copy()
    vals = {c: np.zeros(len(out), dtype=np.float32) for c in FEATURE_COLS if c != "score_source_mean_z"}
    for row_pos, (uid, gid) in enumerate(out[["userID", "gameID"]].astype(str).itertuples(index=False, name=None)):
        j = item_to_idx.get(gid)
        hist = histories.get(uid, [])
        if j is None or not hist:
            continue
        hist_arr = np.asarray(hist, dtype=np.int32)
        cvals = co[hist_arr, j]
        hvals = hrs[hist_arr, j]
        vals["score_coplay_sum"][row_pos] = float(cvals.sum())
        vals["score_coplay_max"][row_pos] = float(cvals.max(initial=0.0))
        if cvals.size:
            vals["score_coplay_top5_mean"][row_pos] = float(np.sort(cvals)[-min(5, cvals.size):].mean())
        vals["score_hours_sum"][row_pos] = float(hvals.sum())
        vals["score_hours_max"][row_pos] = float(hvals.max(initial=0.0))
        recent = hist_arr[-20:]
        decay = np.power(0.85, np.arange(len(recent) - 1, -1, -1, dtype=np.float32))
        vals["score_forward_recent"][row_pos] = float((fwd[recent, j] * decay).sum())
        vals["score_reverse_recent"][row_pos] = float((rev[recent, j] * decay).sum())
        last5 = hist_arr[-5:]
        last_decay = np.power(0.75, np.arange(len(last5) - 1, -1, -1, dtype=np.float32))
        vals["score_last5_coplay"][row_pos] = float((co[last5, j] * last_decay).sum())
        vals["score_last5_forward"][row_pos] = float((fwd[last5, j] * last_decay).sum())
    for col, arr in vals.items():
        out[col] = arr
    out = normalize_within_user(out, [c for c in vals], user_col="userID")
    z_cols = [f"z_{c}" for c in vals]
    out["score_source_mean_z"] = out[z_cols].mean(axis=1)
    return out


def exact_binom_two_sided(successes: int, trials: int) -> float | None:
    if trials <= 0:
        return None
    k = min(successes, trials - successes)
    cdf = sum(math.comb(trials, i) for i in range(k + 1)) / (2 ** trials)
    return float(min(1.0, 2.0 * cdf))


def predict_vec(df: pd.DataFrame, score_col: str) -> np.ndarray:
    return evaluate_tophalf(df, score_col, label_col="Label", user_col="userID", id_col="ID")[1]["Pred"].to_numpy(np.int8)


def run() -> dict[str, Any]:
    ensure_dir(ARTIFACT_DIR)
    split_payloads = []
    variant_rows: dict[str, list[dict[str, Any]]] = {}
    source_note = "source-separated co-visitation: normalized co-play, hours co-play, ordered forward/reverse transitions, last-K history reductions"
    for split in SPLITS:
        split_dir = ROOT / "artifacts" / "validation" / split
        train = load_train_interactions(split_dir / "train_interactions.csv")
        candidates = load_pairs_csv(split_dir / "candidates.csv")
        base = load_base(split)
        candidates = candidates.merge(base[["ID", "score_base"]], on="ID", validate="one_to_one")
        _, item_to_idx, _ = build_indices(train)
        sources = build_sources(train, item_to_idx)
        scored = score_candidates(train, candidates, item_to_idx, sources)
        scored = normalize_within_user(scored, ["score_base"], user_col="userID")
        base_summary, base_pred_df = evaluate_tophalf(scored, "score_base", label_col="Label", user_col="userID", id_col="ID")
        base_pred = base_pred_df["Pred"].to_numpy(np.int8)
        y = scored["Label"].to_numpy(np.int8)
        split_out = ensure_dir(ARTIFACT_DIR / "otto_source_covisit" / split)
        keep_cols = ["ID", "userID", "gameID", "Label", "score_base"] + FEATURE_COLS + [f"z_{c}" for c in FEATURE_COLS if c != "score_source_mean_z"]
        scored[keep_cols].to_csv(split_out / "validation_otto_source_scores.csv", index=False)
        split_payload = {"split": split, "base_accuracy": base_summary["row_accuracy"], "score_artifact": str(split_out / "validation_otto_source_scores.csv"), "variants": []}
        for feature in FEATURE_COLS:
            z_feature = feature if feature == "score_source_mean_z" else f"z_{feature}"
            for w in WEIGHTS:
                variant = f"base_plus_{feature}_w{w:g}"
                score_col = f"score_{variant}"
                scored[score_col] = scored["z_score_base"] + w * scored[z_feature]
                summary, pred_df = evaluate_tophalf(scored, score_col, label_col="Label", user_col="userID", id_col="ID")
                pred = pred_df["Pred"].to_numpy(np.int8)
                fixes = int(((pred == y) & (base_pred != y)).sum())
                breaks = int(((pred != y) & (base_pred == y)).sum())
                row = {
                    "split": split,
                    "variant": variant,
                    "feature": feature,
                    "weight": w,
                    "accuracy": float(cast(Any, summary["row_accuracy"])),
                    "delta_vs_base": float(cast(Any, summary["row_accuracy"])) - float(cast(Any, base_summary["row_accuracy"])),
                    "fixes": fixes,
                    "breaks": breaks,
                }
                split_payload["variants"].append(row)
                variant_rows.setdefault(variant, []).append(row)
        split_payload["top_by_delta"] = sorted(split_payload["variants"], key=lambda r: r["delta_vs_base"], reverse=True)[:10]
        split_payloads.append(split_payload)

    aggregate = []
    for variant, rows in variant_rows.items():
        if len(rows) != len(SPLITS):
            continue
        deltas = [float(r["delta_vs_base"]) for r in rows]
        fixes = sum(int(r["fixes"]) for r in rows)
        breaks = sum(int(r["breaks"]) for r in rows)
        p = exact_binom_two_sided(max(fixes, breaks), fixes + breaks)
        aggregate.append({
            "variant": variant,
            "mean_delta_vs_base": float(np.mean(deltas)),
            "min_delta_vs_base": float(np.min(deltas)),
            "max_delta_vs_base": float(np.max(deltas)),
            "positive_splits": int(sum(d > 0 for d in deltas)),
            "fixes": int(fixes),
            "breaks": int(breaks),
            "pooled_p_exact": p,
            "split_deltas": {r["split"]: r["delta_vs_base"] for r in rows},
        })
    aggregate = sorted(aggregate, key=lambda r: (r["mean_delta_vs_base"], r["min_delta_vs_base"], r["fixes"] - r["breaks"]), reverse=True)
    strict = [r for r in aggregate if r["mean_delta_vs_base"] >= 0.0015 and r["min_delta_vs_base"] >= 0 and r["positive_splits"] == 3 and r["fixes"] > r["breaks"] and r["pooled_p_exact"] is not None and r["pooled_p_exact"] < 0.05]
    top = aggregate[0] if aggregate else {}
    verdict = "STRICT_PASS" if strict else ("WEAK_SIGNAL" if top and top.get("mean_delta_vs_base", 0) > 0 else "REJECT")
    payload = {
        "safety_flags": {
            "validation_only": True,
            "candidate_csv_written": False,
            "full_test_candidate_or_submission_csv_created": False,
            "kaggle_submit_executed": False,
            "hidden_labels_used": False,
            "private_answers_used": False,
            "external_steam_scraping_used": False,
            "credentials_or_tokens_printed": False,
            "quarantine_or_guard_logic_weakened": False,
            "git_stage_commit_push_executed": False,
            "recursive_cron_scheduled": False,
        },
        "axis_decision": "launched_fresh_bounded_validation_only_otto_source_separated_covisit_smoke",
        "new_probe": {
            "launched": True,
            "status": "completed",
            "command": "timeout 600 env -u VIRTUAL_ENV UV_NO_ACTIVE_VENV=1 uv run --with numpy --with pandas --with scipy python scripts/otto_source_covisit_smoke.py",
            "pid_file": None,
            "log": str(ROOT / "logs" / f"{RUN_TS}_otto_source_covisit_smoke.log"),
            "report_json": str(REPORT_JSON.relative_to(ROOT)),
            "report_md": str(REPORT_MD.relative_to(ROOT)),
            "artifact_dir": str(ARTIFACT_DIR.relative_to(ROOT)),
        },
        "best_or_top_metrics": top,
        "closed_or_rejected_axes_checked": [
            {"axis": "UserKNN gated residual fine-grid", "evidence": "missing expected reports; stalled warning-dominated log; not relaunched"},
            {"axis": "jackknife uncertainty boundary expanded", "evidence": "missing expected reports; 12-line log stopped mid split"},
            {"axis": "jackknife uncertainty boundary smoke", "evidence": "WEAK_SIGNAL mean +0.00036674, min -0.00120024, 2/3 positive, p 0.338815"},
            {"axis": "boundary/frontier/rankblend/TAG-CF", "evidence": "public-negative/quarantined family conflicts in aggressive_quota_runner_state.json"},
            {"axis": "plain/time-decay ItemKNN/BM25", "evidence": "already tested in paper_guided_next_steps and Stage3 blend; this probe keeps separate OTTO-style source features and residual-gates only"},
        ],
        "ranked_next_axis_hypotheses": [
            {"rank": 1, "hypothesis": "If source-separated co-visitation is only weak here, try no expansion; strict gate requires 3/3 positive and p<0.05."},
            {"rank": 2, "hypothesis": "A cleaned one-split UserKNN diagnostic is lower priority because prior broad grid stalled and prior smoke was below strict gate."},
            {"rank": 3, "hypothesis": "Text/semantic residuals remain low priority because prior train-review semantic probes were weak/redundant."},
        ],
        "artifacts_reports_produced": {
            "report_json": str(REPORT_JSON.relative_to(ROOT)),
            "report_md": str(REPORT_MD.relative_to(ROOT)),
            "artifact_dir": str(ARTIFACT_DIR.relative_to(ROOT)),
            "split_score_files": [sp["score_artifact"] for sp in split_payloads],
        },
        "source_note": source_note,
        "split_results": split_payloads,
        "aggregate_top10": aggregate[:10],
        "strict_pass_count": len(strict),
        "verdict": verdict,
    }
    write_json(REPORT_JSON, payload)
    write_md(payload)
    return payload


def write_md(payload: dict[str, Any]) -> None:
    top = cast(dict[str, Any], payload.get("best_or_top_metrics") or {})
    lines = [
        "# KMURecSys26 Steam no-submit improvement-axis loop",
        "",
        f"- Timestamp: {RUN_TS}",
        "- Safety: validation-only; no Kaggle submit; no candidate/submission CSV; no hidden/private labels; no external Steam scraping.",
        f"- Verdict: `{payload['verdict']}`",
        "",
        "## Axis decision",
        "",
        str(payload["axis_decision"]),
        "",
        "The launched smoke is a bounded OTTO-style source-separated co-visitation residual gate. It differs from prior plain/time-decay ItemKNN/BM25 by keeping separate co-play, ordered transition, last-K, and hours-weighted source scores, then testing small residual weights against the emb128 4-seed LightGCN reference across three uniform validation splits.",
        "",
        "## Top aggregate metric",
        "",
    ]
    if top:
        lines += [
            f"- Variant: `{top.get('variant')}`",
            f"- mean Δ vs base: {float(top.get('mean_delta_vs_base', 0.0)):+.10f}",
            f"- min split Δ: {float(top.get('min_delta_vs_base', 0.0)):+.10f}",
            f"- positive splits: {top.get('positive_splits')}/3",
            f"- fixes/breaks: {top.get('fixes')}/{top.get('breaks')}",
            f"- pooled exact p: {top.get('pooled_p_exact')}",
            f"- split deltas: `{top.get('split_deltas')}`",
        ]
    else:
        lines.append("No aggregate rows were produced.")
    lines += ["", "## Closed/rejected axes checked", ""]
    for row in cast(list[dict[str, Any]], payload["closed_or_rejected_axes_checked"]):
        lines.append(f"- {row['axis']}: {row['evidence']}")
    lines += ["", "## Produced artifacts", ""]
    produced = cast(dict[str, Any], payload["artifacts_reports_produced"])
    for k, v in produced.items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Strict gate status", "", f"Strict pass count: `{payload['strict_pass_count']}`"]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = run()
    print(json.dumps({"verdict": p["verdict"], "top": p["best_or_top_metrics"]}, indent=2, ensure_ascii=False))
