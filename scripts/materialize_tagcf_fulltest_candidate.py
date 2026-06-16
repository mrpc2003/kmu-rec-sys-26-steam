#!/usr/bin/env python3
"""Materialize a full-test TAG-CF-style candidate.

Safety: uses only official train.json and pairs.csv; no hidden labels, no external
Steam scraping, and no Kaggle submission. The submission decision remains with the
autorun guard.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lightgcn_train import score_candidates, train_lightgcn
from recsys_played_utils import DEFAULT_DATA_DIR, build_user_item_matrix, ensure_dir, load_pairs_csv, load_train_interactions
from tagcf_testtime_aggregation_probe import make_tag_embeddings


def predict_tophalf_no_labels(df: pd.DataFrame, score_col: str) -> np.ndarray:
    pred = np.zeros(len(df), dtype=np.int8)
    values = df[score_col].to_numpy(dtype=float)
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx = np.asarray(idx)
        k = len(idx) // 2
        order = np.lexsort((df.loc[idx, "ID"].to_numpy(), -values[idx]))
        pred[idx[order[:k]]] = 1
    return pred


def preflight_submission(path: Path, pairs: pd.DataFrame) -> dict[str, Any]:
    d = pd.read_csv(path)
    out = {
        "rows": int(len(d)),
        "expected_rows": int(len(pairs)),
        "columns": d.columns.tolist(),
        "id_unique": bool(d["ID"].is_unique) if "ID" in d.columns else False,
        "labels_binary": bool(set(d.get("Played", pd.Series(dtype=int)).dropna().unique()).issubset({0, 1})),
        "label_1": int((d.get("Played", pd.Series(dtype=int)) == 1).sum()),
        "label_0": int((d.get("Played", pd.Series(dtype=int)) == 0).sum()),
        "bad_users_tophalf": None,
    }
    if {"ID", "Played"}.issubset(d.columns):
        m = pairs[["ID", "userID"]].merge(d, on="ID", how="left", validate="one_to_one")
        bad = 0
        for _, g in m.groupby("userID", sort=False):
            if int(g["Played"].sum()) != len(g) // 2:
                bad += 1
        out["bad_users_tophalf"] = bad
    return out


def row_diff(path: Path, ref: Path) -> int | None:
    if not ref.exists():
        return None
    a = pd.read_csv(path).sort_values("ID")["Played"].astype(int).to_numpy()
    b = pd.read_csv(ref).sort_values("ID")["Played"].astype(int).to_numpy()
    if len(a) != len(b):
        return None
    return int((a != b).sum())


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out_dir = ensure_dir(Path(args.out_dir))
    data_dir = Path(args.data_dir)
    train_df = load_train_interactions(data_dir / "train.json")
    pairs = load_pairs_csv(data_dir / "pairs.csv")
    mat, user_to_idx, item_to_idx, users, items = build_user_item_matrix(train_df, binary=True)

    user_emb, item_emb, train_meta = train_lightgcn(
        mat,
        len(users),
        len(items),
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        lr=args.lr,
        reg=args.reg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    u_tag, i_tag = make_tag_embeddings(mat, user_emb, item_emb, alpha=args.alpha, mode=args.mode, normalize=args.normalize)
    pairs = pairs.copy().sort_values("ID", kind="mergesort").reset_index(drop=True)
    pairs["score_base"] = score_candidates(pairs, user_emb, item_emb, user_to_idx, item_to_idx)
    tag_col = f"score_tag_{args.mode}_a{str(args.alpha).replace('.', 'p')}_{'l2' if args.normalize else 'raw'}"
    pairs[tag_col] = score_candidates(pairs, u_tag, i_tag, user_to_idx, item_to_idx)
    if args.blend_z:
        g = pairs.groupby("userID", sort=False)
        z_base = (pairs["score_base"] - g["score_base"].transform("mean")) / g["score_base"].transform("std").replace(0, np.nan).fillna(1.0)
        z_tag = (pairs[tag_col] - g[tag_col].transform("mean")) / g[tag_col].transform("std").replace(0, np.nan).fillna(1.0)
        score_col = "score_tagcf_zblend"
        pairs[score_col] = (1.0 - args.blend_weight) * z_base + args.blend_weight * z_tag
    else:
        score_col = tag_col
    pairs["Played"] = predict_tophalf_no_labels(pairs, score_col)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs[["ID", "Played"]].to_csv(out_path, index=False)
    score_path = out_dir / "tagcf_fulltest_scores.csv"
    pairs[["ID", "userID", "gameID", "score_base", tag_col, score_col]].to_csv(score_path, index=False)

    comparisons = {}
    refs = {
        "rankblend_public_best": ROOT / "submissions/candidate_rank_blend_emb128_emb192.csv",
        "boundary_w_minus_0p75": ROOT / "submissions/candidate_autorun_boundary_scoreblend_z128_z192_z64_w-0.75.csv",
        "boundary_w2": ROOT / "submissions/candidate_autorun_boundary_scoreblend_z128_z192_z64_w2.csv",
        "frontier_z_w1920_w64_minus_0p25": ROOT / "submissions/candidate_autorun_frontier_z_w1920_w64-0.25.csv",
    }
    for name, ref in refs.items():
        comparisons[name] = {"path": str(ref), "row_diff": row_diff(out_path, ref)}
    result = {
        "kind": "tagcf_fulltest_candidate",
        "safety": {"hidden_label_access": False, "external_steam_scraping": False, "kaggle_submit_executed": False},
        "variant": f"tagcf_fulltest_seed{args.seed}_{args.mode}_a{args.alpha:g}_{'l2' if args.normalize else 'raw'}_{'zblend' if args.blend_z else 'direct'}_bw{args.blend_weight:g}",
        "output": str(out_path),
        "score_path": str(score_path),
        "train_meta": train_meta,
        "args": vars(args),
        "preflight": preflight_submission(out_path, pairs),
        "comparisons": comparisons,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# TAG-CF full-test candidate\n\n",
        f"- variant: `{result['variant']}`\n",
        f"- output: `{out_path}`\n",
        f"- safety: no hidden labels / no external scraping / no submit\n",
        f"- preflight: `{result['preflight']}`\n",
        "\n## Row diffs\n\n",
    ]
    for name, c in comparisons.items():
        lines.append(f"- {name}: `{c['row_diff']}`\n")
    Path(args.md).write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"variant": result["variant"], "output": str(out_path), "preflight": result["preflight"], "comparisons": comparisons, "json": args.json}, indent=2, ensure_ascii=False), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out-dir", default=str(ROOT / "artifacts/tagcf_fulltest_candidate"))
    p.add_argument("--out", default=str(ROOT / "submissions/candidate_autorun_tagcf_fulltest_latest.csv"))
    p.add_argument("--json", default=str(ROOT / "reports/20260603_tagcf_fulltest_candidate.json"))
    p.add_argument("--md", default=str(ROOT / "reports/20260603_tagcf_fulltest_candidate.md"))
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--mode", choices=["mean", "sym"], default="sym")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--blend-z", action="store_true", default=True)
    p.add_argument("--no-blend-z", action="store_false", dest="blend_z")
    p.add_argument("--blend-weight", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
