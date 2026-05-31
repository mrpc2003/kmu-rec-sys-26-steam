#!/usr/bin/env python3
"""AlphaRec-core hypothesis probe (ICLR'25 Oral: "Language Representations Can be What
Recommenders Need") — UNIFORM gate. CPU. No Kaggle submission.

AlphaRec's central finding: frozen-LM item-text representations, mapped into a behavior
space, encode collaborative signals. Distinct from my earlier MiniLM probe (which used the
user's OWN review text vs item review text, raw cosine -> solo 0.639, corr_z 0.461: orthogonal
but too weak). AlphaRec instead builds the USER representation by AGGREGATING the language
representations of the ITEMS the user interacted with (collaborative co-occurrence in LM space).

Cheap faithful test of the core hypothesis (no MLP/InfoNCE machinery, which I've shown hurts):
  item_lm[i]   = L2-normalize( mean of MiniLM embeddings of reviews FOR item i )      # item semantics
  user_beh[u]  = L2-normalize( mean over items i in train(u) of item_lm[i] )          # AlphaRec behavior aggregation
  score(u,i)   = cos( user_beh[u], item_lm[i] )
Also a ridge-mapped variant: learn W mapping item_lm -> (no labels) is skipped to avoid
redundancy-by-construction; we only test the parameter-free aggregation form.

Gate (same bar as every axis): solo must clear popularity floor 0.684 to matter; a 50/50
within-user z-blend with emb128 must beat ref 0.76505 by > noise 0.0007 to be adopted. The
decisive number is corr_z vs emb128 (orthogonality). Honest prior: LOW, because (a) item text
here is noisy USER REVIEWS not clean titles, (b) all test users/items are seen so AlphaRec's
zero-shot/cold-start advantage doesn't apply, (c) it shares the LM-semantic axis my MiniLM
probe already found too weak. Running it cleanly closes the LM-representation-CF axis.
"""
from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import (  # noqa: E402
    load_train_interactions, load_pairs_csv, evaluate_tophalf, ensure_dir, write_json,
)

SPLIT = "val_random_uniform_seed42"
RAW_TRAIN = ROOT / "data/raw/public/data/train.json"
EMB128_SEED42_UNI = ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / SPLIT / "lightgcn_scores.csv"
EMB128_ENS_REF = 0.76505
POP_FLOOR = 0.684
NOISE = 0.0007
OUT = ensure_dir(ROOT / "artifacts/alpharec_core_uniform")
OUT_JSON = ROOT / "reports/20260531_alpharec_core_uniform_gate.json"
OUT_MD = ROOT / "reports/20260531_alpharec_core_uniform_gate.md"


def load_text_by_row(needed: set[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    with RAW_TRAIN.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx not in needed:
                continue
            try:
                d = ast.literal_eval(line)
            except Exception:
                d = {}
            out[idx] = str(d.get("text") or "")
            if len(out) == len(needed):
                break
    return out


def zscore_within_user(df, col, user_col="userID"):
    g = df.groupby(user_col)[col]
    mu = g.transform("mean"); sd = g.transform("std").replace(0, 1.0).fillna(1.0)
    return ((df[col] - mu) / sd).to_numpy()


def l2n(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():
    from sentence_transformers import SentenceTransformer

    tr = load_train_interactions(ROOT / "artifacts/validation" / SPLIT / "train_interactions.csv")
    cand = load_pairs_csv(ROOT / "artifacts/validation" / SPLIT / "candidates.csv")
    needed = set(tr["row_idx"].astype(int).tolist())
    text_by_row = load_text_by_row(needed)
    tr = tr.copy()
    tr["text"] = tr["row_idx"].astype(int).map(text_by_row).fillna("")

    nonempty = tr[tr["text"].str.len() > 0].copy()
    print(f"[alpharec] encoding {len(nonempty)} reviews", flush=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    emb = model.encode(nonempty["text"].tolist(), batch_size=256, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
    dim = emb.shape[1]
    print(f"[alpharec] encoded -> {emb.shape}", flush=True)

    # item language representation = mean review embedding per item (L2-normalized)
    i_sum = defaultdict(lambda: np.zeros(dim, np.float32)); i_cnt = defaultdict(int)
    for k, gid in enumerate(nonempty["gameID"].astype(str).tolist()):
        i_sum[gid] += emb[k]; i_cnt[gid] += 1
    item_lm = {g: l2n(i_sum[g] / max(i_cnt[g], 1)) for g in i_sum}

    # AlphaRec user behavior repr = mean of item_lm over items the user interacted with (train)
    u_items = defaultdict(list)
    for uid, gid in tr[["userID", "gameID"]].astype(str).itertuples(index=False):
        if gid in item_lm:
            u_items[uid].append(gid)
    user_beh = {}
    for u, gids in u_items.items():
        if gids:
            user_beh[u] = l2n(np.mean([item_lm[g] for g in gids], axis=0))

    scores = np.full(len(cand), np.nan, np.float32)
    for n, (uid, gid) in enumerate(cand[["userID", "gameID"]].astype(str).itertuples(index=False)):
        ub = user_beh.get(uid); il = item_lm.get(gid)
        if ub is not None and il is not None:
            scores[n] = float(np.dot(ub, il))
    known = ~np.isnan(scores)
    cand = cand.copy(); cand["score_alpha"] = scores
    cand["score_alpha"] = cand["score_alpha"].fillna(np.nanmin(scores) - 1.0)

    solo, _ = evaluate_tophalf(cand, "score_alpha", label_col="Label", user_col="userID", id_col="ID")
    solo_acc = round(float(solo["row_accuracy"]), 5)

    e128 = pd.read_csv(EMB128_SEED42_UNI)[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": "score_cf"})
    m = cand.merge(e128, on="ID", how="inner")
    m["zt"] = zscore_within_user(m, "score_alpha")
    m["zc"] = zscore_within_user(m, "score_cf")
    corr_z = float(np.corrcoef(m["zt"], m["zc"])[0, 1])
    m["zb"] = 0.5 * m["zt"] + 0.5 * m["zc"]
    b, _ = evaluate_tophalf(m, "zb", label_col="Label", user_col="userID", id_col="ID")
    blend50 = round(float(b["row_accuracy"]), 5)
    blend_vs_ref = round(blend50 - EMB128_ENS_REF, 5)

    if solo_acc < POP_FLOOR and blend_vs_ref <= NOISE:
        tier = "REJECT_WEAK"
        verdict = (f"AlphaRec-core solo {solo_acc} < floor {POP_FLOOR}; corr_z {corr_z:.3f}; "
                   f"blend50 {blend50} (vs ref {blend_vs_ref:+.5f}). LM-representation CF axis is "
                   f"orthogonal-ish but too weak on noisy review text -> closed, like MiniLM/TF-IDF.")
    elif blend_vs_ref > NOISE and corr_z < 0.6:
        tier = "ADOPT_CHECK"
        verdict = (f"blend50 {blend50} beats ref by {blend_vs_ref:+.5f} with corr_z {corr_z:.3f} "
                   f"-> potential orthogonal find, escalate to full AlphaRec.")
    else:
        tier = "REJECT_TIED"
        verdict = (f"solo {solo_acc}, corr_z {corr_z:.3f}, blend50 {blend50} vs ref {blend_vs_ref:+.5f} "
                   f"within/under noise -> not a parameter-free upgrade.")

    summary = {"note": "AlphaRec-core (LM item-rep behavior aggregation) uniform gate. CPU. No submission.",
               "split": SPLIT, "rows": int(len(m)), "text_known_rate": float(known.mean()),
               "solo_uniform": solo_acc, "corr_withinuser_z": round(corr_z, 4),
               "blend50_uniform": blend50, "emb128_ens_ref": EMB128_ENS_REF,
               "blend50_vs_ref": blend_vs_ref, "pop_floor": POP_FLOOR, "noise": NOISE,
               "tier": tier, "verdict": verdict}
    m[["ID", "userID", "gameID", "Label", "score_alpha"]].to_csv(OUT / "alpharec_scores_uniform.csv", index=False)
    write_json(OUT_JSON, summary)
    md = ["# AlphaRec-core (ICLR'25) — UNIFORM gate (LM item-rep behavior aggregation)\n",
          f"- split `{SPLIT}` rows={len(m)} | text known {known.mean():.4f}",
          f"- **solo uniform: {solo_acc}** (pop floor {POP_FLOOR})",
          f"- corr(within-user z) vs emb128 = **{corr_z:.4f}**",
          f"- **50/50 z-blend: {blend50}** vs emb128 ref {EMB128_ENS_REF} → **{blend_vs_ref:+.5f}**",
          f"- **tier: {tier}** — {verdict}\n",
          "## vs prior MiniLM probe\n",
          "MiniLM probe used the user's OWN review text (solo 0.639, corr_z 0.461). AlphaRec-core "
          "uses behavior aggregation of item language reps. Both test the LM-semantic axis."]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\n[ALPHAREC GATE] solo={solo_acc} corr_z={corr_z:.3f} blend50={blend50} "
          f"vs_ref={blend_vs_ref:+.5f} tier={tier}", flush=True)
    print(f"saved: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
