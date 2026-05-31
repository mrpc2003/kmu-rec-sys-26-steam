"""MiniLM semantic-embedding orthogonal axis — UNIFORM gate (last unverified signal source).

Context
-------
CF family (BPR-LightGCN / ALS / EASE / ItemKNN) is saturated; SGL/DirectAU failed the
strong+orthogonal bet. Review TF-IDF was weak as a standalone scorer. The last untested
orthogonal signal is *pretrained semantic* text embeddings (all-MiniLM-L6-v2, cached),
which capture review meaning rather than lexical overlap.

Method (test rows have NO text, so train-only profiles)
-------------------------------------------------------
- Encode every train review with MiniLM (mean-pooled sentence embedding).
- user profile = mean of the user's review embeddings; item profile = mean of the item's
  received review embeddings (L2-normalized).
- candidate score = cosine(user_profile, item_profile).

Adoption gate (parameter-free, same bar as every other axis)
------------------------------------------------------------
On the uniform public-surrogate split, report:
  - solo uniform row_acc (must clear the popularity floor 0.684 to even be interesting),
  - corr(z_text, z_emb128) (orthogonality),
  - 50/50 z-blend(text, emb128) uniform row_acc vs emb128 ens ref 0.76505.
Only a parameter-free blend that BEATS emb128 by > noise (0.0007) justifies adoption.
Grid-tuned blend weights are reported as diagnostic only (stacker-trap risk).

CPU-only (GPU busy with seed-expansion workers). No Kaggle submission. Report-only.
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
OUT = ensure_dir(ROOT / "artifacts/semantic_minilm_uniform")
OUT_JSON = ROOT / "reports/20260531_semantic_minilm_uniform.json"
OUT_MD = ROOT / "reports/20260531_semantic_minilm_uniform.md"


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


def zscore_within_user(df: pd.DataFrame, col: str, user_col: str = "userID") -> np.ndarray:
    g = df.groupby(user_col)[col]
    mu = g.transform("mean"); sd = g.transform("std").replace(0, 1.0).fillna(1.0)
    return ((df[col] - mu) / sd).to_numpy()


def main() -> None:
    from sentence_transformers import SentenceTransformer

    tr = load_train_interactions(ROOT / "artifacts/validation" / SPLIT / "train_interactions.csv")
    cand = load_pairs_csv(ROOT / "artifacts/validation" / SPLIT / "candidates.csv")
    needed = set(tr["row_idx"].astype(int).tolist())
    text_by_row = load_text_by_row(needed)
    tr = tr.copy()
    tr["text"] = tr["row_idx"].astype(int).map(text_by_row).fillna("")

    # encode unique non-empty reviews once
    nonempty = tr[tr["text"].str.len() > 0].copy()
    print(f"[minilm] reviews to encode: {len(nonempty)} (of {len(tr)} train rows)", flush=True)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    texts = nonempty["text"].tolist()
    emb = model.encode(texts, batch_size=256, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
    print(f"[minilm] encoded -> {emb.shape}", flush=True)

    # per-user / per-item mean profiles
    dim = emb.shape[1]
    u_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(dim, np.float32))
    u_cnt: dict[str, int] = defaultdict(int)
    i_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(dim, np.float32))
    i_cnt: dict[str, int] = defaultdict(int)
    for k, (uid, gid) in enumerate(nonempty[["userID", "gameID"]].astype(str).itertuples(index=False)):
        v = emb[k]
        u_sum[uid] += v; u_cnt[uid] += 1
        i_sum[gid] += v; i_cnt[gid] += 1

    def norm_profile(s, c):
        v = s / max(c, 1)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    u_prof = {u: norm_profile(u_sum[u], u_cnt[u]) for u in u_sum}
    i_prof = {g: norm_profile(i_sum[g], i_cnt[g]) for g in i_sum}

    scores = np.full(len(cand), np.nan, np.float32)
    for n, (uid, gid) in enumerate(cand[["userID", "gameID"]].astype(str).itertuples(index=False)):
        up = u_prof.get(uid); ip = i_prof.get(gid)
        if up is not None and ip is not None:
            scores[n] = float(np.dot(up, ip))
    known = ~np.isnan(scores)
    # impute missing with per-user min so they sort last
    cand = cand.copy()
    cand["score_text"] = scores
    fill = np.nanmin(scores) - 1.0
    cand["score_text"] = cand["score_text"].fillna(fill)

    solo, _ = evaluate_tophalf(cand, "score_text", label_col="Label", user_col="userID", id_col="ID")
    solo_acc = round(float(solo["row_accuracy"]), 5)

    # merge emb128 seed42 uniform for corr + blend
    e128 = pd.read_csv(EMB128_SEED42_UNI)[["ID", "score_lightgcn"]].rename(columns={"score_lightgcn": "score_cf"})
    m = cand.merge(e128, on="ID", how="inner")
    m["zt"] = zscore_within_user(m, "score_text")
    m["zc"] = zscore_within_user(m, "score_cf")
    corr = float(np.corrcoef(m["score_text"], m["score_cf"])[0, 1])
    corr_z = float(np.corrcoef(m["zt"], m["zc"])[0, 1])

    cf_solo, _ = evaluate_tophalf(m, "score_cf", label_col="Label", user_col="userID", id_col="ID")
    cf_acc = round(float(cf_solo["row_accuracy"]), 5)

    m["z_blend50"] = 0.5 * m["zt"] + 0.5 * m["zc"]
    b50, _ = evaluate_tophalf(m, "z_blend50", label_col="Label", user_col="userID", id_col="ID")
    blend50_acc = round(float(b50["row_accuracy"]), 5)

    # diagnostic grid (NOT a decision criterion)
    grid = {}
    for w in [0.05, 0.1, 0.2, 0.3]:
        m["zg"] = w * m["zt"] + (1 - w) * m["zc"]
        gg, _ = evaluate_tophalf(m, "zg", label_col="Label", user_col="userID", id_col="ID")
        grid[w] = round(float(gg["row_accuracy"]), 5)

    blend_vs_ref = round(blend50_acc - EMB128_ENS_REF, 5)
    if solo_acc < POP_FLOOR:
        verdict = (f"text solo {solo_acc} < popularity floor {POP_FLOOR} -> too weak. "
                   f"corr_z={corr_z:.3f}. 50/50 z-blend {blend50_acc} vs emb128 ref {EMB128_ENS_REF} "
                   f"({blend_vs_ref:+.5f}). REJECT as axis (orthogonal but too weak, same as TF-IDF/DirectAU).")
        tier = "REJECT_WEAK"
    elif blend_vs_ref > NOISE:
        verdict = (f"50/50 z-blend {blend50_acc} beats emb128 ref {EMB128_ENS_REF} by {blend_vs_ref:+.5f} "
                   f"> noise {NOISE} (corr_z={corr_z:.3f}) -> ADOPT as orthogonal axis.")
        tier = "ADOPT"
    else:
        verdict = (f"text solo {solo_acc} (>= floor) but 50/50 z-blend {blend50_acc} vs ref "
                   f"{EMB128_ENS_REF} = {blend_vs_ref:+.5f}, within/under noise {NOISE} "
                   f"(corr_z={corr_z:.3f}). Grid-max {max(grid.values())} is stacker-trap, not trusted. "
                   f"REJECT: not a parameter-free upgrade.")
        tier = "REJECT_TIED"

    summary = {
        "note": "MiniLM semantic axis uniform gate. CPU. No submission.",
        "split": SPLIT, "rows": int(len(m)),
        "text_known_rate": float(known.mean()),
        "text_solo_uniform": solo_acc, "cf_solo_uniform": cf_acc,
        "corr_raw": round(corr, 4), "corr_withinuser_z": round(corr_z, 4),
        "blend50_uniform": blend50_acc, "emb128_ens_ref": EMB128_ENS_REF,
        "blend50_vs_ref": blend_vs_ref, "pop_floor": POP_FLOOR, "noise": NOISE,
        "diagnostic_grid_w_text": grid, "tier": tier, "verdict": verdict,
    }
    m[["ID", "userID", "gameID", "Label", "score_text"]].to_csv(OUT / "text_scores_uniform.csv", index=False)
    write_json(OUT_JSON, summary)
    md = ["# MiniLM Semantic Axis — UNIFORM gate (last orthogonal signal)\n",
          f"- split `{SPLIT}` rows={len(m)} | text known rate {known.mean():.4f}",
          f"- **text solo uniform: {solo_acc}** (pop floor {POP_FLOOR}, cf solo {cf_acc})",
          f"- corr(raw)={corr:.4f}  corr(within-user z)=**{corr_z:.4f}**",
          f"- **50/50 z-blend uniform: {blend50_acc}** vs emb128 ref {EMB128_ENS_REF} → **{blend_vs_ref:+.5f}**",
          f"- diagnostic grid (w_text→acc, stacker-trap, not trusted): {grid}",
          f"- **tier: {tier}** — {verdict}\n"]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"\n[MINILM GATE] solo={solo_acc} corr_z={corr_z:.3f} blend50={blend50_acc} "
          f"vs_ref={blend_vs_ref:+.5f} tier={tier}", flush=True)
    print(f"saved: {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
