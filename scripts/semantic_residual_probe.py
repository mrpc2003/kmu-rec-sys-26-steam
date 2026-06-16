#!/usr/bin/env python3
"""Validation-only semantic residual probe for KMURecSys26 Steam.

This script tests the latest-paper/HF-model hypothesis in its cheapest safe form:
fold-train-only review text -> item semantic embeddings -> user positive centroids ->
validation candidate semantic scores -> residual/blend against the canonical emb128
LightGCN 4-seed ensemble.

Safety contract:
- no Kaggle submission;
- no real test/pairs.csv decoding;
- no candidate/submission CSV materialization;
- validation outputs only under artifacts/semantic_residual_probe and reports/.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import evaluate_tophalf, ensure_dir, write_json  # noqa: E402

SPLITS = ("val_random_uniform_seed42", "val_random_uniform_seed7", "val_random_uniform_seed123")
SEEDS = (42, 123, 2024, 7)
BASE_EXPECTED = {
    "val_random_uniform_seed42": 0.7650530106021204,
    "val_random_uniform_seed7": 0.7609521904380876,
    "val_random_uniform_seed123": 0.7599519903980796,
}
MDE = 0.00355
ALPHAS = (-0.10, -0.05, -0.02, -0.01, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
RAW_TRAIN = ROOT / "data/raw/public/data/train.json"


def slugify_model_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")[:96]


def exact_two_sided_binom_p(k: int, n: int) -> float:
    """Exact two-sided sign-test/McNemar p under p=0.5 for discordant flips."""
    if n <= 0:
        return 1.0
    kk = min(k, n - k)
    logs = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0)
        for i in range(kk + 1)
    ]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(v - m) for v in logs)
    return min(1.0, 2.0 * tail)


def fisher_pvalue(p_values: Iterable[float]) -> float:
    vals = [max(min(float(p), 1.0), 1e-300) for p in p_values]
    if not vals:
        return 1.0
    try:
        from scipy.stats import chi2

        stat = -2.0 * sum(math.log(v) for v in vals)
        return float(chi2.sf(stat, 2 * len(vals)))
    except Exception:
        return float("nan")


def score_col(df: pd.DataFrame) -> str:
    for c in ("score_layermix_uniform", "score_lightgcn", "score"):
        if c in df.columns:
            return c
    raise ValueError(f"No known score column found in {df.columns.tolist()}")


def e128_seed_path(split: str, seed: int) -> Path:
    if split == "val_random_uniform_seed42":
        if seed == 42:
            return ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03" / split / "lightgcn_scores.csv"
        return ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}" / split / "lightgcn_scores.csv"
    return ROOT / f"artifacts/split_panel_emb128/{split}/seed{seed}/lightgcn_scores.csv"


def load_base_frame(split: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    score_cols: list[str] = []
    for seed in SEEDS:
        p = e128_seed_path(split, seed)
        if not p.exists():
            raise FileNotFoundError(f"Missing canonical emb128 score file for split={split}, seed={seed}: {p}")
        d = pd.read_csv(p)
        sc = score_col(d)
        need = {"ID", "userID", "gameID", "Label", sc}
        if not need.issubset(d.columns):
            raise ValueError(f"Missing columns in {p}: {need - set(d.columns)}")
        col = f"score_e128_seed{seed}"
        part = d[["ID", "userID", "gameID", "Label", sc]].rename(columns={sc: col})
        if merged is None:
            merged = part
        else:
            before = len(merged)
            merged = merged.merge(part[["ID", col]], on="ID", how="inner", validate="one_to_one")
            if len(merged) != before:
                raise RuntimeError(f"Row alignment changed while merging {p}")
        score_cols.append(col)
    assert merged is not None
    merged = merged.sort_values("ID", kind="mergesort").reset_index(drop=True)
    merged["score_base"] = merged[score_cols].mean(axis=1)
    summ, _ = evaluate_tophalf(merged, "score_base", label_col="Label", user_col="userID", id_col="ID")
    got = float(summ["row_accuracy"])
    exp = BASE_EXPECTED[split]
    if abs(got - exp) > 5e-10:
        raise RuntimeError(f"Base reproduction mismatch for {split}: got={got}, expected={exp}")
    return merged[["ID", "userID", "gameID", "Label", "score_base"]]


def make_group_cache(df: pd.DataFrame) -> list[tuple[np.ndarray, int, np.ndarray]]:
    y = df["Label"].to_numpy(dtype=np.int8)
    ids = df["ID"].to_numpy(dtype=np.int64)
    cache: list[tuple[np.ndarray, int, np.ndarray]] = []
    for _, idx in df.groupby("userID", sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        cache.append((idx_arr, int(y[idx_arr].sum()), ids[idx_arr]))
    return cache


def stable_predict_tophalf_cached(group_cache: list[tuple[np.ndarray, int, np.ndarray]], score: np.ndarray, n_rows: int) -> np.ndarray:
    pred = np.zeros(n_rows, dtype=np.int8)
    values = np.asarray(score, dtype=float)
    for idx_arr, k, ids in group_cache:
        order_local = np.lexsort((ids, -values[idx_arr]))
        pred[idx_arr[order_local[:k]]] = 1
    return pred


def stable_predict_tophalf(df: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    return stable_predict_tophalf_cached(make_group_cache(df), score, len(df))


def metric_against_base(
    df: pd.DataFrame,
    score: np.ndarray,
    base_pred: np.ndarray,
    group_cache: list[tuple[np.ndarray, int, np.ndarray]] | None = None,
) -> dict[str, Any]:
    if group_cache is None:
        group_cache = make_group_cache(df)
    pred = stable_predict_tophalf_cached(group_cache, score, len(df))
    y = df["Label"].to_numpy(dtype=np.int8)
    ok = pred == y
    base_ok = base_pred == y
    fixes = int((~base_ok & ok).sum())
    breaks = int((base_ok & ~ok).sum())
    p = exact_two_sided_binom_p(fixes, fixes + breaks)
    return {
        "accuracy": float(ok.mean()),
        "delta": float(ok.mean() - base_ok.mean()),
        "fixes": fixes,
        "breaks": breaks,
        "discordant": fixes + breaks,
        "changed": int((pred != base_pred).sum()),
        "mcnemar_exact_p": p,
    }


def within_user_z(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    tmp = pd.DataFrame({"userID": df["userID"].to_numpy(), "v": np.asarray(values, dtype=float)})
    g = tmp.groupby("userID", sort=False)["v"]
    mu = g.transform("mean").to_numpy(dtype=float)
    sd = g.transform(lambda s: float(s.std(ddof=0))).to_numpy(dtype=float)
    out = np.zeros(len(tmp), dtype=float)
    m = sd > 1e-12
    v = tmp["v"].to_numpy(dtype=float)
    out[m] = (v[m] - mu[m]) / sd[m]
    out[~np.isfinite(out)] = 0.0
    return out


def residualize(y: np.ndarray, cols: list[np.ndarray]) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    xs = [np.ones_like(y)] + [np.asarray(c, dtype=float) for c in cols]
    x = np.column_stack(xs)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    out = y - x @ beta
    out[~np.isfinite(out)] = 0.0
    return out


def load_raw_text_table(raw_train: Path = RAW_TRAIN) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with raw_train.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if not line.strip():
                continue
            d = ast.literal_eval(line)
            text = str(d.get("text") or "")
            rows.append({"row_idx": row_idx, "text": text})
    return pd.DataFrame(rows)


def clean_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return text


def build_item_texts(
    train_df: pd.DataFrame,
    raw_texts: pd.DataFrame,
    reviews_per_item: int,
    max_review_chars: int,
    min_chars: int,
) -> pd.DataFrame:
    tr = train_df.merge(raw_texts, on="row_idx", how="left", validate="many_to_one")
    tr["text_clean"] = tr["text"].fillna("").map(lambda s: clean_text(str(s), max_review_chars))
    tr["text_chars"] = tr["text_clean"].str.len()
    tr = tr[tr["text_chars"] >= min_chars].copy()
    # Long-ish reviews are usually more descriptive than +1 / "good"; deterministic fold-train-only selection.
    tr = tr.sort_values(["gameID", "text_chars", "row_idx"], ascending=[True, False, True], kind="mergesort")
    item_rows: list[dict[str, Any]] = []
    for game_id, g in tr.groupby("gameID", sort=True):
        seen: set[str] = set()
        snippets: list[str] = []
        for _, r in g.head(max(reviews_per_item * 3, reviews_per_item)).iterrows():
            txt = str(r["text_clean"])
            if not txt or txt in seen:
                continue
            seen.add(txt)
            snippets.append(txt)
            if len(snippets) >= reviews_per_item:
                break
        if snippets:
            item_rows.append({"gameID": game_id, "item_text": " Review: ".join(["In-bundle Steam reviews."] + snippets)})
    out = pd.DataFrame(item_rows)
    if out.empty:
        raise RuntimeError("No item texts built from fold train interactions")
    return out


def sha256_texts(ids: list[str], texts: list[str]) -> str:
    h = hashlib.sha256()
    for i, t in zip(ids, texts, strict=True):
        h.update(i.encode("utf-8")); h.update(b"\0"); h.update(t.encode("utf-8")); h.update(b"\n")
    return h.hexdigest()


def embed_tfidf_svd(texts: list[str], dim: int) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(max_features=50000, min_df=2, max_df=0.95, ngram_range=(1, 2), dtype=np.float32)
    x = vec.fit_transform(texts)
    n_comp = min(dim, max(2, min(x.shape) - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    emb = svd.fit_transform(x).astype(np.float32)
    if n_comp < dim:
        pad = np.zeros((emb.shape[0], dim - n_comp), dtype=np.float32)
        emb = np.hstack([emb, pad])
    return l2_normalize(emb)


def embed_sentence_transformer(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_seq_length: int,
    device: str,
    trust_remote_code: bool,
) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {}
    if device.startswith("cuda") and torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
    try:
        model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
            model_kwargs=kwargs if kwargs else None,
        )
    except TypeError:
        model = SentenceTransformer(model_name, device=device)
    if max_seq_length > 0:
        model.max_seq_length = max_seq_length
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def encode_item_texts(
    item_texts: pd.DataFrame,
    model_name: str,
    batch_size: int,
    max_seq_length: int,
    device: str,
    trust_remote_code: bool,
    tfidf_dim: int,
) -> np.ndarray:
    texts = item_texts["item_text"].fillna("").astype(str).tolist()
    if model_name == "tfidf-svd-smoke":
        return embed_tfidf_svd(texts, tfidf_dim)
    return embed_sentence_transformer(texts, model_name, batch_size, max_seq_length, device, trust_remote_code)


def build_user_centroid_arrays(
    train_df: pd.DataFrame,
    item_to_pos: dict[str, int],
    item_emb: np.ndarray,
    weighted: bool,
) -> tuple[dict[str, int], np.ndarray, dict[str, int]]:
    """Build one L2-normalized centroid per user with vectorized per-user blocks.

    The first implementation used per-interaction Python vector additions; with 165k
    interactions x 1k-dim embeddings that is unnecessarily slow.  This version loops
    over users and lets NumPy sum each user's item-embedding block.
    """
    hist = train_df[["userID", "gameID", "hours_transformed"]].copy()
    hist["item_pos"] = hist["gameID"].astype(str).map(item_to_pos)
    hist = hist.dropna(subset=["item_pos"]).copy()
    hist["item_pos"] = hist["item_pos"].astype(np.int64)
    if hist.empty:
        return {}, np.zeros((0, item_emb.shape[1]), dtype=np.float32), {}

    user_to_pos: dict[str, int] = {}
    counts: dict[str, int] = {}
    centroids: list[np.ndarray] = []
    dim = int(item_emb.shape[1])
    for user_id, g in hist.groupby("userID", sort=False):
        pos = g["item_pos"].to_numpy(dtype=np.int64)
        if weighted:
            w = g["hours_transformed"].to_numpy(dtype=np.float32)
            w = np.where(np.isfinite(w) & (w > 0), w, 1.0).astype(np.float32)
        else:
            w = np.ones(len(pos), dtype=np.float32)
        block = item_emb[pos]
        vec = (block * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-12)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            continue
        user_to_pos[str(user_id)] = len(centroids)
        counts[str(user_id)] = int(len(pos))
        centroids.append((vec / norm).astype(np.float32))
    if not centroids:
        return {}, np.zeros((0, dim), dtype=np.float32), {}
    return user_to_pos, np.vstack(centroids).astype(np.float32), counts


def score_candidates_semantic(
    df: pd.DataFrame,
    item_to_pos: dict[str, int],
    item_emb: np.ndarray,
    user_to_pos: dict[str, int],
    user_centroids: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    u_pos = df["userID"].astype(str).map(user_to_pos)
    i_pos = df["gameID"].astype(str).map(item_to_pos)
    ok = (~u_pos.isna()) & (~i_pos.isna())
    scores = np.zeros(len(df), dtype=np.float32)
    if ok.any():
        uu = u_pos[ok].to_numpy(dtype=np.int64)
        ii = i_pos[ok].to_numpy(dtype=np.int64)
        scores[np.flatnonzero(ok.to_numpy())] = np.einsum("ij,ij->i", user_centroids[uu], item_emb[ii]).astype(np.float32)
    return scores, {
        "covered_rows": int(ok.sum()),
        "missing_user_rows": int((u_pos.isna()).sum()),
        "missing_item_rows": int((~u_pos.isna() & i_pos.isna()).sum()),
    }


def item_logpop(train_df: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    pop = train_df.groupby("gameID").size().astype(float)
    return np.log1p(df["gameID"].map(pop).fillna(0.0).to_numpy(dtype=float))


def run_split(args: argparse.Namespace, split: str, raw_texts: pd.DataFrame, model_slug: str) -> dict[str, Any]:
    split_dir = ROOT / "artifacts/validation" / split
    train_path = split_dir / "train_interactions.csv"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    train_df = pd.read_csv(train_path)
    train_df["row_idx"] = train_df["row_idx"].astype(int)
    base = load_base_frame(split)
    base_pred = stable_predict_tophalf(base, base["score_base"].to_numpy(dtype=float))
    y = base["Label"].to_numpy(dtype=np.int8)
    if abs(float((base_pred == y).mean()) - BASE_EXPECTED[split]) > 5e-10:
        raise RuntimeError(f"Local base predict mismatch for {split}")

    out_dir = ensure_dir(ROOT / args.out_dir / model_slug / split)
    item_texts = build_item_texts(train_df, raw_texts, args.reviews_per_item, args.max_review_chars, args.min_review_chars)
    item_texts = item_texts.sort_values("gameID", kind="mergesort").reset_index(drop=True)
    text_hash = sha256_texts(item_texts["gameID"].astype(str).tolist(), item_texts["item_text"].astype(str).tolist())

    emb_path = out_dir / "item_embeddings.npy"
    ids_path = out_dir / "item_ids.json"
    meta_path = out_dir / "embedding_meta.json"
    reuse = False
    if args.reuse_embeddings and emb_path.exists() and ids_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        old_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if meta.get("text_hash") == text_hash and old_ids == item_texts["gameID"].astype(str).tolist():
            reuse = True
    if reuse:
        item_emb = np.load(emb_path).astype(np.float32)
    else:
        item_emb = encode_item_texts(
            item_texts,
            args.model_name,
            args.batch_size,
            args.max_seq_length,
            args.device,
            args.trust_remote_code,
            args.tfidf_dim,
        )
        item_emb = l2_normalize(item_emb)
        np.save(emb_path, item_emb.astype(np.float32))
        ids_path.write_text(json.dumps(item_texts["gameID"].astype(str).tolist(), ensure_ascii=False), encoding="utf-8")
        write_json(meta_path, {
            "split": split,
            "model_name": args.model_name,
            "text_hash": text_hash,
            "items_embedded": int(len(item_texts)),
            "embedding_dim": int(item_emb.shape[1]),
            "reviews_per_item": args.reviews_per_item,
            "max_review_chars": args.max_review_chars,
            "min_review_chars": args.min_review_chars,
            "validation_only": True,
        })
    item_to_pos = {str(g): i for i, g in enumerate(item_texts["gameID"].astype(str).tolist())}

    u2p_bin, cent_bin, hist_bin = build_user_centroid_arrays(train_df, item_to_pos, item_emb, weighted=False)
    u2p_htr, cent_htr, hist_htr = build_user_centroid_arrays(train_df, item_to_pos, item_emb, weighted=True)
    sem_bin, cov_bin = score_candidates_semantic(base, item_to_pos, item_emb, u2p_bin, cent_bin)
    sem_htr, cov_htr = score_candidates_semantic(base, item_to_pos, item_emb, u2p_htr, cent_htr)

    frame = base.copy()
    frame["score_sem_bin"] = sem_bin
    frame["score_sem_htr"] = sem_htr
    frame["score_logpop"] = item_logpop(train_df, frame)
    frame["z_base"] = within_user_z(frame, frame["score_base"].to_numpy(dtype=float))
    frame["z_sem_bin"] = within_user_z(frame, sem_bin)
    frame["z_sem_htr"] = within_user_z(frame, sem_htr)
    frame["z_logpop"] = within_user_z(frame, frame["score_logpop"].to_numpy(dtype=float))
    frame["sem_bin_resid_base_pop"] = residualize(frame["z_sem_bin"].to_numpy(dtype=float), [frame["z_base"].to_numpy(dtype=float), frame["z_logpop"].to_numpy(dtype=float)])
    frame["sem_htr_resid_base_pop"] = residualize(frame["z_sem_htr"].to_numpy(dtype=float), [frame["z_base"].to_numpy(dtype=float), frame["z_logpop"].to_numpy(dtype=float)])
    frame["z_sem_bin_resid_base_pop"] = within_user_z(frame, frame["sem_bin_resid_base_pop"].to_numpy(dtype=float))
    frame["z_sem_htr_resid_base_pop"] = within_user_z(frame, frame["sem_htr_resid_base_pop"].to_numpy(dtype=float))

    variants: dict[str, np.ndarray] = {
        "sem_bin_only": frame["z_sem_bin"].to_numpy(dtype=float),
        "sem_htr_only": frame["z_sem_htr"].to_numpy(dtype=float),
        "sem_bin_resid_only": frame["z_sem_bin_resid_base_pop"].to_numpy(dtype=float),
        "sem_htr_resid_only": frame["z_sem_htr_resid_base_pop"].to_numpy(dtype=float),
    }
    for a in ALPHAS:
        tag = f"a{a:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
        variants[f"base_plus_{tag}_sem_bin"] = frame["z_base"].to_numpy(dtype=float) + a * frame["z_sem_bin"].to_numpy(dtype=float)
        variants[f"base_plus_{tag}_sem_htr"] = frame["z_base"].to_numpy(dtype=float) + a * frame["z_sem_htr"].to_numpy(dtype=float)
        variants[f"base_plus_{tag}_sem_bin_resid"] = frame["z_base"].to_numpy(dtype=float) + a * frame["z_sem_bin_resid_base_pop"].to_numpy(dtype=float)
        variants[f"base_plus_{tag}_sem_htr_resid"] = frame["z_base"].to_numpy(dtype=float) + a * frame["z_sem_htr_resid_base_pop"].to_numpy(dtype=float)

    split_metrics = []
    group_cache = make_group_cache(frame)
    for name, score in variants.items():
        m = metric_against_base(frame, score, base_pred, group_cache)
        m.update({"split": split, "variant": name, "base_accuracy": BASE_EXPECTED[split]})
        split_metrics.append(m)

    # Validation-score artifact, not a submission candidate: includes labels and no ID/Label-only schema.
    score_cols = [
        "ID", "userID", "gameID", "Label", "score_base", "score_sem_bin", "score_sem_htr",
        "score_logpop", "z_base", "z_sem_bin", "z_sem_htr", "z_sem_bin_resid_base_pop", "z_sem_htr_resid_base_pop",
    ]
    frame[score_cols].to_csv(out_dir / "validation_semantic_scores.csv", index=False)
    item_texts.assign(text_chars=item_texts["item_text"].str.len()).drop(columns=["item_text"]).to_csv(out_dir / "item_text_index.csv", index=False)

    return {
        "split": split,
        "base_accuracy": BASE_EXPECTED[split],
        "rows": int(len(frame)),
        "users": int(frame["userID"].nunique()),
        "items_in_fold_train": int(train_df["gameID"].nunique()),
        "items_with_text_profile": int(len(item_texts)),
        "embedding_dim": int(item_emb.shape[1]),
        "embedding_reused": reuse,
        "coverage_binary": cov_bin,
        "coverage_hours_weighted": cov_htr,
        "user_centroid_count_binary": int(len(cent_bin)),
        "user_centroid_count_hours_weighted": int(len(cent_htr)),
        "split_metrics": split_metrics,
    }


def aggregate(split_results: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for sr in split_results:
        rows.extend(sr["split_metrics"])
    split_df = pd.DataFrame(rows)
    agg_rows = []
    for variant, g in split_df.groupby("variant", sort=False):
        pvals = g["mcnemar_exact_p"].tolist()
        agg_rows.append({
            "variant": variant,
            "mean_accuracy": float(g["accuracy"].mean()),
            "mean_delta": float(g["delta"].mean()),
            "min_delta": float(g["delta"].min()),
            "max_delta": float(g["delta"].max()),
            "positive_splits": int((g["delta"] > 0).sum()),
            "nonnegative_splits": int((g["delta"] >= 0).sum()),
            "total_fixes": int(g["fixes"].sum()),
            "total_breaks": int(g["breaks"].sum()),
            "total_discordant": int(g["discordant"].sum()),
            "fisher_mcnemar_p": fisher_pvalue(pvals),
            "pass_strict_mde": bool((g["delta"].mean() >= MDE) and ((g["delta"] > 0).sum() >= 2) and (g["fixes"].sum() > g["breaks"].sum()) and (fisher_pvalue(pvals) < 0.05)),
        })
    agg = pd.DataFrame(agg_rows).sort_values(["mean_delta", "positive_splits", "total_fixes"], ascending=[False, False, False])
    return split_df, agg


def write_markdown_report(path: Path, args: argparse.Namespace, model_slug: str, split_results: list[dict[str, Any]], split_df: pd.DataFrame, agg: pd.DataFrame) -> None:
    top = agg.head(15).copy()
    lines = []
    lines.append(f"# Semantic Residual Probe — {args.model_name}\n")
    lines.append("Validation-only probe using fold-train in-bundle review text. No Kaggle submit and no candidate/submission CSV were created.\n")
    lines.append("## Safety flags\n")
    lines.append("- validation_only: true")
    lines.append("- kaggle_submit_executed: false")
    lines.append("- candidate_csv_written: false")
    lines.append("- hidden_label_access: false")
    lines.append("- external_steam_scraping: false")
    lines.append("- text source: `data/raw/public/data/train.json` joined only by each fold's `train_interactions.csv.row_idx`\n")
    lines.append("## Coverage\n")
    lines.append("| split | base acc | items w/text | emb dim | bin covered rows | missing user | missing item | reused |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for sr in split_results:
        cov = sr["coverage_binary"]
        lines.append(
            f"| {sr['split']} | {sr['base_accuracy']:.5f} | {sr['items_with_text_profile']} | {sr['embedding_dim']} | "
            f"{cov['covered_rows']} | {cov['missing_user_rows']} | {cov['missing_item_rows']} | {sr['embedding_reused']} |"
        )
    lines.append("\n## Top variants by mean delta\n")
    lines.append("| variant | mean Δ | pos splits | fixes | breaks | Fisher p | strict pass |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in top.iterrows():
        lines.append(
            f"| `{r['variant']}` | {r['mean_delta']:+.6f} | {int(r['positive_splits'])}/3 | "
            f"{int(r['total_fixes'])} | {int(r['total_breaks'])} | {float(r['fisher_mcnemar_p']):.4g} | {bool(r['pass_strict_mde'])} |"
        )
    best = agg.iloc[0].to_dict() if len(agg) else {}
    strict_any = bool(agg["pass_strict_mde"].any()) if len(agg) else False
    verdict = "STRICT_PASS" if strict_any else "NO_STRICT_PASS"
    lines.append("\n## Verdict\n")
    lines.append(f"- verdict: `{verdict}`")
    if best:
        lines.append(f"- best variant: `{best['variant']}`")
        lines.append(f"- best mean Δ: {best['mean_delta']:+.6f} versus MDE {MDE:+.5f}")
        lines.append(f"- best positive splits: {int(best['positive_splits'])}/3")
        lines.append(f"- best flips: fixes={int(best['total_fixes'])}, breaks={int(best['total_breaks'])}")
    lines.append("\n## Notes\n")
    lines.append("- `sem_*_resid` variants remove linear association with within-user base score and log-popularity before blending.")
    lines.append("- The grid is exploratory/predeclared for validation triage only; below-MDE positives are manual-risk/no-submit signals, not candidates.")
    lines.append("- Output score artifacts are validation-labeled diagnostics under `artifacts/semantic_residual_probe/`, not submission-like `ID,Label` files.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B", help="HF SentenceTransformer model or tfidf-svd-smoke")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--reviews-per-item", type=int, default=8)
    ap.add_argument("--max-review-chars", type=int, default=512)
    ap.add_argument("--min-review-chars", type=int, default=20)
    ap.add_argument("--tfidf-dim", type=int, default=256)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--reuse-embeddings", action="store_true")
    ap.add_argument("--out-dir", default="artifacts/semantic_residual_probe")
    ap.add_argument("--report-prefix", default="reports/20260602_semantic_residual_probe")
    args = ap.parse_args()

    model_slug = slugify_model_name(args.model_name)
    print(f"[semantic] model={args.model_name} slug={model_slug} splits={args.splits}", flush=True)
    raw_texts = load_raw_text_table()
    split_results = []
    for split in args.splits:
        print(f"[semantic] split={split} start", flush=True)
        sr = run_split(args, split, raw_texts, model_slug)
        split_results.append(sr)
        best_split = max(sr["split_metrics"], key=lambda x: x["delta"])
        print(f"[semantic] split={split} best={best_split['variant']} delta={best_split['delta']:+.6f}", flush=True)

    split_df, agg = aggregate(split_results)
    out_root = ensure_dir(ROOT / args.out_dir / model_slug)
    split_df.to_csv(out_root / "semantic_residual_split_metrics.csv", index=False)
    agg.to_csv(out_root / "semantic_residual_aggregate_metrics.csv", index=False)

    report_json = ROOT / f"{args.report_prefix}_{model_slug}.json"
    report_md = ROOT / f"{args.report_prefix}_{model_slug}.md"
    best = agg.iloc[0].to_dict() if len(agg) else {}
    summary = {
        "model_name": args.model_name,
        "model_slug": model_slug,
        "splits": args.splits,
        "mde": MDE,
        "safety": {
            "validation_only": True,
            "kaggle_submit_executed": False,
            "candidate_csv_written": False,
            "hidden_label_access": False,
            "external_steam_scraping": False,
        },
        "best_variant": best,
        "strict_pass_any": bool(agg["pass_strict_mde"].any()) if len(agg) else False,
        "split_results": split_results,
        "artifacts_dir": str(out_root.relative_to(ROOT)),
    }
    write_json(report_json, summary)
    write_markdown_report(report_md, args, model_slug, split_results, split_df, agg)
    print(f"[semantic] report_json={report_json}", flush=True)
    print(f"[semantic] report_md={report_md}", flush=True)
    if best:
        print(f"[semantic] BEST {best['variant']} mean_delta={best['mean_delta']:+.6f} pos={best['positive_splits']}/3 strict={best['pass_strict_mde']}", flush=True)


if __name__ == "__main__":
    main()
