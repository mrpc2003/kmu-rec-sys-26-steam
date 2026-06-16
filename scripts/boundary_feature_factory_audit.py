#!/usr/bin/env python3
"""Boundary Feature Factory Audit (validation-only, no-submit).

목적
----
마지막 제출권을 쓰기 전에, 이미 닫힌 LightGCN/후처리 축 밖에 남은
"경계(row rank K/K+1) 오답을 설명하는 train-only residual feature"가
실제로 존재하는지 3개 uniform validation split으로 검증한다.

핵심 안전 규칙
--------------
* real hidden test/pairs는 읽지 않는다. artifacts/validation/* 의 synthetic validation만 사용한다.
* Kaggle 제출 파일을 만들지 않는다. 출력은 artifacts/boundary_feature_factory 및 reports 뿐이다.
* public-transfer 가능한 후보로 인정하려면 fixed unsupervised variant가 3-split aggregate gate를
  통과해야 한다. cross-fit/label-trained 결과는 "headroom 진단"으로만 보고, 제출 후보가 아니다.

검증 구성
---------
1. emb128 LightGCN 4-seed raw-mean base 재현: split별 canonical accuracy 확인.
2. train-only feature factory:
   - item/user aggregate: pop, hours, text length, funny, early access, date.
   - user-item compatibility: hours/date/textlen closeness, hours product.
   - conditional graph residual: user's train history와 후보 item의 co-occurrence/cosine coverage.
3. residualization:
   - 모든 novelty feature는 within-user z 후, base score z와 log-pop z를 전역 OLS로 제거하고
     다시 within-user z로 바꾼다. 즉 model-circular + popularity-trap 성분을 최대한 제거한다.
4. fixed unsupervised variants:
   - score = z_user(base) + lambda * z_user(residual_feature), lambda in {0.05, 0.10, 0.20}.
   - feature 방향은 사전에 "값이 클수록 더 positive"가 되도록 정의한다. negative sign sweep은
     제출 후보로 쓰지 않고 cross-fit 진단에만 쓴다.
5. boundary diagnostics:
   - base rank K/K+1 pair에서 feature difference가 selected-side positive 여부를 맞히는 AUC.
6. cross-fit discovery diagnostics:
   - per-feature lambda sign/size는 validation user-half에서만 고르고 반대 half에 적용한다.
   - integrated ridge도 user-half cross-fit으로만 평가한다. 이는 hidden test에 deploy 불가하므로
     "실마리 존재 여부" 확인용이며, fixed gate와 분리해 보고한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from recsys_played_utils import evaluate_tophalf, ensure_dir, load_train_interactions  # noqa: E402

SPLITS = [
    "val_random_uniform_seed42",
    "val_random_uniform_seed7",
    "val_random_uniform_seed123",
]
MODEL_SEEDS = [42, 7, 123, 2024]
EXPECTED_BASE_ACCURACY = {
    "val_random_uniform_seed42": 0.7650530106021204,
    "val_random_uniform_seed7": 0.7609521904380876,
    "val_random_uniform_seed123": 0.7599519903980796,
}
MDE = 0.00355
FIXED_LAMBDAS = [0.05, 0.10, 0.20]
CROSSFIT_LAMBDAS = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]
RIDGE_ALPHA = 5.0
BASE_SCORE_COL = "score_base_mean"
BASE_Z_COL = "z_base_user"
LOGPOP_Z_COL = "z_feat_item_logpop"

# Raw feature definitions whose sign is pre-normalized to "higher => more plausible positive".
RAW_FEATURES = [
    "feat_item_logpop",              # trap/control; excluded from novelty pass
    "feat_item_hours_mean",
    "feat_item_hours_median",
    "feat_item_hours_std",
    "feat_item_textlen_mean",
    "feat_item_funny_rate",
    "feat_item_early_access_rate",
    "feat_item_date_mean",
    "feat_item_date_std",
    "feat_hours_compat",            # -abs(user mean hours - item mean hours)
    "feat_date_compat",             # -abs(user mean date - item mean date)
    "feat_textlen_compat",          # -abs(user mean text length - item mean text length)
    "feat_hours_prod",
    "feat_cooc_raw_sum",
    "feat_cooc_norm_sum",
    "feat_cooc_norm_mean",
    "feat_cooc_norm_max",
    "feat_cooc_coverage",
]
TRAP_OR_CIRCULAR = {"feat_item_logpop"}


@dataclass
class MetricBundle:
    accuracy: float
    delta: float
    fixes: int
    breaks: int
    discordant: int
    p_value: float
    changed_predictions: int
    direction_positive: bool


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True, default=str), encoding="utf-8")


def exact_two_sided_binom_p(k: int, n: int) -> float:
    """Exact two-sided binomial p-value for H0 p=0.5."""
    if n <= 0:
        return 1.0
    kk = min(k, n - k)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2.0) for i in range(kk + 1)]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(v - m) for v in logs)
    return min(1.0, 2.0 * tail)


def source_path_for(split: str, seed: int) -> tuple[Path, str]:
    """Return the validation-only score source for canonical emb128/L4/reg=1e-3."""
    if split == "val_random_uniform_seed42":
        if seed == 42:
            # This is the same seed42 checkpoint as the canonical base, with an already verified
            # ordinary uniform layer average score column. It keeps seed42 aligned with prior panels.
            return ROOT / "artifacts/layermix_probe/emb128_L4_r3_seed42/layermix_validation_scores.csv", "score_layermix_uniform"
        return ROOT / f"artifacts/lightgcn_emb128L4r3_ens/seed{seed}/{split}/lightgcn_scores.csv", "score_lightgcn"
    return ROOT / f"artifacts/split_panel_emb128/{split}/seed{seed}/lightgcn_scores.csv", "score_lightgcn"


def load_split_scores(split: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for seed in MODEL_SEEDS:
        path, score_col = source_path_for(split, seed)
        low = str(path).lower()
        if any(token in low for token in ["hidden", "leaderboard"]):
            raise RuntimeError(f"Forbidden score source path: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Missing score file for {split} seed={seed}: {path}")
        df = pd.read_csv(path)
        if score_col not in df.columns:
            raise ValueError(f"Missing score column {score_col} in {path}")
        required_identity_cols = ["ID", "userID", "gameID", "Label"]
        missing_identity = [c for c in required_identity_cols if c not in df.columns]
        if missing_identity:
            raise ValueError(f"Missing identity columns {missing_identity} in {path}")
        if merged is None:
            part = df[["ID", "userID", "gameID", "Label", score_col]].copy()
        else:
            # Hard safety check: score files must refer to the exact same validation rows, not just
            # an accidentally matching ID sequence.
            identity = df[["ID", "userID", "gameID", "Label"]].copy()
            chk = merged[["ID", "userID", "gameID", "Label"]].merge(
                identity,
                on="ID",
                how="inner",
                suffixes=("_base", "_seed"),
                validate="one_to_one",
            )
            if len(chk) != len(merged):
                raise RuntimeError(f"Identity alignment changed for {split} seed={seed}: {len(merged)}->{len(chk)}")
            for c in ["userID", "gameID", "Label"]:
                if not (chk[f"{c}_base"].astype(str).to_numpy() == chk[f"{c}_seed"].astype(str).to_numpy()).all():
                    raise RuntimeError(f"Identity mismatch for {split} seed={seed} column={c}")
            part = df[["ID", score_col]].copy()
        part = part.rename(columns={score_col: f"score_seed{seed}"})
        if merged is None:
            merged = part
        else:
            before = len(merged)
            merged = merged.merge(part, on="ID", how="inner", validate="one_to_one")
            if len(merged) != before:
                raise RuntimeError(f"Row alignment changed while merging {split} seed={seed}: {before}->{len(merged)}")
    assert merged is not None
    merged = merged.sort_values("ID", kind="mergesort").reset_index(drop=True)
    merged["ID"] = merged["ID"].astype(int)
    merged["Label"] = merged["Label"].astype(int)
    merged[BASE_SCORE_COL] = merged[[f"score_seed{s}" for s in MODEL_SEEDS]].mean(axis=1)
    return merged


def stable_user_fold(user_id: str) -> int:
    h = hashlib.blake2b(str(user_id).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(h, byteorder="little") % 2


def within_user_z(df: pd.DataFrame, col: str) -> np.ndarray:
    grouped = df.groupby("userID", sort=False)[col]
    mean = grouped.transform("mean").to_numpy(dtype=float)
    std = grouped.transform(lambda s: float(s.std(ddof=0))).to_numpy(dtype=float)
    x = df[col].to_numpy(dtype=float)
    out = np.zeros(len(df), dtype=float)
    mask = std > 1e-12
    out[mask] = (x[mask] - mean[mask]) / std[mask]
    out[~np.isfinite(out)] = 0.0
    return out


def global_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu = float(out.mean()) if len(out) else 0.0
    sd = float(out.std(ddof=0)) if len(out) else 0.0
    if sd <= 1e-12:
        return np.zeros_like(out)
    return (out - mu) / sd


def residualize_against_controls(feature_z: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    y = np.asarray(feature_z, dtype=float)
    cols = [np.ones(len(y), dtype=float)] + [np.asarray(c, dtype=float) for c in controls]
    X = np.column_stack(cols)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    resid = np.zeros(len(y), dtype=float)
    if int(mask.sum()) >= X.shape[1] + 2:
        coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        resid = y - X @ coef
    resid[~np.isfinite(resid)] = 0.0
    return resid


def validate_per_user_half(df: pd.DataFrame, split: str) -> None:
    g = df.groupby("userID", sort=False).agg(n=("ID", "size"), positives=("Label", "sum"))
    bad = g[(g["n"] % 2 != 0) | (g["positives"] * 2 != g["n"])]
    if not bad.empty:
        raise RuntimeError(f"{split}: invalid per-user 1:1 validation cardinality for {len(bad)} users")


def auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = labels == 1
    neg = labels == 0
    npos = int(pos.sum())
    nneg = int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    ss = scores[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (ranks[order[i]] + ranks[order[j]]) / 2.0
        i = j + 1
    return float((ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def evaluate_variant(df: pd.DataFrame, score_col: str, base_pred: pd.DataFrame) -> tuple[MetricBundle, pd.DataFrame]:
    summary, pred = evaluate_tophalf(df, score_col, label_col="Label", user_col="userID", id_col="ID")
    pred = pred.sort_values("ID", kind="mergesort")
    base_aligned = base_pred.sort_values("ID", kind="mergesort")
    if not np.array_equal(pred["ID"].to_numpy(), base_aligned["ID"].to_numpy()):
        raise RuntimeError("Prediction row alignment mismatch")
    base_ok = base_aligned["Correct"].astype(bool).to_numpy()
    var_ok = pred["Correct"].astype(bool).to_numpy()
    fixes = int((~base_ok & var_ok).sum())
    breaks = int((base_ok & ~var_ok).sum())
    discordant = fixes + breaks
    changed = int((pred["Pred"].astype(int).to_numpy() != base_aligned["Pred"].astype(int).to_numpy()).sum())
    base_acc = float(base_ok.mean())
    acc = float(summary["row_accuracy"])
    mb = MetricBundle(
        accuracy=acc,
        delta=acc - base_acc,
        fixes=fixes,
        breaks=breaks,
        discordant=discordant,
        p_value=exact_two_sided_binom_p(fixes, discordant),
        changed_predictions=changed,
        direction_positive=fixes > breaks,
    )
    return mb, pred


def metric_to_dict(m: MetricBundle) -> dict[str, Any]:
    return {
        "accuracy": m.accuracy,
        "delta": m.delta,
        "fixes": m.fixes,
        "breaks": m.breaks,
        "discordant": m.discordant,
        "paired_exact_binom_p_two_sided": m.p_value,
        "changed_predictions": m.changed_predictions,
        "direction_positive": m.direction_positive,
    }


def build_train_only_features(df: pd.DataFrame, fold_train: pd.DataFrame, split: str) -> pd.DataFrame:
    out = df.copy()
    tr = fold_train.copy()
    tr["date_ord"] = pd.to_datetime(tr["date"], errors="coerce").map(lambda d: float(d.toordinal()) if pd.notna(d) else np.nan)
    tr["funny_pos"] = (tr.get("found_funny", 0.0).fillna(0.0).astype(float) > 0).astype(float)
    tr["early_float"] = tr.get("early_access", False).astype(float)

    item_agg = tr.groupby("gameID").agg(
        item_count=("gameID", "size"),
        item_hours_mean=("hours_transformed", "mean"),
        item_hours_median=("hours_transformed", "median"),
        item_hours_std=("hours_transformed", "std"),
        item_textlen_mean=("text_len", "mean"),
        item_funny_rate=("funny_pos", "mean"),
        item_early_access_rate=("early_float", "mean"),
        item_date_mean=("date_ord", "mean"),
        item_date_std=("date_ord", "std"),
    )
    user_agg = tr.groupby("userID").agg(
        user_count=("userID", "size"),
        user_hours_mean=("hours_transformed", "mean"),
        user_textlen_mean=("text_len", "mean"),
        user_date_mean=("date_ord", "mean"),
    )

    # Fill missing aggregate values with neutral global values. Missing item means item has no fold-train
    # evidence after holdout; those rows should not create artificial extremes.
    item_fill = item_agg.median(numeric_only=True).to_dict()
    item_fill["item_count"] = 0.0
    item_agg = item_agg.fillna(item_fill)
    user_agg = user_agg.fillna(user_agg.median(numeric_only=True))

    out = out.merge(item_agg, left_on="gameID", right_index=True, how="left")
    out = out.merge(user_agg, left_on="userID", right_index=True, how="left")
    for col, val in item_fill.items():
        if col in out.columns:
            out[col] = out[col].fillna(val)
    for col in ["user_count", "user_hours_mean", "user_textlen_mean", "user_date_mean"]:
        out[col] = out[col].fillna(float(user_agg[col].median()) if col in user_agg else 0.0)

    out["feat_item_logpop"] = np.log1p(out["item_count"].to_numpy(dtype=float))
    out["feat_item_hours_mean"] = out["item_hours_mean"].astype(float)
    out["feat_item_hours_median"] = out["item_hours_median"].astype(float)
    out["feat_item_hours_std"] = out["item_hours_std"].fillna(0.0).astype(float)
    out["feat_item_textlen_mean"] = out["item_textlen_mean"].astype(float)
    out["feat_item_funny_rate"] = out["item_funny_rate"].astype(float)
    out["feat_item_early_access_rate"] = out["item_early_access_rate"].astype(float)
    out["feat_item_date_mean"] = out["item_date_mean"].astype(float)
    out["feat_item_date_std"] = out["item_date_std"].fillna(0.0).astype(float)
    out["feat_hours_compat"] = -np.abs(out["user_hours_mean"].astype(float) - out["item_hours_mean"].astype(float))
    out["feat_date_compat"] = -np.abs(out["user_date_mean"].astype(float) - out["item_date_mean"].astype(float)) / 365.25
    out["feat_textlen_compat"] = -np.abs(out["user_textlen_mean"].astype(float) - out["item_textlen_mean"].astype(float))
    out["feat_hours_prod"] = out["user_hours_mean"].astype(float) * out["item_hours_mean"].astype(float)

    print(f"[bffa:{split}] building sparse co-occurrence features...", flush=True)
    users = np.sort(tr["userID"].unique())
    items = np.sort(tr["gameID"].unique())
    u2r = {u: i for i, u in enumerate(users)}
    i2r = {g: i for i, g in enumerate(items)}
    row = tr["userID"].map(u2r).to_numpy(dtype=np.int32)
    col = tr["gameID"].map(i2r).to_numpy(dtype=np.int32)
    R = sp.csr_matrix((np.ones(len(row), dtype=np.float32), (row, col)), shape=(len(users), len(items)), dtype=np.float32)
    R.data[:] = 1.0
    R.eliminate_zeros()
    C = (R.T @ R).tocsr().astype(np.float32)
    C.setdiag(0)
    C.eliminate_zeros()
    item_pop = np.asarray(R.sum(axis=0)).ravel().astype(float)
    scale = 1.0 / np.sqrt(item_pop + 1.0)
    C_norm = C.multiply(scale[:, None]).multiply(scale[None, :]).tocsr()
    user_hist = {
        ur: set(R.indices[R.indptr[ur] : R.indptr[ur + 1]].tolist())
        for ur in range(R.shape[0])
    }

    def cooc_stats(uid: str, gid: str) -> tuple[float, float, float, float, float]:
        ur = u2r.get(uid)
        ir = i2r.get(gid)
        if ur is None or ir is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        hist = user_hist.get(ur, set())
        if not hist:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        row_raw = C.getrow(ir)
        row_norm = C_norm.getrow(ir)
        if row_raw.nnz == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        mask_raw = np.fromiter((j in hist for j in row_raw.indices), dtype=bool, count=len(row_raw.indices))
        if not mask_raw.any():
            return 0.0, 0.0, 0.0, 0.0, 0.0
        raw_vals = row_raw.data[mask_raw].astype(float)
        # C and C_norm share sparsity after diagonal removal, but use separate mask for safety.
        mask_norm = np.fromiter((j in hist for j in row_norm.indices), dtype=bool, count=len(row_norm.indices))
        norm_vals = row_norm.data[mask_norm].astype(float) if mask_norm.any() else np.array([], dtype=float)
        raw_sum = float(raw_vals.sum())
        norm_sum = float(norm_vals.sum()) if len(norm_vals) else 0.0
        norm_mean = float(norm_sum / max(1, len(hist)))
        norm_max = float(norm_vals.max()) if len(norm_vals) else 0.0
        coverage = float(len(norm_vals) / max(1, len(hist)))
        return raw_sum, norm_sum, norm_mean, norm_max, coverage

    cooc_records = [cooc_stats(u, g) for u, g in zip(out["userID"].tolist(), out["gameID"].tolist(), strict=True)]
    cooc_arr = np.asarray(cooc_records, dtype=float)
    out["feat_cooc_raw_sum"] = np.log1p(cooc_arr[:, 0])
    out["feat_cooc_norm_sum"] = cooc_arr[:, 1]
    out["feat_cooc_norm_mean"] = cooc_arr[:, 2]
    out["feat_cooc_norm_max"] = cooc_arr[:, 3]
    out["feat_cooc_coverage"] = cooc_arr[:, 4]
    return out


def add_standardized_and_residual_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df.copy()
    out[BASE_Z_COL] = within_user_z(out, BASE_SCORE_COL)
    seed_cols = [f"score_seed{s}" for s in MODEL_SEEDS]
    out["score_seed_std"] = out[seed_cols].std(axis=1, ddof=0)
    for seed in MODEL_SEEDS:
        out[f"rank_seed{seed}"] = out.groupby("userID", sort=False)[f"score_seed{seed}"].rank(method="first", ascending=False)
    out["n_user"] = out.groupby("userID", sort=False)["ID"].transform("size").astype(int)
    out["h_user"] = out["n_user"] // 2
    out["seed_vote_tophalf"] = 0
    for seed in MODEL_SEEDS:
        out["seed_vote_tophalf"] += (out[f"rank_seed{seed}"] <= out["h_user"]).astype(int)
    out["feat_model_seed_std_neg"] = -out["score_seed_std"].astype(float)  # higher = more stable/confident
    out["feat_model_vote"] = out["seed_vote_tophalf"].astype(float)

    all_raw = RAW_FEATURES + ["feat_model_seed_std_neg", "feat_model_vote"]
    for feat in all_raw:
        out[f"z_{feat}"] = within_user_z(out, feat)
    if LOGPOP_Z_COL not in out.columns:
        raise RuntimeError("logpop z feature missing")

    # Residualize novelty features against base score and log-pop. The model-derived controls are
    # retained only as diagnostics and are not treated as orthogonal public-transferable features.
    controls = [out[BASE_Z_COL].to_numpy(dtype=float), out[LOGPOP_Z_COL].to_numpy(dtype=float)]
    residual_cols: list[str] = []
    raw_z_cols: list[str] = []
    for feat in all_raw:
        zcol = f"z_{feat}"
        raw_z_cols.append(zcol)
        resid = residualize_against_controls(out[zcol].to_numpy(dtype=float), controls)
        tmp_col = f"tmp_resid_{feat}"
        out[tmp_col] = resid
        rcol = f"zr_{feat}"
        out[rcol] = within_user_z(out, tmp_col)
        out = out.drop(columns=[tmp_col])
        residual_cols.append(rcol)
    return out, raw_z_cols, residual_cols


def boundary_auc_table(df: pd.DataFrame, feature_cols: list[str]) -> list[dict[str, Any]]:
    ranked = df.sort_values(["userID", BASE_SCORE_COL, "ID"], ascending=[True, False, True], kind="mergesort").copy()
    ranked["rank_base"] = ranked.groupby("userID", sort=False).cumcount() + 1
    ku = ranked.groupby("userID", sort=False)["Label"].sum().astype(int).to_dict()
    ranked["Ku"] = ranked["userID"].map(ku)
    left_cols = ["userID", "ID", "gameID", "Label", *feature_cols]
    right_cols = ["userID", "ID", "gameID", "Label", *feature_cols]
    sel = ranked[ranked["rank_base"] == ranked["Ku"]][left_cols].copy()
    rej = ranked[ranked["rank_base"] == ranked["Ku"] + 1][right_cols].copy()
    rename_sel = {"ID": "ID_in", "gameID": "gameID_in", "Label": "label_in"} | {c: f"{c}_in" for c in feature_cols}
    rename_rej = {"ID": "ID_out", "gameID": "gameID_out", "Label": "label_out"} | {c: f"{c}_out" for c in feature_cols}
    bnd = sel.rename(columns=rename_sel).merge(rej.rename(columns=rename_rej), on="userID", how="inner")
    bnd = bnd[(bnd["label_in"] + bnd["label_out"]) == 1].copy()
    bnd["selected_side_positive"] = (bnd["label_in"] == 1).astype(int)
    rows = []
    y = bnd["selected_side_positive"].to_numpy(dtype=int)
    for c in feature_cols:
        diff = bnd[f"{c}_in"].to_numpy(dtype=float) - bnd[f"{c}_out"].to_numpy(dtype=float)
        auc = auc_rank(diff, y)
        rows.append(
            {
                "feature_col": c,
                "boundary_auc_selected_positive": auc,
                "abs_dev_from_half": abs(auc - 0.5) if auc == auc else None,
                "orientation": "supports_base_selected" if auc >= 0.5 else "supports_rejected_side",
                "informative_pairs": int(len(bnd)),
            }
        )
    return rows


def score_and_metric(df: pd.DataFrame, base_pred: pd.DataFrame, score_values: np.ndarray, score_name: str) -> MetricBundle:
    tmp = df[["ID", "userID", "gameID", "Label"]].copy()
    tmp[score_name] = score_values
    metric, _ = evaluate_variant(tmp, score_name, base_pred)
    return metric


def crossfit_single_feature(df: pd.DataFrame, base_pred: pd.DataFrame, feature_col: str) -> dict[str, Any]:
    folds = df["userID"].map(stable_user_fold).to_numpy(dtype=int)
    score = np.zeros(len(df), dtype=float)
    fold_choices = []
    for heldout_fold in [0, 1]:
        train_mask = folds != heldout_fold
        eval_mask = folds == heldout_fold
        best: tuple[float, float] | None = None
        for lam in CROSSFIT_LAMBDAS:
            s_train = df.loc[train_mask, BASE_Z_COL].to_numpy(dtype=float) + lam * df.loc[train_mask, feature_col].to_numpy(dtype=float)
            train_df = df.loc[train_mask, ["ID", "userID", "gameID", "Label"]].copy()
            train_df["score_tmp"] = s_train
            base_train_pred = base_pred[base_pred["ID"].isin(train_df["ID"])].copy()
            m, _ = evaluate_variant(train_df, "score_tmp", base_train_pred)
            if best is None or m.delta > best[1]:
                best = (lam, m.delta)
        assert best is not None
        lam = best[0]
        score[eval_mask] = df.loc[eval_mask, BASE_Z_COL].to_numpy(dtype=float) + lam * df.loc[eval_mask, feature_col].to_numpy(dtype=float)
        fold_choices.append({"heldout_fold": heldout_fold, "lambda_chosen_on_other_fold": lam, "train_delta": best[1]})
    metric = score_and_metric(df, base_pred, score, f"score_crossfit_{feature_col}")
    return {"feature_col": feature_col, "fold_choices": fold_choices, **metric_to_dict(metric)}


def crossfit_integrated_ridge(df: pd.DataFrame, base_pred: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    folds = df["userID"].map(stable_user_fold).to_numpy(dtype=int)
    X_all = np.column_stack([df[BASE_Z_COL].to_numpy(dtype=float)] + [df[c].to_numpy(dtype=float) for c in feature_cols])
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    y_all = df["Label"].to_numpy(dtype=float) - 0.5
    score = np.zeros(len(df), dtype=float)
    fold_weights = []
    for heldout_fold in [0, 1]:
        train_mask = folds != heldout_fold
        eval_mask = folds == heldout_fold
        X = X_all[train_mask]
        y = y_all[train_mask]
        # Do not regularize intercept because there is no intercept; columns are already centered-ish.
        XtX = X.T @ X + RIDGE_ALPHA * np.eye(X.shape[1])
        Xty = X.T @ y
        try:
            w = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        # Preserve the base as anchor by using learned residual only at small calibrated scale.
        residual_score = X_all[eval_mask] @ w
        residual_score = global_z(residual_score)
        score[eval_mask] = df.loc[eval_mask, BASE_Z_COL].to_numpy(dtype=float) + 0.10 * residual_score
        top_weights = sorted(
            [
                {"col": col, "weight": float(val)}
                for col, val in zip([BASE_Z_COL] + feature_cols, w, strict=True)
            ],
            key=lambda r: abs(r["weight"]),
            reverse=True,
        )[:12]
        fold_weights.append({"heldout_fold": heldout_fold, "top_weights": top_weights})
    metric = score_and_metric(df, base_pred, score, "score_integrated_ridge_cf")
    return {"feature_cols": feature_cols, "fold_weights": fold_weights, **metric_to_dict(metric)}


def run_split(split: str, out_dir: Path) -> dict[str, Any]:
    print(f"\n[bffa] === split {split} ===", flush=True)
    split_out = ensure_dir(out_dir / split)
    scores = load_split_scores(split)
    validate_per_user_half(scores, split)
    base_summary, base_pred = evaluate_tophalf(scores, BASE_SCORE_COL, label_col="Label", user_col="userID", id_col="ID")
    base_acc = float(base_summary["row_accuracy"])
    expected = EXPECTED_BASE_ACCURACY[split]
    if abs(base_acc - expected) > 1e-9:
        raise RuntimeError(f"{split}: base accuracy mismatch {base_acc:.12f} != expected {expected:.12f}")
    print(f"[bffa:{split}] base acc reproduced = {base_acc:.5f}", flush=True)

    fold_path = ROOT / "artifacts/validation" / split / "train_interactions.csv"
    if not fold_path.exists():
        raise FileNotFoundError(f"Missing fold train file: {fold_path}")
    fold_train = load_train_interactions(fold_path)
    feat_df = build_train_only_features(scores, fold_train, split)
    feat_df, raw_z_cols, residual_cols = add_standardized_and_residual_features(feat_df)

    # Novel residual columns exclude logpop trap and model-derived circular diagnostics.
    novel_residual_cols = [
        f"zr_{feat}"
        for feat in RAW_FEATURES
        if feat not in TRAP_OR_CIRCULAR
    ]
    control_cols = ["zr_feat_item_logpop", "zr_feat_model_seed_std_neg", "zr_feat_model_vote"]
    diagnostic_cols = novel_residual_cols + control_cols

    boundary_rows = boundary_auc_table(feat_df, diagnostic_cols)
    pd.DataFrame(boundary_rows).to_csv(split_out / "boundary_auc_table.csv", index=False)

    fixed_rows = []
    for rcol in novel_residual_cols:
        for lam in FIXED_LAMBDAS:
            score = feat_df[BASE_Z_COL].to_numpy(dtype=float) + lam * feat_df[rcol].to_numpy(dtype=float)
            metric = score_and_metric(feat_df, base_pred, score, f"score_fixed_{rcol}_{lam}")
            fixed_rows.append({"variant_id": f"fixed__{rcol}__lam{lam:.2f}", "feature_col": rcol, "lambda": lam, **metric_to_dict(metric)})
    fixed_df = pd.DataFrame(fixed_rows).sort_values(["delta", "fixes"], ascending=[False, False], kind="mergesort")
    fixed_df.to_csv(split_out / "fixed_unsupervised_metrics.csv", index=False)

    crossfit_rows = []
    for rcol in novel_residual_cols:
        crossfit_rows.append(crossfit_single_feature(feat_df, base_pred, rcol))
    crossfit_df = pd.DataFrame(crossfit_rows).sort_values(["delta", "fixes"], ascending=[False, False], kind="mergesort")
    crossfit_df.to_csv(split_out / "crossfit_single_feature_metrics.csv", index=False)

    integrated = crossfit_integrated_ridge(feat_df, base_pred, novel_residual_cols)

    # Compact summary only; no per-row feature dump to avoid accidental leakage-like artifacts.
    summary = {
        "split": split,
        "base_accuracy": base_acc,
        "rows": int(len(feat_df)),
        "users": int(feat_df["userID"].nunique()),
        "feature_counts": {
            "raw_features": len(RAW_FEATURES),
            "novel_residual_features": len(novel_residual_cols),
            "diagnostic_cols": len(diagnostic_cols),
        },
        "top_boundary_auc_absdev": sorted(boundary_rows, key=lambda r: (r["abs_dev_from_half"] or 0.0), reverse=True)[:10],
        "top_fixed_unsupervised": fixed_df.head(12).to_dict(orient="records"),
        "top_crossfit_single_feature": crossfit_df.head(12).to_dict(orient="records"),
        "integrated_ridge_crossfit": integrated,
        "output_files": {
            "boundary_auc_table": str(split_out / "boundary_auc_table.csv"),
            "fixed_unsupervised_metrics": str(split_out / "fixed_unsupervised_metrics.csv"),
            "crossfit_single_feature_metrics": str(split_out / "crossfit_single_feature_metrics.csv"),
        },
    }
    write_json(split_out / "summary.json", summary)
    print(f"[bffa:{split}] best fixed Δ={fixed_df.iloc[0]['delta']:+.5f}; best crossfit Δ={crossfit_df.iloc[0]['delta']:+.5f}; integrated Δ={integrated['delta']:+.5f}", flush=True)
    return summary


def aggregate_metric_rows(rows: list[dict[str, Any]], key_col: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out_rows = []
    for key, g in df.groupby(key_col, sort=False):
        fixes = int(g["fixes"].sum())
        breaks = int(g["breaks"].sum())
        discordant = fixes + breaks
        deltas = g["delta"].astype(float).to_numpy()
        out_rows.append(
            {
                key_col: key,
                "feature_col": g["feature_col"].iloc[0] if "feature_col" in g.columns else key,
                "mean_delta": float(deltas.mean()),
                "min_delta": float(deltas.min()),
                "max_delta": float(deltas.max()),
                "positive_splits": int((deltas > 0).sum()),
                "fixes": fixes,
                "breaks": breaks,
                "discordant": discordant,
                "paired_exact_binom_p_two_sided": exact_two_sided_binom_p(fixes, discordant),
                "splits": g["split"].tolist(),
                "split_deltas": [float(x) for x in deltas],
            }
        )
    return pd.DataFrame(out_rows).sort_values(["mean_delta", "fixes"], ascending=[False, False], kind="mergesort")


def write_markdown_report(report_path: Path, aggregate: dict[str, Any]) -> None:
    fixed = pd.DataFrame(aggregate["fixed_gate_aggregate"])
    cross = pd.DataFrame(aggregate["crossfit_headroom_aggregate"])
    integ = pd.DataFrame(aggregate["integrated_ridge_by_split"])
    bauc = pd.DataFrame(aggregate["boundary_auc_aggregate"])

    def table(df: pd.DataFrame, cols: list[str], n: int = 12) -> str:
        """Tiny dependency-free Markdown table writer (avoid optional tabulate dependency)."""
        if df.empty:
            return "(empty)"
        d = df[cols].head(n).copy()
        for c in d.columns:
            if pd.api.types.is_float_dtype(d[c]):
                d[c] = d[c].map(lambda x: f"{x:+.5f}" if "delta" in c or c in {"mean_delta", "min_delta", "max_delta"} else f"{x:.5f}")
            else:
                d[c] = d[c].astype(str)
        headers = [str(c) for c in d.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in d.iterrows():
            vals = [str(row[c]).replace("|", "\\|") for c in d.columns]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    pass_fixed = aggregate["fixed_gate_pass_variants"]
    pass_cross = aggregate["crossfit_headroom_pass_features"]
    verdict = aggregate["verdict"]
    text = f"""# Boundary Feature Factory Audit — 3-uniform validation panel

## Verdict

**{verdict}**

- Fixed unsupervised gate pass variants: `{len(pass_fixed)}`
- Cross-fit headroom pass features: `{len(pass_cross)}` (diagnostic only; not submission-ready)
- Gate: mean Δ ≥ {MDE:.5f}, fixes > breaks, p < 0.05, positive split count ≥ 2/3.
- Safety: validation artifacts only; no Kaggle submission file is generated by this script.

## Fixed unsupervised variants (submission-candidate gate)

{table(fixed, ["variant_id", "feature_col", "mean_delta", "min_delta", "max_delta", "positive_splits", "fixes", "breaks", "paired_exact_binom_p_two_sided"], 18)}

## Cross-fit single-feature discovery (headroom diagnostic only)

{table(cross, ["feature_col", "mean_delta", "min_delta", "max_delta", "positive_splits", "fixes", "breaks", "paired_exact_binom_p_two_sided"], 18)}

## Integrated ridge cross-fit by split (headroom diagnostic only)

{table(integ, ["split", "delta", "fixes", "breaks", "paired_exact_binom_p_two_sided"], 10)}

## Boundary K/K+1 feature AUC aggregate

AUC는 base가 선택한 K번째 row와 탈락한 K+1번째 row 중 선택된 쪽이 실제 positive인지 예측하는 정도다.
`zr_*`는 base score와 log-pop 성분을 제거한 residual feature다.

{table(bauc, ["feature_col", "mean_abs_dev_from_half", "mean_auc", "max_abs_dev_from_half", "splits_evaluated"], 18)}

## Notes

- `feat_item_logpop`은 control/trap으로만 사용했고 novelty gate에서 제외했다.
- `feat_model_*`은 model-circular diagnostic으로만 사용했고 fixed candidate gate에서 제외했다.
- cross-fit 결과가 좋아도 real hidden test에서는 validation label로 lambda/weights를 학습할 수 없으므로 제출 후보가 아니다. 실제 후보는 fixed unsupervised table의 gate pass만 인정한다.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def aggregate_and_report(split_summaries: list[dict[str, Any]], out_dir: Path, report_path: Path) -> dict[str, Any]:
    fixed_rows = []
    cross_rows = []
    integrated_rows = []
    boundary_rows = []
    for summ in split_summaries:
        split = summ["split"]
        split_dir = out_dir / split
        fdf = pd.read_csv(split_dir / "fixed_unsupervised_metrics.csv")
        cdf = pd.read_csv(split_dir / "crossfit_single_feature_metrics.csv")
        bdf = pd.read_csv(split_dir / "boundary_auc_table.csv")
        fdf["split"] = split
        cdf["split"] = split
        bdf["split"] = split
        fixed_rows.extend(fdf.to_dict(orient="records"))
        cross_rows.extend(cdf.to_dict(orient="records"))
        integ = summ["integrated_ridge_crossfit"].copy()
        integ["split"] = split
        integrated_rows.append(integ)
        boundary_rows.extend(bdf.to_dict(orient="records"))

    fixed_agg = aggregate_metric_rows(fixed_rows, "variant_id")
    cross_agg = aggregate_metric_rows(cross_rows, "feature_col")

    bdf_all = pd.DataFrame(boundary_rows)
    bauc_rows = []
    for feat, g in bdf_all.groupby("feature_col", sort=False):
        aucs = g["boundary_auc_selected_positive"].astype(float).to_numpy()
        devs = np.abs(aucs - 0.5)
        bauc_rows.append(
            {
                "feature_col": feat,
                "mean_auc": float(np.nanmean(aucs)),
                "mean_abs_dev_from_half": float(np.nanmean(devs)),
                "max_abs_dev_from_half": float(np.nanmax(devs)),
                "splits_evaluated": int(len(g)),
            }
        )
    bauc_agg = pd.DataFrame(bauc_rows).sort_values(["mean_abs_dev_from_half", "max_abs_dev_from_half"], ascending=[False, False], kind="mergesort")

    fixed_pass = []
    for _, r in fixed_agg.iterrows():
        if (
            float(r["mean_delta"]) >= MDE
            and int(r["fixes"]) > int(r["breaks"])
            and float(r["paired_exact_binom_p_two_sided"]) < 0.05
            and int(r["positive_splits"]) >= 2
        ):
            fixed_pass.append(r.to_dict())
    cross_pass = []
    for _, r in cross_agg.iterrows():
        if (
            float(r["mean_delta"]) >= MDE
            and int(r["fixes"]) > int(r["breaks"])
            and float(r["paired_exact_binom_p_two_sided"]) < 0.05
            and int(r["positive_splits"]) >= 2
        ):
            cross_pass.append(r.to_dict())

    integ_df = pd.DataFrame(integrated_rows).sort_values("split", kind="mergesort")
    integ_fixes = int(integ_df["fixes"].sum())
    integ_breaks = int(integ_df["breaks"].sum())
    integ_delta_mean = float(integ_df["delta"].astype(float).mean())
    integ_p = exact_two_sided_binom_p(integ_fixes, integ_fixes + integ_breaks)

    if fixed_pass:
        verdict = "CANDIDATE_FOUND_FIXED_GATE_PASS — fixed residual feature variant cleared the strict gate; manual review required before any last submission."
    elif cross_pass:
        verdict = "HEADROOM_DIAGNOSTIC_ONLY — cross-fit found label-trained residual signal, but no fixed deployable variant passed; do not submit yet."
    elif integ_delta_mean >= MDE and integ_fixes > integ_breaks and integ_p < 0.05 and int((integ_df["delta"] > 0).sum()) >= 2:
        verdict = "HEADROOM_DIAGNOSTIC_ONLY — integrated cross-fit passed, but it is validation-label-trained and not submission-ready."
    else:
        verdict = "NO_SUBMIT_CEILING_CONFIRMED — no fixed residual feature or cross-fit diagnostic clears the 3-split gate; preserve final submission slot."

    aggregate = {
        "mde": MDE,
        "splits": SPLITS,
        "fixed_gate_aggregate": fixed_agg.to_dict(orient="records"),
        "crossfit_headroom_aggregate": cross_agg.to_dict(orient="records"),
        "integrated_ridge_by_split": integ_df.to_dict(orient="records"),
        "integrated_ridge_aggregate": {
            "mean_delta": integ_delta_mean,
            "fixes": integ_fixes,
            "breaks": integ_breaks,
            "paired_exact_binom_p_two_sided": integ_p,
            "positive_splits": int((integ_df["delta"] > 0).sum()),
        },
        "boundary_auc_aggregate": bauc_agg.to_dict(orient="records"),
        "fixed_gate_pass_variants": fixed_pass,
        "crossfit_headroom_pass_features": cross_pass,
        "verdict": verdict,
        "safety_note": "Validation-only synthetic split audit. No hidden test read and no Kaggle submission artifact generated.",
    }
    write_json(report_path.with_suffix(".json"), aggregate)
    write_markdown_report(report_path, aggregate)
    print(f"\n[bffa] VERDICT: {verdict}", flush=True)
    print(f"[bffa] wrote {report_path} and {report_path.with_suffix('.json')}", flush=True)
    return aggregate


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="*", default=SPLITS, help="Validation splits to audit; default is the 3-uniform panel.")
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/boundary_feature_factory"), help="Audit artifact directory.")
    ap.add_argument("--report", default=str(ROOT / "reports/20260601_boundary_feature_factory_audit.md"), help="Markdown report path; JSON uses same stem.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    splits = list(args.splits)
    for split in splits:
        if split not in SPLITS:
            raise ValueError(f"Unexpected split {split}; allowed: {SPLITS}")
    out_dir = ensure_dir(Path(args.out_dir))
    summaries = [run_split(split, out_dir) for split in splits]
    aggregate_and_report(summaries, out_dir, Path(args.report))
    print("DONE_BOUNDARY_FEATURE_FACTORY_AUDIT", flush=True)


if __name__ == "__main__":
    main()
