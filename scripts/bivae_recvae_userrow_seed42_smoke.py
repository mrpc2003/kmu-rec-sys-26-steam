#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from recsys_played_utils import build_user_item_matrix, evaluate_tophalf, load_pairs_csv, load_train_interactions, write_json

ROOT = Path("/opt/data/kaggle/kmu-rec-sys-26-steam")
SPLIT = "val_random_uniform_seed42"
BASE_REF = 0.7650530106021204
SOLO_GATE = 0.735
DELTA_GATE = 0.0007
PASS_BLEND = BASE_REF + DELTA_GATE
DESCRIPTION = "BiVAE/RecVAE-style user-row reconstruction seed42 validation-only smoke. No submit, no submissions writes, no full-test materialization."
SEED_SCORE_PATHS = {
    42: ROOT / "artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03/val_random_uniform_seed42/lightgcn_scores.csv",
    123: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed123/val_random_uniform_seed42/lightgcn_scores.csv",
    2024: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed2024/val_random_uniform_seed42/lightgcn_scores.csv",
    7: ROOT / "artifacts/lightgcn_emb128L4r3_ens/seed7/val_random_uniform_seed42/lightgcn_scores.csv",
}


def module(name: str) -> Any:
    return importlib.import_module(name)


def assert_not_submissions(path: Path) -> None:
    resolved = path.resolve()
    submissions = (ROOT / "submissions").resolve()
    if resolved == submissions or submissions in resolved.parents:
        raise ValueError(f"Refusing to write under submissions/: {path}")


def finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): finite_or_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_or_none(v) for v in value]
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def metric_float(summary: dict[str, object], key: str) -> float:
    value = summary[key]
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def within_user_z(df: Any, col: str, np: Any) -> Any:
    grouped = df.groupby("userID", sort=False)[col]
    std = grouped.transform("std").replace(0, np.nan).fillna(1.0)
    return ((df[col] - grouped.transform("mean")) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def within_user_pct_rank(df: Any, col: str, np: Any) -> Any:
    grouped = df.groupby("userID", sort=False)[col]
    denom = grouped.transform("size") - 1
    rank = grouped.rank(method="first", ascending=False)
    return np.where(denom > 0, (rank - 1) / denom, 0.0)


def load_base_scores(pd: Any, np: Any) -> Any:
    merged = None
    cols: list[str] = []
    for seed, path in SEED_SCORE_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        col = f"score_emb128_seed{seed}"
        cols.append(col)
        part = pd.read_csv(path)[["ID", "userID", "gameID", "Label", "score_lightgcn"]].rename(columns={"score_lightgcn": col})
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part[["ID", col]], on="ID", validate="one_to_one")
    if merged is None:
        raise RuntimeError("No base score files loaded")
    merged["score_emb128_4seed"] = merged[cols].mean(axis=1).astype(np.float32)
    return merged[["ID", "score_emb128_4seed"]]


def build_recvae(torch: Any, n_items: int, hidden: int, latent: int, dropout: float) -> Any:
    nn = torch.nn

    class ResidualBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(dim, dim)
            self.norm = nn.LayerNorm(dim)

        def forward(self, x: Any) -> Any:
            return x + torch.nn.functional.silu(self.norm(self.linear(x)))

    class RecVAEUserRow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.drop = nn.Dropout(dropout)
            self.input = nn.Linear(n_items, hidden)
            self.block1 = ResidualBlock(hidden)
            self.block2 = ResidualBlock(hidden)
            self.mu = nn.Linear(hidden, latent)
            self.logvar = nn.Linear(hidden, latent)
            self.dec1 = nn.Linear(latent, hidden)
            self.dec2 = nn.Linear(hidden, n_items)
            for layer in [self.input, self.mu, self.logvar, self.dec1, self.dec2]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        def encode(self, x: Any) -> tuple[Any, Any]:
            h = torch.nn.functional.normalize(x, dim=1)
            h = self.drop(h)
            h = torch.nn.functional.silu(self.input(h))
            h = self.block2(self.block1(h))
            return self.mu(h), self.logvar(h).clamp(min=-8.0, max=8.0)

        def forward(self, x: Any) -> tuple[Any, Any, Any]:
            mu, logvar = self.encode(x)
            if self.training:
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            else:
                z = mu
            h = torch.nn.functional.silu(self.dec1(z))
            return self.dec2(h), mu, logvar

    return RecVAEUserRow()


def recvae_loss(torch: Any, logits: Any, x: Any, mu: Any, logvar: Any, beta: float, l2: float) -> Any:
    nll = -(torch.nn.functional.log_softmax(logits, dim=1) * x).sum(dim=1).mean()
    kld = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    return nll + beta * kld + l2 * mu.pow(2).mean()


def train_reconstruct(args: Any, matrix: Any, n_items: int, torch: Any, np: Any) -> tuple[Any, float]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = build_recvae(torch, n_items, args.hidden, args.latent, args.dropout).to(device)
    dense = torch.tensor(matrix.toarray(), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_users = dense.shape[0]
    rng = np.random.default_rng(args.seed)
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        beta = min(args.beta_cap, args.beta_cap * (epoch + 1) / max(1, args.beta_warmup_epochs))
        perm = rng.permutation(n_users)
        last_loss = 0.0
        for start_idx in range(0, n_users, args.batch):
            idx = torch.as_tensor(perm[start_idx:start_idx + args.batch], dtype=torch.long, device=device)
            xb = dense.index_select(0, idx)
            logits, mu, logvar = model(xb)
            loss = recvae_loss(torch, logits, xb, mu, logvar, beta, args.latent_l2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(f"epoch={epoch + 1}/{args.epochs} beta={beta:.4f} loss={last_loss:.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(dense)
    return logits.detach().cpu().numpy(), round(time.time() - start, 2)


def score_candidates(candidates: Any, recon: Any, user_to_idx: dict[str, int], item_to_idx: dict[str, int], np: Any) -> Any:
    scores = np.full(len(candidates), -1e30, dtype=np.float32)
    for row_idx, row in enumerate(candidates[["userID", "gameID"]].astype(str).itertuples(index=False)):
        ui = user_to_idx.get(row.userID)
        ii = item_to_idx.get(row.gameID)
        if ui is not None and ii is not None:
            scores[row_idx] = float(recon[ui, ii])
    return scores


def label_predictions(df: Any, score_col: str) -> Any:
    _, pred = evaluate_tophalf(df, score_col, label_col="Label", user_col="userID", id_col="ID")
    return pred[["ID", "Pred", "Correct", "rank_in_user"]].rename(
        columns={"Pred": f"pred_{score_col}", "Correct": f"correct_{score_col}", "rank_in_user": f"rank_{score_col}"}
    )


def prediction_delta(base_pred: Any, recvae_pred: Any) -> dict[str, Any]:
    merged = base_pred.merge(recvae_pred, on="ID", validate="one_to_one")
    base_correct = merged["correct_score_emb128_4seed"].astype(bool)
    recvae_correct = merged["correct_score_recvae_userrow"].astype(bool)
    fixes = int((~base_correct & recvae_correct).sum())
    breaks = int((base_correct & ~recvae_correct).sum())
    changed = int((merged["pred_score_emb128_4seed"] != merged["pred_score_recvae_userrow"]).sum())
    return {"fixes": fixes, "breaks": breaks, "net_fixes": fixes - breaks, "changed_rows_vs_base": changed}


def item_degree_deciles(train_df: Any, pd: Any) -> Any:
    counts = train_df.groupby("gameID").size().rename("item_degree").reset_index()
    ranks = counts["item_degree"].rank(method="first")
    counts["item_degree_decile"] = pd.qcut(ranks, q=10, labels=False, duplicates="drop").astype(int)
    return counts


def degree_bucket_audit(scored: Any, changed_ids: set[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    changed = scored[scored["ID"].astype(int).isin(changed_ids)].copy()
    if changed.empty:
        return rows
    changed["base_correct"] = changed["pred_score_emb128_4seed"] == changed["Label"]
    changed["recvae_correct"] = changed["pred_score_recvae_userrow"] == changed["Label"]
    for decile, group in changed.groupby("item_degree_decile", sort=True):
        fixes = int((~group["base_correct"] & group["recvae_correct"]).sum())
        breaks = int((group["base_correct"] & ~group["recvae_correct"]).sum())
        rows.append({"item_degree_decile": int(decile), "changed_rows": int(len(group)), "fixes": fixes, "breaks": breaks, "net_fixes": fixes - breaks})
    return rows


def head_only_lift(bucket_rows: list[dict[str, Any]]) -> bool:
    total_net = sum(max(0, int(row["net_fixes"])) for row in bucket_rows)
    head_net = sum(max(0, int(row["net_fixes"])) for row in bucket_rows if int(row["item_degree_decile"]) >= 8)
    return bool(total_net > 0 and head_net / total_net >= 0.8)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# bivae_recvae_userrow_seed42_smoke",
        "",
        "- validation_only: true",
        "- no_kaggle_submit: true",
        "- candidate_csv_written: false",
        "- full_test_candidate_materialized: false",
        "- split: `val_random_uniform_seed42`",
        "",
        "## verdict",
        "",
        f"`{payload['tier']}`",
        "",
        payload["verdict"],
        "",
        "## metrics",
        "",
        "| metric | value | gate |",
        "|---|---:|---:|",
        f"| solo accuracy | {payload['solo_accuracy']:.6f} | >= {SOLO_GATE:.6f} |",
        f"| base emb128 4-seed accuracy | {payload['base_accuracy']:.6f} | ref {BASE_REF:.6f} |",
        f"| 50/50 z-blend accuracy | {payload['blend50_accuracy']:.6f} | >= {PASS_BLEND:.6f} |",
        f"| blend delta vs base | {payload['blend50_delta_vs_base']:+.6f} | > +{DELTA_GATE:.6f} |",
        f"| corr_z vs base | {payload['corr_z_vs_base']:.6f} | <= 0.950000 |",
        f"| rank corr vs base | {payload['rank_corr_vs_base']:.6f} | diagnostic |",
        f"| fixes / breaks | {payload['fixes']} / {payload['breaks']} | fixes > breaks |",
        "",
        "## item-degree bucket audit",
        "",
        "| item degree decile | changed rows | fixes | breaks | net |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in payload["item_degree_bucket_rows"]:
        lines.append(f"| {row['item_degree_decile']} | {row['changed_rows']} | {row['fixes']} | {row['breaks']} | {row['net_fixes']} |")
    lines.extend(
        [
            "",
            "## outputs",
            "",
            f"- `{payload['score_csv']}`",
            f"- `{payload['report_json']}`",
            f"- `{payload['report_md']}`",
            "",
            "BIVAE_RECVAE_USERROW_SEED42_SMOKE_DONE",
        ]
    )
    assert_not_submissions(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--hidden", type=int, default=600)
    parser.add_argument("--latent", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--beta-cap", type=float, default=0.15)
    parser.add_argument("--beta-warmup-epochs", type=int, default=60)
    parser.add_argument("--latent-l2", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=40)
    parser.add_argument("--out-dir", default="artifacts/bivae_recvae_userrow_seed42_smoke")
    args = parser.parse_args()

    np = module("numpy")
    pd = module("pandas")
    torch = module("torch")
    out_root = ROOT / args.out_dir
    score_dir = out_root / args.split
    score_csv = score_dir / "recvae_userrow_validation_scores.csv"
    report_json = ROOT / "reports/20260616T_bivae_recvae_userrow_seed42_smoke.json"
    report_md = ROOT / "reports/20260616T_bivae_recvae_userrow_seed42_smoke.md"
    for path in (out_root, score_dir, score_csv, report_json, report_md):
        assert_not_submissions(path)

    split_dir = ROOT / "artifacts/validation" / args.split
    train_df = load_train_interactions(split_dir / "train_interactions.csv")
    candidates = load_pairs_csv(split_dir / "candidates.csv")
    matrix, user_to_idx, item_to_idx, _, items = build_user_item_matrix(train_df, binary=True)
    recon, train_seconds = train_reconstruct(args, matrix.tocsr(), len(items), torch, np)
    scored = candidates.copy()
    scored["score_recvae_userrow"] = score_candidates(scored, recon, user_to_idx, item_to_idx, np)
    scored = scored.merge(load_base_scores(pd, np), on="ID", validate="one_to_one")
    scored = scored.merge(item_degree_deciles(train_df, pd), on="gameID", how="left", validate="many_to_one")
    scored["item_degree"] = scored["item_degree"].fillna(0).astype(int)
    scored["item_degree_decile"] = scored["item_degree_decile"].fillna(0).astype(int)
    scored["z_recvae_userrow"] = within_user_z(scored, "score_recvae_userrow", np)
    scored["z_emb128_4seed"] = within_user_z(scored, "score_emb128_4seed", np)
    scored["blend50_z_recvae_emb128"] = 0.5 * scored["z_recvae_userrow"] + 0.5 * scored["z_emb128_4seed"]
    scored["pct_rank_recvae_userrow"] = within_user_pct_rank(scored, "score_recvae_userrow", np)
    scored["pct_rank_emb128_4seed"] = within_user_pct_rank(scored, "score_emb128_4seed", np)

    solo_summary, _ = evaluate_tophalf(scored, "score_recvae_userrow", label_col="Label", user_col="userID", id_col="ID")
    base_summary, _ = evaluate_tophalf(scored, "score_emb128_4seed", label_col="Label", user_col="userID", id_col="ID")
    blend_summary, _ = evaluate_tophalf(scored, "blend50_z_recvae_emb128", label_col="Label", user_col="userID", id_col="ID")
    base_pred = label_predictions(scored, "score_emb128_4seed")
    recvae_pred = label_predictions(scored, "score_recvae_userrow")
    blend_pred = label_predictions(scored, "blend50_z_recvae_emb128")
    scored = scored.merge(base_pred, on="ID", validate="one_to_one").merge(recvae_pred, on="ID", validate="one_to_one").merge(blend_pred, on="ID", validate="one_to_one")
    deltas = prediction_delta(base_pred, recvae_pred)
    changed_ids = set(scored.loc[scored["pred_score_emb128_4seed"] != scored["pred_score_recvae_userrow"], "ID"].astype(int).tolist())
    bucket_rows = degree_bucket_audit(scored, changed_ids)
    head_only = head_only_lift(bucket_rows)
    corr_z = float(np.corrcoef(scored["z_recvae_userrow"].to_numpy(dtype=float), scored["z_emb128_4seed"].to_numpy(dtype=float))[0, 1])
    rank_corr = float(np.corrcoef(scored["pct_rank_recvae_userrow"].astype(float), scored["pct_rank_emb128_4seed"].astype(float))[0, 1])
    solo_acc = metric_float(solo_summary, "row_accuracy")
    base_acc = metric_float(base_summary, "row_accuracy")
    blend_acc = metric_float(blend_summary, "row_accuracy")
    blend_delta = blend_acc - base_acc
    pass_gate = bool(solo_acc >= SOLO_GATE and blend_delta > DELTA_GATE and deltas["fixes"] > deltas["breaks"] and corr_z <= 0.95 and not head_only)
    if pass_gate:
        tier = "PASS_SEED42_ESCALATE_TO_PANEL"
        verdict = "RecVAE-style user-row model cleared the seed42 smoke gate; next safe step is fixed 3-split validation panel, still no submit."
    elif solo_acc < SOLO_GATE:
        tier = "KILL_WEAK_SOLO"
        verdict = "RecVAE-style user-row solo accuracy missed the 0.735 smoke floor."
    elif blend_delta <= DELTA_GATE:
        tier = "KILL_TINY_OR_NEGATIVE_BLEND"
        verdict = "RecVAE-style user-row score did not improve the emb128 4-seed base by more than the +0.0007 noise band."
    else:
        tier = "KILL_GUARD_OR_BUCKET_FAIL"
        verdict = "RecVAE-style user-row score failed fixes/breaks, correlation, or head-only guard checks."

    score_dir.mkdir(parents=True, exist_ok=True)
    scored[
        [
            "ID",
            "userID",
            "gameID",
            "Label",
            "score_recvae_userrow",
            "score_emb128_4seed",
            "z_recvae_userrow",
            "z_emb128_4seed",
            "blend50_z_recvae_emb128",
            "pred_score_recvae_userrow",
            "pred_score_emb128_4seed",
            "pred_blend50_z_recvae_emb128",
            "item_degree",
            "item_degree_decile",
        ]
    ].to_csv(score_csv, index=False)
    payload = {
        "artifact": "bivae_recvae_userrow_seed42_smoke",
        "validation_only": True,
        "no_kaggle_submit": True,
        "candidate_csv_written": False,
        "full_test_candidate_materialized": False,
        "split": args.split,
        "params": {
            "hidden": args.hidden,
            "latent": args.latent,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "batch": args.batch,
            "beta_cap": args.beta_cap,
            "beta_warmup_epochs": args.beta_warmup_epochs,
            "latent_l2": args.latent_l2,
            "seed": args.seed,
            "device": args.device,
        },
        "train_seconds": train_seconds,
        "rows": int(len(scored)),
        "users": int(scored["userID"].nunique()),
        "items": int(len(items)),
        "solo_accuracy": solo_acc,
        "base_accuracy": base_acc,
        "base_ref": BASE_REF,
        "base_accuracy_delta_vs_ref": base_acc - BASE_REF,
        "blend50_accuracy": blend_acc,
        "blend50_delta_vs_base": blend_delta,
        "solo_gate": SOLO_GATE,
        "blend_gate_accuracy": PASS_BLEND,
        "delta_gate": DELTA_GATE,
        "corr_z_vs_base": corr_z,
        "rank_corr_vs_base": rank_corr,
        **deltas,
        "head_only_lift": head_only,
        "item_degree_bucket_rows": bucket_rows,
        "tier": tier,
        "verdict": verdict,
        "score_csv": str(score_csv.relative_to(ROOT)),
        "report_json": str(report_json.relative_to(ROOT)),
        "report_md": str(report_md.relative_to(ROOT)),
    }
    write_json(report_json, finite_or_none(payload))
    write_md(report_md, payload)
    print(json.dumps(finite_or_none({k: payload[k] for k in ["artifact", "tier", "solo_accuracy", "base_accuracy", "blend50_accuracy", "blend50_delta_vs_base", "corr_z_vs_base", "fixes", "breaks", "score_csv"]}), indent=2))


if __name__ == "__main__":
    main()
