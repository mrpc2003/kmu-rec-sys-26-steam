"""3-split rank-blend gate: does rank(emb128_4seed + emb192_4seed) beat emb128_4seed
on ALL THREE independent uniform splits (seed42/seed7/seed123)?

seed42 surrogate showed rank(128+192) = +0.0017 vs emb128. But emb192 already LOST to
emb128 on the real public LB (0.77715 < 0.77745), and the dominant variance is between-split
data-draw (std ~0.0027). So a single-split gain is suspect. This paired 3-split test cancels
data-draw variance: a REAL blend gain must be sign-stable across all three splits.

VERDICT:
  rank(128+192) > emb128 on ALL 3 splits, mean Δ > noise  -> REAL signal, escalate to user
  otherwise                                                -> mirage, close ensemble track

numpy+pandas only. No quota. Validation-only."""
import os, numpy as np, pandas as pd
R="/opt/data/kaggle/kmu-rec-sys-26-steam"
NOISE=0.0007

# Per-split member file resolution.
def emb128_files(split):
    if split=="val_random_uniform_seed42":
        return [f"{R}/artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03/{split}/lightgcn_scores.csv",
                f"{R}/artifacts/lightgcn_emb128L4r3_ens/seed123/{split}/lightgcn_scores.csv",
                f"{R}/artifacts/lightgcn_emb128L4r3_ens/seed2024/{split}/lightgcn_scores.csv",
                f"{R}/artifacts/lightgcn_emb128L4r3_ens/seed7/{split}/lightgcn_scores.csv"]
    return [f"{R}/artifacts/split_panel_emb128/{split}/seed{s}/lightgcn_scores.csv" for s in (42,123,2024,7)]

def emb192_files(split):
    if split=="val_random_uniform_seed42":
        return [f"{R}/artifacts/capacity_uniform/emb192_L4_r3/{split}/lightgcn_scores.csv",
                f"{R}/artifacts/capacity_uniform/emb192_L4_r3_seed123/{split}/lightgcn_scores.csv",
                f"{R}/artifacts/capacity_uniform/emb192_L4_r3_seed2024/{split}/lightgcn_scores.csv",
                f"{R}/artifacts/capacity_uniform/emb192_L4_r3_seed7/{split}/lightgcn_scores.csv"]
    return [f"{R}/artifacts/split_panel_emb192/{split}/seed{s}/lightgcn_scores.csv" for s in (42,123,2024,7)]

def sc(df):
    for c in ("score_lightgcn","score"):
        if c in df.columns: return c
    raise SystemExit(df.columns.tolist())
def ens(files):
    miss=[f for f in files if not os.path.exists(f)]
    if miss:
        return None, miss
    b=pd.read_csv(files[0]); s=sc(b)
    a=b[["ID","userID","gameID","Label"]].copy(); a["s"]=b[s].astype(float).values
    for f in files[1:]:
        d=pd.read_csv(f); a=a.merge(d[["ID"]].assign(x=d[sc(d)].astype(float).values),on="ID"); a["s"]+=a.pop("x")
    a["s"]/=len(files); return a, []
def th(df,col):
    p=np.zeros(len(df),int); v=df[col].values
    for u,ix in df.groupby("userID").indices.items():
        ix=np.asarray(ix); k=len(ix)//2; p[ix[np.argsort(v[ix])[::-1][:k]]]=1
    return (p==df["Label"].values).mean()
def urank(df,col):
    r=np.zeros(len(df)); v=df[col].values
    for u,ix in df.groupby("userID").indices.items():
        ix=np.asarray(ix); r[ix[np.argsort(v[ix])]]=np.arange(len(ix))
    return r

SPLITS=["val_random_uniform_seed42","val_random_uniform_seed7","val_random_uniform_seed123"]
print(f"{'split':32s} {'emb128':>8s} {'emb192':>8s} {'rank(128+192)':>14s} {'Δ vs128':>9s}  verdict")
deltas=[]; ok=True
for sp in SPLITS:
    A,m1=ens(emb128_files(sp)); B,m2=ens(emb192_files(sp))
    if A is None or B is None:
        print(f"{sp:32s} MISSING FILES: {(m1+m2)[:2]} ... (training incomplete)")
        ok=False; continue
    B=B[["ID","s"]].rename(columns={"s":"b"}); A=A.rename(columns={"s":"a"})
    df=A.merge(B,on="ID")
    accA=th(df.assign(x=df.a),"x"); accB=th(df.assign(x=df.b),"x")
    df["rb"]=urank(df,"a")+urank(df,"b")
    accR=th(df,"rb"); d=accR-accA; deltas.append(d)
    v="WIN" if d>0 else ("tie" if abs(d)<1e-9 else "LOSE")
    print(f"{sp:32s} {accA:8.5f} {accB:8.5f} {accR:14.5f} {d:+9.5f}  {v}")

if ok and len(deltas)==3:
    allwin=all(d>0 for d in deltas); mean=np.mean(deltas)
    print(f"\nmean Δ = {mean:+.5f}  | all-3-splits-win = {allwin}  | noise band = {NOISE}")
    if allwin and mean>NOISE:
        print("VERDICT: REAL signal — rank(128+192) sign-stable across 3 splits. ESCALATE to user.")
    elif allwin:
        print("VERDICT: sign-stable but sub-noise mean. Marginal; report honestly, do not over-claim.")
    else:
        print("VERDICT: MIRAGE — gain not sign-stable across splits (seed42 was a lucky draw). Close ensemble track.")
print("\nDONE_3SPLIT_GATE")
