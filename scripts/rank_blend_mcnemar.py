"""Paired McNemar test for rank(emb128_4seed + emb192_4seed) vs emb128_4seed, per split.
Completes the promised '3-split + paired McNemar' gate. numpy/scipy only, no quota."""
import os, numpy as np, pandas as pd
from scipy.stats import chi2
R="/opt/data/kaggle/kmu-rec-sys-26-steam"

def emb128_files(sp):
    if sp=="val_random_uniform_seed42":
        return [f"{R}/artifacts/lightgcn_sweep_uniform_eval/emb128_L4_reg1e-03/{sp}/lightgcn_scores.csv",
                f"{R}/artifacts/lightgcn_emb128L4r3_ens/seed123/{sp}/lightgcn_scores.csv",
                f"{R}/artifacts/lightgcn_emb128L4r3_ens/seed2024/{sp}/lightgcn_scores.csv",
                f"{R}/artifacts/lightgcn_emb128L4r3_ens/seed7/{sp}/lightgcn_scores.csv"]
    return [f"{R}/artifacts/split_panel_emb128/{sp}/seed{s}/lightgcn_scores.csv" for s in (42,123,2024,7)]
def emb192_files(sp):
    if sp=="val_random_uniform_seed42":
        return [f"{R}/artifacts/capacity_uniform/emb192_L4_r3/{sp}/lightgcn_scores.csv",
                f"{R}/artifacts/capacity_uniform/emb192_L4_r3_seed123/{sp}/lightgcn_scores.csv",
                f"{R}/artifacts/capacity_uniform/emb192_L4_r3_seed2024/{sp}/lightgcn_scores.csv",
                f"{R}/artifacts/capacity_uniform/emb192_L4_r3_seed7/{sp}/lightgcn_scores.csv"]
    return [f"{R}/artifacts/split_panel_emb192/{sp}/seed{s}/lightgcn_scores.csv" for s in (42,123,2024,7)]

def sc(df):
    for c in ("score_lightgcn","score"):
        if c in df.columns: return c
def ens(fs):
    b=pd.read_csv(fs[0]); s=sc(b); a=b[["ID","userID","gameID","Label"]].copy(); a["s"]=b[s].astype(float).values
    for f in fs[1:]:
        d=pd.read_csv(f); a=a.merge(d[["ID"]].assign(x=d[sc(d)].astype(float).values),on="ID"); a["s"]+=a.pop("x")
    a["s"]/=len(fs); return a
def correct_vec(df,col):
    """row-level correct(1)/wrong(0) under per-user top-half."""
    p=np.zeros(len(df),int); v=df[col].values
    for u,ix in df.groupby("userID").indices.items():
        ix=np.asarray(ix); k=len(ix)//2; p[ix[np.argsort(v[ix])[::-1][:k]]]=1
    return (p==df["Label"].values).astype(int)
def urank(df,col):
    r=np.zeros(len(df)); v=df[col].values
    for u,ix in df.groupby("userID").indices.items():
        ix=np.asarray(ix); r[ix[np.argsort(v[ix])]]=np.arange(len(ix))
    return r

SPLITS=["val_random_uniform_seed42","val_random_uniform_seed7","val_random_uniform_seed123"]
print(f"{'split':30s} {'b(base✓,blend✗)':>16s} {'c(base✗,blend✓)':>16s} {'McNemar χ²':>11s} {'p':>8s}")
ps=[]
for sp in SPLITS:
    A=ens(emb128_files(sp)).rename(columns={"s":"a"})
    B=ens(emb192_files(sp))[["ID","s"]].rename(columns={"s":"b"})
    df=A.merge(B,on="ID")
    df["rb"]=urank(df,"a")+urank(df,"b")
    cb=correct_vec(df,"a"); cr=correct_vec(df,"rb")
    b=int(np.sum((cb==1)&(cr==0)))   # base right, blend wrong
    c=int(np.sum((cb==0)&(cr==1)))   # base wrong, blend right (blend wins these)
    n=b+c
    stat=(abs(b-c)-1)**2/n if n>0 else 0.0   # continuity-corrected
    p=1-chi2.cdf(stat,1)
    ps.append(p)
    print(f"{sp:30s} {b:16d} {c:16d} {stat:11.3f} {p:8.4f}  {'SIG' if p<0.05 else 'ns'}")

print(f"\nFisher combined across 3 splits:")
stat_f=-2*np.sum(np.log(np.clip(ps,1e-300,1)))
p_f=1-chi2.cdf(stat_f,2*len(ps))
print(f"  χ²={stat_f:.3f} df={2*len(ps)} p={p_f:.4f}  {'SIGNIFICANT' if p_f<0.05 else 'not significant'}")
print("\nDONE_MCNEMAR")
