#!/usr/bin/env python3
from __future__ import annotations
import ast, json
from pathlib import Path
import numpy as np, pandas as pd
root=Path('data/raw/public/data')
data=[]
for line in open(root/'train.json',encoding='utf-8'):
    d=ast.literal_eval(line); data.append((d['userID'],d['gameID'],float(d.get('hours_transformed',0)),d['date']))
df=pd.DataFrame(data,columns=['userID','gameID','htr','date'])
pairs=pd.read_csv(root/'pairs.csv')
user_n=df.groupby('userID').size().rename('train_n')
item_n=df.groupby('gameID').size().rename('item_n')
cand_n=pairs.groupby('userID').size().rename('candidate_n')
user_diag=pd.concat([user_n,cand_n],axis=1).dropna().astype(int)
user_diag['hidden_pos_count_known_from_structure']=user_diag['candidate_n']//2
user_diag['test_frac_vs_train']=user_diag['hidden_pos_count_known_from_structure']/user_diag['train_n']
# formula probes
best=[]
for mode in ['floor','round','ceil']:
    for frac in np.linspace(0.01,0.3,291):
        raw=user_diag['train_n']*frac
        if mode=='floor': pred=np.floor(raw)
        elif mode=='round': pred=np.round(raw)
        else: pred=np.ceil(raw)
        pred=np.maximum(1,pred).astype(int)
        acc=float((pred==user_diag['hidden_pos_count_known_from_structure']).mean())
        mae=float(np.mean(np.abs(pred-user_diag['hidden_pos_count_known_from_structure'])))
        best.append((acc,-mae,mode,frac,mae))
best=sorted(best, reverse=True)[:10]
bins=[0,10,20,30,40,60,80,120,200,999]
user_diag['train_n_bin']=pd.cut(user_diag['train_n'], bins=bins, right=True)
bin_table=user_diag.groupby('train_n_bin', observed=True).agg(users=('train_n','size'),train_n_median=('train_n','median'),hidden_pos_median=('hidden_pos_count_known_from_structure','median'),hidden_pos_mean=('hidden_pos_count_known_from_structure','mean'),test_frac_median=('test_frac_vs_train','median'),candidate_n_max=('candidate_n','max')).reset_index()
bin_table['train_n_bin']=bin_table['train_n_bin'].astype(str)
# item sampling exponent fit: user-conditioned unseen negatives q ∝ pop^alpha.
all_games=np.array(item_n.index.tolist(),dtype=object); pop=np.array([item_n[g] for g in all_games],dtype=float)
game_to_idx={g:i for i,g in enumerate(all_games)}
user_hist_idx={u: np.fromiter((game_to_idx[g] for g in s), dtype=int) for u,s in df.groupby('userID')['gameID'].apply(set).items()}
pair_user_counts=pairs.groupby('userID').size().to_dict()
actual=np.sort(pairs.join(item_n,on='gameID')['item_n'].values)
rng=np.random.default_rng(123)
alpha_scores=[]
for alpha in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5]:
    weights_base=pop**alpha
    vals=[]
    for u,c in pair_user_counts.items():
        mask=np.ones(len(all_games), dtype=bool)
        mask[user_hist_idx[u]]=False
        unseen_idx=np.flatnonzero(mask)
        w=weights_base[unseen_idx]; w=w/w.sum()
        sampled=rng.choice(unseen_idx,size=c,replace=True,p=w)
        vals.extend(pop[sampled].tolist())
    vals=np.sort(np.asarray(vals))
    merged=np.sort(np.unique(np.concatenate([actual,vals])))
    fa=np.searchsorted(actual,merged,side='right')/len(actual)
    fv=np.searchsorted(vals,merged,side='right')/len(vals)
    ks=float(np.max(np.abs(fa-fv)))
    alpha_scores.append((ks,alpha,float(np.median(vals)),float(vals.mean()),float(np.percentile(vals,95))))
alpha_scores=sorted(alpha_scores)[:8]
# pair row order structure
order=pairs.copy(); order['prev_same_user']=order['userID'].eq(order['userID'].shift(1))
consecutive_same=int(order['prev_same_user'].sum())
pos_spread=[int(gp['ID'].max()-gp['ID'].min()) for _,gp in pairs.groupby('userID')]
# correlations from feature preview
pair_feat=pd.read_csv('reports/deep_eda/candidate_pair_engineered_features_preview.csv')
corrs=pair_feat[['item_n','hist_item_cos_max','hist_item_cos_top3_mean','hist_item_cooc_sum','hist_htr_weighted_cos','candidate_pop_z_vs_user_hist','user_n','candidate_count']].corr(numeric_only=True)
out={
  'hidden_positive_count_from_candidate_half': {
    'stats': {k: float(v) for k,v in user_diag['hidden_pos_count_known_from_structure'].describe(percentiles=[.25,.5,.75,.9,.95,.99]).to_dict().items()},
    'corr_with_train_n': float(user_diag['hidden_pos_count_known_from_structure'].corr(user_diag['train_n'])),
    'test_frac_stats': {k: float(v) for k,v in user_diag['test_frac_vs_train'].describe(percentiles=[.25,.5,.75,.9,.95,.99]).to_dict().items()},
    'best_simple_formula_matches': [{'mode':m,'frac':round(float(f),3),'match_rate':round(float(a),4),'mae':round(float(mae),3)} for a,negmae,m,f,mae in best[:6]],
    'bin_table': bin_table.to_dict('records')
  },
  'candidate_negative_sampling_exponent_fit_by_item_pop': [{'alpha':float(a),'ks':round(float(ks),4),'median':float(med),'mean':round(float(mean),2),'p95':float(p95)} for ks,a,med,mean,p95 in alpha_scores],
  'pair_row_order': {'consecutive_same_user_rows': consecutive_same, 'median_id_span_per_user': float(np.median(pos_spread)), 'p95_id_span_per_user': float(np.percentile(pos_spread,95))},
  'selected_pair_feature_corr': corrs.round(4).to_dict()
}
Path('reports/deep_eda/deep_structure_diagnostics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=2))
