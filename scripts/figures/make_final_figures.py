#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FINAL=Path('/root/autodl-tmp/final_analysis'); OUT=FINAL/'figures'; OUT.mkdir(parents=True,exist_ok=True)
paths={
 'Raw + RC':FINAL/'heldout_ensembles/raw_rc_5fold_ensemble/metrics.json',
 'Raw, no RC':FINAL/'heldout_ensembles/raw_no_rc_5fold_ensemble/metrics.json',
 'Frozen NT':FINAL/'heldout_ensembles/frozen_nt_attnpool_5fold_ensemble/metrics.json',
 'Motif transformer':FINAL/'heldout_ensembles/motif_transformer_5fold_ensemble/metrics.json'}
metrics={k:json.load(open(v)) for k,v in paths.items()}; boot=np.load(FINAL/'bootstrap/poisson_group_bootstrap_2000.npz')
colors={'Raw + RC':'#1b6ca8','Raw, no RC':'#64a4cc','Frozen NT':'#e78b3e','Motif transformer':'#6a4c93'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

# Held-out performance with grouped-bootstrap intervals.
names=list(paths); keys=['raw_rc','raw_no_rc','frozen_nt','motif']; means=[metrics[n]['mean_r2'] for n in names]; cis=np.array([[np.quantile(boot[f'r2_{k}'],.025),np.quantile(boot[f'r2_{k}'],.975)] for k in keys])
fig,ax=plt.subplots(figsize=(7.2,4.5)); x=np.arange(len(names)); ax.bar(x,means,color=[colors[n] for n in names],width=.68); ax.errorbar(x,means,yerr=np.vstack([np.array(means)-cis[:,0],cis[:,1]-np.array(means)]),fmt='none',ecolor='black',capsize=4,lw=1.2)
ax.set_xticks(x,names); ax.set_ylabel('Mean per-bin $R^2$'); ax.set_ylim(0,.68)
for i,v in enumerate(means): ax.text(i,v+.014,f'{v:.3f}',ha='center')
fig.tight_layout(); fig.savefig(OUT/'heldout_model_performance.png',dpi=300); fig.savefig(OUT/'heldout_model_performance.pdf'); plt.close(fig)

# Per-pseudotime performance.
fig,ax=plt.subplots(figsize=(8.2,4.6)); bins=np.arange(1,21)
for n in names: ax.plot(bins,metrics[n]['r2_bins'],marker='o',ms=3,lw=1.8,label=n,color=colors[n])
ax.set_xlabel('Pseudotime bin'); ax.set_ylabel('$R^2$'); ax.set_xticks(bins); ax.legend(frameon=False,ncol=2); fig.tight_layout(); fig.savefig(OUT/'heldout_r2_by_pseudotime.png',dpi=300); fig.savefig(OUT/'heldout_r2_by_pseudotime.pdf'); plt.close(fig)

# Target variance and model performance.
target=pd.read_csv(FINAL/'development_target_diagnostics.csv'); fig,ax1=plt.subplots(figsize=(8.2,4.6)); ax2=ax1.twinx(); ax1.bar(bins,target['variance'],color='#d9d9d9',label='Development target variance'); ax2.plot(bins,metrics['Raw + RC']['r2_bins'],color=colors['Raw + RC'],marker='o',label='Held-out Raw + RC $R^2$'); ax1.set_xlabel('Pseudotime bin'); ax1.set_ylabel('Target variance'); ax2.set_ylabel('$R^2$'); ax1.set_xticks(bins); lines=ax1.get_legend_handles_labels(); lines2=ax2.get_legend_handles_labels(); ax2.legend(lines[0]+lines2[0],lines[1]+lines2[1],frameon=False,loc='upper right'); fig.tight_layout(); fig.savefig(OUT/'target_variance_and_predictability.png',dpi=300); fig.savefig(OUT/'target_variance_and_predictability.pdf'); plt.close(fig)

# Fold diagnostic showing the unusual variance in Fold 2.
fold=pd.read_csv(FINAL/'fold_diagnostics.csv'); fig,ax=plt.subplots(figsize=(6.6,4.3)); bars=ax.bar(fold['fold'],fold['target_grand_variance'],color=['#bdbdbd','#d95f02','#bdbdbd','#bdbdbd','#bdbdbd']); ax.set_xlabel('Validation fold'); ax.set_ylabel('Mean target variance across bins'); ax.set_xticks(fold['fold']);
for b,v in zip(bars,fold['target_grand_variance']): ax.text(b.get_x()+b.get_width()/2,v+90,f'{v:,.0f}',ha='center',fontsize=9)
fig.tight_layout(); fig.savefig(OUT/'fold_target_variance.png',dpi=300); fig.savefig(OUT/'fold_target_variance.pdf'); plt.close(fig)

# Paired differences.
summary=json.load(open(FINAL/'bootstrap/summary.json')); labels=['vs no RC','vs Frozen NT','vs motif']; comps=['raw_rc_minus_raw_no_rc','raw_rc_minus_frozen_nt','raw_rc_minus_motif']; vals=[summary['comparisons'][c]['mean_difference'] for c in comps]; ci=np.array([summary['comparisons'][c]['ci95'] for c in comps]); y=np.arange(3)
fig,ax=plt.subplots(figsize=(7,3.8)); ax.errorbar(vals,y,xerr=np.vstack([np.array(vals)-ci[:,0],ci[:,1]-np.array(vals)]),fmt='o',color=colors['Raw + RC'],ecolor='black',capsize=4); ax.axvline(0,color='black',lw=.8); ax.set_yticks(y,labels); ax.set_xlabel('Difference in mean per-bin $R^2$ (Raw + RC minus comparator)'); ax.invert_yaxis(); fig.tight_layout(); fig.savefig(OUT/'bootstrap_model_differences.png',dpi=300); fig.savefig(OUT/'bootstrap_model_differences.pdf'); plt.close(fig)

with open(OUT/'README.txt','w') as h: h.write('Figures generated only from locked development diagnostics and single held-out ensemble evaluations.\n')
print('\n'.join(str(p) for p in sorted(OUT.iterdir())))
