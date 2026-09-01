#!/usr/bin/env python3
"""Shape prediction with a fair shared ResCNN trunk for raw DNA and motif grids.

Raw:   [B,4,201]    -> multiscale stem -> [B,128,201]
Motif: [B,10,1443] -> motif projection -> [B,128,10]
Both:  four dilated residual blocks -> mean/max pool -> 20-bin profile logits.

The profile objectives include multinomial NLL, equal-region cross entropy,
and an unconstrained joint Poisson rate model over the original count matrix.
Only development folds are used by this script.
"""

import argparse, copy, json, math, pickle, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

P = argparse.ArgumentParser()
P.add_argument("--modality", choices=["raw", "motif", "motif_nopos"], required=True)
P.add_argument("--fold", type=int, default=0)
P.add_argument("--objective", choices=["mnll", "equal_ce", "normalized_poisson", "normalized_poisson_fixed", "poisson", "factorized_poisson", "factorized_balanced"], default="mnll")
P.add_argument("--shape-total", type=float, default=100.0,
               help="Fixed pseudo-count total for normalized_poisson")
P.add_argument("--shape-weight", type=float, default=1.0,
               help="Weight of equal-region shape CE for factorized_balanced")
P.add_argument("--activation", choices=["relu", "softplus"], default="relu",
               help="Hidden-layer nonlinearity; Poisson uses a positive rate link")
P.add_argument("--backbone", choices=["linear", "cnn", "transformer"], default="cnn")
P.add_argument("--transformer-d-model", type=int, default=256)
P.add_argument("--transformer-layers", type=int, default=3)
P.add_argument("--transformer-heads", type=int, default=8)
P.add_argument("--epochs", type=int, default=30)
P.add_argument("--patience", type=int, default=5)
P.add_argument("--batch-size", type=int, default=1024)
P.add_argument("--accum-steps", type=int, default=4)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--p-threshold", type=float, default=1e-3)
P.add_argument("--max-train", type=int, default=0, help="Optional deterministic cap for smoke tests")
P.add_argument("--max-val", type=int, default=0, help="Optional deterministic cap for smoke tests")
P.add_argument("--rc-prob", type=float, default=0.5)
P.add_argument("--presence", action="store_true", help="Add score>0 as an ablation channel; off by default for pure-score models")
P.add_argument("--num-workers", type=int, default=0)
P.add_argument("--output-root", default="/root/autodl-tmp/shape_motif_revision")
A = P.parse_args()

ROOT = Path("/root/sc-motif-open/R")
OUT = Path(A.output_root) / f"{A.modality}_{A.objective}" / f"fold{A.fold+1}"
OUT.mkdir(parents=True, exist_ok=True)
N_MOTIFS, N_POS, N_BINS = 1443, 10, 20
seed = 2026 + A.fold
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda:0")

with open(ROOT / "totmat_shape.txt") as f:
    n_regions, n_bins = map(int, f.read().split())
assert n_bins == N_BINS
Y = np.memmap(ROOT / "totmat_f64.bin", dtype="<f8", mode="r",
              shape=(n_regions, n_bins), order="F")
SEQ = np.memmap(ROOT / "fullseqs.txt", dtype="S202", mode="r", shape=(n_regions,))
with open(ROOT / "folds.pkl", "rb") as f:
    folds = pickle.load(f)
tr0, va0 = [np.asarray(x, dtype=np.int64) for x in folds[A.fold]]
pvals = pd.read_csv(ROOT / "peaktest.tsv", sep="\t", usecols=["p.value"])["p.value"].to_numpy()
selected = np.isfinite(pvals) & (pvals < A.p_threshold)

def filter_idx(idx, chunk=200_000):
    ans=[]
    for s in range(0,len(idx),chunk):
        z=idx[s:s+chunk]; y=np.asarray(Y[z],dtype=np.float32)
        ans.append(z[np.isfinite(y).all(1) & (y.sum(1)>0) & (y>=0).all(1)])
    return np.concatenate(ans)

train_idx, val_idx = filter_idx(tr0[selected[tr0]]), filter_idx(va0[selected[va0]])
if A.max_train>0: train_idx=train_idx[:A.max_train]
if A.max_val>0: val_idx=val_idx[:A.max_val]

def mean_total(idx, chunk=200_000):
    total = 0.0
    for s in range(0, len(idx), chunk):
        total += float(np.asarray(Y[idx[s:s+chunk]], dtype=np.float64).sum())
    return total / len(idx)

# A fixed fold-level constant only rescales the Poisson objective and gradients;
# unlike batch-wise normalization, it does not change the likelihood optimum.
POISSON_SCALE = mean_total(train_idx) if A.objective in {"poisson", "factorized_poisson", "factorized_balanced"} else 1.0

MOTIF = ROOT / "motif_csr"
offsets=np.load(MOTIF/"offsets.npy",mmap_mode="r")
motif_ids=np.load(MOTIF/"motif_ids.npy",mmap_mode="r")
position_bins=np.load(MOTIF/"position_bins.npy",mmap_mode="r")
scores=np.load(MOTIF/"scores.npy",mmap_mode="r")

def onehot(idx):
    s=np.ascontiguousarray(np.asarray(SEQ[idx]))
    b=s.view(np.uint8).reshape(len(idx),202)[:,:201]
    return torch.from_numpy(np.stack([b==65,b==67,b==71,b==84],1).astype(np.float32))

def motif_grid(idx):
    idx=np.asarray(idx,dtype=np.int64); B=len(idx)
    starts=np.asarray(offsets[idx],dtype=np.int64); ends=np.asarray(offsets[idx+1],dtype=np.int64)
    counts=ends-starts; nh=int(counts.sum()); out=np.zeros((B,N_POS,N_MOTIFS),np.float32)
    if nh==0: return torch.from_numpy(out)
    cum=np.cumsum(counts); prev=np.r_[0,cum[:-1]]
    rows=np.repeat(np.arange(B),counts)
    hi=np.arange(nh)-np.repeat(prev,counts)+np.repeat(starts,counts)
    mids=np.asarray(motif_ids[hi],dtype=np.int64)-1
    pos=np.asarray(position_bins[hi],dtype=np.int64)-1
    vals=np.asarray(scores[hi],dtype=np.float32)
    ok=(mids>=0)&(mids<N_MOTIFS)&(pos>=0)&(pos<N_POS)&np.isfinite(vals)
    flat=out.reshape(B,-1); cols=pos[ok]*N_MOTIFS+mids[ok]
    np.maximum.at(flat,(rows[ok],cols),vals[ok])
    if A.modality == "motif_nopos":
        pooled=out.max(1,keepdims=True)
        out=np.broadcast_to(pooled,(B,N_POS,N_MOTIFS)).copy()
    return torch.from_numpy(out)

def get_x(idx, augment=False):
    if A.modality == "raw":
        x=onehot(idx).to(device,non_blocking=True)
        if augment:
            m=torch.rand(len(x),device=device)<A.rc_prob
            if m.any(): x[m]=torch.flip(x[m][:,[3,2,1,0]],dims=[2])
        return x
    x=motif_grid(idx).to(device,non_blocking=True)
    if augment:
        m=torch.rand(len(x),device=device)<A.rc_prob
        if m.any(): x[m]=torch.flip(x[m],dims=[1])
    return x

def activation_module():
    return nn.Softplus(beta=1.0, threshold=20.0) if A.activation == "softplus" else nn.ReLU()


def activate(x):
    return F.softplus(x, beta=1.0, threshold=20.0) if A.activation == "softplus" else F.relu(x)


class ResBlock(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.c1=nn.Conv1d(128,128,3,padding=d,dilation=d); self.b1=nn.BatchNorm1d(128)
        self.c2=nn.Conv1d(128,128,3,padding=d,dilation=d); self.b2=nn.BatchNorm1d(128)
    def forward(self,x): return activate(self.b2(self.c2(activate(self.b1(self.c1(x)))))+x)

class SharedShapeNet(nn.Module):
    def __init__(self, modality):
        super().__init__(); self.modality=modality
        if A.backbone=="linear":
            # Required low-capacity baseline.  No hidden layer or interaction is
            # introduced: this is multinomial/Poisson linear regression on the
            # exact input representation used by the corresponding deep model.
            in_dim=4*201 if modality=="raw" else N_POS*N_MOTIFS
            self.linear=nn.Linear(in_dim,N_BINS)
            pool_dim=None
        elif modality=="raw":
            if A.backbone != "cnn": raise ValueError("raw modality currently supports only the CNN backbone")
            self.branches=nn.ModuleList([nn.Sequential(nn.Conv1d(4,64,k,padding=k//2),nn.BatchNorm1d(64),activation_module()) for k in (7,15,31)])
            self.stem=nn.Sequential(nn.Conv1d(192,128,1),nn.BatchNorm1d(128),activation_module())
            self.blocks=nn.Sequential(*[ResBlock(d) for d in (1,2,4,8)])
            pool_dim=256
        elif A.backbone=="transformer":
            d=A.transformer_d_model
            if d % A.transformer_heads: raise ValueError("transformer d_model must be divisible by heads")
            self.score_proj=nn.Linear(N_MOTIFS,d,bias=False)
            self.pres_proj=nn.Linear(N_MOTIFS,d,bias=False) if A.presence else None
            self.input_norm=nn.LayerNorm(d)
            self.pos_embed=nn.Parameter(torch.randn(1,N_POS,d)*0.02)
            layer=nn.TransformerEncoderLayer(d_model=d,nhead=A.transformer_heads,
                dim_feedforward=4*d,dropout=.1,activation="gelu",batch_first=True,norm_first=True)
            self.transformer=nn.TransformerEncoder(layer,num_layers=A.transformer_layers,norm=nn.LayerNorm(d))
            pool_dim=2*d
        else:
            self.score_proj=nn.Linear(N_MOTIFS,128,bias=False)
            self.pres_proj=nn.Linear(N_MOTIFS,128,bias=False) if A.presence else None
            self.stem_norm=nn.BatchNorm1d(128)
            self.blocks=nn.Sequential(*[ResBlock(d) for d in (1,2,4,8)])
            pool_dim=256
        self.head=(None if A.backbone=="linear" else
                   nn.Sequential(nn.Linear(pool_dim,128),activation_module(),nn.Dropout(.1),nn.Linear(128,N_BINS)))
        self.total_head=(nn.Sequential(nn.Linear(pool_dim,128),activation_module(),nn.Dropout(.1),nn.Linear(128,1))
                         if A.objective.startswith("factorized_") and A.backbone!="linear" else None)
    def forward(self,x):
        if A.backbone=="linear":
            profile=self.linear(x.reshape(x.shape[0],-1))
            return profile
        if self.modality=="raw": x=self.stem(torch.cat([b(x) for b in self.branches],1))
        elif A.backbone=="transformer":
            x=self.score_proj(x)+(self.pres_proj((x>0).to(x.dtype)) if self.pres_proj is not None else 0)
            x=self.transformer(self.input_norm(x)+self.pos_embed)
            z=torch.cat([x.mean(1),x.max(1).values],1)
            profile=self.head(z)
            return (profile,self.total_head(z).squeeze(1)) if self.total_head is not None else profile
        else:
            x=self.score_proj(x) + (self.pres_proj((x>0).to(x.dtype)) if self.pres_proj is not None else 0)
            x=activate(self.stem_norm(x.transpose(1,2)))
        x=self.blocks(x); z=torch.cat([x.mean(2),x.max(2).values],1)
        profile=self.head(z)
        return (profile,self.total_head(z).squeeze(1)) if self.total_head is not None else profile

net=SharedShapeNet(A.modality).to(device)
opt=torch.optim.AdamW(net.parameters(),lr=A.lr,weight_decay=1e-4)
scaler=torch.amp.GradScaler("cuda")

def targets(idx):
    c=torch.as_tensor(np.asarray(Y[idx],dtype=np.float32),device=device)
    p=c/c.sum(1,keepdim=True).clamp_min(1.)
    return c,p

def loss_fn(output,c,p):
    if A.objective.startswith("factorized_"):
        logits,total_raw=output
        total=F.softplus(total_raw.float(),beta=1.0,threshold=20.0)+1e-4
        n=c.sum(1); logq=F.log_softmax(logits.float(),1)
        magnitude=(total-n*torch.log(total)).mean()/POISSON_SCALE
        if A.objective=="factorized_poisson":
            shape=-(c*logq).sum(1).mean()/POISSON_SCALE
        else:
            shape=-(p*logq).sum(1).mean()*A.shape_weight
        return magnitude+shape
    logits=output
    if A.objective=="normalized_poisson":
        # Literal reading of the recommendation: normalize each observed row
        # to 100 pseudo-counts, then fit 20 free Poisson rates through a
        # Softplus link.  Their predicted sum is learned rather than imposed.
        pseudo_count=p*A.shape_total
        mu=F.softplus(logits.float(),beta=1.0,threshold=20.0)+1e-4
        return (mu-pseudo_count*torch.log(mu)).sum(1).mean()/A.shape_total
    if A.objective=="normalized_poisson_fixed":
        # Each row is converted to pseudo-counts with a fixed total (100 by
        # default), and the predicted rates are constrained to the same total.
        # Dividing by the total makes this exactly equal to equal-region CE up
        # to constants, while retaining the requested Poisson formulation.
        logq=F.log_softmax(logits.float(),1)
        pseudo_count=p*A.shape_total
        log_mu=math.log(A.shape_total)+logq
        return (A.shape_total-(pseudo_count*log_mu).sum(1)).mean()/A.shape_total
    if A.objective=="poisson":
        mu=F.softplus(logits.float(),beta=1.0,threshold=20.0)+1e-4
        return (mu-c*torch.log(mu)).sum(1).mean()/POISSON_SCALE
    logq=F.log_softmax(logits.float(),1)
    if A.objective=="mnll": return -(c*logq).sum(1).mean()/c.sum(1).mean().detach().clamp_min(1.)
    return -(p*logq).sum(1).mean()

@torch.no_grad()
def evaluate(idx, save_predictions=False):
    net.eval(); ps=[]; qs=[]; ids=[]; losses=[]; true_totals=[]; pred_totals=[]
    for s in tqdm(range(0,len(idx),A.batch_size),desc="eval"):
        ii=idx[s:s+A.batch_size]; x=get_x(ii); c,p=targets(ii)
        with torch.autocast("cuda",dtype=torch.float16): output=net(x)
        losses.append((loss_fn(output,c,p).item(),len(ii)))
        if A.objective.startswith("factorized_"):
            logits,total_raw=output
            q=torch.softmax(logits.float(),1)
            total=F.softplus(total_raw.float(),beta=1.0,threshold=20.0)+1e-4
            true_totals.append(c.sum(1).cpu().numpy()); pred_totals.append(total.cpu().numpy())
        elif A.objective=="normalized_poisson":
            logits=output
            mu=F.softplus(logits.float(),beta=1.0,threshold=20.0)+1e-4
            q=mu/mu.sum(1,keepdim=True).clamp_min(1e-8)
            pred_totals.append(mu.sum(1).cpu().numpy())
        elif A.objective=="normalized_poisson_fixed":
            logits=output
            q=torch.softmax(logits.float(),1)
        elif A.objective=="poisson":
            logits=output
            mu=F.softplus(logits.float(),beta=1.0,threshold=20.0)+1e-4
            q=mu/mu.sum(1,keepdim=True).clamp_min(1e-8)
            true_totals.append(c.sum(1).cpu().numpy()); pred_totals.append(mu.sum(1).cpu().numpy())
        else:
            logits=output
            q=torch.softmax(logits.float(),1)
        ps.append(p.cpu().numpy()); qs.append(q.cpu().numpy()); ids.append(ii)
    p=np.vstack(ps); q=np.vstack(qs); ids=np.concatenate(ids); eps=1e-8
    pc=p-p.mean(1,keepdims=True); qc=q-q.mean(1,keepdims=True)
    corr=(pc*qc).sum(1)/np.maximum(np.sqrt((pc*pc).sum(1)*(qc*qc).sum(1)),eps)
    js=.5*((p*np.log((p+eps)/(.5*(p+q)+eps))).sum(1)+(q*np.log((q+eps)/(.5*(p+q)+eps))).sum(1))
    cdf_err=np.abs(np.cumsum(p,1)-np.cumsum(q,1)).sum(1)
    com=np.abs((p*np.arange(N_BINS)).sum(1)-(q*np.arange(N_BINS)).sum(1))
    peak=np.abs(np.argmax(p,1)-np.argmax(q,1))
    sse=((p-q)**2).sum(0); sst=((p-p.mean(0))**2).sum(0); r2=1-sse/np.maximum(sst,eps)
    m={"loss":sum(x*n for x,n in losses)/sum(n for _,n in losses),"n":len(p),"mean_bin_r2":float(r2.mean()),
       "r2_bins":r2.tolist(),"rmse":float(np.sqrt(np.mean((p-q)**2))),"pearson_mean":float(corr.mean()),
       "pearson_median":float(np.median(corr)),"js_mean":float(js.mean()),"wasserstein_bins_mean":float(cdf_err.mean()),
       "center_of_mass_mae_bins":float(com.mean()),"peak_bin_mae":float(peak.mean()),"peak_bin_accuracy":float((peak==0).mean())}
    if A.objective=="normalized_poisson":
        u=np.concatenate(pred_totals)
        m.update({"predicted_pseudocount_total_mean":float(u.mean()),
                  "predicted_pseudocount_total_sd":float(u.std())})
    extra={}
    if A.objective=="poisson" or A.objective.startswith("factorized_"):
        t=np.concatenate(true_totals); u=np.concatenate(pred_totals)
        t_sst=((t-t.mean())**2).sum(); lt=np.log1p(t); lu=np.log1p(u)
        m.update({"magnitude_r2":float(1-((t-u)**2).sum()/max(t_sst,eps)),
                  "magnitude_rmse":float(np.sqrt(np.mean((t-u)**2))),
                  "log_magnitude_r2":float(1-((lt-lu)**2).sum()/max(((lt-lt.mean())**2).sum(),eps)),
                  "log_magnitude_pearson":float(np.corrcoef(lt,lu)[0,1])})
        true_count=p*t[:,None]; pred_count=q*u[:,None]
        count_sse=((true_count-pred_count)**2).sum(0)
        count_sst=((true_count-true_count.mean(0))**2).sum(0)
        count_r2=1-count_sse/np.maximum(count_sst,eps)
        m.update({"count_mean_bin_r2":float(count_r2.mean()),
                  "count_r2_bins":count_r2.tolist(),
                  "count_rmse":float(np.sqrt(np.mean((true_count-pred_count)**2)))})
        extra={"true_total":t,"pred_total":u}
    if save_predictions: np.savez_compressed(OUT/"val_predictions.npz",indices=ids,true_profile=p,pred_profile=q,pearson=corr,js=js,wasserstein=cdf_err,com_error=com,peak_error=peak,**extra)
    return m

cfg=vars(A)|{"seed":seed,"train_n":len(train_idx),"val_n":len(val_idx),"n_parameters":sum(x.numel() for x in net.parameters()),"selected_total":int(selected.sum()),"poisson_scale":POISSON_SCALE}
(OUT/"config.json").write_text(json.dumps(cfg,indent=2))
print(json.dumps(cfg,indent=2)); best=float("inf"); state=None; wait=0; best_epoch=0
for epoch in range(1,A.epochs+1):
    net.train(); order=np.random.permutation(train_idx); opt.zero_grad(set_to_none=True)
    bar=tqdm(range(0,len(order),A.batch_size),desc=f"epoch {epoch}")
    for step,s in enumerate(bar,1):
        ii=order[s:s+A.batch_size]; x=get_x(ii,True); c,p=targets(ii)
        with torch.autocast("cuda",dtype=torch.float16): raw=loss_fn(net(x),c,p); loss=raw/A.accum_steps
        scaler.scale(loss).backward()
        if step%A.accum_steps==0 or s+A.batch_size>=len(order):
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(),1.); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        bar.set_postfix(loss=f"{raw.item():.4f}")
    met=evaluate(val_idx); print("VAL",json.dumps(met))
    if met["loss"]<best-1e-6: best=met["loss"]; state=copy.deepcopy(net.state_dict()); best_epoch=epoch; wait=0
    else: wait+=1
    if wait>=A.patience: break
net.load_state_dict(state); torch.save({"state_dict":state,"config":cfg,"best_epoch":best_epoch},OUT/"model.pt")
metrics=evaluate(val_idx,True)|{"best_epoch":best_epoch}
(OUT/"metrics.json").write_text(json.dumps(metrics,indent=2)); print("FINAL",json.dumps(metrics,indent=2))
