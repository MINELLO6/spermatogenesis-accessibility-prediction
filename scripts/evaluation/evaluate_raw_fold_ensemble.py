#!/usr/bin/env python3
import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path('/root/sc-motif-open/R')
FINAL = Path('/root/autodl-tmp/final_analysis')
parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=['raw_rc', 'raw'], required=True)
parser.add_argument('--batch-size', type=int, default=4096)
args = parser.parse_args()

name = 'raw_rc_5fold_ensemble' if args.variant == 'raw_rc' else 'raw_no_rc_5fold_ensemble'
out = FINAL / 'heldout_ensembles' / name
out.mkdir(parents=True, exist_ok=True)
if (out / 'LOCKED_COMPLETE').exists():
    raise SystemExit('Already complete; refusing to overwrite')

with open(ROOT / 'totmat_shape.txt') as h:
    n, t = map(int, h.read().split())
targets = np.memmap(ROOT / 'totmat_f64.bin', dtype='<f8', mode='r', shape=(n, t), order='F')
seqs = np.memmap(ROOT / 'fullseqs.txt', dtype='S202', mode='r', shape=(n,))
heldout = np.load(FINAL / 'heldout_test_idx_LOCKED.npy')

class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__(); self.conv1=nn.Conv1d(c,c,3,padding=d,dilation=d); self.conv2=nn.Conv1d(c,c,3,padding=d,dilation=d); self.bn1=nn.BatchNorm1d(c); self.bn2=nn.BatchNorm1d(c)
    def forward(self,x):
        r=x; x=F.relu(self.bn1(self.conv1(x))); return F.relu(self.bn2(self.conv2(x))+r)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def br(k): return nn.Sequential(nn.Conv1d(4,64,k,padding=k//2),nn.BatchNorm1d(64),nn.ReLU())
        self.branch7,self.branch15,self.branch31=br(7),br(15),br(31)
        self.project=nn.Sequential(nn.Conv1d(192,128,1),nn.BatchNorm1d(128),nn.ReLU())
        self.blocks=nn.Sequential(*[Block(128,d) for d in (1,2,4,8)])
        self.head=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,20))
    def forward(self,x):
        x=self.project(torch.cat([self.branch7(x),self.branch15(x),self.branch31(x)],1)); x=self.blocks(x)
        return self.head(torch.cat([x.mean(2),x.max(2).values],1))
def encode(idx):
    raw=np.ascontiguousarray(np.asarray(seqs[idx])); b=raw.view(np.uint8).reshape(len(idx),202)[:,:201]
    x=np.stack([b==ord('A'),b==ord('C'),b==ord('G'),b==ord('T')],1)
    return torch.from_numpy(x.astype(np.float32,copy=False))

device=torch.device('cuda:0')
ensemble=np.lib.format.open_memmap(out/'heldout_predictions_f32.npy',mode='w+',dtype=np.float32,shape=(len(heldout),20))
ensemble[:] = 0
for fold in range(1,6):
    path=ROOT/'results'/f'{args.variant}_fold{fold}.pt'
    state=torch.load(path,map_location='cpu',weights_only=True)
    net=Net().to(device); net.load_state_dict(state); net.eval()
    with torch.inference_mode():
        for start in range(0,len(heldout),args.batch_size):
            idx=heldout[start:start+args.batch_size]
            with torch.autocast('cuda',dtype=torch.float16): p=net(encode(idx).to(device)).float().cpu().numpy()
            ensemble[start:start+len(idx)] += p/5.0
    ensemble.flush(); del net; torch.cuda.empty_cache(); print('fold',fold,'complete',flush=True)

sse=np.zeros(20); sy=np.zeros(20); sy2=np.zeros(20)
for start in range(0,len(heldout),100_000):
    idx=heldout[start:start+100_000]; y=np.asarray(targets[idx],dtype=np.float64); p=np.asarray(ensemble[start:start+len(idx)],dtype=np.float64)
    sse+=np.square(p-y).sum(0); sy+=y.sum(0); sy2+=np.square(y).sum(0)
sst=sy2-np.square(sy)/len(heldout); r2=1-sse/sst; mse=sse.sum()/(len(heldout)*20)
metrics={'name':name,'folds':5,'aggregation':'equal prediction mean','heldout_evaluations':1,'mse':float(mse),'rmse':float(math.sqrt(mse)),'mean_r2':float(r2.mean()),'r2_bins':r2.tolist()}
with open(out/'metrics.json','w') as h: json.dump(metrics,h,indent=2)
with open(out/'LOCKED_COMPLETE','w') as h: h.write('Five-fold heldout ensemble complete.\n')
print(json.dumps(metrics,indent=2))
