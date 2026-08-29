#!/usr/bin/env python3
import json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

ROOT=Path('/root/sc-motif-open/R'); FINAL=Path('/root/autodl-tmp/final_analysis')
OUT=FINAL/'heldout_ensembles/frozen_nt_attnpool_5fold_ensemble'; OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'LOCKED_COMPLETE').exists(): raise SystemExit('Already complete')
MODEL_PATH=Path('/root/.cache/huggingface/hub/models--InstaDeepAI--nucleotide-transformer-v2-50m-multi-species/snapshots/81b29e5786726d891dbf929404ef20adca5b36f1')
with open(ROOT/'totmat_shape.txt') as h: n,t=map(int,h.read().split())
targets=np.memmap(ROOT/'totmat_f64.bin',dtype='<f8',mode='r',shape=(n,t),order='F')
seqs=np.memmap(ROOT/'fullseqs.txt',dtype='S202',mode='r',shape=(n,))
heldout=np.load(FINAL/'heldout_test_idx_LOCKED.npy')

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.LayerNorm(256))
        self.attn=nn.Sequential(nn.Linear(256,128),nn.Tanh(),nn.Linear(128,1))
        self.head=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,20))
    def forward(self,x,mask):
        x=self.proj(x); score=self.attn(x).squeeze(-1).masked_fill(~mask,-1e4); w=torch.softmax(score,1)
        ap=(x*w.unsqueeze(-1)).sum(1); mp=x.masked_fill(~mask.unsqueeze(-1),-1e4).max(1).values
        return self.head(torch.cat([ap,mp],-1))

device=torch.device('cuda:0'); tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,trust_remote_code=True,local_files_only=True)
backbone=AutoModelForMaskedLM.from_pretrained(MODEL_PATH,trust_remote_code=True,local_files_only=True).to(device).eval()
heads=[]
for fold in range(1,6):
    h=Head().to(device); h.load_state_dict(torch.load(ROOT/'results'/f'nt_attnpool_fold{fold}.pt',map_location='cpu',weights_only=True)); h.eval(); heads.append(h)
pred=np.lib.format.open_memmap(OUT/'heldout_predictions_f32.npy',mode='w+',dtype=np.float32,shape=(len(heldout),20))
batch=512
with torch.inference_mode():
    for start in range(0,len(heldout),batch):
        idx=heldout[start:start+batch]; strings=[seqs[i].decode('ascii').strip() for i in idx]
        enc=tokenizer(strings,padding=True,truncation=True,return_tensors='pt',return_special_tokens_mask=True)
        enc={k:v.to(device) for k,v in enc.items()}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            o=backbone(input_ids=enc['input_ids'],attention_mask=enc['attention_mask'],output_hidden_states=True,return_dict=True)
            x=o.hidden_states[-1]; mask=enc['attention_mask'].bool().clone(); mask[:,0]=False
            p=sum(h(x,mask).float() for h in heads)/5
        pred[start:start+len(idx)]=p.cpu().numpy()
        if start%(batch*100)==0: print(start,'/',len(heldout),flush=True)
pred.flush()
sse=np.zeros(20); sy=np.zeros(20); sy2=np.zeros(20)
for start in range(0,len(heldout),100000):
    idx=heldout[start:start+100000]; y=np.asarray(targets[idx],dtype=np.float64); p=np.asarray(pred[start:start+len(idx)],dtype=np.float64)
    sse+=np.square(p-y).sum(0); sy+=y.sum(0); sy2+=np.square(y).sum(0)
sst=sy2-sy**2/len(heldout); r2=1-sse/sst; mse=sse.sum()/(len(heldout)*20)
m={'name':'frozen_nt_attnpool_5fold_ensemble','folds':5,'aggregation':'equal prediction mean','heldout_evaluations':1,'mse':float(mse),'rmse':float(math.sqrt(mse)),'mean_r2':float(r2.mean()),'r2_bins':r2.tolist()}
with open(OUT/'metrics.json','w') as h: json.dump(m,h,indent=2)
with open(OUT/'LOCKED_COMPLETE','w') as h: h.write('Five-fold heldout ensemble complete.\n')
print(json.dumps(m,indent=2))
