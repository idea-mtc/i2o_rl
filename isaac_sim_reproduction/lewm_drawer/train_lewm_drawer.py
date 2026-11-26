"""Standalone bounded LeWM run on the previously audited drawer split."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from transformers import ViTConfig, ViTModel
from lewm_drawer_data import mosaic, DrawerTransitions

UPSTREAM = Path(__file__).resolve().parent
DATA = Path(os.environ.get("LEWM_DRAWER_DATA",
            UPSTREAM / "data"))
OUT = None
SEED = 3072


def save(name, data):
    temp = OUT / (name + ".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(OUT / name)


def load(rows):
    adapter = DrawerTransitions(DATA, rows)
    images = []
    starts, actions, pulls, episodes = [], [], [], []
    offset = 0
    for row in rows:
        with np.load(DATA / row["file"]) as d:
            raw, state = d["raw_actions"], d["state"]
            n = len(raw)
            images.extend(np.asarray(Image.fromarray(mosaic(frame)).resize(
                (224,224),Image.Resampling.BILINEAR)) for frame in d["rgb"])
            starts.extend(range(offset,offset+n))
            actions.extend(2*(raw-adapter.low)/(adapter.high-adapter.low)-1)
            # World -x opens the native drawer. Closure and TCP motion are a proxy.
            pulls.extend((raw[:,7]<=.5) & (np.diff(state[:,8]) < -.001))
            episodes.extend([row["file"]]*n)
            offset += n+1
    return (torch.from_numpy(np.stack(images)),torch.tensor(starts),
            torch.from_numpy(np.stack(actions)),np.array(pulls),np.array(episodes))


def batch(data, indices):
    frames, starts, actions = data[:3]
    t = starts[indices]
    images = frames[torch.stack([t,t+1],1)].to("cuda").permute(0,1,4,2,3).float()/255
    images = (images - MEAN)/STD
    return dict(pixels=images,action=actions[indices,None].to("cuda"))


def forward(model, inputs):
    emb = model.encode(inputs)
    prediction = model.predict(emb["emb"][:,:1],emb["act_emb"])
    return emb["emb"].float(),prediction[:,0].float()


def evaluate(model,data):
    errors=[]
    model.eval()
    with torch.inference_mode():
        for idx in torch.arange(len(data[1])).split(32):
            z,p=forward(model,batch(data,idx))
            errors.extend(torch.stack(((p-z[:,1]).square().mean(1),
                                       (z[:,0]-z[:,1]).square().mean(1)),1).cpu().tolist())
    errors=np.array(errors)
    with (OUT/"validation.csv").open("w") as f:
        writer=csv.writer(f)
        writer.writerow(["episode","step","pull_proxy","model_mse","persistence_mse"])
        last=None
        for name,pull,pair in zip(data[4],data[3],errors):
            step=0 if name!=last else step+1
            writer.writerow([name,step,int(pull),*pair])
            last=name
    report={}
    for label,mask in [("all",np.ones(len(errors),bool)),("pull_proxy",data[3]),("other",~data[3])]:
        e=errors[mask]
        report[label]=dict(n=len(e),episodes=len(set(data[4][mask])),model=e[:,0].mean(),
                            persistence=e[:,1].mean(),fraction_better=(e[:,0]<e[:,1]).mean())
    return report


def build_model():
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    proj=lambda: MLP(input_dim=192,output_dim=192,hidden_dim=2048,norm_fn=nn.BatchNorm1d)
    return JEPA(ViTModel(ViTConfig(image_size=224,patch_size=14,hidden_size=192,
              intermediate_size=768,num_attention_heads=3,num_hidden_layers=12)),
              ARPredictor(num_frames=1,depth=6,heads=16,mlp_dim=2048,input_dim=192,
              hidden_dim=192,output_dim=192,dropout=.1),Embedder(input_dim=8,emb_dim=192),proj(),proj())


if __name__=="__main__":
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP, SIGReg
    OUT = Path(sys.argv[1])
    OUT.mkdir(exist_ok=False)
    uuid=os.environ["CUDA_VISIBLE_DEVICES"]
    gpu=int(subprocess.check_output(["nvidia-smi","-i",uuid,"--query-gpu=index","--format=csv,noheader"]))
    assert gpu in [2,3,4] and torch.cuda.device_count()==1
    assert str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-")==uuid.lower().removeprefix("gpu-")
    torch.manual_seed(SEED)
    torch.set_num_threads(3)
    audit=UPSTREAM/"audit"
    splits=json.loads((audit/"splits.json").read_text())["splits"]
    for file,digest in json.loads((audit/"batch.json").read_text())["upstream_sha256"].items():
        assert hashlib.sha256((UPSTREAM/file).read_bytes()).hexdigest()==digest
    save("config.json", dict(gpu=gpu,uuid=uuid,seed=SEED,batch_size=32,updates=1000,lr=5e-5,
        weight_decay=.001,sigreg=.09,history=1,frameskip=1,split_sha256=hashlib.sha256((audit/"splits.json").read_bytes()).hexdigest()))
    MEAN=torch.tensor([.485,.456,.406],device="cuda").view(3,1,1)
    STD=torch.tensor([.229,.224,.225],device="cuda").view(3,1,1)
    model=build_model().cuda()
    reg=SIGReg().cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.001)
    print("Caching train episodes",flush=True)
    train=load(splits["train"])
    start=time.monotonic()
    for step in range(1,1001):
        idx=torch.randint(len(train[1]),(32,))
        with torch.autocast("cuda",dtype=torch.bfloat16):
            z,p=forward(model,batch(train,idx))
        mse=(p-z[:,1]).square().mean()
        loss=mse+.09*reg(z.transpose(0,1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        opt.step()
        if step%50==0 or step==1:
            print(step, float(loss.detach()),float(mse.detach()),round(time.monotonic()-start,1),flush=True)
    torch.save(dict(model=model.state_dict(),step=step),OUT/"checkpoint.pt")
    del train
    metrics=evaluate(model,load(splits["val"]))
    save("result.json",dict(updates=step,metrics=metrics))
    print(metrics,flush=True)
