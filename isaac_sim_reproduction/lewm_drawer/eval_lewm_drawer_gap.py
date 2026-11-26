"""Post-hoc front-view opening evidence and train-only closed-form linear readout."""
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import train_lewm_drawer as run
from lewm_drawer_data import DrawerTransitions


def summarize(records):
    result={}
    for name, rows in records.items():
        a=np.asarray(rows)
        result[name]=dict(n=len(a),model_latent_mse=a[:,0].mean(),persistence_latent_mse=a[:,1].mean(),
                          model_gap_delta_mae=a[:,2].mean(),persistence_gap_delta_mae=a[:,3].mean(),
                          true_frame_readout_gap_delta_mae=a[:,4].mean(),true_mean_delta=a[:,5].mean(),
                          model_mean_delta=a[:,6].mean(),actual_next_gap_readout_mae=a[:,7].mean())
    return result


@torch.inference_mode()
def encode_episode(model, root, row, adapter):
    with np.load(root/row["file"]) as f:
        frames=f["rgb"]
        action=torch.from_numpy(2*(f["raw_actions"]-adapter.low)/(adapter.high-adapter.low)-1).cuda()
        state=f["state"]
        pulling=(f["raw_actions"][:,7]<=.5)&(np.diff(state[:,8])<-.001)
    # Interior dark slit on front camera; a unit is mean grayscale deficit below 100.
    gap=np.maximum(0,100-frames[:,0,:,21:42,41:53].mean(1)).mean((1,2))
    from lewm_drawer_data import pixels
    zs=[]
    for start in range(0,len(frames),48):
        images=torch.stack([pixels(x) for x in frames[start:start+48]]).cuda()
        zs.append(model.encode({"pixels":images[:,None]})["emb"][:,0])
    z=torch.cat(zs)
    p=model.predict(z[:-1,None],model.action_encoder(action[:,None]))[:,0]
    return z.cpu().numpy(),p.cpu().numpy(),gap,pulling


if __name__=="__main__":
    out=Path(sys.argv[1])
    uuid=os.environ["CUDA_VISIBLE_DEVICES"]
    gpu=int(subprocess.check_output(["nvidia-smi","-i",uuid,"--query-gpu=index","--format=csv,noheader"]))
    assert gpu in (2,3,4) and torch.cuda.device_count()==1
    assert str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-")==uuid.lower().removeprefix("gpu-")
    torch.set_num_threads(2)
    with open(run.UPSTREAM/"audit/splits.json") as f: splits=json.load(f)["splits"]
    model=run.build_model().cuda().eval()
    ckpt=torch.load(out/"checkpoint.pt",weights_only=True,map_location="cpu")
    assert ckpt["step"]==1000
    model.load_state_dict(ckpt["model"])
    adapter=DrawerTransitions(run.DATA,splits["train"])
    # Deterministic train-only readout calibration, uniformly across manifest rows.
    X,Y=[],[]
    for row in splits["train"][::6]:
        z,_,gap,_=encode_episode(model,run.DATA,row,adapter)
        X.append(z);Y.append(gap)
    x=np.concatenate(X).astype(float);y=np.concatenate(Y)
    center=x.mean(0);offset=y.mean();x-=center
    weights=np.linalg.solve(x.T@x+np.eye(x.shape[1])*len(x)*.0001,x.T@(y-offset))
    metrics=defaultdict(list)
    with (out/"gap_validation.csv").open("w") as f:
        writer=csv.writer(f)
        writer.writerow(["episode","t","gap_now","gap_next","pull_proxy","visible_open","true_delta","predicted_delta","model_latent_mse","persistence_latent_mse"])
        for row in splits["val"]:
            z,p,gap,pull=encode_episode(model,run.DATA,row,adapter)
            d=np.diff(gap)
            # Local 3-transition slope rejects isolated flicker; require actual outward TCP motion too.
            visible=(np.convolve(d,[1/3]*3,"same")>.10)&pull if len(d)>=3 else np.zeros(len(d),bool)
            pred=(p-z[:-1])@weights
            oracle=(z[1:]-z[:-1])@weights
            m=((p-z[1:])**2).mean(1);b=((z[:-1]-z[1:])**2).mean(1)
            for i in range(len(d)):
                writer.writerow([row["file"],i,gap[i],gap[i+1],int(pull[i]),int(visible[i]),d[i],pred[i],m[i],b[i]])
                values=[m[i],b[i],abs(pred[i]-d[i]),abs(d[i]),abs(oracle[i]-d[i]),d[i],pred[i],abs((z[i+1]-center)@weights+offset-gap[i+1])]
                metrics["all"].append(values)
                if pull[i]:metrics["pull_proxy"].append(values)
                if visible[i]:metrics["visible_open_and_pull"].append(values)
    (out/"gap_report.json").write_text(json.dumps(summarize(metrics),indent=2))
