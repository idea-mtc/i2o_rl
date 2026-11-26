"""Episode splits, official JEPA inputs and a CPU-only alignment check for open_drawer."""
import argparse
import bisect
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, default_collate

COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
CAMERAS = ("front", "wrist", "left_shoulder", "right_shoulder")


def prepare(root, seed):
    groups = defaultdict(list)
    experts = json.loads((root / "dataset/manifest.json").read_text())["episodes"]
    paths = sorted((root / "dataset").glob("*.npz"))
    assert {x["file"] for x in experts} == {p.name for p in paths} and paths
    online = sorted((root / "online_episodes").glob("*.npz"))
    assert online, "No online trajectory files"
    for path in paths + online:
        with np.load(path, allow_pickle=False) as data:
            actions = data["raw_actions"]
            n = len(actions)
            state, times = data["state"], data["sim_time"]
            assert n > 0 and actions.shape == (n, 8) and state.shape == (n+1, 15), path
            assert np.isfinite(actions).all() and np.isfinite(state).all(), path
            assert times.shape == (n+1,) and np.all(np.diff(times) > 0), path
            assert all(data[k].shape == (n,) for k in ("rewards", "terminated", "truncated")), path
            assert not (data["terminated"][:-1] | data["truncated"][:-1]).any(), path
            group = "expert" if path.parent.name == "dataset" else (
                "online_positive" if data["rewards"][-1] > 0 else "online_other")
            groups[group].append(dict(file=str(path.relative_to(root)), steps=n, group=group))
    rng = random.Random(seed)
    splits = {name: [] for name in ("train", "val", "test")}
    for group in sorted(groups):
        rows = groups[group]
        rng.shuffle(rows)
        hold = round(len(rows)*0.1)
        for name, subset in zip(splits, (rows[2*hold:], rows[:hold], rows[hold:2*hold])):
            splits[name].extend(subset)
    return dict(seed=seed, upstream_commit=COMMIT, cameras=CAMERAS, frameskip=1, splits=splits)


def mosaic(frame):
    assert frame.shape == (4, 3, 84, 84) and frame.dtype == np.uint8
    return frame.reshape(2, 2, 3, 84, 84).transpose(0, 3, 1, 4, 2).reshape(168, 168, 3)


def pixels(frame):
    image = Image.fromarray(mosaic(frame)).resize((224, 224), Image.Resampling.BILINEAR)
    rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
    return (rgb - torch.tensor([.485,.456,.406])[:,None,None]) / torch.tensor([.229,.224,.225])[:,None,None]


class DrawerTransitions(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = root, rows
        self.ends = np.cumsum([r["steps"] for r in rows])
        stats = json.loads((root / "action_statistics.json").read_text())
        self.low = np.asarray(stats["expanded_min"], np.float32)
        self.high = np.asarray(stats["expanded_max"], np.float32)

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = bisect.bisect_right(self.ends, index)
        t = int(index - (self.ends[row-1] if row else 0))
        file = self.rows[row]["file"]
        with np.load(self.root / file, allow_pickle=False) as data:
            frames = data["rgb"]
            assert len(frames) == self.rows[row]["steps"] + 1
            action = data["raw_actions"][t:t+1]
            return dict(pixels=torch.stack([pixels(f) for f in frames[t:t+2]]),
                        action=torch.from_numpy(2*(action-self.low)/(self.high-self.low)-1),
                        file=file, t=t)


def check(root, manifest, out):
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
    from transformers import ViTConfig, ViTModel
    rows = manifest["splits"]["val"]
    dataset = DrawerTransitions(root, rows)
    selected = [i for group in ("expert", "online_positive", "online_other")
                for i in [next(i for i,r in enumerate(rows) if r["group"] == group)]]
    batch = default_collate([dataset[int(dataset.ends[i]-rows[i]["steps"] + rows[i]["steps"]//2)] for i in selected])
    sheet = Image.new("RGB",(4*84,3*2*108),"white")
    draw = ImageDraw.Draw(sheet)
    samples = []
    for b,(file,tensor_t) in enumerate(zip(batch["file"],batch["t"])):
        t = int(tensor_t)
        with np.load(root / file) as data:
            frames = data["rgb"][t:t+2]
            times = data["sim_time"][t:t+2]
            raw = data["raw_actions"][t]
            decoded = (batch["action"][b,0].numpy()+1)*.5*(dataset.high-dataset.low)+dataset.low
            np.testing.assert_allclose(decoded,raw,atol=1e-6)
            changes = np.abs(frames[1].astype(float)-frames[0]).mean((1,2,3))
            for k in range(2):
                torch.testing.assert_close(batch["pixels"][b,k], pixels(frames[k]),rtol=0,atol=0)
                y=(2*b+k)*108
                for j,name in enumerate(CAMERAS):
                    frame = frames[k,j].transpose(1,2,0)
                    np.testing.assert_array_equal(mosaic(frames[k])[(j//2)*84:(j//2+1)*84,(j%2)*84:(j%2+1)*84],frame)
                    sheet.paste(Image.fromarray(frame),(84*j,y+24))
                    draw.text((84*j,y+11),name,fill="black")
                draw.text((0,y),f"{rows[selected[b]]['group']}  frame {t+k}  sim {times[k]:.3f}",fill="black")
            samples.append(dict(file=file,t=t,times=times.tolist(),raw_action=raw.tolist(),
                                measured_joint_delta=np.diff(data["state"][t:t+2,:7],axis=0)[0].tolist(),
                                camera_mean_abs_pixel_change=changes.tolist()))
    sheet.save(out/"batch.png")
    projector=lambda: MLP(input_dim=192,output_dim=192,hidden_dim=2048,norm_fn=torch.nn.BatchNorm1d)
    model=JEPA(ViTModel(ViTConfig(image_size=224,patch_size=14,hidden_size=192,
                    intermediate_size=768,num_attention_heads=3,num_hidden_layers=12)),
               ARPredictor(num_frames=1,depth=6,heads=16,mlp_dim=2048,input_dim=192,
                           hidden_dim=192,output_dim=192,dropout=.1),
               Embedder(input_dim=8,emb_dim=192),projector(),projector()).eval()
    with torch.inference_mode():
        result=model.encode(dict(pixels=batch["pixels"],action=batch["action"]))
        pred=model.predict(result["emb"][:,:1],result["act_emb"])
        assert pred.shape==result["emb"][:,1:].shape and torch.isfinite(pred).all()
    return dict(samples=samples, shapes={k:list(v.shape) for k,v in result.items()},
                prediction=list(pred.shape), updates=0,device="cpu")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("root","upstream","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.upstream.resolve()))
    torch.set_num_threads(2)
    torch.manual_seed(3072)
    manifest=prepare(args.root,3072)
    (args.out/"splits.json").write_text(json.dumps(manifest,indent=2))
    report=check(args.root,manifest,args.out)
    report["groups"]={k:dict(Counter(r["group"] for r in v)) for k,v in manifest["splits"].items()}
    report["steps"]={k:sum(r["steps"] for r in v) for k,v in manifest["splits"].items()}
    report["upstream_sha256"]={f:hashlib.sha256((args.upstream/f).read_bytes()).hexdigest() for f in ("jepa.py","module.py")}
    (args.out/"batch.json").write_text(json.dumps(report,indent=2))
    print(report["groups"],report["steps"],report["shapes"])
