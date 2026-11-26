"""Three-frame LeWM on complete, stride-five windows of audited drawer episodes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch import nn
from transformers import ViTConfig, ViTModel

import train_lewm_drawer as base
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg

STRIDE = 5
HISTORY = 3
SPAN = HISTORY * STRIDE
BATCH = 32


def build_model():
    def projector():
        return MLP(input_dim=192, output_dim=192, hidden_dim=2048, norm_fn=nn.BatchNorm1d)

    encoder = ViTModel(ViTConfig(image_size=224, patch_size=14, hidden_size=192,
                                 intermediate_size=768, num_attention_heads=3, num_hidden_layers=12))
    predictor = ARPredictor(num_frames=HISTORY, depth=6, heads=16, mlp_dim=2048,
                            input_dim=192, hidden_dim=192, output_dim=192, dropout=.1)
    return JEPA(encoder, predictor, Embedder(input_dim=8 * STRIDE, emb_dim=192),
                projector(), projector())


def windows(data):
    names = data[4]
    starts = torch.arange(max(0, len(names) - SPAN + 1))
    # Each filename occurs in one contiguous run. The last required action is start+14.
    return starts[names[starts.numpy()] == names[(starts + SPAN - 1).numpy()]]


def make_batch(data, indices):
    frames, frame_indices, actions = data[:3]
    frame_ids = frame_indices[indices, None] + torch.arange(0, SPAN + 1, STRIDE)
    action_ids = indices[:, None] + torch.arange(SPAN)
    images = frames[frame_ids].cuda().permute(0, 1, 4, 2, 3).float() / 255
    images = (images - base.MEAN) / base.STD
    blocks = actions[action_ids].reshape(len(indices), HISTORY, STRIDE * 8).cuda()
    return dict(pixels=images, action=blocks)


def forward(model, inputs):
    encoded = model.encode(inputs)
    z = encoded["emb"]
    predicted = model.predict(z[:, :HISTORY], encoded["act_emb"])
    return z.float(), predicted.float()


@torch.inference_mode()
def evaluate(model, rows, out):
    model.eval()
    data = base.load(rows)
    starts = windows(data)
    results = []
    for indices in starts.split(BATCH):
        z, prediction = forward(model, make_batch(data, indices))
        target = z[:, -1]
        model_error = (prediction[:, -1] - target).square().mean(-1)
        persistence = (z[:, -2] - target).square().mean(-1)
        results.extend(torch.stack((model_error, persistence), -1).cpu().tolist())
    records = []
    offset = 0
    for row in rows:
        with np.load(base.DATA / row["file"]) as episode:
            raw = episode["raw_actions"]
            region = episode["rgb"][:, 0, :, 21:42, 41:53]
            gap = np.maximum(0, 100 - region.mean(1)).mean((1, 2))
            for s in range(len(raw) - SPAN + 1):
                t = s + (HISTORY - 1) * STRIDE
                end = t + STRIDE
                pull_steps = int(data[3][offset + t:offset + end].sum())
                delta = float(gap[end] - gap[t])
                i = len(records)
                assert starts[i].item() == offset + s
                records.append((row["file"], t, end, pull_steps, delta, *results[i]))
            offset += len(raw)
    assert len(records) == len(results)
    with (out / "validation.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "context_end", "target_step", "pull_steps", "gap_change",
                         "model_mse", "persistence_mse"])
        writer.writerows(records)
    groups = dict(all=records, pull_majority=[r for r in records if r[3] >= 3],
                  visible_open_and_pull=[r for r in records if r[3] >= 3 and r[4] > .5])
    report = {}
    for name, group in groups.items():
        errors = np.array([r[-2:] for r in group])
        assert len(group) and np.isfinite(errors).all()
        report[name] = dict(n=len(group), episodes=len(set(r[0] for r in group)),
                            model=errors[:, 0].mean(), persistence=errors[:, 1].mean(),
                            fraction_better=(errors[:, 0] < errors[:, 1]).mean())
    return report


def main():
    out = Path(sys.argv[1])
    uuid = os.environ["CUDA_VISIBLE_DEVICES"]
    gpu = int(subprocess.check_output(["nvidia-smi", "-i", uuid, "--query-gpu=index",
                                       "--format=csv,noheader"]))
    assert gpu in (2, 3, 4) and torch.cuda.device_count() == 1
    assert str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-") == uuid.lower().removeprefix("gpu-")
    out.mkdir(exist_ok=False)
    audit = base.UPSTREAM / "audit"
    manifest = audit / "splits.json"
    split_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for file, digest in json.loads((audit / "batch.json").read_text())["upstream_sha256"].items():
        assert hashlib.sha256((base.UPSTREAM / file).read_bytes()).hexdigest() == digest
    splits = json.loads(manifest.read_text())["splits"]
    torch.manual_seed(base.SEED)
    torch.set_num_threads(3)
    base.MEAN = torch.tensor([.485, .456, .406], device="cuda").view(3, 1, 1)
    base.STD = torch.tensor([.229, .224, .225], device="cuda").view(3, 1, 1)
    data = base.load(splits["train"])
    starts = windows(data)
    first = splits["train"][0]
    with np.load(base.DATA / first["file"]) as episode:
        check = make_batch(data, starts[:1])
        adapter = base.DrawerTransitions(base.DATA, [first])
        expected = 2 * (episode["raw_actions"][:SPAN] - adapter.low) / (adapter.high - adapter.low) - 1
        np.testing.assert_allclose(check["action"].cpu().numpy().reshape(SPAN, 8), expected)
        from lewm_drawer_data import pixels
        expected_images = torch.stack([pixels(frame) for frame in episode["rgb"][:SPAN+1:STRIDE]])
        torch.testing.assert_close(check["pixels"][0].cpu(), expected_images)
    config = dict(gpu=gpu, uuid=uuid, history=HISTORY, frameskip=STRIDE,
                  seed=base.SEED, batch_size=BATCH, updates=1000, lr=5e-5,
                  weight_decay=.001, sigreg=.09, split_sha256=split_hash,
                  train_windows=len(starts), action_block_dim=40, pull_min_steps=3,
                  visible_min_gap_change=.5, audit_episode=first["file"])
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print("Window and alignment checks passed", config, flush=True)
    model = build_model().cuda()
    reg = SIGReg().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.001)
    for step in range(1, 1001):
        indices = starts[torch.randint(len(starts), (BATCH,))]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, prediction = forward(model, make_batch(data, indices))
        mse = (prediction - z[:, 1:]).square().mean()
        loss = mse + .09 * reg(z.transpose(0, 1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(step, loss.item(), mse.item(), flush=True)
    torch.save(dict(model=model.state_dict(), step=step), out / "checkpoint.pt")
    del data
    report = evaluate(model, splits["val"], out)
    (out / "result.json").write_text(json.dumps(dict(updates=step, metrics=report), indent=2))
    print(report, flush=True)


if __name__ == "__main__":
    main()
