"""Recompute BatchNorm statistics from train windows, then score held-out episodes."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

import train_lewm_drawer as single
import train_lewm_drawer_cube as five


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["single", "five"])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    uuid = os.environ["CUDA_VISIBLE_DEVICES"]
    physical = int(subprocess.check_output(
        ["nvidia-smi", "-i", uuid, "--query-gpu=index", "--format=csv,noheader"]))
    assert physical in (2, 3, 4) and torch.cuda.device_count() == 1
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-")
    assert actual_uuid == uuid.lower().removeprefix("gpu-")
    torch.manual_seed(single.SEED)
    torch.set_num_threads(3)
    args.out.mkdir(exist_ok=False, parents=True)
    splits = json.loads((single.UPSTREAM / "audit/splits.json").read_text())["splits"]
    single.MEAN = torch.tensor([.485, .456, .406], device="cuda").view(3, 1, 1)
    single.STD = torch.tensor([.229, .224, .225], device="cuda").view(3, 1, 1)

    if args.variant == "five":
        # Historical diagnostic drew a random validation order before constructing
        # the model. Reproduce its CPU RNG position using only manifest lengths.
        n_val = sum(max(0, row["steps"] - five.SPAN + 1) for row in splits["val"])
        torch.randperm(n_val)
        model = five.build_model().cuda()
        batch, forward = five.make_batch, five.forward
    else:
        model = single.build_model().cuda()
        batch, forward = single.batch, single.forward
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 1000
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm1d):
            layer.train()
            layer.reset_running_stats()
            layer.momentum = None

    data = single.load(splits["train"])
    starts = five.windows(data) if args.variant == "five" else torch.arange(len(data[1]))
    for _ in range(128):
        indices = starts[torch.randint(len(starts), (32,))]
        forward(model, batch(data, indices))
    del data
    model.eval()
    if args.variant == "five":
        metrics = five.evaluate(model, splits["val"], args.out)
    else:
        single.OUT = args.out
        metrics = single.evaluate(model, single.load(splits["val"]))
    (args.out / "result.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
