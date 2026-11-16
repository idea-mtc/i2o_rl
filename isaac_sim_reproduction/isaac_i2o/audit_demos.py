"""Offline full dataset audit and actual replay/batch compatibility check."""
import argparse
import json
from pathlib import Path
import numpy as np
from demonstrations import audit_dataset,load_into_replay,read_episode,stacked_observations,file_hash
from replay import Replay

parser=argparse.ArgumentParser()
parser.add_argument("directory",type=Path)
parser.add_argument("--report",type=Path)
args=parser.parse_args()
manifest,audit=audit_dataset(args.directory)
replay=Replay(seed=1)
_,_,episodes=load_into_replay(args.directory,replay,manifest["task"],2)
batch=next(replay.batches(256))
assert batch[0].shape==(256,4,6,84,84)
assert batch[1].shape==(256,30) and batch[6].shape==(256,8)
assert np.all(batch[11]==1) and np.isfinite(batch[6]).all()
sample=read_episode(args.directory/manifest["episodes"][0]["file"])
stacked=stacked_observations(sample,8)
assert stacked[0][0].shape==(4,24,84,84) and stacked[0][1].shape==(120,)
for i in range(8):np.testing.assert_array_equal(stacked[0][0][:,3*i:3*i+3],sample["rgb"][0])
report={k:v for k,v in audit.items() if k!="per_episode"}
report.update(dataset=str(args.directory.resolve()),manifest_sha256=file_hash(args.directory/"manifest.json"),
    loader_episodes=episodes,replay_transitions=len(replay.demos),
    cpu_batch_256_rgb_shape=list(batch[0].shape),cpu_batch_256_state_shape=list(batch[1].shape),
    eight_frame_padding_verified=True,validation_code_sha256=file_hash(Path(__file__).with_name("demonstrations.py")),
    scope="data and CPU replay checks, not batch-256 GPU memory acceptance")
if args.report:args.report.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
