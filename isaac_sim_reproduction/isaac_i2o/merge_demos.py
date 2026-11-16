"""Merge fully audited collection batches without silently dropping failures."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import numpy as np
from demonstrations import audit_dataset, file_hash, FORMAT

parser=argparse.ArgumentParser()
parser.add_argument("--sources",type=Path,nargs="+",required=True)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--expected-episodes",type=int,default=100)
args=parser.parse_args()
out=args.output.resolve()
assert not out.exists(),"refusing to overwrite an existing fixed dataset"
sources=[p.resolve() for p in args.sources]
audited=[audit_dataset(p) for p in sources]
first=audited[0][0]
for manifest,_ in audited:
    assert manifest["format"]==FORMAT and manifest["task"]==first["task"]
    assert manifest["environment_config"]==first["environment_config"]
    assert manifest["horizon"]==first["horizon"]
    assert manifest["source_sha256"]==first["source_sha256"],"collection code differed between batches"
assert sum(len(m["episodes"]) for m,_ in audited)==args.expected_episodes
all_seeds=[row["seed"] for m,_ in audited for row in m["episodes"]]
assert len(all_seeds)==len(set(all_seeds)),"overlapping seeds"
(out/"episodes").mkdir(parents=True)
merged={k:v for k,v in first.items() if k not in ("episodes","attempts","sessions","root_seed")}
merged.update(status="collecting",target_episodes=args.expected_episodes,root_seed=None,
    episodes=[],attempts=[],sessions=[],source_datasets=[],pilot_passed=True,
    collection_seed_policy="unique per-episode seeds from separately recorded seed blocks")
for directory,(manifest,audit) in zip(sources,audited):
    merged["source_datasets"].append({"directory":str(directory),
        "manifest_sha256":file_hash(directory/"manifest.json"),"root_seed":manifest["root_seed"],
        "episodes":len(manifest["episodes"]),"attempts":len(manifest["attempts"])})
    merged["attempts"].extend({**row,"source_dataset":str(directory)} for row in manifest["attempts"])
    for record in manifest["episodes"]:
        index=len(merged["episodes"]);relative=f"episodes/episode_{index:04d}.npz"
        shutil.copy2(directory/record["file"],out/relative)
        assert file_hash(out/relative)==record["sha256"]
        merged["episodes"].append({**record,"file":relative,"source_file":record["file"],
                                   "source_dataset":str(directory)})
merged.update(status="complete",accepted_count=len(merged["episodes"]),
              merged_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/"manifest.json").write_text(json.dumps(merged,indent=2))
_,audit=audit_dataset(out)
xy=np.array([e["initial_xy"] for e in merged["episodes"]])
audit.update(attempts=len(merged["attempts"]),
    rejected=sum(not r["accepted"] for r in merged["attempts"]),
    unique_episode_seeds=len(all_seeds),initial_xy_min=xy.min(axis=0).tolist(),
    initial_xy_max=xy.max(axis=0).tolist(),bytes=sum(p.stat().st_size for p in (out/"episodes").glob("*.npz")),
    manifest_sha256=file_hash(out/"manifest.json"))
(out/"audit.json").write_text(json.dumps(audit,indent=2))
print(json.dumps({k:v for k,v in audit.items() if k!="per_episode"},indent=2))
