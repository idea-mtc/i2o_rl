"""Versioned raw-frame demonstration format and strict episode validation.

No simulator imports: datasets can be audited without launching Isaac Sim.
Stacking happens per episode on load, never across a reset boundary.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

FORMAT = "isaac_i2o_raw_episode_v1"


def file_hash(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def read_episode(path):
    with np.load(path,allow_pickle=False) as data:
        return {name:data[name] for name in data.files}


def validate_episode(data, task):
    n=len(data["actions"])
    views=4 if task=="take_lid_off_saucepan" else 1
    assert n>0 and data["rgb"].shape==(n+1,views,3,84,84)
    assert data["rgb"].dtype==np.uint8
    assert data["state"].shape==(n+1,15)
    assert data["actions"].shape==(n,8)
    for name in ("state","actions","rewards","discounts","object_positions","tcp_positions"):
        assert np.isfinite(data[name]).all(),name
    assert np.abs(data["actions"]).max()<=1.000001
    for name in ("rewards","discounts","terminated","truncated"):
        assert data[name].shape==(n,),name
    assert bool(data["terminated"][-1]) and not data["terminated"][:-1].any()
    assert not data["truncated"].any()
    assert data["rewards"][-1]==1 and np.all(data["rewards"][:-1]==0)
    assert data["discounts"][-1]==0 and np.allclose(data["discounts"][:-1],0.99)
    assert data["object_positions"].shape==(n+1,3)
    assert data["tcp_positions"].shape==(n+1,3)
    assert data["sim_time"].shape==(n+1,) and np.isfinite(data["sim_time"]).all()
    dt=np.diff(data["sim_time"])
    assert (dt>0).all() and np.allclose(dt,0.1,atol=1e-5),"physical/control time mismatch"
    changes=np.abs(data["rgb"][-1].astype(float)-data["rgb"][0].astype(float)).mean(axis=(1,2,3))
    assert (changes>0.1).all(),f"camera frames did not change: {changes}"
    std=data["rgb"].astype(np.float32).std(axis=(2,3,4))
    assert (std>3).all(),"blank observation"
    checks={"transitions":n,"camera_mean_abs_change":changes.tolist(),
            "rgb_shape":list(data["rgb"].shape),"actions_bounded":True,
            "terminal_reward_consistent":True,"finite":True,"all_views_nonblank":True,
            "sim_control_period_s":float(dt.mean())}
    if task=="take_lid_off_saucepan":
        assert data["knob_positions"].shape==(n+1,3)
        assert np.isfinite(data["knob_positions"]).all()
        lift=data["object_positions"][1:,2]-data["object_positions"][0,2]
        xy=np.linalg.norm(data["object_positions"][1:,:2]-data["object_positions"][0,:2],axis=1)
        distance=np.linalg.norm(data["knob_positions"][1:]-data["tcp_positions"][1:],axis=1)
        opening=data["state"][1:,7]*0.04
        assert n>=30 and (lift[-30:]>0.15).all()
        assert (xy[-30:]<0.05).all() and (distance[-30:]<0.05).all()
        assert (opening[-30:]<0.025).all()
        relative=data["knob_positions"][-30:]-data["tcp_positions"][-30:]
        slip=float(np.ptp(relative,axis=0).max())
        assert slip<0.015
        checks.update(min_lift_last_3s_m=float(lift[-30:].min()),final_lift_m=float(lift[-1]),
                      relative_knob_tcp_range_m=slip,physical_success_recomputed=True)
    return checks


def audit_dataset(directory, task=None, require_complete=True):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text())
    assert manifest["format"]==FORMAT
    if task is not None:assert manifest["task"]==task,"dataset/task mismatch"
    if require_complete:
        assert manifest["status"]=="complete" and len(manifest["episodes"])==manifest["target_episodes"]
    rows=[];seeds=set();transitions=0
    for record in manifest["episodes"]:
        path=directory/record["file"]
        assert file_hash(path)==record["sha256"],f"checksum mismatch: {path}"
        assert record["seed"] not in seeds,"duplicate episode seed"
        seeds.add(record["seed"])
        checks=validate_episode(read_episode(path),manifest["task"])
        assert checks["transitions"]==record["checks"]["transitions"]
        rows.append(checks);transitions+=checks["transitions"]
    return manifest,{"pass":True,"episodes":len(rows),"transitions":transitions,
                     "sha256_verified":True,"physical_success_recomputed":True,
                     "per_episode":rows}


def stacked_observations(data, frame_stack):
    assert frame_stack>=1
    rgb,state=data["rgb"],data["state"]
    observations=[]
    for t in range(len(rgb)):
        ids=np.maximum(0,np.arange(t-frame_stack+1,t+1))
        observations.append((np.concatenate([rgb[i] for i in ids],axis=1),
                             np.concatenate([state[i] for i in ids]).astype(np.float32)))
    return observations


def load_into_replay(directory,replay,task,frame_stack,env_config=None,limit=None,observation_transform=None,action_transform=None):
    manifest,audit=audit_dataset(directory,task)
    if env_config is not None:
        # Exact scene/control/camera contract, independent of training horizon/seed.
        assert manifest["environment_config"]==env_config,"dataset/environment configuration mismatch"
    records=manifest["episodes"] if limit is None else manifest["episodes"][:limit]
    if limit is not None:assert len(records)==limit,"not enough stored demos"
    for record in records:
        data=read_episode(Path(directory)/record["file"])
        observations=stacked_observations(data,frame_stack)
        if observation_transform is not None:
            observations=[observation_transform(obs,i) for i,obs in enumerate(observations)]
        actions=action_transform(data) if action_transform is not None else data["actions"]
        for i,action in enumerate(actions):
            replay.add(observations[i],action,data["rewards"][i],data["discounts"][i],observations[i+1],True)
    return manifest,audit,len(records)
