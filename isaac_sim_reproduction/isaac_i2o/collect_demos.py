"""Collect immutable, audited saucepan expert episodes independently of training."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
parser=argparse.ArgumentParser()
parser.add_argument("--task",choices=("take_lid_off_saucepan",),default="take_lid_off_saucepan")
parser.add_argument("--episodes",type=int,default=100)
parser.add_argument("--pilot",type=int,default=10)
parser.add_argument("--seed",type=int,default=12000)
parser.add_argument("--randomization",type=float,default=0.02)
parser.add_argument("--horizon",type=int,default=260)
parser.add_argument("--resume",type=Path)
parser.add_argument("--max-attempts",type=int,default=300)
args=parser.parse_args()
assert args.episodes>=args.pilot>0 and args.max_attempts>=args.episodes
ROOT=Path(os.environ.get("PINGANDOG_ROOT","/home/ubuntu/pingandog"))
OUT=args.resume or ROOT/"datasets"/args.task/datetime.datetime.now().strftime("demos_%Y%m%d_%H%M%S")
OUT=OUT.resolve();OUT.mkdir(parents=True,exist_ok=True)
if not args.resume and (OUT/"manifest.json").exists():raise RuntimeError("dataset already exists")
GPU=int(os.environ.get("PINGANDOG_GPU","0"))
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"renderer":"RaytracedLighting","multi_gpu":False,
    "active_gpu":GPU,"physics_gpu":GPU,"width":960,"height":720,
    "extra_args":["--/physics/updateToUsd=true"]})
env=None;manifest=None;result={"pass":False,"directory":str(OUT)}
try:
    import numpy as np
    from tasks import make_env
    from demonstrations import FORMAT,file_hash,validate_episode,audit_dataset,stacked_observations,read_episode
    from replay import Replay
    env=make_env(args.task,1,args.horizon,args.seed,args.randomization)
    sources={p.name:file_hash(p) for p in [HERE/"collect_demos.py",HERE/"tasks.py",HERE/"saucepan_env.py",
                                           HERE/"panda_env.py",HERE/"demonstrations.py"]}
    def atomic_json(path,value):
        tmp=path.with_suffix(path.suffix+".tmp")
        tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
    if args.resume:
        manifest=json.loads((OUT/"manifest.json").read_text())
        assert manifest["format"]==FORMAT and manifest["task"]==args.task
        assert manifest["root_seed"]==args.seed and manifest["source_sha256"]==sources
        assert manifest["environment_config"]==env.config and manifest["horizon"]==args.horizon
        assert len(manifest["episodes"])<=args.episodes
        audit_dataset(OUT,args.task,require_complete=False)
    else:
        manifest={"format":FORMAT,"task":args.task,"root_seed":args.seed,
            "environment_config":env.config,"horizon":args.horizon,"source_sha256":sources,
            "raw_frames":True,"state_fields":["7_joint_positions","normalized_finger_opening","tcp_xyz","tcp_quat_xyzw"],
            "actions":"7 commanded joint increments /0.04 rad; absolute gripper -1 closed,+1 open",
            "discount":0.99,"episodes":[],"attempts":[],"sessions":[],"pilot_passed":False}
    manifest.update(status="collecting",target_episodes=args.episodes)
    manifest["sessions"].append({"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target":args.episodes,"existing":len(manifest["episodes"]),"gpu":GPU})
    atomic_json(OUT/"manifest.json",manifest)
    print("DEMO_DIRECTORY",str(OUT),flush=True)
    while len(manifest["episodes"])<args.episodes and len(manifest["attempts"])<args.max_attempts:
        attempt=len(manifest["attempts"]);seed=args.seed+attempt
        env.rng=np.random.default_rng(seed)
        obs=env.reset()
        initial=env.status()
        rgb=[obs[0].copy()];states=[obs[1].copy()]
        positions=[initial["lid"]];knobs=[initial["knob"]];tcps=[initial["tcp"]]
        times=[float(env.world.current_time)]
        actions=[];rewards=[];discounts=[];terminateds=[];truncateds=[];phases=[]
        info=initial
        for action in env.expert_actions():
            obs,reward,terminated,truncated,info=env.step(action)
            rgb.append(obs[0].copy());states.append(obs[1].copy())
            positions.append(info["lid"]);knobs.append(info["knob"]);tcps.append(info["tcp"])
            times.append(float(env.world.current_time))
            actions.append(action.copy());rewards.append(reward);discounts.append(0. if terminated else 0.99)
            terminateds.append(terminated);truncateds.append(truncated);phases.append(env.expert_phase)
            if terminated or truncated:break
        record={"attempt":attempt,"seed":seed,"initial_xy":env.pot_xy.tolist(),
                "steps":len(actions),"success":bool(info.get("success",False)),"final":info,"accepted":False}
        if record["success"]:
            data={"rgb":np.stack(rgb).astype(np.uint8),"state":np.stack(states).astype(np.float32),
                "actions":np.asarray(actions,np.float32),"rewards":np.asarray(rewards,np.float32),
                "discounts":np.asarray(discounts,np.float32),"terminated":np.asarray(terminateds,bool),
                "truncated":np.asarray(truncateds,bool),"object_positions":np.asarray(positions,np.float32),
                "knob_positions":np.asarray(knobs,np.float32),"tcp_positions":np.asarray(tcps,np.float32),
                "sim_time":np.asarray(times,np.float64),"phase":np.asarray(phases,dtype="U16")}
            try:
                checks=validate_episode(data,args.task)
                dt=np.diff(data["sim_time"])
                assert (dt>0).all() and np.allclose(dt,0.1,atol=1e-5),f"control clock mismatch: {dt.min()}, {dt.max()}"
                checks["sim_control_period_s"]=float(dt.mean())
                index=len(manifest["episodes"]);relative=f"episodes/episode_{index:04d}.npz"
                path=OUT/relative;path.parent.mkdir(exist_ok=True)
                assert not path.exists(),"refusing to overwrite a collected episode"
                temporary=path.with_suffix(".npz.tmp")
                with open(temporary,"wb") as stream:np.savez_compressed(stream,**data)
                temporary.replace(path)
                # Verify what was actually stored, not just in-memory buffers.
                validate_episode(read_episode(path),args.task)
                accepted={"file":relative,"sha256":file_hash(path),"seed":seed,"attempt":attempt,
                    "initial_xy":record["initial_xy"],"checks":checks}
                manifest["episodes"].append(accepted)
                record["accepted"]=True
                if index<10 or (index+1)%10==0:
                    preview=OUT/"previews";preview.mkdir(exist_ok=True)
                    env.save_image(preview/f"episode_{index:04d}_success.png")
            except AssertionError as error:
                record["rejection_reason"]=str(error)
        else:record["rejection_reason"]="expert did not satisfy task success"
        manifest["attempts"].append(record)
        manifest["accepted_count"]=len(manifest["episodes"])
        atomic_json(OUT/"manifest.json",manifest)
        atomic_json(OUT/"progress.json",{"accepted":len(manifest["episodes"]),"target":args.episodes,
            "attempts":len(manifest["attempts"]),"pilot_passed":manifest["pilot_passed"]})
        print("DEMO_PROGRESS",json.dumps({"accepted":len(manifest["episodes"]),"target":args.episodes,
            "attempt":attempt,"seed":seed,"accepted_this_attempt":record["accepted"],
            "rejection_reason":record.get("rejection_reason")}),flush=True)
        if len(manifest["episodes"])>=args.pilot and not manifest["pilot_passed"]:
            _,pilot=audit_dataset(OUT,args.task,require_complete=False)
            # Explicitly check stack padding and replay tensor shape at an episode boundary.
            sample=read_episode(OUT/manifest["episodes"][0]["file"])
            stacked=stacked_observations(sample,2)
            assert np.array_equal(stacked[0][0][:,:3],stacked[0][0][:,3:])
            assert np.array_equal(stacked[1][0][:,:3],sample["rgb"][0])
            assert np.array_equal(stacked[1][0][:,3:],sample["rgb"][1])
            replay=Replay()
            for i in range(min(4,len(sample["actions"]))):
                replay.add(stacked[i],sample["actions"][i],sample["rewards"][i],sample["discounts"][i],stacked[i+1],True)
            batch=next(replay.batches(4))
            assert batch[0].shape==(4,4,6,84,84) and batch[1].shape==(4,30)
            pilot.update(stack_padding_verified=True,replay_batch_shapes_verified=True)
            atomic_json(OUT/"pilot_audit.json",pilot)
            manifest["pilot_passed"]=True;atomic_json(OUT/"manifest.json",manifest)
            print("DEMO_PILOT_PASS",len(manifest["episodes"]),flush=True)
        if len(manifest["attempts"])>=20 and len(manifest["episodes"])<args.pilot:
            raise RuntimeError("pilot failed to collect 10 valid episodes within 20 attempts")
    assert len(manifest["episodes"])==args.episodes,"collection attempt budget exhausted"
    manifest["status"]="complete";atomic_json(OUT/"manifest.json",manifest)
    _,audit=audit_dataset(OUT,args.task)
    atomic_json(OUT/"audit.json",audit)
    result.update({"pass":True,"episodes":len(manifest["episodes"]),"attempts":len(manifest["attempts"]),
        "rejected":len(manifest["attempts"])-len(manifest["episodes"]),"transitions":audit["transitions"],
        "pilot_passed":manifest["pilot_passed"],"dataset_sha256":file_hash(OUT/"manifest.json")})
except Exception:
    result["error"]=traceback.format_exc();print(result["error"],flush=True)
    if manifest is not None:
        manifest["status"]="failed"
        tmp=OUT/"manifest.json.tmp";tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(OUT/"manifest.json")
finally:
    (OUT/"collection_result.json").write_text(json.dumps(result,indent=2))
    print("DEMO_RESULT",json.dumps(result),flush=True)
    if env is not None:env.close()
    app.close()
sys.exit(0 if result["pass"] else 3)
