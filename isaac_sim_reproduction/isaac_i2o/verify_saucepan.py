"""Verify physical lid grasp, four camera streams and a rigid wrist mount."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
parser=argparse.ArgumentParser()
parser.add_argument("--episodes",type=int,default=3)
parser.add_argument("--seed",type=int,default=12)
parser.add_argument("--randomization",type=float,default=0.02)
args=parser.parse_args()
assert args.episodes>0
ROOT=Path(os.environ.get("PINGANDOG_ROOT","/home/ubuntu/pingandog"))
OUT=ROOT/"outputs/panda_saucepan"/datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
OUT.mkdir(parents=True)
GPU=int(os.environ.get("PINGANDOG_GPU","0"))
R={"pass":False,"output":str(OUT),"scope":"procedural Isaac Sim lid task; expert and cameras only, no I2O training",
   "args":vars(args),"gpu":GPU}
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"renderer":"RaytracedLighting","multi_gpu":False,
    "active_gpu":GPU,"physics_gpu":GPU,"width":960,"height":720,
    "extra_args":["--/physics/updateToUsd=true"]})
env=None
writers=[]
try:
    import numpy as np
    from PIL import Image, ImageDraw
    from pxr import UsdGeom, UsdPhysics
    from saucepan_env import SaucepanEnv
    env=SaucepanEnv(seed=args.seed,randomization=args.randomization)
    R["config"]=env.config
    (OUT/"config.json").write_text(json.dumps(env.config,indent=2))
    asset=ROOT/"assets/tasks/take_lid_off_saucepan"
    asset.mkdir(parents=True,exist_ok=True)
    (asset/"config.json").write_text(json.dumps(env.config,indent=2))
    trajectory=[];snapshots=[]
    class Writer:
        def __init__(self,path):
            self.p=subprocess.Popen(["/usr/bin/ffmpeg","-y","-loglevel","error","-f","rawvideo",
                "-pix_fmt","rgb24","-s","1280x960","-r","10","-i","-","-an","-c:v","libx264",
                "-pix_fmt","yuv420p",str(path)],stdin=subprocess.PIPE)
            self.closed=False
        def write(self,a):self.p.stdin.write(a.tobytes())
        def close(self):
            if self.closed:return
            self.closed=True;self.p.stdin.close()
            if self.p.wait()!=0:raise RuntimeError("ffmpeg failed")
    def mosaic(images,label):
        canvas=Image.new("RGB",(1280,960));draw=ImageDraw.Draw(canvas)
        for i,name in enumerate(env.camera_names):
            x=(i%2)*640;y=(i//2)*480
            canvas.paste(Image.fromarray(images[name]),(x,y))
            draw.rectangle((x,y,x+300,y+25),fill=(15,15,15))
            draw.text((x+8,y+6),name+" / "+label,fill=(255,255,255))
        return np.asarray(canvas)
    def save_snapshot(ep,label):
        directory=OUT/f"episode_{ep}";directory.mkdir(exist_ok=True)
        high=env.images("preview");low=env.images("rgb")
        for name in env.camera_names:
            Image.fromarray(high[name]).save(directory/f"{label}_{name}.png")
            Image.fromarray(low[name]).save(directory/f"{label}_{name}_84.png")
        Image.fromarray(mosaic(high,label)).save(directory/f"{label}_four_views.png")
        snapshots.append({"episode":ep,"phase":label,"camera_poses":env.camera_poses(),
                          "status":env.status(),"pixel_std":{k:float(v.std()) for k,v in low.items()}})
    # Open-jaw counterfactual: moving upward alone must not raise the lid.
    env.reset(nominal=True)
    base_prim=next(p for p in env.world.stage.Traverse() if p.GetName()=="panda_link0")
    def base_position():
        return np.array(UsdGeom.XformCache().GetLocalToWorldTransform(base_prim).ExtractTranslation())
    initial_base=base_position()
    initial_height=float(env.initial_lid[2])
    assert abs(initial_height-(env.table_top+env.pot_height+0.004))<0.004,"lid is not resting on the pot rim"
    negative_max=0.
    for action in env.expert_actions(close=False):
        _,_,terminated,truncated,info=env.step(action)
        negative_max=max(negative_max,info["lift_m"])
        if terminated or truncated:break
    R["open_gripper_control"]={"maximum_lift_m":negative_max,"pass":negative_max<0.01}
    assert R["open_gripper_control"]["pass"],"open-jaw control unexpectedly lifted lid"
    successes=[];camera_checks=[]
    for episode in range(args.episodes):
        obs=env.reset(nominal=(episode==0))
        assert obs[0].shape==(4,6,84,84) and obs[1].shape==(30,)
        camera_start=env.camera_poses();images_start=env.images()
        wrist_local_start=np.array(UsdGeom.Xformable(env.world.stage.GetPrimAtPath(env.camera_paths["wrist"])).GetLocalTransformation())
        if episode==0:
            stage=env.world.stage
            stage.GetRootLayer().Export(str(asset/"scene.usda"))
            stage.GetRootLayer().Export(str(OUT/"scene_initial.usda"))
        save_snapshot(episode,"initial")
        writer=Writer(OUT/f"episode_{episode}"/"expert_four_views.mp4");writers.append(writer)
        writer.write(mosaic(env.images("preview"),"initial"))
        prior_phase=None;episode_rows=[];phase_boundaries=[]
        for action in env.expert_actions():
            label=env.expert_phase
            obs,reward,terminated,truncated,info=env.step(action)
            row={"episode":episode,"phase":label,"action":action.tolist(),"reward":reward,**info,
                 "base":base_position().tolist()}
            trajectory.append(row);episode_rows.append(row)
            writer.write(mosaic(env.images("preview"),label))
            if label!=prior_phase:
                phase_boundaries.append({"phase":label,"step":info["step"]})
                save_snapshot(episode,label+"_start")
                prior_phase=label
            if terminated or truncated:break
        writer.close();save_snapshot(episode,"final_hold")
        cameras_end=env.camera_poses();images_end=env.images()
        fixed_drift={name:float(np.abs(np.array(cameras_end[name])-np.array(camera_start[name])).max())
                     for name in env.camera_names if name!="wrist"}
        wrist_travel=float(np.linalg.norm(np.array(cameras_end["wrist"])[3,:3]-np.array(camera_start["wrist"])[3,:3]))
        wrist_local_end=np.array(UsdGeom.Xformable(env.world.stage.GetPrimAtPath(env.camera_paths["wrist"])).GetLocalTransformation())
        wrist_mount_error=float(np.abs(wrist_local_end-wrist_local_start).max())
        pixel_change={name:float(np.abs(images_end[name].astype(float)-images_start[name].astype(float)).mean())
                      for name in env.camera_names}
        checks={"fixed_camera_poses":max(fixed_drift.values())<1e-8,
            "wrist_camera_moves_with_arm":wrist_travel>0.03,
            "wrist_mount_constant":wrist_mount_error<1e-10,
            "all_camera_images_change":all(v>0.1 for v in pixel_change.values()),
            "valid_observation_shape":obs[0].shape==(4,6,84,84)}
        camera_checks.append({"checks":checks,"fixed_pose_drift":fixed_drift,"wrist_translation_m":wrist_travel,
            "wrist_local_mount_error":wrist_mount_error,"image_mean_abs_change":pixel_change})
        hold=[row for row in episode_rows if row["held_steps"]>0]
        # Require actual 3-second elevated grasp, not only a maximum-height sample.
        last_hold=episode_rows[-30:]
        relative=np.array([np.array(row["knob"])-np.array(row["tcp"]) for row in last_hold])
        min_lift=min(row["lift_m"] for row in last_hold)
        slip=float(np.ptp(relative,axis=0).max())
        base_drift=max(float(np.linalg.norm(np.array(row["base"])-initial_base)) for row in episode_rows)
        passed=bool(info["success"] and len(last_hold)==30 and min_lift>0.15 and slip<0.015 and base_drift<0.0001)
        result={"episode":episode,"pass":passed,"final":info,"min_lift_during_last_3s_m":min_lift,
            "relative_knob_tcp_range_m":slip,"base_translation_m":base_drift,"control_steps":len(episode_rows),
            "phase_boundaries":phase_boundaries,"initial_xy":env.pot_xy.tolist()}
        successes.append(result);print("LID_EXPERT",json.dumps(result),flush=True)
        print("LID_CAMERAS",json.dumps(camera_checks[-1]),flush=True)
    # Inspect the scene graph: there must be no fixed/grasp attachment involving Lid.
    lid_joints=[]
    for prim in env.world.stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint=UsdPhysics.Joint(prim)
            targets=list(joint.GetBody0Rel().GetTargets())+list(joint.GetBody1Rel().GetTargets())
            if any(str(p).startswith("/World/Lid") for p in targets):lid_joints.append(str(prim.GetPath()))
    R.update(episodes=successes,cameras=camera_checks,attachment_joints=lid_joints,
        expert_successes=sum(r["pass"] for r in successes),expert_episodes=args.episodes,
        rgb_shape=[4,6,84,84],rgb_raw_shape=[4,3,84,84],
        scene_usd=str(asset/"scene.usda"),physical_grasp=True)
    R["pass"]=bool(all(r["pass"] for r in successes) and all(all(c["checks"].values()) for c in camera_checks)
        and not lid_joints and R["open_gripper_control"]["pass"])
    (OUT/"trace.json").write_text(json.dumps(trajectory))
    (OUT/"camera_snapshots.json").write_text(json.dumps(snapshots,indent=2))
except Exception:
    R["error"]=traceback.format_exc();print(R["error"],flush=True)
finally:
    for writer in writers:
        try:writer.close()
        except Exception:pass
    (OUT/"result.json").write_text(json.dumps(R,indent=2))
    latest=OUT.parent/"latest"
    if latest.is_symlink():latest.unlink()
    if not latest.exists():latest.symlink_to(OUT.name)
    print("LID_RESULT",json.dumps(R),flush=True)
    if env is not None:env.close()
    app.close()
sys.exit(0 if R["pass"] else 3)
