"""I2O-RL/Isaac Sim Panda integration and bounded smoke-training entry point."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=128)
parser.add_argument("--task", choices=("cube", "take_lid_off_saucepan"), default="cube")
parser.add_argument("--demo-dir", type=Path, help="load an audited fixed dataset instead of collecting again")
parser.add_argument("--demos", type=int, help="limit stored episodes; default all, or 2 for live collection")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--frame-stack", type=int, default=2)
parser.add_argument("--horizon", type=int, help="default 220 for cube, 260 for saucepan")
parser.add_argument("--pretrain-updates", type=int, default=8)
parser.add_argument("--eval-episodes", type=int, default=1)
parser.add_argument("--capacity", type=int, default=10000)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--randomization", type=float, default=0.02)
parser.add_argument("--checkpoint-every", type=int, default=1000)
parser.add_argument("--resume", type=Path)
parser.add_argument("--output-dir", type=Path, help="explicit new output directory for supervised recovery")
parser.add_argument("--action-mode", choices=("soft", "hard"), default="soft")
parser.add_argument("--softargmax-beta", type=float, default=1.0)
parser.add_argument("--soft-endpoint-rescale", action="store_true")
parser.add_argument("--phase-clock", action="store_true")
parser.add_argument("--absolute-joints", action="store_true")
parser.add_argument("--phase-heads",type=int,default=1)
parser.add_argument("--exploration-std",type=float,default=0.01)
parser.add_argument("--demo-phase-goals",action="store_true")
parser.add_argument("--demo-bc-only",action="store_true")
parser.add_argument("--cartesian-goals",action="store_true")
parser.add_argument("--augmentation-pad",type=int,default=4)
parser.add_argument("--pretrain-mode", choices=("mse", "combined"), default="combined")
parser.add_argument("--eval-every", type=int, default=0)
parser.add_argument("--eval-at", type=int, default=0, help="additional early validation at this online step")
parser.add_argument("--periodic-eval-episodes", type=int, default=10)
parser.add_argument("--eval-seed", type=int, default=100000)
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--run-label", default="integration")
args = parser.parse_args()
assert args.steps > 0 and (args.demos is None or args.demos > 0) and args.batch_size >= 2
ROOT = Path(os.environ.get("PINGANDOG_ROOT", "/home/ubuntu/pingandog"))
OUT = args.output_dir or ROOT / "outputs/i2o_panda" / datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
OUT.mkdir(parents=True, exist_ok=bool(args.output_dir))
assert not (OUT/"metrics.jsonl").exists(), "refusing to overwrite existing training output"
GPU = int(os.environ.get("PINGANDOG_GPU", "0"))
R = {"pass": False, "scope": "memory/speed benchmark" if args.benchmark else args.run_label,
     "args": {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
     "output": str(OUT), "gpu": GPU}
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "multi_gpu": False,
    "active_gpu": GPU, "physics_gpu": GPU, "width": 960, "height": 720,
    "extra_args": ["--/physics/updateToUsd=true"]})
env = None
agent = None
updates = 0
online_step = 0
start_wall = time.monotonic()
def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)

def progress(stage, **extra):
    write_json(OUT/"progress.json", {"stage":stage,"updates":updates,"online_step":online_step,
        "output":str(OUT),"gpu":GPU,"updated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_seconds":time.monotonic()-start_wall,**extra})

progress("initializing")
try:
    import numpy as np
    import torch
    from vendor import utils
    from vendor.i2o_rl import I2OAgent
    from tasks import make_env
    from demonstrations import load_into_replay, file_hash
    from replay import Replay
    torch.set_num_threads(4)
    utils.set_seed_everywhere(args.seed)
    device = f"cuda:{GPU}"
    assert torch.cuda.is_available()
    probe = torch.randn(32,32,device=device)
    assert torch.isfinite(probe @ probe).all()
    R.update(torch_version=torch.__version__, torch_cuda=torch.version.cuda,
             cuda_arch=torch.cuda.get_device_capability(GPU))
    print("I2O_RUNTIME", json.dumps(R), flush=True)
    data_start = time.monotonic()
    env = make_env(args.task, args.frame_stack, args.horizon, args.seed, args.randomization)
    assert args.phase_heads == 1 or (args.phase_heads == 5 and args.phase_clock)
    assert not args.demo_phase_goals or ((args.absolute_joints or args.cartesian_goals) and args.demo_bc_only), "relabeled phase goals cannot be used as factual TD transitions"
    data_environment_config = getattr(env,"config",None)
    if args.phase_clock:
        from phase_clock import ClockConditionedEnv, condition
        assert args.task == "take_lid_off_saucepan"
        env = ClockConditionedEnv(env)
    action_transform = None
    if args.absolute_joints:
        from absolute_joints import AbsoluteJointEnv, absolute_episode_actions
        absolute_base = env.base if args.phase_clock else env
        action_transform = lambda data:absolute_episode_actions(data, absolute_base,args.demo_phase_goals)
        assert not args.demo_phase_goals or args.demo_dir
        env = AbsoluteJointEnv(env)
    if args.cartesian_goals:
        assert args.demo_dir and not args.absolute_joints
        from cartesian_goals import CartesianGoalEnv,cartesian_episode_actions
        cartesian_base = env.base if args.phase_clock else env
        action_transform=lambda data:cartesian_episode_actions(data,cartesian_base,args.demo_phase_goals)
        env=CartesianGoalEnv(env)
    R["task"] = args.task
    R["environment_config"] = getattr(env,"config",None)
    replay = Replay(args.capacity, args.seed)
    successes = 0
    demo_results = []
    if args.demo_dir:
        manifest,audit,successes=load_into_replay(args.demo_dir,replay,args.task,args.frame_stack,
            data_environment_config,args.demos, condition if args.phase_clock else None, action_transform)
        requested_demos=successes
        assert env.horizon>=max(e["checks"]["transitions"] for e in manifest["episodes"])
        R["demo_source"]={"directory":str(args.demo_dir.resolve()),
            "manifest_sha256":file_hash(args.demo_dir/"manifest.json"),"loaded_episodes":successes,
            "audit_episodes":audit["episodes"],"recollected":False}
        demo_results=[{"source":e["file"],"seed":e["seed"],"success":True}
                      for e in manifest["episodes"][:successes]]
        print("I2O_DEMOS_LOADED",json.dumps(R["demo_source"]),flush=True)
    else:
        requested_demos=args.demos or 2
        for attempt in range(requested_demos * 3):
            obs = env.reset()
            transitions = []
            for action in env.expert_actions():
                nxt, reward, terminated, truncated, info = env.step(action)
                transitions.append((obs, action, reward, 0.0 if terminated else 0.99, nxt))
                obs = nxt
                if terminated or truncated:
                    break
            demo_results.append(info)
            print("I2O_DEMO", attempt, json.dumps(info), flush=True)
            if info["success"]:
                for transition in transitions:
                    replay.add(*transition, demo=True)
                successes += 1
                env.save_image(OUT / f"demo_{successes}.png")
            if successes >= requested_demos:
                break
    R.update(demo_results=demo_results, successful_demos=successes,
             demo_transitions=len(replay.demos))
    assert successes == requested_demos, "expert did not collect enough successful grasp episodes"
    if not args.demo_dir:
        replay.save(OUT / "demonstrations.npz")
    R["dataset_load_seconds"] = time.monotonic()-data_start
    progress("data_loaded", demo_episodes=successes, demo_transitions=len(replay.demos))
    torch.cuda.reset_peak_memory_stats(GPU)
    agent = I2OAgent(rgb_obs_shape=env.rgb_shape, low_dim_obs_shape=env.state_shape,
        action_shape=env.action_shape, device=device, lr=5e-5, weight_decay=0.1,
        feature_dim=128, hidden_dim=512, levels=3, bins=5, atoms=51,
        v_min=-2., v_max=2., bc_lambda=1., bc_margin=0.01, critic_lambda=0.1,
        critic_target_tau=0.02, num_expl_steps=0, update_every_steps=1,
        stddev_schedule=str(args.exploration_std), use_logger=True,
        action_mode=args.action_mode, softargmax_beta=args.softargmax_beta,
        soft_endpoint_rescale=args.soft_endpoint_rescale,phase_heads=args.phase_heads,demo_bc_only=args.demo_bc_only,augmentation_pad=args.augmentation_pad)
    resume_step = 0
    resume_updates = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        assert ck["rgb_shape"] == env.rgb_shape and ck["state_shape"] == env.state_shape
        assert ck.get("task", args.task) == args.task
        assert ck.get("config", {}).get("phase_clock", False) == args.phase_clock
        assert ck.get("config", {}).get("absolute_joints", False) == args.absolute_joints
        assert ck.get("config", {}).get("phase_heads", 1) == args.phase_heads
        assert ck.get("config", {}).get("demo_phase_goals", False) == args.demo_phase_goals
        assert ck.get("config", {}).get("cartesian_goals", False) == args.cartesian_goals
        assert ck.get("config", {}).get("soft_endpoint_rescale", False) == args.soft_endpoint_rescale
        assert ck.get("softargmax_beta", args.softargmax_beta) == args.softargmax_beta
        assert ck.get("demo_source", {}).get("manifest_sha256") == R.get("demo_source", {}).get("manifest_sha256")
        resume_step = int(ck.get("step", 0))
        resume_updates = int(ck.get("updates", 0))
        assert 0 <= resume_step <= args.steps
        if resume_step:
            assert args.pretrain_updates == 0, "online resume must skip pretraining"
        online_step = resume_step
        R["resume_source"] = {"checkpoint":str(args.resume), "step":resume_step,
            "updates":resume_updates, "online_replay_restored":False, "rng_restored":False}
        print("I2O_CHECKPOINT_VERIFIED", json.dumps(R["resume_source"]), flush=True)
        for name in ("encoder", "critic", "critic_target"):
            assert all(torch.isfinite(v).all().item() for v in ck[name].values()), name
            getattr(agent, name).load_state_dict(ck[name])
        if "encoder_opt" in ck and "critic_opt" in ck:
            agent.encoder_opt.load_state_dict(ck["encoder_opt"])
            agent.critic_opt.load_state_dict(ck["critic_opt"])
            R["resume_mode"] = "model_and_optimizer"
        else:
            R["resume_mode"] = "model_only_new_optimizers"
        print("I2O_RESUME", R["resume_mode"], flush=True)
    source_dir=OUT/"source"
    source_dir.mkdir()
    import shutil
    for source in list(HERE.glob("*.py"))+list((HERE/"vendor").glob("*.py")):
        dest=source_dir/("vendor_"+source.name if source.parent.name=="vendor" else source.name)
        shutil.copy2(source,dest)
    R["command"] = sys.argv
    R["model_config"]={"lr":5e-5,"weight_decay":0.1,"optimizer":"AdamW","feature_dim":128,
        "hidden_dim":512,"levels":3,"bins":5,"atoms":51,"v_min":-2.,"v_max":2.,
        "bc_lambda":1.,"bc_margin":0.01,"critic_lambda":0.1,"discount":0.99,"target_tau":0.02,
        "rgb_shape":list(env.rgb_shape),"state_shape":list(env.state_shape),"action_shape":list(env.action_shape),"phase_heads":args.phase_heads}
    R["experiment_notes"]={"paper_matched":"Panda, four RGB 84x84, 100 demos, AdamW lr5e-5 wd0.1, batch256 when selected",
        "frame_stack":"8 follows repository configuration; paper does not specify it clearly",
        "schedule":"steps are environment steps, one combined update per step, plus separately counted pretraining",
        "phase_clock":args.phase_clock,"demo_bc_only":args.demo_bc_only,"demo_phase_goals":args.demo_phase_goals,"absolute_joints":args.absolute_joints,"soft_endpoint_rescale":args.soft_endpoint_rescale,
        "softargmax_beta":args.softargmax_beta,
        "limitations":"procedural Isaac Sim task, sparse +1 reward, retained C51/FOSD; not full VL-I2O reproduction"}
    write_json(OUT/"config.json",R)
    iterator = replay.batches(args.batch_size)
    before = next(agent.critic.network.adv_head.parameters()).detach().clone()
    encoder_before = next(agent.encoder.parameters()).detach().clone()
    metrics_log = open(OUT / "metrics.jsonl", "w", buffering=1)
    updates = resume_updates
    def update(stage, step):
        global updates
        metrics = agent.update(iterator, updates)
        assert all(np.isfinite(v) for v in metrics.values()), metrics
        metrics_log.write(json.dumps({"stage":stage,"step":step,"update":updates,**metrics})+"\n")
        updates += 1
        return metrics
    def checkpoint(step, path):
        payload = {name:getattr(agent,name).state_dict() for name in ("encoder","critic","critic_target")}
        payload.update(encoder_opt=agent.encoder_opt.state_dict(),critic_opt=agent.critic_opt.state_dict(),
            step=step, updates=updates, rgb_shape=env.rgb_shape, state_shape=env.state_shape,
            config=R["args"], action_mode=args.action_mode, softargmax_beta=args.softargmax_beta,
            algorithm_variant="C51 I2O with differentiable soft MSE imitation and inherited margin/FOSD losses",
            task=args.task,demo_source=R.get("demo_source"),
            note="online replay/RNG/environment restart; step/update counters restore; fixed demos reloaded")
        payload["pretrain_mode"] = args.pretrain_mode
        temporary=path.with_suffix(path.suffix+".tmp")
        torch.save(payload,temporary)
        temporary.replace(path)
    pretrain_start=time.monotonic()
    for i in range(args.pretrain_updates):
        if args.pretrain_mode=="mse":
            batch=next(iterator)
            metrics=agent.imitation_update(batch[0],batch[1],batch[6])
            assert all(np.isfinite(v) for v in metrics.values())
            metrics_log.write(json.dumps({"stage":"mse_pretrain","step":i,"update":updates,**metrics})+"\n")
            updates+=1
        else:
            metrics=update("demo_pretrain",i)
        if (i+1)%32==0 or i+1==args.pretrain_updates:
            progress("pretraining",pretrain_update=i+1,total_pretrain_updates=args.pretrain_updates,metrics=metrics)
        if (i+1)%250==0:
            checkpoint(0,OUT/"pretrain_checkpoint.pt")
    if args.pretrain_mode=="mse" and args.pretrain_updates > 0:
        agent.critic_target.load_state_dict(agent.critic.state_dict())
    R["pretrain_seconds"]=time.monotonic()-pretrain_start
    checkpoint(resume_step,OUT/"checkpoint.pt")
    env.rng=np.random.default_rng(args.seed)
    obs = env.reset()
    episodes = []
    policy_actions = []
    started = time.monotonic()
    step_times=[];eval_history=[];eval_timings=[]
    episode_log=open(OUT/"episodes.jsonl","w",buffering=1)
    def evaluate(count, label):
        # Preserve RNG streams and never add evaluation data to training replay.
        # The shared physical environment is reset for eval and again afterward.
        evaluation_start=time.monotonic();evaluation_steps=0
        py_state=utils.random.getstate();np_state=np.random.get_state()
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        saved_env_rng=env.rng
        out=[]
        try:
            env.rng=np.random.default_rng(args.eval_seed)
            for ep in range(count):
                evaluation_obs=env.reset()
                for t in range(env.horizon):
                    with torch.no_grad(),utils.eval_mode(agent):
                        action=agent.act(*evaluation_obs,step=online_step,eval_mode=True)
                    evaluation_obs,reward,terminated,truncated,info=env.step(action)
                    evaluation_steps+=1
                    if terminated or truncated:break
                out.append(info)
                print("I2O_EVAL",label,ep,json.dumps(info),flush=True)
            if out:env.save_image(OUT/f"eval_{label}.png")
        finally:
            env.rng=saved_env_rng
            utils.random.setstate(py_state);np.random.set_state(np_state)
            torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
        eval_timings.append({"label":label,"episodes":count,"steps":evaluation_steps,
                             "seconds":time.monotonic()-evaluation_start})
        return out
    progress("online_training",total_online_steps=args.steps)
    info = {}
    for step in range(resume_step, args.steps):
        if args.benchmark:torch.cuda.synchronize(GPU)
        step_start=time.monotonic()
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(*obs, step=step, eval_mode=False)
        assert action.shape==env.action_shape and np.isfinite(action).all() and np.max(np.abs(action))<=1.00001
        nxt, reward, terminated, truncated, info = env.step(action)
        replay.add(obs,action,reward,0. if terminated else 0.99,nxt)
        obs = nxt
        policy_actions.append(action.tolist())
        metrics = update("online",step)
        online_step=step+1
        if args.benchmark:torch.cuda.synchronize(GPU)
        step_times.append(time.monotonic()-step_start)
        if terminated or truncated:
            episodes.append(info)
            episode_log.write(json.dumps({"at_online_step":online_step,**info})+"\n")
            obs = env.reset()
        if (step+1)%32==0 or step+1==args.steps:
            progress("online_training",total_online_steps=args.steps,metrics=metrics,
                recent_step_seconds=float(np.mean(step_times[-32:])),
                cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(GPU)/2**30,
                cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(GPU)/2**30)
        if (step+1)%32==0:
            print("I2O_TRAIN",step+1,"updates",updates,"loss",metrics.get("q_critic_loss"),flush=True)
        if (step+1)%args.checkpoint_every==0:
            checkpoint(step+1,OUT/"checkpoint.pt")
            if (step+1)%5000==0:checkpoint(step+1,OUT/f"checkpoint_{step+1}.pt")
        if ((args.eval_every and (step+1)%args.eval_every==0) or step+1==args.eval_at) and step+1<args.steps:
            # Reset only at a real boundary or explicitly end this partial episode
            # as a time-limit truncation before evaluation, so replay never links resets.
            eval_start=time.monotonic();progress("evaluating",at_step=step+1)
            if not (terminated or truncated):
                cut={**info,"truncated_for_evaluation":True,"at_online_step":online_step}
                episodes.append(cut);episode_log.write(json.dumps(cut)+"\n")
            results=evaluate(args.periodic_eval_episodes,str(step+1))
            rate=sum(x["success"] for x in results)/len(results) if results else None
            eval_history.append({"step":step+1,"success_rate":rate,"episodes":results,
                                 "seconds":time.monotonic()-eval_start})
            write_json(OUT/"evaluation_history.json",eval_history)
            obs=env.reset()
            checkpoint(step+1,OUT/"checkpoint.pt")
    checkpoint(args.steps,OUT/"checkpoint.pt")
    online_last_info = info.copy()
    delta = float((next(agent.critic.network.adv_head.parameters()).detach()-before).abs().max())
    encoder_delta=float((next(agent.encoder.parameters()).detach()-encoder_before).abs().max())
    assert resume_step == args.steps or (delta>0 and encoder_delta>0), "network weights did not update"
    (OUT/"policy_actions.json").write_text(json.dumps(policy_actions))
    env.save_image(OUT/"policy_rollout.png")
    progress("final_evaluation")
    evaluations=evaluate(args.eval_episodes,"final")
    # Evaluation reset the physical environment; take a fresh consistent observation.
    if args.eval_episodes:obs=env.observe()
    # Verify saved weights reload and yield identical predictions on the same observation.
    with torch.no_grad():
        rgb=torch.as_tensor(obs[0],device=device).unsqueeze(0).float()
        reference=agent.encoder(rgb).clone()
        saved=torch.load(OUT/"checkpoint.pt",map_location=device,weights_only=False)
        agent.encoder.load_state_dict(saved["encoder"])
        reload_error=float((agent.encoder(rgb)-reference).abs().max())
    assert reload_error==0
    metrics_log.close()
    episode_log.close()
    R.update(pass_checks={"successful_demonstrations":successes==requested_demos,
        "online_policy_steps":len(policy_actions)==args.steps-resume_step,"finite_losses":True,
        "critic_updated":delta>0 or resume_step==args.steps,"encoder_updated":encoder_delta>0 or resume_step==args.steps,"checkpoint_reload":reload_error==0},
        updates=updates, critic_parameter_max_change=delta, encoder_parameter_max_change=encoder_delta,
        online_episodes=episodes, online_last_info=online_last_info, eval_results=evaluations,
        eval_success_rate=sum(x["success"] for x in evaluations)/len(evaluations) if evaluations else None,
        training_seconds=time.monotonic()-started, evaluation_history=eval_history,evaluation_timings=eval_timings,
        step_seconds_mean=float(np.mean(step_times)) if step_times else None,
        step_seconds_median=float(np.median(step_times[4:] or step_times)) if step_times else None,
        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated(GPU)/2**30,
        cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(GPU)/2**30,
        checkpoint=str(OUT/"checkpoint.pt"),
        upstream_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/"vendor").glob("*.py")})
    R["pass"]=all(R["pass_checks"].values())
    if args.benchmark:
        import subprocess
        R["gpu_memory_after_mib"]=subprocess.check_output(["nvidia-smi",f"--id={GPU}",
            "--query-gpu=memory.used,memory.total","--format=csv,noheader,nounits"],text=True).strip()
        write_json(OUT/"benchmark.json",R)
    progress("complete",pass_checks=R["pass_checks"])
except Exception:
    R["error"]=traceback.format_exc()
    print(R["error"],flush=True)
    progress("failed",error=R["error"])
finally:
    (OUT/"result.json").write_text(json.dumps(R,indent=2))
    latest=OUT.parent/"latest"
    if latest.is_symlink():latest.unlink()
    if not latest.exists():latest.symlink_to(OUT.name)
    print("I2O_RESULT",json.dumps(R),flush=True)
    if env is not None:env.close()
    app.close()
sys.exit(0 if R["pass"] else 3)
