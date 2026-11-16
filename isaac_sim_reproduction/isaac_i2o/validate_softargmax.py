"""Isolate soft imitation gradients, fit fixed demos, then execute the learned policy."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--updates", type=int, default=1000)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--fit-samples", type=int, default=64)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--eval-episodes", type=int, default=2)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()
assert args.updates >= 20 and args.batch_size >= 2 and args.beta > 0
ROOT = Path(os.environ.get("PINGANDOG_ROOT", "/home/ubuntu/pingandog"))
OUT = ROOT / "outputs/softargmax_panda" / datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
OUT.mkdir(parents=True)
GPU = int(os.environ.get("PINGANDOG_GPU", "0"))
R = {"pass":False,"output":str(OUT),"data":str(args.data),
     "scope":"MSE-only soft imitation verification; learner grasp performance reported separately",
     "args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
from isaacsim import SimulationApp
app = SimulationApp({"headless":True,"renderer":"RaytracedLighting","multi_gpu":False,
    "active_gpu":GPU,"physics_gpu":GPU,"width":960,"height":720,
    "extra_args":["--/physics/updateToUsd=true"]})
env = None
try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from vendor import utils
    from vendor.i2o_rl import I2OAgent
    from vendor.cqn_utils import soft_zoom_in
    from panda_env import PandaLiftEnv
    torch.set_num_threads(4)
    utils.set_seed_everywhere(args.seed)
    rng=np.random.default_rng(args.seed)
    device=f"cuda:{GPU}"
    # Finite-difference test on the complete 3-level soft decoding chain.
    logits=torch.randn(3,2,8,5,dtype=torch.double,requires_grad=True)
    def decode_chain(scores):
        lo=torch.full((2,8),-1.,dtype=scores.dtype)
        hi=torch.ones_like(lo)
        for level in range(3):lo,hi=soft_zoom_in(lo,hi,scores[level],5,args.beta)
        return (lo+hi)/2
    numerical_gradcheck=torch.autograd.gradcheck(decode_chain,(logits,),fast_mode=True)
    uniform=decode_chain(torch.zeros_like(logits))
    assert numerical_gradcheck and torch.allclose(uniform,torch.zeros_like(uniform),atol=1e-12)
    bounded=decode_chain(logits)
    assert bool((bounded>=-1).all() and (bounded<=1).all())
    data=np.load(args.data,allow_pickle=False)
    ends=np.flatnonzero(data["discounts"]==0)
    assert len(ends)>=2,"two successful episodes required for episode-separated holdout"
    train_end=int(ends[0])+1
    train_ids=np.linspace(0,train_end-1,min(args.fit_samples,train_end)).astype(int)
    test_ids=np.arange(train_end,len(data["actions"]))
    rgb_shape=tuple(data["rgb"].shape[1:]);state_shape=tuple(data["state"].shape[1:])
    assert rgb_shape[0]==1 and rgb_shape[-2:]==(84,84)
    agent=I2OAgent(rgb_shape,state_shape,(8,),device,5e-5,128,512,3,5,51,-2.,2.,
        1.,0.01,0.1,0.02,0.1,0,1,"0.01",True,
        softargmax_beta=args.beta,action_mode="soft")
    rgb=torch.as_tensor(data["rgb"],dtype=torch.float32,device=device)
    state=torch.as_tensor(data["state"],dtype=torch.float32,device=device)
    actions=torch.as_tensor(data["actions"],dtype=torch.float32,device=device)
    assert float((rgb[0]-rgb[train_end-1]).abs().mean())>0.1,"stale camera observations"
    def measures(ids):
        with torch.no_grad():
            prediction=agent.critic.get_soft_action(agent.encoder(rgb[ids]),state[ids],args.beta)
            assert prediction.shape==actions[ids].shape and torch.isfinite(prediction).all()
            assert prediction.abs().max()<=1.000001
            return {"mse":float(F.mse_loss(prediction,actions[ids])),
                "arm_mse":float(F.mse_loss(prediction[:,:7],actions[ids,:7])),
                "gripper_mse":float(F.mse_loss(prediction[:,7],actions[ids,7]))}
    before_train=measures(train_ids);before_test=measures(test_ids)
    before_weights={"cnn":next(agent.encoder.parameters()).detach().clone(),
        "head":agent.critic.network.adv_head.weight.detach().clone()}
    # Show that the previous hard action path cannot propagate an action loss.
    hard,_=agent.critic.get_action(agent.encoder(rgb[train_ids[:8]]),state[train_ids[:8]])
    hard_loss_has_gradient=F.mse_loss(hard,actions[train_ids[:8]]).requires_grad
    assert not hard_loss_has_gradient
    gradients=[]
    def audit_gradient(iteration):
        agent.encoder_opt.zero_grad(set_to_none=True);agent.critic_opt.zero_grad(set_to_none=True)
        pred,levels=agent.critic.get_soft_action(agent.encoder(rgb[train_ids]),state[train_ids],args.beta,True)
        loss=F.mse_loss(pred,actions[train_ids]);loss.backward()
        groups={"cnn":agent.encoder,"visual_projection":agent.critic.network.adv_rgb_encoder,
            "state_projection":agent.critic.network.adv_low_dim_encoder,
            "fusion":agent.critic.network.adv_fusion,"q_trunk":agent.critic.network.adv_net,
            "q_head":agent.critic.network.adv_head}
        norms={name:sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None)**0.5
               for name,module in groups.items()}
        norms.update({f"level_{i+1}_q":float(q.grad.norm()) for i,q in enumerate(levels)})
        assert all(np.isfinite(v) for v in norms.values())
        row={"iteration":iteration,"loss":float(loss),"norms":norms}
        gradients.append(row);print("SOFT_GRADIENT",json.dumps(row),flush=True)
        return norms
    audit_gradient(0)
    log=open(OUT/"metrics.jsonl","w",buffering=1)
    started=time.monotonic()
    active_gradients=None
    for iteration in range(args.updates):
        indices=rng.choice(train_ids,size=args.batch_size,replace=True)
        metrics=agent.imitation_update(rgb[indices],state[indices],actions[indices])
        assert np.isfinite(metrics["action_mse"])
        log.write(json.dumps({"iteration":iteration,**metrics})+"\n")
        if iteration==19:
            active_gradients=audit_gradient(iteration+1)
            assert all(v>1e-12 for v in active_gradients.values()),active_gradients
        if (iteration+1)%100==0:
            print("SOFT_FIT",iteration+1,json.dumps(measures(train_ids)),flush=True)
    after_train=measures(train_ids);after_test=measures(test_ids)
    weight_changes={"cnn":float((next(agent.encoder.parameters()).detach()-before_weights["cnn"]).abs().max()),
        "head":float((agent.critic.network.adv_head.weight.detach()-before_weights["head"]).abs().max())}
    # Exercise the real combined TD + margin + soft action imitation update.
    # Back up and restore the MSE-only model so rollout measures only imitation.
    import copy
    from replay import Replay
    model_before_combined={n:copy.deepcopy(getattr(agent,n).state_dict()) for n in ("encoder","critic","critic_target")}
    replay=Replay(seed=args.seed)
    for idx in train_ids:
        replay.add((data["rgb"][idx],data["state"][idx]),data["actions"][idx],data["rewards"][idx],
            data["discounts"][idx],(data["next_rgb"][idx],data["next_state"][idx]),True)
    combined_metrics=agent.update(replay.batches(8),0)
    assert "bc_loss" in combined_metrics and all(np.isfinite(v) for v in combined_metrics.values())
    combined_cnn_grad=sum(float(p.grad.norm()) for p in agent.encoder.parameters() if p.grad is not None)
    assert combined_cnn_grad>0
    for name,value in model_before_combined.items():getattr(agent,name).load_state_dict(value)
    agent.critic_target.load_state_dict(agent.critic.state_dict())
    checkpoint={name:getattr(agent,name).state_dict() for name in ("encoder","critic","critic_target")}
    checkpoint.update(rgb_shape=rgb_shape,state_shape=state_shape,action_mode="soft",beta=args.beta,
                      updates=args.updates,training="MSE-only, no TD/margin, isolated optimizer state not exported")
    torch.save(checkpoint,OUT/"soft_imitation.pt")
    np.savez_compressed(OUT/"predictions.npz",expert=data["actions"][train_ids],
        prediction=agent.critic.get_soft_action(agent.encoder(rgb[train_ids]),state[train_ids],args.beta).detach().cpu().numpy(),
        train_indices=train_ids,heldout_indices=test_ids)
    R.update(numerical_gradcheck=bool(numerical_gradcheck),hard_action_loss_requires_grad=hard_loss_has_gradient,
        gradients=gradients,before_train=before_train,after_train=after_train,
        before_heldout=before_test,after_heldout=after_test,weight_changes=weight_changes,
        fit_samples=len(train_ids),heldout_samples=len(test_ids),split="first episode train, remaining episodes held out",
        beta=args.beta,combined_update_metrics=combined_metrics,combined_cnn_grad_norm=combined_cnn_grad,
        fit_seconds=time.monotonic()-started)
    checks={"finite_difference_gradient":bool(numerical_gradcheck),
        "all_gradient_groups_nonzero":all(v>1e-12 for v in active_gradients.values()),
        "train_mse_reduced_by_half":after_train["mse"]<before_train["mse"]*0.5,
        "arm_mse_reduced":after_train["arm_mse"]<before_train["arm_mse"],
        "gripper_mse_reduced":after_train["gripper_mse"]<before_train["gripper_mse"],
        "weights_updated":all(v>0 for v in weight_changes.values()),"combined_update_finite":True}
    # Reload before actual simulation rollout; no teacher is called in evaluation.
    saved=torch.load(OUT/"soft_imitation.pt",map_location=device,weights_only=False)
    for name in ("encoder","critic","critic_target"):getattr(agent,name).load_state_dict(saved[name])
    env=PandaLiftEnv(rgb_shape[1]//3,220,args.seed+100,0.02)
    evaluations=[];policy_trace=[]
    for ep in range(args.eval_episodes):
        obs=env.reset()
        max_lift=0.0
        for step in range(env.horizon):
            with torch.no_grad(),utils.eval_mode(agent):action=agent.act(*obs,step=step,eval_mode=True)
            assert np.isfinite(action).all() and np.abs(action).max()<=1.000001
            obs,reward,terminated,truncated,info=env.step(action)
            max_lift=max(max_lift,info["lift_m"])
            policy_trace.append({"episode":ep,"step":step,"action":action.tolist(),**info})
            if terminated or truncated:break
        evaluations.append({**info,"maximum_lift_m":max_lift})
        env.save_image(OUT/f"policy_eval_{ep}.png")
        print("SOFT_POLICY_EVAL",json.dumps(evaluations[-1]),flush=True)
    (OUT/"policy_trace.json").write_text(json.dumps(policy_trace))
    checks["policy_rollout_finite"]=bool(policy_trace) or args.eval_episodes==0
    R.update(checks=checks,eval_results=evaluations,
        policy_success_rate=sum(i["success"] for i in evaluations)/len(evaluations) if evaluations else None)
    R["pass"]=all(checks.values())
    log.close()
except Exception:
    R["error"]=traceback.format_exc();print(R["error"],flush=True)
finally:
    (OUT/"result.json").write_text(json.dumps(R,indent=2))
    latest=OUT.parent/"latest"
    if latest.is_symlink():latest.unlink()
    if not latest.exists():latest.symlink_to(OUT.name)
    print("SOFT_RESULT",json.dumps(R),flush=True)
    if env is not None:env.close()
    app.close()
sys.exit(0 if R["pass"] else 3)
