"""Frozen evaluation worker for bounded ablation snapshots; never trains."""
import argparse,os,sys,json,time,hashlib,datetime,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
p=argparse.ArgumentParser();p.add_argument('--job-dir',type=Path,required=True);p.add_argument('--branch',required=True,choices=['C_replay_verified','D_no_promotion']);args=p.parse_args();OUT=args.job_dir.resolve();OUT.mkdir(parents=True,exist_ok=True);RUN=OUT.parent;GPU=int(os.environ['PINGANDOG_GPU']);ROOT=Path('/home/ubuntu/pingandog');SOURCE=ROOT/'outputs/extended_asset_training/run_20261114/open_microwave';SEEDS=[900000+5*i for i in range(20)]
def write(name,data):
 f=OUT/name;f.parent.mkdir(parents=True,exist_ok=True);tmp=f.with_suffix(f.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(f)
def progress(stage,**data):write('evaluation_progress.json',dict(stage=stage,branch=args.branch,gpu=GPU,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**data))
progress('initializing')
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','multi_gpu':False,'active_gpu':GPU,'physics_gpu':GPU,'extra_args':['--/physics/updateToUsd=true']});env=None
try:
 import numpy as np,torch
 from aligned.extended_tasks import ExtendedTaskEnv
 from aligned.contract import ActionCodec
 from vendor.i2o_rl import I2OAgent
 from vendor import utils
 torch.set_num_threads(3);torch.set_float32_matmul_precision('high');utils.set_seed_everywhere(1);device=f'cuda:{GPU}'
 env=ExtendedTaskEnv('open_microwave');env.diagnostics_dir=OUT/'camera_diagnostics'
 agent=I2OAgent(rgb_obs_shape=(4,24,84,84),low_dim_obs_shape=(120,),action_shape=(8,),device=device,lr=5e-5,weight_decay=.1,feature_dim=128,hidden_dim=512,levels=3,bins=5,atoms=51,v_min=-2.,v_max=2.,bc_lambda=1.,bc_margin=.01,critic_lambda=.1,critic_target_tau=.02,num_expl_steps=0,update_every_steps=1,stddev_schedule='.01',use_logger=True,action_mode='soft',softargmax_beta=1.,soft_endpoint_rescale=False,phase_heads=1)
 summary=[]
 for updates in [1000]:
  checkpoint=SOURCE/'validated_start.pt' if updates==0 else OUT/f'checkpoint_{updates}.pt';started=time.monotonic()
  while not checkpoint.exists():
   if (RUN/'error.json').exists():raise RuntimeError('producer failed: '+(RUN/'error.json').read_text())
   if time.monotonic()-started>1800:raise TimeoutError('bounded checkpoint not ready in30minutes')
   progress('waiting_checkpoint',updates=updates);time.sleep(10)
  digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();saved=torch.load(checkpoint,map_location=device,weights_only=False)
  for k in ['encoder','critic','critic_target']:getattr(agent,k).load_state_dict(saved[k])
  c=saved['codec'];env.codec=ActionCodec(c['min'],c['max']);del saved
  results=[]
  for i,seed in enumerate(SEEDS):
   output=OUT/'evaluation'/f'updates_{updates}_{seed}.json'
   if output.exists():
    x=json.loads(output.read_text());assert x['checkpoint_sha256']==digest;results.append(x);continue
   env.rng=np.random.default_rng(seed);obs=env.reset();rows=[];initial=np.asarray(env.status()['tcp'])
   for _ in range(125):
    with torch.no_grad(),utils.eval_mode(agent):action=agent.act(*obs,step=updates,eval_mode=True)
    obs,r,term,trunc,info=env.step(action);rows.append(dict(action=action.tolist(),**info))
    if term or trunc:break
   closes=[x for x in rows if x['requested_raw_action'][-1]<=.5]
   result=dict(seed=seed,branch=args.branch,updates=updates,checkpoint_sha256=digest,success=bool(info['success']),steps=len(rows),max_angle=max(x['opening_deg'] for x in rows),first_close_step=closes[0]['step'] if closes else None,first_close_distance=closes[0]['handle_tcp_distance_m'] if closes else None,tcp_motion_m=float(np.linalg.norm(np.ptp(np.vstack([initial,[x['tcp'] for x in rows]]),axis=0))),final=info)
   write(str(output.relative_to(OUT)),result);write(f'evaluation/updates_{updates}_{seed}_trace.json',rows);results.append(result)
   if i==0:env.save_image(OUT/f'updates_{updates}_seed_{seed}.png')
   progress('evaluating',updates=updates,completed=len(results),total=20,successes=sum(x['success'] for x in results))
  entry=dict(branch=args.branch,updates=updates,episodes=20,successes=sum(x['success'] for x in results),success_rate=sum(x['success'] for x in results)/20,minimum_motion_m=min(x['tcp_motion_m'] for x in results),checkpoint_sha256=digest,results=results)
  summary.append(entry);write('evaluations.json',summary);print('ABLATION_EVAL',json.dumps({k:v for k,v in entry.items() if k!='results'}),flush=True)
 progress('complete',evaluations=[{k:x[k] for k in ['updates','successes','episodes']} for x in summary])
except Exception:
 error=traceback.format_exc();write('evaluation_error.json',dict(error=error));progress('failed',error=error);raise
finally:
 if env is not None:env.close()
 app.close()
