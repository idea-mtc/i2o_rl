"""Bounded three-arm diagnostic: offline IL, online-matched IL, online IL+TD.

B alone collects experience. B and C consume identical tensors/augmentations;
C's only loss difference is rl=False. A samples original demos exclusively.
This is a diagnostic, not a replacement production training configuration.
"""
import os,sys,json,time,random,hashlib,copy,datetime,traceback,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
p=argparse.ArgumentParser();p.add_argument('--job-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=1000);args=p.parse_args();assert args.steps==1000
ROOT=Path('/home/ubuntu/pingandog');SOURCE=ROOT/'outputs/extended_asset_training/run_20261114/open_microwave';OUT=args.job_dir.resolve();OUT.mkdir(parents=True,exist_ok=True);GPU=int(os.environ['PINGANDOG_GPU'])
def write(name,data):
 f=OUT/name;f.parent.mkdir(exist_ok=True,parents=True);t=f.with_suffix(f.suffix+'.tmp');t.write_text(json.dumps(data,indent=2));t.replace(f)
def progress(stage,**data):write('progress.json',dict(stage=stage,gpu=GPU,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**data))
progress('initializing')
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','multi_gpu':False,'active_gpu':GPU,'physics_gpu':GPU,'extra_args':['--/physics/updateToUsd=true']})
env=None
try:
 import numpy as np,torch
 from vendor.i2o_rl import I2OAgent
 from vendor import utils
 from aligned.extended_tasks import ExtendedTaskEnv
 from aligned.contract import ActionCodec
 from aligned.gpu_replay import GPUReplay
 from aligned.paper_update import update
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high');utils.set_seed_everywhere(1);device=f'cuda:{GPU}'
 saved=torch.load(SOURCE/'validated_start.pt',map_location=device,weights_only=False);assert saved['online_step']==0
 manifest=json.loads((SOURCE/'dataset/manifest.json').read_text());assert len(manifest['episodes'])==100
 codec=ActionCodec(saved['codec']['min'],saved['codec']['max']);replay=GPUReplay(codec,device,capacity=18000,seed=1)
 for e in manifest['episodes']:
  path=SOURCE/'dataset'/e['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==e['sha256']
  with np.load(path,allow_pickle=False) as f:data={k:f[k] for k in f.files}
  replay.add(data,True,path)
 agents={}
 for branch in ['A_offline_il','B_online_td','C_matched_il']:
  a=I2OAgent(rgb_obs_shape=(4,24,84,84),low_dim_obs_shape=(120,),action_shape=(8,),device=device,lr=5e-5,weight_decay=.1,feature_dim=128,hidden_dim=512,levels=3,bins=5,atoms=51,v_min=-2.,v_max=2.,bc_lambda=1.,bc_margin=.01,critic_lambda=.1,critic_target_tau=.02,num_expl_steps=0,update_every_steps=1,stddev_schedule='.01',use_logger=True,action_mode='soft',softargmax_beta=1.,soft_endpoint_rescale=False,phase_heads=1)
  for k in ['encoder','critic','critic_target','encoder_opt','critic_opt']:getattr(a,k).load_state_dict(copy.deepcopy(saved[k]))
  agents[branch]=a
 # Ensure both parameters and optimizer moments match before the intervention.
 for branch in ['A_offline_il','C_matched_il']:
  for k in ['encoder','critic','critic_target']:
   assert all(torch.equal(v,getattr(agents[branch],k).state_dict()[n]) for n,v in getattr(agents['B_online_td'],k).state_dict().items())
 env=ExtendedTaskEnv('open_microwave');env.codec=codec;env.diagnostics_dir=OUT/'camera_diagnostics'
 replay.rng.bit_generator.state=copy.deepcopy(saved['replay_rng']);a_rng=np.random.default_rng();a_rng.bit_generator.state=copy.deepcopy(saved['replay_rng'])
 random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),GPU)
 # The source checkpoint follows gate seed810004, after one XY draw.
 env.rng=np.random.default_rng(810004);env.rng.uniform(-.02,.02,2)
 write('contract.json',dict(source=str(SOURCE/'validated_start.pt'),source_sha256=hashlib.sha256((SOURCE/'validated_start.pt').read_bytes()).hexdigest(),steps=1000,checkpoints=[500,1000],seed=1,lr=5e-5,td_scale=.1,batch=256,demos=100,branches={'A_offline_il':'demo-only, IL loss','B_online_td':'original mixed replay, IL+.1TD, source online collection','C_matched_il':'identical B batch and augmentations, IL only'},evaluation_seeds=[900000+5*i for i in range(20)],evaluation='separate frozen-checkpoint workers; never resets collection environment',notes=['A replays source demos only even after online successful episodes are added.','C follows the B-generated fixed data stream; C does not collect its own behavior.','Evaluation at500 is external, so collection is not interrupted; source first1000 also has no periodic evaluation before1000.','All models, optimizer moments and initial RNG restored; original source checkpoint remains immutable.']))
 def start(obs):return dict(rgb=[obs[0][:,-3:].copy()],state=[obs[1][-15:].copy()],raw_actions=[],rewards=[],terminated=[],truncated=[],sim_time=[float(env.world.current_time)])
 def append(ep,raw,obs,reward,term,trunc):
  ep['rgb'].append(obs[0][:,-3:].copy());ep['state'].append(obs[1][-15:].copy());ep['raw_actions'].append(raw.copy());ep['rewards'].append(reward);ep['terminated'].append(term);ep['truncated'].append(trunc);ep['sim_time'].append(float(env.world.current_time))
 def store_episode(ep,step,success):
  data={k:np.asarray(v,dtype=np.uint8 if k=='rgb' else np.bool_ if k in ['terminated','truncated'] else np.float64 if k=='sim_time' else np.float32) for k,v in ep.items()}
  path=OUT/'online_episodes'/f'episode_{step}.npz';path.parent.mkdir(exist_ok=True)
  with path.open('wb') as f:np.savez_compressed(f,**data)
  replay.add(data,success,path)
 def snapshot(step):
  for branch,a in agents.items():
   path=OUT/branch;path.mkdir(exist_ok=True)
   payload={k:getattr(a,k).state_dict() for k in ['encoder','critic','critic_target','encoder_opt','critic_opt']}
   payload.update(online_step=step,warm_updates=saved['warm_updates'],codec=codec.json(),branch=branch,diagnostic_updates=step,source_checkpoint_sha256=hashlib.sha256((SOURCE/'validated_start.pt').read_bytes()).hexdigest())
   temp=path/f'checkpoint_{step}.pt.tmp';torch.save(payload,temp);temp.replace(path/f'checkpoint_{step}.pt')
 obs=env.reset();episode=start(obs);metrics_file=(OUT/'metrics.jsonl').open('a',buffering=1);began=time.monotonic();successes=0;episodes=0
 for step in range(1,1001):
  b=agents['B_online_td']
  with torch.no_grad(),utils.eval_mode(b):normalized=b.act(*obs,step=step-1,eval_mode=False)
  raw=codec.decode(normalized);obs,r,term,trunc,info=env.step_raw(raw);append(episode,raw,obs,r,term,trunc)
  batch=replay.sample(256)
  before=(torch.get_rng_state(),torch.cuda.get_rng_state(GPU));mb=update(b,batch,rl=True);after=(torch.get_rng_state(),torch.cuda.get_rng_state(GPU))
  torch.set_rng_state(before[0]);torch.cuda.set_rng_state(before[1],GPU);mc=update(agents['C_matched_il'],batch,rl=False)
  if step==1:
   assert abs(mb['action_mse']-mc['action_mse'])<1e-7 and abs(mb['margin']-mc['margin'])<1e-7
   write('initial_equality_check.json',dict(parameters_and_optimizers_cloned=True,matched_batch=True,matched_augmentation=True,B=mb,C=mc))
  # Sample A from the100 original episodes, not the growing successful pool.
  rng_b=replay.rng;pool_b=replay.success_ids;replay.rng=a_rng;replay.success_ids=list(range(100))
  try:batch_a=replay.sample(256,demonstrations_only=True)
  finally:replay.rng=rng_b;replay.success_ids=pool_b
  torch.set_rng_state(before[0]);torch.cuda.set_rng_state(before[1],GPU);ma=update(agents['A_offline_il'],batch_a,rl=False)
  torch.set_rng_state(after[0]);torch.cuda.set_rng_state(after[1],GPU)
  metrics_file.write(json.dumps(dict(step=step,A=ma,B=mb,C=mc))+'\n')
  if term or trunc:
   store_episode(episode,step,bool(info['success']));episodes+=1;successes+=int(info['success']);obs=env.reset();episode=start(obs)
  if step%25==0:progress('bounded_updates',updates=step,total=1000,online_episodes=episodes,online_successes=successes,seconds=time.monotonic()-began,losses={'A':ma,'B':mb,'C':mc})
  if step in [500,1000]:snapshot(step)
 if episode['raw_actions']:
  episode['truncated'][-1]=True;store_episode(episode,1000,False)
 progress('updates_complete',updates=1000,online_episodes=episodes,online_successes=successes,seconds=time.monotonic()-began)
except Exception:
 error=traceback.format_exc();write('error.json',dict(error=error));progress('failed',error=error);raise
finally:
 if env is not None:env.close()
 app.close()
