"""Reconstruct the fixed B stream; isolate successful-online demo promotion.
Two1000-update IL-only branches share every tensor and augmentation. No rollout.
"""
import os,sys,json,copy,hashlib,traceback,datetime,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
p=argparse.ArgumentParser();p.add_argument('--job-dir',type=Path,required=True);a=p.parse_args();OUT=a.job_dir.resolve();OUT.mkdir(parents=True,exist_ok=True);RUN=OUT.parent;ROOT=Path('/home/ubuntu/pingandog');SOURCE=ROOT/'outputs/extended_asset_training/run_20261114/open_microwave';GPU=int(os.environ['PINGANDOG_GPU'])
def write(n,d):
 f=OUT/n;f.parent.mkdir(parents=True,exist_ok=True);t=f.with_suffix(f.suffix+'.tmp');t.write_text(json.dumps(d,indent=2));t.replace(f)
def progress(stage,**d):write('progress.json',dict(stage=stage,gpu=GPU,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**d))
progress('initializing')
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'multi_gpu':False,'active_gpu':GPU,'physics_gpu':GPU})
try:
 import numpy as np,torch
 from vendor.i2o_rl import I2OAgent
 from vendor import utils
 from aligned.contract import ActionCodec
 from aligned.gpu_replay import GPUReplay
 from aligned.paper_update import update
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high');utils.set_seed_everywhere(1);device=f'cuda:{GPU}';saved=torch.load(SOURCE/'validated_start.pt',map_location=device,weights_only=False)
 codec=ActionCodec(saved['codec']['min'],saved['codec']['max']);replay=GPUReplay(codec,device,18000,1)
 def load(path):
  with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
 manifest=json.loads((SOURCE/'dataset/manifest.json').read_text())
 for e in manifest['episodes']:replay.add(load(SOURCE/'dataset'/e['file']),True,SOURCE/'dataset'/e['file'])
 original_end=replay.used
 files=sorted((RUN/'online_episodes').glob('episode_*.npz'),key=lambda p:int(p.stem.split('_')[-1]));pending=[(int(p.stem.split('_')[-1]),p) for p in files]
 agents={}
 for branch in ['C_replay_verified','D_no_promotion']:
  agent=I2OAgent(rgb_obs_shape=(4,24,84,84),low_dim_obs_shape=(120,),action_shape=(8,),device=device,lr=5e-5,weight_decay=.1,feature_dim=128,hidden_dim=512,levels=3,bins=5,atoms=51,v_min=-2.,v_max=2.,bc_lambda=1.,bc_margin=.01,critic_lambda=.1,critic_target_tau=.02,num_expl_steps=0,update_every_steps=1,stddev_schedule='.01',use_logger=True,action_mode='soft',softargmax_beta=1.,soft_endpoint_rescale=False,phase_heads=1)
  for k in ['encoder','critic','critic_target','encoder_opt','critic_opt']:getattr(agent,k).load_state_dict(copy.deepcopy(saved[k]))
  agents[branch]=agent
 replay.rng.bit_generator.state=copy.deepcopy(saved['replay_rng']);torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),GPU)
 write('contract.json',dict(updates=1000,online_collection=False,source=str(SOURCE/'validated_start.pt'),data_stream=str(RUN/'online_episodes'),difference='D masks IL gradients for online samples; C promotes successful online episodes as before; B sampling pools unchanged for both',matching='same sampled inputs/actions and augmentation RNG; consume B exploration normal draw each step to reconstruct C',original_demo_frames=original_end))
 log=(OUT/'metrics.jsonl').open('a',buffering=1);began=time.monotonic();pointer=0;maxdiffs={}
 for step in range(1,1001):
  while pointer<len(pending) and pending[pointer][0]<step:
   _,path=pending[pointer];data=load(path);replay.add(data,bool(data['rewards'][-1]>0),path);pointer+=1
  # B consumes one8-dimensional exploration draw before its update.
  torch.empty((1,8),device=device).normal_()
  rng=copy.deepcopy(replay.rng.bit_generator.state);batch=replay.sample(256);rng_after=copy.deepcopy(replay.rng.bit_generator.state)
  masks=replay.demo;replay.demo=torch.zeros_like(masks);replay.demo[:original_end]=masks[:original_end];replay.rng.bit_generator.state=rng
  try:batch_d=replay.sample(256)
  finally:replay.demo=masks;replay.rng.bit_generator.state=rng_after
  if step==1:
   for i in [0,1,6,7,8,9,10]:assert torch.equal(batch[i],batch_d[i])
  before=(torch.get_rng_state(),torch.cuda.get_rng_state(GPU));mc=update(agents['C_replay_verified'],batch,rl=False);after=(torch.get_rng_state(),torch.cuda.get_rng_state(GPU));torch.set_rng_state(before[0]);torch.cuda.set_rng_state(before[1],GPU);md=update(agents['D_no_promotion'],batch_d,rl=False);torch.set_rng_state(after[0]);torch.cuda.set_rng_state(after[1],GPU)
  log.write(json.dumps(dict(step=step,C=mc,D=md,promoted_count=int((batch[11]-batch_d[11]).sum())))+'\n')
  if step%25==0:progress('bounded_replay_updates',updates=step,total=1000,seconds=time.monotonic()-began)
  if step in [500,1000]:
   reference=torch.load(RUN/'C_matched_il'/f'checkpoint_{step}.pt',map_location=device,weights_only=False);maxdiff=max(float((v-reference[k][n]).abs().max()) for k in ['encoder','critic','critic_target'] for n,v in getattr(agents['C_replay_verified'],k).state_dict().items());maxdiffs[step]=maxdiff;del reference
   for branch,agent in agents.items():
    folder=RUN/branch;folder.mkdir(exist_ok=True);payload={k:getattr(agent,k).state_dict() for k in ['encoder','critic','critic_target']};payload.update(online_step=step,warm_updates=saved['warm_updates'],codec=codec.json(),branch=branch,diagnostic_updates=step)
    tmp=folder/f'checkpoint_{step}.pt.tmp';torch.save(payload,tmp);tmp.replace(folder/f'checkpoint_{step}.pt')
   write('reconstruction_check.json',dict(max_parameter_differences=maxdiffs,note='Near-zero difference verifies replay and augmentation reconstruction; otherwise interpret C/D as a new paired fixed-stream ablation.'))
 progress('updates_complete',updates=1000,seconds=time.monotonic()-began,max_parameter_differences=maxdiffs)
except Exception:
 error=traceback.format_exc();write('error.json',dict(error=error));progress('failed',error=error);raise
finally:app.close()
