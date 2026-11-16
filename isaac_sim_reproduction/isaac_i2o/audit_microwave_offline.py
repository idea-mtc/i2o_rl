"""Read-only comparison of checkpoints on identical stored expert observations."""
import os,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from isaacsim import SimulationApp
gpu=int(os.environ['PINGANDOG_GPU']);app=SimulationApp({'headless':True,'multi_gpu':False,'active_gpu':gpu,'physics_gpu':gpu})
import numpy as np,torch
from vendor.i2o_rl import I2OAgent
from vendor import utils
from aligned.contract import ActionCodec
root=Path('/home/ubuntu/pingandog');job=root/'outputs/extended_asset_training/run_20261114/open_microwave';out=root/'outputs/microwave_diagnosis';gpu=int(os.environ['PINGANDOG_GPU']);device=f'cuda:{gpu}'
torch.set_num_threads(4);torch.set_float32_matmul_precision('high');utils.set_seed_everywhere(1)
m=json.loads((job/'dataset/manifest.json').read_text());examples=[]
for entry in m['episodes'][::8]:
 with np.load(job/'dataset'/entry['file'],allow_pickle=False) as f:
  rgb=f['rgb'];state=f['state'];actions=f['raw_actions'];closed=np.flatnonzero(actions[:,-1]<.5);switch=int(closed[0])
  ts=sorted(set([0,10,25,40,50,min(len(actions)-1,100)]+[int(t) for t in range(max(0,switch-5),min(len(actions),switch+6))]))
  for t in ts:
   ids=np.maximum(0,np.arange(t-7,t+1));examples.append((np.concatenate(rgb[ids],axis=1),state[ids].reshape(120),actions[t],t-switch))
agent=I2OAgent(rgb_obs_shape=(4,24,84,84),low_dim_obs_shape=(120,),action_shape=(8,),device=device,lr=5e-5,weight_decay=.1,feature_dim=128,hidden_dim=512,levels=3,bins=5,atoms=51,v_min=-2.,v_max=2.,bc_lambda=1.,bc_margin=.01,critic_lambda=.1,critic_target_tau=.02,num_expl_steps=0,update_every_steps=1,stddev_schedule='.01',use_logger=True,action_mode='soft',softargmax_beta=1.,soft_endpoint_rescale=False,phase_heads=1)
results=[]
for name in ['validated_start.pt','checkpoint_5000.pt','checkpoint_10000.pt','checkpoint_20000.pt','latest.pt']:
 s=torch.load(job/name,map_location=device,weights_only=False);codec=ActionCodec(s['codec']['min'],s['codec']['max'])
 for k in ['encoder','critic','critic_target']:getattr(agent,k).load_state_dict(s[k])
 preds=[]
 for i in range(0,len(examples),24):
  chunk=examples[i:i+24];rgb=torch.as_tensor(np.stack([x[0] for x in chunk]),device=device);low=torch.as_tensor(np.stack([x[1] for x in chunk]),device=device)
  with torch.no_grad(),utils.eval_mode(agent):a=agent.critic_target.get_soft_action(agent.encoder(rgb),low,beta=1.,endpoint_rescale=False)
  preds.append(a.cpu().numpy())
 pred=np.concatenate(preds);raw=codec.decode(pred);target=np.stack([x[2] for x in examples]);normalized=codec.encode(target);relative=np.array([x[3] for x in examples]);groups={}
 for key,mask in [('all',np.ones(len(target),bool)),('before_close',relative<0),('switch_neighborhood',abs(relative)<=5),('after_close',relative>=0)]:
  groups[key]=dict(n=int(mask.sum()),normalized_mse=float(((pred[mask]-normalized[mask])**2).mean()),arm_raw_rmse=float(np.sqrt(((raw[mask,:7]-target[mask,:7])**2).mean())),gripper_binary_accuracy=float(((raw[mask,-1]>.5)==(target[mask,-1]>.5)).mean()),gripper_raw_mae=float(abs(raw[mask,-1]-target[mask,-1]).mean()))
 files=s['replay_files'];result=dict(checkpoint=name,online_step=s['online_step'],warm_updates=s['warm_updates'],examples=len(examples),groups=groups,replay_episodes=len(files),successful_pool=sum(x['successful'] for x in files),online_successes=sum(x['successful'] for x in files[100:]),online_episodes=len(files)-100)
 results.append(result);print(json.dumps(result),flush=True);del s
(out/'offline_checkpoint_comparison.json').write_text(json.dumps(results,indent=2))

app.close()
