"""Clock-conditioned soft I2O imitation, with an explicit physical success gate."""
import argparse, datetime, json, os, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
p=argparse.ArgumentParser();p.add_argument('--updates',type=int,default=6000)
p.add_argument('--batch-size',type=int,default=128);p.add_argument('--beta',type=float,default=5.)
p.add_argument('--resume-repair',type=Path);p.add_argument('--dagger',type=int,default=6)
p.add_argument('--absolute-joints',action='store_true')
p.add_argument('--phase-heads',type=int,default=1)
p.add_argument('--demo-phase-goals',action='store_true')
p.add_argument('--cartesian-goals',action='store_true')
p.add_argument('--validate-first',action='store_true')
a=p.parse_args();assert not a.demo_phase_goals or ((a.absolute_joints or a.cartesian_goals) and a.dagger==0)
ROOT=Path('/home/ubuntu/pingandog');GPU=int(os.environ['PINGANDOG_GPU'])
OUT=ROOT/'outputs/imitation_repair'/datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S');OUT.mkdir(parents=True)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','multi_gpu':False,'active_gpu':GPU,'physics_gpu':GPU,
                  'width':960,'height':720,'extra_args':['--/physics/updateToUsd=true']})
env=None
try:
 import numpy as np,torch,subprocess
 from PIL import Image,ImageDraw
 from tasks import make_env
 from phase_clock import ClockConditionedEnv,condition
 from demonstrations import load_into_replay,file_hash
 from replay import Replay
 from vendor.i2o_rl import I2OAgent
 from vendor import utils
 torch.set_num_threads(4);utils.set_seed_everywhere(17)
 def write(name,value):
  path=OUT/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
 def progress(stage,**values):write('progress.json',dict(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),output=str(OUT),**values))
 progress('initializing')
 base=make_env('take_lid_off_saucepan',8,260,17,.02);env=ClockConditionedEnv(base)
 if a.absolute_joints:
  from absolute_joints import AbsoluteJointEnv,absolute_episode_actions,encode_joints
  env=AbsoluteJointEnv(env)
 if a.cartesian_goals:
  from cartesian_goals import CartesianGoalEnv,cartesian_episode_actions
  env=CartesianGoalEnv(env)
 dataset=ROOT/'datasets/take_lid_off_saucepan/fixed_100_20261112'
 replay=Replay(50000,17)
 action_transform=(lambda d:absolute_episode_actions(d,base,a.demo_phase_goals)) if a.absolute_joints else None
 if a.cartesian_goals:action_transform=lambda d:cartesian_episode_actions(d,base,a.demo_phase_goals)
 manifest,audit,n=load_into_replay(dataset,replay,'take_lid_off_saucepan',8,base.config,100,condition,action_transform)
 agent=I2OAgent(rgb_obs_shape=env.rgb_shape,low_dim_obs_shape=env.state_shape,action_shape=env.action_shape,
  device=f'cuda:{GPU}',lr=5e-5,weight_decay=.1,feature_dim=128,hidden_dim=512,levels=3,bins=5,atoms=51,
  v_min=-2.,v_max=2.,bc_lambda=1.,bc_margin=.01,critic_lambda=.1,critic_target_tau=.02,
  num_expl_steps=0,update_every_steps=1,stddev_schedule='.01',use_logger=True,
  action_mode='soft',softargmax_beta=a.beta,soft_endpoint_rescale=True,phase_heads=a.phase_heads)
 # Preserve learned visual features, extend state projections with zero-initialized clock columns.
 old=torch.load(a.resume_repair or ROOT/'outputs/i2o_panda/run_20261112_103403/checkpoint_5000.pt',map_location=agent.device,weights_only=False)
 start_update=int(old['updates']) if a.resume_repair else 0
 agent.encoder.load_state_dict(old['encoder'])
 for name in (() if (a.absolute_joints or a.cartesian_goals) and not a.resume_repair else ('critic','critic_target')):
  state=getattr(agent,name).state_dict();src=old[name]
  for k,v in state.items():
   if v.shape==src[k].shape:state[k]=src[k]
   else:
    assert v.ndim==2 and v.shape[1]==src[k].shape[1]+6 and 'low_dim_encoder' in k,(k,v.shape)
    state[k]=torch.zeros_like(v);state[k][:,:src[k].shape[1]]=src[k]
  getattr(agent,name).load_state_dict(state)
 if a.resume_repair:
  agent.encoder_opt.load_state_dict(old['encoder_opt']);agent.critic_opt.load_state_dict(old['critic_opt'])
 # Tests run against the actual CUDA/runtime implementation before spending on training.
 from vendor.cqn_utils import soft_zoom_in
 q=torch.zeros((1,8,5),dtype=torch.float64,device=agent.device,requires_grad=True)
 lo=-torch.ones((1,8),dtype=torch.float64,device=agent.device);hi=-lo
 assert torch.autograd.gradcheck(lambda x:soft_zoom_in(lo,hi,x,5,5.)[0],(q,),atol=1e-5)
 with torch.no_grad():
  test_q=torch.full((1,8,5),-2.,device=agent.device);test_q[:,:,4]=2.
  l=-torch.ones((1,8),device=agent.device);h=-l
  for _ in range(3):l,h=soft_zoom_in(l,h,test_q,5,a.beta)
  endpoint=(l+h)/2/(1-5**-3)
  assert (endpoint>.999).all() and (endpoint<=1.00001).all(),endpoint
 write('config.json',dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},environment=env.config,demo_source=str(dataset),
  dataset_manifest_sha256=file_hash(dataset/'manifest.json'),mode='MSE only, no RL updates',
  success_gate='20/20 unseen randomized seeds, real physical success hold 30 steps',
  warm_start=str(a.resume_repair or 'checkpoint_5000: encoder, critic; state columns extended, new optimizers'),
  gate_seeds=list(range(400000,400020)),dagger='expert labels on learner-visited states; learner actions executed'))
 # Phase-balanced sampling avoids underweighting transitions and the short hold phase.
 buckets=[[] for _ in range(5)]
 for item in replay.demos:buckets[int(np.argmax(item[0][1][-6:-1]))].append(item)
 rng=np.random.default_rng(17);log=open(OUT/'metrics.jsonl','w',buffering=1)
 def save_checkpoint(update,path):
  payload={name:getattr(agent,name).state_dict() for name in ('encoder','critic','critic_target')}
  payload.update(encoder_opt=agent.encoder_opt.state_dict(),critic_opt=agent.critic_opt.state_dict(),
   step=0,updates=update,rgb_shape=env.rgb_shape,state_shape=env.state_shape,task='take_lid_off_saucepan',
   config={'phase_clock':True,'soft_endpoint_rescale':True,'absolute_joints':a.absolute_joints,'phase_heads':a.phase_heads,'demo_phase_goals':a.demo_phase_goals,'cartesian_goals':a.cartesian_goals},action_mode='soft',softargmax_beta=a.beta,
   demo_source={'directory':str(dataset),'manifest_sha256':file_hash(dataset/'manifest.json')},
   note='clock-conditioned imitation repair; no online RL performed')
  temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)
 def frame(info,label):
  canvas=Image.new('RGB',(672,548));draw=ImageDraw.Draw(canvas)
  for i,name in enumerate(env.camera_names):
   x=i%2*336;y=i//2*252
   canvas.paste(Image.fromarray(env.images('preview')[name]).resize((336,252)),(x,y))
   draw.rectangle((x,y,x+335,y+18),fill='black');draw.text((x+3,y+3),name,fill='white')
  draw.text((4,508),label,fill='white');draw.text((4,528),f"step {info['step']} lift {info['lift_m']:.3f}m distance {info['knob_tcp_distance_m']:.3f}m",fill='white')
  return np.asarray(canvas)
 def evaluate(update,seeds,record=False):
  agent.critic_target.load_state_dict(agent.critic.state_dict());results=[]
  for index,seed in enumerate(seeds):
   env.rng=np.random.default_rng(seed);obs=env.reset();rows=[];writer=None
   if record:
    writer=subprocess.Popen(['/usr/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','672x548','-r','10','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/f'eval_{update}_{seed}.mp4')],stdin=subprocess.PIPE)
   for t in range(env.horizon):
    with torch.no_grad(),utils.eval_mode(agent):action=agent.act(*obs,step=0,eval_mode=True)
    assert np.isfinite(action).all() and np.max(np.abs(action))<=1.00001
    obs,r,term,trunc,info=env.step(action);rows.append(dict(action=action.tolist(),**info))
    if writer:writer.stdin.write(frame(info,f'learned imitation | seed {seed}').tobytes())
    if term or trunc:break
   if writer:writer.stdin.close();assert writer.wait()==0
   result=dict(seed=seed,final=info,max_lift_m=max(x['lift_m'] for x in rows),
               min_distance_m=min(x['knob_tcp_distance_m'] for x in rows))
   results.append(result);write(f'trace_{update}_{seed}.json',rows)
   progress('evaluating',update=update,completed=len(results),total=len(seeds),results=results)
   print('REPAIR_EVAL',update,json.dumps(result),flush=True)
   if len(seeds)==20 and not info['success']:break
  write(f'eval_{update}.json',results);return results
 dagger_buckets=[[] for _ in range(5)]
 def collect_dagger(update):
  agent.critic_target.load_state_dict(agent.critic.state_dict())
  records=[]
  for ep,seed in enumerate(([300006,300009]+list(range(210000,210000+max(0,a.dagger-2))))[:a.dagger]):
   env.rng=np.random.default_rng(seed);obs=env.reset();expert=env.expert_actions();count=0;raws=[];states=[];labels=[];executed=[]
   while True:
    try:label=next(expert)
    except StopIteration:break
    phase=int(np.argmax(obs[1][-6:-1]))
    # Teacher sees simulator geometry only to create labels. Rollout actions are network predictions.
    dagger_buckets[phase].append((obs,label.copy()))
    with torch.no_grad(),utils.eval_mode(agent):action=agent.act(*obs,step=0,eval_mode=True)
    raws.append(obs[0][:,-3:].copy());states.append(obs[1].copy());labels.append(label.copy());executed.append(action.copy())
    obs,r,term,trunc,info=env.step(action);count+=1
    if term or trunc:break
   np.savez_compressed(OUT/f'dagger_{update}_{seed}.npz',rgb=np.stack(raws),state=np.stack(states),
                       expert_labels=np.stack(labels),executed_policy_actions=np.stack(executed))
   records.append(dict(seed=seed,labels=count,final=info))
   progress('collecting_closed_loop_corrections',update=update,completed=ep+1,total=a.dagger)
  write(f'dagger_{update}.json',records)
 history=[];passed=False;update=start_update
 if a.validate_first:
  results=evaluate(update,[200000,200001,300006]);rate=sum(x['final']['success'] for x in results)/3
  history.append(dict(update=update,success_rate=rate));write('history.json',history)
  if rate==1:
   results=evaluate(update,list(range(400000,400020)),record=True)
   passed=len(results)==20 and all(x['final']['success'] for x in results)
   if passed:save_checkpoint(update,OUT/'validated_imitation.pt')
 if a.dagger:collect_dagger(start_update)
 for update in ([] if passed else range(start_update+1,a.updates+1)):
  rows=[buckets[k][rng.integers(len(buckets[k]))] for k in rng.integers(5,size=a.batch_size)]
  if a.dagger:
   for j in range(a.batch_size//2):
    phase=int(rng.integers(5));bucket=dagger_buckets[phase]
    if bucket:
     obs,label=bucket[int(rng.integers(len(bucket)))];rows[j]=(obs,label,0.,.99,obs,True)
  rgb=np.stack([x[0][0] for x in rows]);state=np.stack([x[0][1] for x in rows]);target=np.stack([x[1] for x in rows])
  metrics=agent.imitation_update(rgb,state,target);assert np.isfinite(metrics['action_mse'])
  log.write(json.dumps(dict(update=update,**metrics))+'\n')
  if update%32==0:progress('imitation_training',update=update,**metrics)
  if update%250==0:save_checkpoint(update,OUT/'checkpoint.pt')
  if update%500==0:
   results=evaluate(update,[200000,200001,300006]);rate=sum(x['final']['success'] for x in results)/3
   history.append(dict(update=update,success_rate=rate));write('history.json',history)
   if rate==1:
    results=evaluate(update,list(range(400000,400020)),record=True)
    passed=len(results)==20 and all(x['final']['success'] for x in results)
    if passed:
     save_checkpoint(update,OUT/'validated_imitation.pt');break
   if a.dagger:collect_dagger(update)
 save_checkpoint(update,OUT/'checkpoint.pt')
 write('result.json',dict(pass_gate=passed,updates=update,history=history,output=str(OUT)))
 progress('validated' if passed else 'gate_not_met',update=update,pass_gate=passed)
except Exception:
 import traceback
 error=traceback.format_exc();print(error,flush=True)
 (OUT/'error.txt').write_text(error)
 raise
finally:
 if env is not None:env.close()
 app.close()
