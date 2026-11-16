"""Bounded paper-text I2O reconstruction with original action/replay contracts.
No task clock, phase head, Cartesian policy, or scripted learner actions.
"""
import argparse,datetime,hashlib,json,os,random,shutil,sys,time,traceback,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
p=argparse.ArgumentParser();p.add_argument('--job-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=30250)
p.add_argument('--scene',choices=['procedural','native_assets'],default='procedural');p.add_argument('--task',choices=['take_lid_off_saucepan','open_drawer','open_door','open_microwave','put_books_on_bookshelf'],default='take_lid_off_saucepan');p.add_argument('--gate-successes',type=int,default=2);p.add_argument('--demos',type=int,default=100);p.add_argument('--max-warmup',type=int,default=3000)
p.add_argument('--eval-every',type=int,default=1000);p.add_argument('--periodic-eval',type=int,default=5)
p.add_argument('--final-eval',type=int,default=100);p.add_argument('--deadline',default='2026-11-13T09:00:00+08:00')
a=p.parse_args();assert a.steps<=30250 and a.max_warmup<=5000
OUT=a.job_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
GPU=int(os.environ['PINGANDOG_GPU']);ROOT=Path('/home/ubuntu/pingandog')
def write(name,value):
 path=OUT/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def progress(stage,**values):
 write('progress.json',dict(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=GPU,**values))
progress('initializing',deadline=a.deadline)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','multi_gpu':False,'active_gpu':GPU,'physics_gpu':GPU,
 'extra_args':['--/physics/updateToUsd=true']})
env=None;agent=None;replay=None;online_step=0;warm_updates=0;episode=None
try:
 import numpy as np,torch
 from aligned.environment import AlignedLidEnv
 from aligned.contract import ActionCodec
 from aligned.gpu_replay import GPUReplay
 from aligned.source_loader import load_original
 from aligned.paper_update import update as paper_update
 from vendor.i2o_rl import I2OAgent
 from vendor import utils
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high');utils.set_seed_everywhere(1)
 device=f'cuda:{GPU}';started=time.monotonic()
 if a.task=='take_lid_off_saucepan':env=AlignedLidEnv()
 else:
  if a.scene=='native_assets':
   if a.task in ['open_microwave','put_books_on_bookshelf']:
    from aligned.extended_tasks import ExtendedTaskEnv
    env=ExtendedTaskEnv(a.task)
   else:
    from aligned.asset_tasks import AssetTaskEnv
    env=AssetTaskEnv(a.task)
  else:
   from aligned.articulated import ArticulatedEnv
   env=ArticulatedEnv(a.task)
 env.diagnostics_dir=OUT/'camera_diagnostics'
 write('config.json',dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
  environment=env.config,algorithm='paper-text losses on shared original CQN body',
  observation_shape=[4,24,84,84],state_shape=[120],action_shape=[8],phase_clock=False,phase_heads=1,
  softargmax_beta=1,endpoint_rescale=False,lr=5e-5,weight_decay=.1,batch_size=256,
  levels=3,bins=5,atoms=51,support=[-2,2],discount=.99,target_tau=.02,
  losses='action MSE + factorized expert Q margin .01 + .1 scalar final-level Bellman square',
  sampling='episode uniform, ordinary+successful pools, both prefilled, successful online episodes relabeled',
  precision='FP32 storage, high float32 matmul precision (TF32 allowed)',
  remaining_differences=env.config['remaining_differences'],
  caveat='paper decoder/phase update frequency not fully specified; this is an explicit reconstruction, not verified author code'))
 source=OUT/'source'
 if not source.exists():
  source.mkdir()
  for f in list(HERE.glob('*.py'))+list((HERE/'aligned').glob('*.py'))+list((HERE/'vendor').glob('*.py')):
   shutil.copy2(f,source/(f.parent.name+'_'+f.name))
 dataset=OUT/'dataset';dataset.mkdir(exist_ok=True);manifest_path=dataset/'manifest.json'
 manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'format':'rlbench_aligned_isaac_raw_v1','episodes':[],'attempts':0,'status':'collecting'}
 def data_start(obs):
  return dict(rgb=[obs[0][:,-3:].copy()],state=[obs[1][-15:].copy()],raw_actions=[],rewards=[],terminated=[],truncated=[],
              sim_time=[float(env.world.current_time)])
 def data_add(d,raw,obs,r,term,trunc):
  d['rgb'].append(obs[0][:,-3:].copy());d['state'].append(obs[1][-15:].copy());d['raw_actions'].append(np.asarray(raw,np.float32).copy())
  d['rewards'].append(r);d['terminated'].append(term);d['truncated'].append(trunc);d['sim_time'].append(float(env.world.current_time))
 def arrays(d):
  return {k:np.asarray(v,dtype=np.uint8 if k=='rgb' else np.bool_ if k in ['terminated','truncated'] else np.float64 if k=='sim_time' else np.float32) for k,v in d.items()}
 def save_episode(d,directory,label):
  directory.mkdir(exist_ok=True);path=directory/(label+'_'+uuid.uuid4().hex[:10]+'.npz');tmp=path.with_suffix('.tmp')
  data=arrays(d)
  with open(tmp,'wb') as f:np.savez_compressed(f,**data)
  tmp.replace(path);return path,data
 def load_data(path):
  with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
 # Keep original logs, append new attempts. Every stored action is factual, not a phase-end label.
 while len(manifest['episodes'])<a.demos:
  assert manifest['attempts']<a.demos*2,'expert acceptance rate too low'
  seed=700000+manifest['attempts'];env.rng=np.random.default_rng(seed);obs=env.reset();d=data_start(obs)
  expert_failure=None
  try:
   for raw in env.expert_raw_actions():
    obs,r,term,trunc,info=env.step_raw(raw)
    np.testing.assert_allclose(info['applied_delta'],raw[:7],atol=1e-6)
    assert np.max(np.abs(raw[:7]))<=.200001
    assert obs[1].shape==(120,) and obs[0].shape==(4,24,84,84)
    data_add(d,raw,obs,r,term,trunc)
    if term or trunc:break
  except AssertionError as exc:
   if a.scene!='native_assets' or 'native expert IK unreachable' not in str(exc):raise
   expert_failure=str(exc)
  manifest['attempts']+=1
  if expert_failure:
   manifest.setdefault('rejections',[]).append(dict(seed=seed,reason=expert_failure));write('dataset/manifest.json',manifest);progress('collecting_demonstrations',accepted=len(manifest['episodes']),target=a.demos,attempts=manifest['attempts'],rejected=len(manifest['rejections']));continue
  if info['success']:
   data=arrays(d)
   assert np.all(np.diff(data['sim_time'])>0) and set(np.unique(data['state'][:,7]))<=set([0.,1.])
   if not np.allclose(data['raw_actions'][:,7],data['state'][1:,7]):
    manifest.setdefault('rejections',[]).append(dict(seed=seed,reason='gripper command/observed state mismatch'))
    write('dataset/manifest.json',manifest);progress('collecting_demonstrations',accepted=len(manifest['episodes']),target=a.demos,attempts=manifest['attempts'],rejected=len(manifest['rejections']))
    continue
   path,data=save_episode(d,dataset,f'demo_{len(manifest["episodes"]):03d}')
   manifest['episodes'].append(dict(file=path.name,seed=seed,steps=len(data['raw_actions']),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  write('dataset/manifest.json',manifest)
  progress('collecting_demonstrations',accepted=len(manifest['episodes']),target=a.demos,attempts=manifest['attempts'])
  print('ALIGNED_DEMO',len(manifest['episodes']),manifest['attempts'],flush=True)
 demos=[load_data(dataset/x['file']) for x in manifest['episodes']]
 codec=ActionCodec.fit(demos);env.codec=codec
 manifest.update(status='complete',action_statistics=codec.json(),environment=env.config);write('dataset/manifest.json',manifest)
 write('action_statistics.json',codec.json())
 # Raw frames cached once on GPU; stacking and sampling match the CPU/source contract.
 replay=GPUReplay(codec,device,capacity=72000,seed=1)
 def load_replay(entries):
  for entry in entries:replay.add(load_data(Path(entry['file'])),entry['successful'],entry['file'])
 offline=[{'file':str(dataset/e['file']),'successful':True} for e in manifest['episodes']]
 load_replay(offline)
 kwargs=dict(rgb_obs_shape=(4,24,84,84),low_dim_obs_shape=(120,),action_shape=(8,),device=device,
  lr=5e-5,weight_decay=.1,feature_dim=128,hidden_dim=512,levels=3,bins=5,atoms=51,v_min=-2.,v_max=2.,
  bc_lambda=1.,bc_margin=.01,critic_lambda=.1,critic_target_tau=.02,num_expl_steps=0,update_every_steps=1,stddev_schedule='.01',use_logger=True)
 agent=I2OAgent(**kwargs,action_mode='soft',softargmax_beta=1.,soft_endpoint_rescale=False,phase_heads=1)
 assert agent.encoder.num_views==4 and agent.critic.network.phase_heads==1
 metrics_log=open(OUT/'metrics.jsonl','a',buffering=1)
 evaluations=json.loads((OUT/'evaluations.json').read_text()) if (OUT/'evaluations.json').exists() else []
 def checkpoint():
  payload={k:getattr(agent,k).state_dict() for k in ['encoder','critic','critic_target','encoder_opt','critic_opt']}
  payload.update(online_step=online_step,warm_updates=warm_updates,replay_files=replay.files(),partial=episode,
    python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(GPU),
    replay_rng=replay.rng.bit_generator.state,codec=codec.json(),ready=(OUT/'startup_gate.json').exists(),
    algorithm='paper_text_aligned',note='physical episode resets on resume; saved partial transitions are truncated, never linked across reset')
  temp=OUT/'latest.pt.tmp';torch.save(payload,temp)
  if (OUT/'latest.pt').exists():shutil.copy2(OUT/'latest.pt',OUT/'previous.pt')
  temp.replace(OUT/'latest.pt')
 if (OUT/'latest.pt').exists():
  saved=torch.load(OUT/'latest.pt',map_location=device,weights_only=False)
  for k in ['encoder','critic','critic_target','encoder_opt','critic_opt']:getattr(agent,k).load_state_dict(saved[k])
  online_step=saved['online_step'];warm_updates=saved['warm_updates']
  # Offline entries are already loaded; replay order and flags restored exactly.
  for e in saved['replay_files'][len(offline):]:replay.add(load_data(Path(e['file'])),e['successful'],e['file'])
  partial=saved.get('partial')
  if partial and len(partial['raw_actions']):
   partial['truncated'][-1]=True;path,data=save_episode(partial,OUT/'online_episodes',f'interrupted_{online_step}')
   replay.add(data,False,path)
  random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
  torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),GPU);replay.rng.bit_generator.state=saved['replay_rng']
  write('resume.json',dict(online_step=online_step,warm_updates=warm_updates,experience_restored=True,episode_reset=True))
 def evaluate(count,label,seed=800000):
  progress('evaluating',online_step=online_step,warm_updates=warm_updates,label=label,completed=0,total=count)
  results=[];directory=OUT/'evaluation';directory.mkdir(exist_ok=True)
  for ep in range(count):
   output=directory/f'{label}_{ep:03d}.json'
   if output.exists():
    results.append(json.loads(output.read_text()));continue
   env.rng=np.random.default_rng(seed+ep);obs=env.reset();rows=[];initial=np.array(env.status()['tcp'])
   for _ in range(env.horizon):
    with torch.no_grad(),utils.eval_mode(agent):action=agent.act(*obs,step=online_step,eval_mode=True)
    obs,r,term,trunc,info=env.step(action);rows.append(dict(action=action.tolist(),**info))
    if term or trunc:break
   xyz=np.array([x['tcp'] for x in rows]);motion=float(np.linalg.norm(np.ptp(np.vstack([initial,xyz]),axis=0)))
   result=dict(seed=seed+ep,success=info['success'],steps=len(rows),tcp_motion_m=motion,final=info,
               max_lift_m=max(x['lift_m'] for x in rows))
   output.write_text(json.dumps(result,indent=2));(directory/f'{label}_{ep:03d}_trace.json').write_text(json.dumps(rows))
   if ep==0:env.save_image(directory/f'{label}.png')
   results.append(result);progress('evaluating',label=label,online_step=online_step,warm_updates=warm_updates,
       completed=len(results),total=count,successes=sum(x['success'] for x in results))
  entry=dict(label=label,online_step=online_step,warm_updates=warm_updates,episodes=count,
             successes=sum(x['success'] for x in results),success_rate=sum(x['success'] for x in results)/count,
             minimum_motion_m=min(x['tcp_motion_m'] for x in results),results=results)
  if not any(x['label']==label for x in evaluations):evaluations.append(entry);write('evaluations.json',evaluations)
  print('ALIGNED_EVAL',json.dumps({k:v for k,v in entry.items() if k!='results'}),flush=True)
  return entry
 # A bounded original-source gradient/training check is kept distinct from paper reconstruction.
 if not (OUT/'original_source_check.json').exists():
  source_dir=ROOT/'outputs/repro_audit/original';original=load_original(source_dir)
  baseline=original.I2OAgent(**kwargs)
  for k in ['encoder','critic','critic_target']:getattr(baseline,k).load_state_dict(getattr(agent,k).state_dict())
  # Contiguous-output hook is a PyTorch/Isaac layout compatibility fix, not a policy change.
  for net in baseline.encoder.conv_nets:net.register_forward_hook(lambda module,inputs,output:output.contiguous())
  iterator=replay.batches(256);last=None
  for step in range(100):last=baseline.update(iterator,step)
  write('original_source_check.json',dict(updates=100,last_metrics=last,
       source_sha256=hashlib.sha256((source_dir/'i2o_rl.py').read_bytes()).hexdigest(),
       note='original hard decoder/L1/FOSD/C51 objective; separate from paper-text training'))
  del baseline;torch.cuda.empty_cache()
 # Warmup is a startup diagnostic/initialization choice; paper does not give its exact duration.
 while not (OUT/'startup_gate.json').exists() and warm_updates<a.max_warmup:
  metrics=paper_update(agent,replay.sample(256,demonstrations_only=True),rl=False);warm_updates+=1
  metrics_log.write(json.dumps(dict(stage='imitation',update=warm_updates,**metrics))+'\n')
  if warm_updates%32==0:progress('imitation',warm_updates=warm_updates,online_step=online_step,metrics=metrics)
  if warm_updates%100==0:checkpoint()
  if warm_updates%500==0:
   agent.critic_target.load_state_dict(agent.critic.state_dict())
   result=evaluate(3,f'warm_{warm_updates}')
   if result['successes']:
    gate=evaluate(5,f'gate_{warm_updates}',seed=810000)
    if gate['successes']>=a.gate_successes and gate['minimum_motion_m']>.03:
     write('startup_gate.json',dict(pass_gate=True,**gate));checkpoint();shutil.copy2(OUT/'latest.pt',OUT/'validated_start.pt')
 assert (OUT/'startup_gate.json').exists(),'startup gate not met: no unverified long run will be launched'
 episode=None;obs=env.reset();episode=data_start(obs);step_times=[]
 while online_step<a.steps:
  t=time.monotonic()
  with torch.no_grad(),utils.eval_mode(agent):action=agent.act(*obs,step=online_step,eval_mode=False)
  raw=codec.decode(action);obs,r,term,trunc,info=env.step_raw(raw);data_add(episode,raw,obs,r,term,trunc)
  metrics=paper_update(agent,replay.sample(256),rl=True);online_step+=1;step_times.append(time.monotonic()-t)
  metrics_log.write(json.dumps(dict(stage='online',online_step=online_step,**metrics))+'\n')
  if term or trunc:
   path,data=save_episode(episode,OUT/'online_episodes',f'episode_{online_step}')
   replay.add(data,info['success'],path);obs=env.reset();episode=data_start(obs)
  if online_step%32==0:
   speed=float(np.mean(step_times[-32:]));remaining=(a.steps-online_step)*speed
   progress('online_training',online_step=online_step,total=a.steps,warm_updates=warm_updates,metrics=metrics,
     seconds_per_step=speed,estimated_online_seconds_remaining=remaining,
     cuda_peak_reserved_gib=torch.cuda.max_memory_reserved(GPU)/2**30)
  if online_step%250==0:checkpoint()
  if online_step%5000==0:shutil.copy2(OUT/'latest.pt',OUT/f'checkpoint_{online_step}.pt')
  if online_step%a.eval_every==0:
   if episode['raw_actions']:
    episode['truncated'][-1]=True;path,data=save_episode(episode,OUT/'online_episodes',f'eval_cut_{online_step}');replay.add(data,False,path)
   episode=None;checkpoint();result=evaluate(a.periodic_eval,f'step_{online_step}',seed=820000)
   if result['minimum_motion_m']<.005 and result['successes']==0:
    write('motion_alert.json',result)
   obs=env.reset();episode=data_start(obs)
 checkpoint();final=evaluate(a.final_eval,'final',seed=900000)
 write('result.json',dict(complete=True,online_steps=online_step,warm_updates=warm_updates,final=final,
        seconds_this_process=time.monotonic()-started,deadline=a.deadline))
 progress('complete',online_step=online_step,final_success_rate=final['success_rate'],evaluated=a.final_eval)
except Exception:
 error=traceback.format_exc();print(error,flush=True);write('error.json',dict(error=error,online_step=online_step,warm_updates=warm_updates))
 progress('failed',online_step=online_step,warm_updates=warm_updates,error=error)
finally:
 if env is not None:env.close()
 app.close()
