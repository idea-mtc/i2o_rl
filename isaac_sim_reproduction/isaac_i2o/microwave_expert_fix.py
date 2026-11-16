"""Bounded expert-mask validation with conditional full continuation.

Only explicit original expert samples receive IL/Margin; all replay samples
receive unchanged TD. This run starts from an immutable IL checkpoint.
"""
import argparse,copy,datetime,hashlib,json,os,random,shutil,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
p=argparse.ArgumentParser();p.add_argument('--job-dir',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--probe',action='store_true');args=p.parse_args()
assert args.gpu in (2,3,4)
assert os.environ.get('CUDA_VISIBLE_DEVICES')==os.environ.get('I2O_ALLOWED_UUIDS') and os.environ.get('I2O_ALLOWED_UUIDS')
GPU=args.gpu;CUDA=0;OUT=args.job_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
ROOT=Path('/home/ubuntu/pingandog');SOURCE=ROOT/'outputs/extended_asset_training/run_20261114/open_microwave'
KEYS=['encoder','critic','critic_target','encoder_opt','critic_opt'];SEEDS=[900000+5*i for i in range(20)];FINAL_SEEDS=list(range(910000,910100))
DEADLINE=datetime.datetime.fromisoformat('2026-11-16T09:00:00+08:00')
def write(name,data):
 f=OUT/name;f.parent.mkdir(parents=True,exist_ok=True);t=f.with_suffix(f.suffix+'.tmp');t.write_text(json.dumps(data,indent=2));t.replace(f)
def progress(stage,**kw):write('progress.json',dict(stage=stage,gpu=GPU,cuda_index=CUDA,seed=args.seed,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
progress('initializing')
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','multi_gpu':False,'active_gpu':0,'physics_gpu':CUDA,'extra_args':['--/physics/updateToUsd=true','--/renderer/multiGpu/enabled=false','--/renderer/multiGpu/autoEnable=false']})
env=None;step=0
try:
 import numpy as np,torch
 from vendor.i2o_rl import I2OAgent
 from vendor import utils
 from aligned.extended_tasks import ExtendedTaskEnv
 from aligned.contract import ActionCodec
 from aligned.expert_replay import ExpertReplay,migrate_replay_entries
 from aligned.paper_update import update
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.cuda.set_device(CUDA)
 assert torch.cuda.device_count()==1,'CUDA device isolation failed'
 device=f'cuda:{CUDA}';utils.set_seed_everywhere(1)
 import unittest,test_expert_replay
 test_result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_expert_replay))
 assert test_result.wasSuccessful(),'expert provenance tests failed'
 write('provenance_tests.json',dict(passed=True,tests=test_result.testsRun))
 env=ExtendedTaskEnv('open_microwave');env.diagnostics_dir=OUT/'camera_diagnostics'
 if args.probe:
  write('probe.json',dict(pass_probe=True,physical_gpu=GPU,cuda_index=CUDA,cuda_device_count=torch.cuda.device_count(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],pid=os.getpid(),environment=env.config))
  progress('probe_complete');sys.exit(0)
 source_digest=hashlib.sha256((SOURCE/'validated_start.pt').read_bytes()).hexdigest()
 original=torch.load(SOURCE/'validated_start.pt',map_location='cpu',weights_only=False);assert original['online_step']==0
 manifest=json.loads((SOURCE/'dataset/manifest.json').read_text());assert len(manifest['episodes'])==100
 expert_files=[str(SOURCE/'dataset'/e['file']) for e in manifest['episodes']]
 codec=ActionCodec(original['codec']['min'],original['codec']['max']);env.codec=codec
 replay=ExpertReplay(codec,device,capacity=72000,seed=args.seed)
 def load_data(path):
  with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}
 for e,path in zip(manifest['episodes'],expert_files):
  assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==e['sha256']
  replay.add(load_data(path),True,path,is_expert=True)
 agent=I2OAgent(rgb_obs_shape=(4,24,84,84),low_dim_obs_shape=(120,),action_shape=(8,),device=device,lr=5e-5,weight_decay=.1,feature_dim=128,hidden_dim=512,levels=3,bins=5,atoms=51,v_min=-2.,v_max=2.,bc_lambda=1.,bc_margin=.01,critic_lambda=.1,critic_target_tau=.02,num_expl_steps=0,update_every_steps=1,stddev_schedule='.01',use_logger=True,action_mode='soft',softargmax_beta=1.,soft_endpoint_rescale=False,phase_heads=1)
 def load_model(saved):
  for k in KEYS:getattr(agent,k).load_state_dict(saved[k])
 def restore_rng(saved):
  random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),CUDA);replay.rng.bit_generator.state=copy.deepcopy(saved['replay_rng'])
  if 'env_rng' in saved:env.rng.bit_generator.state=copy.deepcopy(saved['env_rng'])
 def save_model(name):
  payload={k:getattr(agent,k).state_dict() for k in KEYS}
  payload.update(schema=2,online_step=step,warm_updates=original['warm_updates'],codec=codec.json(),replay_files=replay.files(),partial=episode,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(CUDA),replay_rng=copy.deepcopy(replay.rng.bit_generator.state),env_rng=copy.deepcopy(env.rng.bit_generator.state),source_checkpoint_sha256=source_digest,seed=args.seed,mask='original_expert_only')
  t=OUT/(name+'.tmp');torch.save(payload,t);t.replace(OUT/name)
 def start(obs):return dict(rgb=[obs[0][:,-3:].copy()],state=[obs[1][-15:].copy()],raw_actions=[],rewards=[],terminated=[],truncated=[],sim_time=[float(env.world.current_time)])
 def append(ep,raw,obs,r,term,trunc):
  ep['rgb'].append(obs[0][:,-3:].copy());ep['state'].append(obs[1][-15:].copy());ep['raw_actions'].append(raw.copy());ep['rewards'].append(r);ep['terminated'].append(term);ep['truncated'].append(trunc);ep['sim_time'].append(float(env.world.current_time))
 def store_episode(ep,success,label):
  data={k:np.asarray(v,dtype=np.uint8 if k=='rgb' else np.bool_ if k in ['terminated','truncated'] else np.float64 if k=='sim_time' else np.float32) for k,v in ep.items()}
  path=OUT/'online_episodes'/f'{label}_{step}_{time.time_ns()}.npz';path.parent.mkdir(exist_ok=True)
  tmp=path.with_suffix('.tmp')
  with tmp.open('wb') as f:np.savez_compressed(f,**data)
  tmp.replace(path);replay.add(data,success,path,is_expert=False)
 def evaluate(checkpoint,label,seeds):
  digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();saved=torch.load(checkpoint,map_location='cpu',weights_only=False);load_model(saved);eval_step=saved['online_step'];del saved
  results=[];began=time.monotonic()
  for i,seed in enumerate(seeds):
   path=OUT/'evaluation'/f'{label}_{seed}.json'
   if path.exists():
    x=json.loads(path.read_text());assert x['checkpoint_sha256']==digest;results.append(x);continue
   env.rng=np.random.default_rng(seed);obs=env.reset();rows=[];initial=np.asarray(env.status()['tcp'])
   for _ in range(env.horizon):
    with torch.no_grad(),utils.eval_mode(agent):action=agent.act(*obs,step=eval_step,eval_mode=True)
    obs,r,term,trunc,info=env.step(action)
    rows.append(dict(step=info['step'],tcp=info['tcp'],opening_deg=info['opening_deg'],handle_tcp_distance_m=info['handle_tcp_distance_m'],gripper=info['requested_raw_action'][-1]))
    if term or trunc:break
   closes=[x for x in rows if x['gripper']<=.5]
   x=dict(seed=seed,success=bool(info['success']),steps=len(rows),max_angle=max(x['opening_deg'] for x in rows),first_close_step=closes[0]['step'] if closes else None,first_close_distance=closes[0]['handle_tcp_distance_m'] if closes else None,tcp_motion_m=float(np.linalg.norm(np.ptp(np.vstack([initial,[x['tcp'] for x in rows]]),axis=0))),checkpoint_sha256=digest,online_step=eval_step)
   write(str(path.relative_to(OUT)),x);write(f'evaluation/{label}_{seed}_trace.json',rows);results.append(x)
   if i==0:env.save_image(OUT/f'{label}.png')
   progress('evaluating',label=label,online_step=step,completed=len(results),total=len(seeds),successes=sum(x['success'] for x in results))
  summary=dict(label=label,online_step=eval_step,episodes=len(results),successes=sum(x['success'] for x in results),success_rate=sum(x['success'] for x in results)/len(results),checkpoint_sha256=digest,seconds_this_evaluation_process=time.monotonic()-began,results=results)
  write(f'{label}.json',summary);return summary
 write('contract.json',dict(source=str(SOURCE/'validated_start.pt'),source_sha256=source_digest,task='open_microwave',physical_gpu=GPU,cuda_index=CUDA,allowed_physical_gpus=[2,3,4],seed=args.seed,mask='original_expert_only',td_scale=.1,batch_size=256,lr=5e-5,short_steps=1000,full_steps=30250,short_seeds=SEEDS,final_seeds=FINAL_SEEDS,gate=dict(baseline_min=14,step500_min=12,step1000_min=14,max_drop_from_baseline=2),deadline=DEADLINE.isoformat(),limitations=['Shared pretrained IL initialization, not three independent from-scratch runs.','Short checkpoint500 evaluated after collection reaches1000 to preserve uninterrupted short collection.','One environment reset/truncated partial at short/full boundary and periodic evaluations; RNG streams restored.','Final tests use held-out910000..910099, distinct from short gate seeds.','Historical uncorrected B1000=3/20 is reference only; not a same-stream intervention.']))
 episode=None
 if (OUT/'latest.pt').exists():
  saved=torch.load(OUT/'latest.pt',map_location='cpu',weights_only=False);assert saved['schema']==2 and saved['source_checkpoint_sha256']==source_digest and saved['seed']==args.seed
  load_model(saved);step=saved['online_step']
  entries=migrate_replay_entries(saved['replay_files'],expert_files)
  assert [e['file'] for e in entries[:100]]==expert_files
  for e in entries[100:]:replay.add(load_data(e['file']),e['successful'],e['file'],is_expert=e['is_expert'])
  restore_rng(saved)
  if saved.get('partial') and len(saved['partial']['raw_actions']):
   ep=saved['partial'];ep['truncated'][-1]=True;store_episode(ep,False,'resume_cut')
  write(f'resume_{time.time_ns()}.json',dict(step=step,expert_entries=sum(e['is_expert'] for e in entries),entries=len(entries),physical_reset=True))
  del saved
 else:
  load_model(original)
  evaluate(SOURCE/'validated_start.pt','baseline20',SEEDS)
  load_model(original)
  if args.seed==1:
   restore_rng(original);env.rng=np.random.default_rng(810004);env.rng.uniform(-.02,.02,2)
  else:utils.set_seed_everywhere(args.seed);replay.rng=np.random.default_rng(args.seed);env.rng=np.random.default_rng(810004+args.seed)
 del original
 # Keep only scalar warm update metadata after releasing original checkpoint.
 original={'warm_updates':3500}
 obs=env.reset();episode=start(obs);save_model('latest.pt')
 metrics=(OUT/'metrics.jsonl').open('a',buffering=1)
 def train_until(target):
  global step,obs,episode
  began=time.monotonic();base=step
  while step<target:
   if shutil.disk_usage(OUT).free<8*2**30:raise RuntimeError('Less than8GiB disk free; stop safely')
   if datetime.datetime.now(datetime.timezone.utc)>DEADLINE:raise TimeoutError('Morning deadline reached; preserved checkpoint')
   with torch.no_grad(),utils.eval_mode(agent):action=agent.act(*obs,step=step,eval_mode=False)
   raw=codec.decode(action);obs,r,term,trunc,info=env.step_raw(raw);append(episode,raw,obs,r,term,trunc)
   batch=replay.sample(256);m=update(agent,batch,rl=True)
   assert all(np.isfinite(v) for v in m.values()),'nonfinite loss/gradient'
   step+=1;m.update(expert_sample_count=int(batch[11].sum().item()),online_step=step)
   metrics.write(json.dumps(m)+'\n')
   if term or trunc:store_episode(episode,bool(info['success']),'episode');obs=env.reset();episode=start(obs)
   if step%25==0:progress('online_training',online_step=step,target=target,seconds_per_step=(time.monotonic()-began)/(step-base),expert_episodes=len(replay.expert_ids),successful_episodes=len(replay.success_ids),metrics=m)
   if step%250==0:save_model('latest.pt')
   if step in [500,1000]:shutil.copy2(OUT/'latest.pt',OUT/f'checkpoint_{step}.pt')
  save_model('latest.pt')
 def return_to_training(saved):
  global obs,episode
  load_model(saved);restore_rng(saved)
  if episode and episode['raw_actions']:
   episode['truncated'][-1]=True;store_episode(episode,False,'eval_cut')
  obs=env.reset();episode=start(obs);save_model('latest.pt')
 if step<1000:train_until(1000)
 if not (OUT/'gate.json').exists():
  saved=torch.load(OUT/'latest.pt',map_location='cpu',weights_only=False)
  e500=evaluate(OUT/'checkpoint_500.pt','short500',SEEDS);e1000=evaluate(OUT/'checkpoint_1000.pt','short1000',SEEDS);baseline=json.loads((OUT/'baseline20.json').read_text())
  passed=baseline['successes']>=14 and e500['successes']>=12 and e1000['successes']>=max(14,baseline['successes']-2)
  write('gate.json',dict(passed=passed,baseline=baseline['successes'],step500=e500['successes'],step1000=e1000['successes'],episodes=20,rule='baseline>=14,500>=12,1000>=max(14,baseline-2)',legacy_reference_successes=3))
  return_to_training(saved);del saved
 if not json.loads((OUT/'gate.json').read_text())['passed']:
  write('result.json',dict(complete=True,stage='short_failed_gate',full_training_started=False,online_steps=step,gate=json.loads((OUT/'gate.json').read_text())))
  progress('short_failed_gate',online_step=step);sys.exit(0)
 if not (OUT/'best.pt').exists():shutil.copy2(OUT/'checkpoint_1000.pt',OUT/'best.pt');write('best.json',dict(step=1000,successes=json.loads((OUT/'short1000.json').read_text())['successes'],episodes=20))
 if not (OUT/'baseline100.json').exists():
  saved=torch.load(OUT/'latest.pt',map_location='cpu',weights_only=False);evaluate(SOURCE/'validated_start.pt','baseline100',FINAL_SEEDS);return_to_training(saved);del saved
 # Best model selection uses development seeds only, never final seeds.
 for target in [5000,10000,15000,20000,25000,30000,30250]:
  if step<target:train_until(target)
  if step==target and target%5000==0 and not (OUT/f'periodic{target}.json').exists():
   frozen=OUT/f'checkpoint_{target}.pt'
   if not frozen.exists():shutil.copy2(OUT/'latest.pt',frozen)
   saved=torch.load(OUT/'latest.pt',map_location='cpu',weights_only=False);entry=evaluate(frozen,f'periodic{target}',SEEDS)
   best=json.loads((OUT/'best.json').read_text())
   if entry['successes']>best['successes']:shutil.copy2(frozen,OUT/'best.pt');write('best.json',dict(step=step,successes=entry['successes'],episodes=20))
   return_to_training(saved);del saved
 # Preserve actual30250-step model separately from best-development model.
 if not (OUT/'final.pt').exists():shutil.copy2(OUT/'latest.pt',OUT/'final.pt')
 final=evaluate(OUT/'final.pt','final100',FINAL_SEEDS);best=evaluate(OUT/'best.pt','best100',FINAL_SEEDS)
 write('result.json',dict(complete=True,stage='full_complete',full_training_started=True,online_steps=step,source_checkpoint_sha256=source_digest,gate=json.loads((OUT/'gate.json').read_text()),baseline=json.loads((OUT/'baseline100.json').read_text()),final=final,best=best,seed=args.seed,physical_gpu=GPU))
 progress('complete',online_step=step,final_success_rate=final['success_rate'],best_success_rate=best['success_rate'])
except Exception:
 err=traceback.format_exc();write(f'error_{time.time_ns()}.json',dict(error=err,step=step));progress('failed',online_step=step,error=err);raise
finally:
 if env is not None:env.close()
 app.close()
