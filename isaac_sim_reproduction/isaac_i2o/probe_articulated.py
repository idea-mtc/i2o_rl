import sys,os,json,datetime,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from isaacsim import SimulationApp
GPU=int(os.environ['PINGANDOG_GPU']);task=sys.argv[1];OUT=Path(sys.argv[2]);OUT.mkdir(parents=True,exist_ok=True)
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','multi_gpu':False,'active_gpu':GPU,'physics_gpu':GPU,'extra_args':['--/physics/updateToUsd=true']})
import faulthandler
faulthandler.dump_traceback_later(90,repeat=True)
env=None
try:
 import numpy as np
 from aligned.articulated import ArticulatedEnv
 if len(sys.argv)>3 and sys.argv[3]=='native':
  from aligned.asset_tasks import AssetTaskEnv
  env=AssetTaskEnv(task)
 else:env=ArticulatedEnv(task)
 results=[]
 env.reset();initial=env.joint_positions().copy()
 for _ in range(20):
  _,_,term,trunc,negative=env.step_raw(np.r_[np.zeros(7),1.])
  assert not term,'uncommanded task success'
  assert not env.unlocked,'door unlocked without contact'
 assert abs(env.joint_positions()[0]-initial[0])<(.003 if task=='open_drawer' else np.deg2rad(1)), 'uncommanded joint drift'
 (OUT/'negative_control.json').write_text(json.dumps(negative,indent=2))
 for seed in [600000,600001,600002]:
  env.rng=np.random.default_rng(seed);obs=env.reset();rows=[]
  previous_phase=None
  for raw in env.expert_raw_actions():
   obs,r,term,trunc,info=env.step_raw(raw)
   if previous_phase!=env.expert_phase:env.save_image(OUT/f'{seed}_{env.expert_phase}.png');previous_phase=env.expert_phase
   rows.append(dict(phase=env.expert_phase,**info));print('PROBE_STEP',env.expert_phase,json.dumps(info),flush=True)
   if term or trunc:break
  results.append(info);(OUT/f'trace_{seed}.json').write_text(json.dumps(rows));env.save_image(OUT/f'final_{seed}.png')
  (OUT/'results.json').write_text(json.dumps(dict(task=task,config=env.config,results=results,successes=sum(x['success'] for x in results)),indent=2))
  if not info['success']:break
except Exception:
 (OUT/'error.txt').write_text(traceback.format_exc());raise
finally:
 if env is not None:env.close()
 app.close()
