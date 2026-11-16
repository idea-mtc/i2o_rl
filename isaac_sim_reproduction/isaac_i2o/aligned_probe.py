"""Bounded control-contract/teacher checks, no policy training."""
import os,sys,json,datetime,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
ROOT=Path('/home/ubuntu/pingandog');OUT=ROOT/'outputs/aligned_probe'/datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S');OUT.mkdir(parents=True)
GPU=int(os.environ['PINGANDOG_GPU'])
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','multi_gpu':False,'active_gpu':GPU,'physics_gpu':GPU,
                  'extra_args':['--/physics/updateToUsd=true']})
env=None;writer=None
try:
 import numpy as np
 from PIL import Image,ImageDraw
 from aligned.environment import AlignedLidEnv
 env=AlignedLidEnv();report={'output':str(OUT),'config':env.config,'tests':[]}
 obs=env.reset();assert obs[0].shape==(4,24,84,84) and obs[1].shape==(120,)
 before=env.status();_,_,_,_,air=env.step_raw(np.r_[np.zeros(7),0.])
 assert not air['attached'] and abs(air['lift_m'])<.01
 report['air_close']=air
 for seed in [600000,600001,600002]:
  env.rng=np.random.default_rng(seed);obs=env.reset();rows=[]
  if seed==600000:
   writer=subprocess.Popen(['/usr/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','672x548','-r','10','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(OUT/'expert.mp4')],stdin=subprocess.PIPE)
  for raw in env.expert_raw_actions():
   obs,r,term,trunc,info=env.step_raw(raw);rows.append(dict(phase=env.expert_phase,raw_action=raw.tolist(),**info))
   np.testing.assert_allclose(np.array(info['applied_delta']),np.array(info['requested_raw_action'])[:7],atol=1e-6)
   if writer:
    canvas=Image.new('RGB',(672,548));draw=ImageDraw.Draw(canvas)
    images=env.images('preview')
    for i,name in enumerate(env.camera_names):
     x=i%2*336;y=i//2*252;canvas.paste(Image.fromarray(images[name]).resize((336,252)),(x,y));draw.text((x+3,y+3),name,fill='red')
    draw.text((4,508),'teacher only | measured-joint delta | attachment proxy',fill='white')
    draw.text((4,528),f"step {env.t} phase {env.expert_phase} lift {info['lift_m']:.3f}m",fill='white');writer.stdin.write(np.asarray(canvas).tobytes())
   if term or trunc:break
  if writer:writer.stdin.close();assert writer.wait()==0;writer=None
  (OUT/f'trace_{seed}.json').write_text(json.dumps(rows))
  report['tests'].append(dict(seed=seed,final=info));print('ALIGNED_EXPERT',seed,json.dumps(info),flush=True)
  if not info['success']:raise AssertionError('state-feedback teacher did not finish within 100 steps')
  env.done=False;_,_,_,_,released=env.step_raw(np.r_[np.zeros(7),1.])
  assert not env.attached
  report['tests'][-1]['released_lift_m']=released['lift_m']
 report['pass']=True
 (OUT/'result.json').write_text(json.dumps(report,indent=2))
 print('ALIGNED_PROBE_PASS',str(OUT),flush=True)
except Exception:
 import traceback
 (OUT/'error.txt').write_text(traceback.format_exc());raise
finally:
 if writer:writer.stdin.close();writer.wait()
 if env is not None:env.close()
 app.close()
