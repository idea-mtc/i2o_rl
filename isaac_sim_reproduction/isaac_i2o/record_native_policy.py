"""Record frozen RL policy rollouts without changing the training run."""
import os,sys,json,subprocess,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
ROOT=Path('/home/ubuntu/pingandog');OUT=Path(sys.argv[1]).resolve();TASK=sys.argv[2];OUT.mkdir(parents=True,exist_ok=True);JOB=ROOT/'outputs/native_asset_training/run_20261113'/TASK;GPU=int(os.environ['PINGANDOG_GPU'])
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','multi_gpu':False,'active_gpu':GPU,'physics_gpu':GPU,'extra_args':['--/physics/updateToUsd=true']})
env=None;writer=None
try:
 import numpy as np,torch
 from PIL import Image,ImageDraw
 from aligned.asset_tasks import AssetTaskEnv
 from aligned.contract import ActionCodec
 from vendor.i2o_rl import I2OAgent
 from vendor import utils
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high');utils.set_seed_everywhere(1)
 saved=torch.load(JOB/'latest.pt',map_location=f'cuda:{GPU}',weights_only=False)
 step=saved['online_step'];c=saved['codec'];env=AssetTaskEnv(TASK);env.codec=ActionCodec(c['min'],c['max'])
 agent=I2OAgent(rgb_obs_shape=(4,24,84,84),low_dim_obs_shape=(120,),action_shape=(8,),device=f'cuda:{GPU}',lr=5e-5,weight_decay=.1,feature_dim=128,hidden_dim=512,levels=3,bins=5,atoms=51,v_min=-2.,v_max=2.,bc_lambda=1.,bc_margin=.01,critic_lambda=.1,critic_target_tau=.02,num_expl_steps=0,update_every_steps=1,stddev_schedule='.01',use_logger=True,action_mode='soft',softargmax_beta=1.,soft_endpoint_rescale=False,phase_heads=1)
 for k in ['encoder','critic','critic_target']:getattr(agent,k).load_state_dict(saved[k])
 results=[]
 for ep,seed in enumerate(json.loads((OUT/'selection.json').read_text())['seeds']):
  env.rng=np.random.default_rng(seed);obs=env.reset();rows=[]
  movie=OUT/f'{TASK}_rl_{step}_seed_{seed}.mp4'
  writer=subprocess.Popen(['/usr/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','672x548','-r','10','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)],stdin=subprocess.PIPE)
  def frame(info):
   canvas=Image.new('RGB',(672,548));draw=ImageDraw.Draw(canvas);images=env.images('preview')
   for i,name in enumerate(env.camera_names):
    x=i%2*336;y=i//2*252;canvas.paste(Image.fromarray(images[name]).resize((336,252)),(x,y));draw.text((x+4,y+4),name,fill='red')
   draw.text((4,508),f'{TASK} | RL {step} | seed {seed} | model-controlled rerun',fill='white')
   draw.text((4,528),f"step {env.t}/{env.horizon} | opening {info.get('opening_m') if TASK=='open_drawer' else info.get('opening_deg')} | success {info.get('success',False)}",fill='white')
   writer.stdin.write(np.asarray(canvas).tobytes())
  frame(env.status())
  for _ in range(env.horizon):
   with torch.no_grad(),utils.eval_mode(agent):action=agent.act(*obs,step=step,eval_mode=True)
   obs,r,term,trunc,info=env.step(action);rows.append(dict(action=action.tolist(),**info));frame(info)
   if term or trunc:break
  
  for _ in range(10):frame(info)
  writer.stdin.close();assert writer.wait()==0;writer=None
  result=dict(seed=seed,online_step=step,success=info['success'],steps=len(rows),max_lift_m=max(x['lift_m'] for x in rows),final=info,movie=movie.name)
  results.append(result);(OUT/f'trace_{seed}.json').write_text(json.dumps(rows));(OUT/'results.json').write_text(json.dumps(dict(checkpoint_sha256=hashlib.sha256((JOB/'latest.pt').read_bytes()).hexdigest(),online_step=step,warm_updates=saved['warm_updates'],results=results),indent=2))
  print('RECORDED_RL',json.dumps(result),flush=True)
except Exception:
 import traceback
 (OUT/'error.txt').write_text(traceback.format_exc());raise
finally:
 if writer:writer.stdin.close();writer.wait()
 if env is not None:env.close()
 app.close()
