"""Read-only checkpoint diagnostics: teacher forcing and matched-seed rollouts."""
import os, sys, json, datetime, subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
ROOT=Path('/home/ubuntu/pingandog')
OUT=ROOT/'outputs/i2o_diagnostics'/datetime.datetime.now().strftime('run_%Y%m%d_%H%M%S')
OUT.mkdir(parents=True)
GPU=int(os.environ['PINGANDOG_GPU'])
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','multi_gpu':False,
                  'active_gpu':GPU,'physics_gpu':GPU,'width':960,'height':720,
                  'extra_args':['--/physics/updateToUsd=true']})
env=None
try:
 import numpy as np, torch
 from PIL import Image,ImageDraw
 from vendor.i2o_rl import I2OAgent
 from vendor import utils
 from tasks import make_env
 from demonstrations import read_episode,stacked_observations
 torch.set_num_threads(4);utils.set_seed_everywhere(1)
 run=ROOT/'outputs/i2o_panda/run_20261112_103403'
 config=json.loads((run/'config.json').read_text())
 env=make_env('take_lid_off_saucepan',8,260,100000,.02)
 kw={k:config['model_config'][k] for k in ['lr','weight_decay','feature_dim','hidden_dim','levels','bins','atoms','v_min','v_max','bc_lambda','bc_margin','critic_lambda']}
 agent=I2OAgent(rgb_obs_shape=env.rgb_shape,low_dim_obs_shape=env.state_shape,action_shape=env.action_shape,
  device=f'cuda:{GPU}',critic_target_tau=.02,num_expl_steps=0,update_every_steps=1,stddev_schedule='.01',
  use_logger=True,action_mode='soft',softargmax_beta=1.,**kw)
 ck=torch.load(run/'checkpoint_5000.pt',map_location=f'cuda:{GPU}',weights_only=False)
 for name in ('encoder','critic','critic_target'):getattr(agent,name).load_state_dict(ck[name]);getattr(agent,name).eval()
 dataset=ROOT/'datasets/take_lid_off_saucepan/fixed_100_20261112'
 manifest=json.loads((dataset/'manifest.json').read_text())
 report={'checkpoint':str(run/'checkpoint_5000.pt'),'step':ck['step'],'gpu':GPU,'output':str(OUT)}
 rows=[];targets=[];preds=[];hard=[];phases=[]
 # Teacher-forcing on complete recorded expert trajectories, not model-generated observations.
 for rec in manifest['episodes'][:5]:
  d=read_episode(dataset/rec['file']);obs=stacked_observations(d,8)[:-1]
  predictions=[];hard_predictions=[]
  for start in range(0,len(obs),16):
   batch=obs[start:start+16]
   with torch.no_grad():
    rgb=torch.as_tensor(np.stack([o[0] for o in batch]),device=agent.device)
    low=torch.as_tensor(np.stack([o[1] for o in batch]),device=agent.device)
    enc=agent.encoder(rgb)
    predictions.extend(agent.critic_target.get_soft_action(enc,low,1.).cpu().numpy())
    hard_predictions.extend(agent.critic_target.get_action(enc,low)[0].cpu().numpy())
  targets.extend(d['actions']);preds.extend(predictions);hard.extend(hard_predictions);phases.extend(d['phase'])
 targets=np.array(targets);preds=np.array(preds);hard=np.array(hard);phases=np.array(phases)
 np.savez_compressed(OUT/'teacher_forcing.npz',expert=targets,soft=preds,hard=hard,phase=phases)
 def stats(ids):
  a=targets[ids];p=preds[ids]
  return {'n':int(len(a)),'mse':float(np.mean((p-a)**2)),'arm_mse':float(np.mean((p[:,:7]-a[:,:7])**2)),
   'gripper_mse':float(np.mean((p[:,7]-a[:,7])**2)),
   'expert_mean':a.mean(0).tolist(),'model_mean':p.mean(0).tolist(),
   'expert_mean_abs':np.abs(a).mean(0).tolist(),'model_mean_abs':np.abs(p).mean(0).tolist(),
   'hard_mse':float(np.mean((hard[ids]-a)**2)),'zero_baseline_mse':float(np.mean(a*a))}
 report['teacher_forcing']={'all':stats(np.ones(len(phases),bool)),**{phase:stats(phases==phase) for phase in np.unique(phases)}}
 def frame(label,info):
  views=env.images('preview');canvas=Image.new('RGB',(672,548));draw=ImageDraw.Draw(canvas)
  for i,name in enumerate(env.camera_names):
   x=i%2*336;y=i//2*252
   canvas.paste(Image.fromarray(views[name]).resize((336,252)),(x,y))
   draw.rectangle((x,y,x+335,y+18),fill='black');draw.text((x+4,y+3),name,fill='white')
  draw.text((5,508),label,fill='white')
  draw.text((5,527),f"step {info.get('step',0)} | lift {info['lift_m']:.3f}m | knob distance {info['knob_tcp_distance_m']:.3f}m",fill='white')
  return np.asarray(canvas)
 for mode in (['hard_policy'] if '--hard-only' in sys.argv else ['policy','expert']):
  env.rng=np.random.default_rng(100000);obs=env.reset();trace=[]
  w=subprocess.Popen(['/usr/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','672x548','-r','10','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/f'{mode}.mp4')],stdin=subprocess.PIPE)
  initial=frame(mode,env.status());Image.fromarray(initial).save(OUT/f'{mode}_initial.png');w.stdin.write(initial.tobytes())
  expert=env.expert_actions() if mode=='expert' else None
  agent.action_mode='hard' if mode=='hard_policy' else 'soft'
  for t in range(260):
   if mode!='expert':
    with torch.no_grad(),utils.eval_mode(agent):action=agent.act(*obs,step=5000,eval_mode=True)
   else:
    try:action=next(expert)
    except StopIteration:break
   before_target=env.target[env.arm].copy()
   obs,reward,term,trunc,info=env.step(action)
   trace.append(dict(action=action.tolist(),target_before=before_target.tolist(),target_after=env.target[env.arm].tolist(),phase=getattr(env,'expert_phase',None) if mode=='expert' else 'policy',**info))
   w.stdin.write(frame(mode,info).tobytes())
   if t in [34,74,99,144,259] or term:Image.fromarray(frame(mode,info)).save(OUT/f'{mode}_{t+1:03d}.png')
   if term or trunc:break
  w.stdin.close();assert w.wait()==0
  (OUT/f'{mode}_trace.json').write_text(json.dumps(trace))
  report[mode]={'final':info,'max_lift_m':max(x['lift_m'] for x in trace),
   'min_knob_distance_m':min(x['knob_tcp_distance_m'] for x in trace),
   'max_arm_target_measured_error_rad':float(max(np.max(np.abs(np.array(x['target_after'])-np.array(x['q'])[:7])) for x in trace))}
  print('DIAG_ROLLOUT',mode,json.dumps(report[mode]),flush=True)
 (OUT/'report.json').write_text(json.dumps(report,indent=2))
 print('DIAG_RESULT',json.dumps(report),flush=True)
finally:
 if env is not None:env.close()
 app.close()
