"""Export actual accepted training observations, without rerunning the simulator."""
import argparse,json,hashlib,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--tasks',nargs='+',default=['open_drawer','open_door']);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
report=[]
for task in a.tasks:
 directory=a.run/task/'dataset';manifest=json.loads((directory/'manifest.json').read_text());entry=manifest['episodes'][0];source=directory/entry['file']
 digest=hashlib.sha256(source.read_bytes()).hexdigest();assert digest==entry['sha256']
 with np.load(source,allow_pickle=False) as f:data={k:f[k] for k in f.files}
 assert data['rgb'].shape[1:]==(4,3,84,84) and len(data['rgb'])==len(data['raw_actions'])+1
 assert data['rewards'][-1]==1
 movie=a.out/f'{task}_expert_demo_000.mp4'
 writer=subprocess.Popen(['/usr/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','672x548','-r','5','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)],stdin=subprocess.PIPE)
 for i,views in enumerate(data['rgb']):
  canvas=Image.new('RGB',(672,548));draw=ImageDraw.Draw(canvas)
  for v,name in enumerate(['front','wrist','left_shoulder','right_shoulder']):
   x=v%2*336;y=v//2*252;canvas.paste(Image.fromarray(views[v].transpose(1,2,0)).resize((336,252),Image.Resampling.BILINEAR),(x,y))
   draw.rectangle((x,y,x+125,y+16),fill='black');draw.text((x+4,y+2),name,fill='white')
  draw.text((5,507),f'EXPERT DEMONSTRATION | {task} | stored training images (84x84/view)',fill='white')
  elapsed=float(data['sim_time'][i]-data['sim_time'][0]);success=i==len(data['rgb'])-1
  draw.text((5,528),f'step {i}/{len(data["raw_actions"])} | sim {elapsed:.2f}s | success {success} | playback 5 frames/s',fill=(60,255,100) if success else 'white')
  frame=np.asarray(canvas).tobytes();writer.stdin.write(frame)
 for _ in range(5):writer.stdin.write(frame)
 writer.stdin.close();assert writer.wait()==0
 report.append(dict(task=task,seed=entry['seed'],demo_index=0,source=str(source),sha256=digest,steps=entry['steps'],recorded_frames=len(data['rgb']),success=True,simulation_seconds=float(data['sim_time'][-1]-data['sim_time'][0]),playback_fps=5,movie=movie.name,notes='Actual dataset RGB; no simulator rerun or policy rollout; final frame held1s.'))
(a.out/'provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
