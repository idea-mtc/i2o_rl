"""Restricted-GPU launcher, recovery and result aggregation for this one run."""
import argparse,datetime,fcntl,json,os,signal,subprocess,time,xml.etree.ElementTree as ET
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');a=p.parse_args()
ROOT=Path('/home/ubuntu/pingandog');RUN=ROOT/'outputs/microwave_expert_fix/run_20261115';CODE=RUN/'code';ISAAC=ROOT/'runtime/isaac-sim-6.2.0'
ALLOWED={2,3,4};DEADLINE=datetime.datetime.fromisoformat('2026-11-16T09:00:00+08:00');jobs={};attempts={g:0 for g in ALLOWED}
def now():return datetime.datetime.now(datetime.timezone.utc)
def write(name,data):
 f=RUN/name;t=f.with_suffix(f.suffix+'.tmp');t.write_text(json.dumps(data,indent=2));t.replace(f)
def event(kind,**kw):
 with (RUN/'supervisor.jsonl').open('a') as f:f.write(json.dumps(dict(event=kind,utc=now().isoformat(),**kw))+'\n')
def devices():
 r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 return {int(x[0]):dict(uuid=x[1].strip(),free=int(x[2]),util=int(x[3])) for line in r.stdout.splitlines() if (x:=line.split(','))}
D=devices();UUIDS=','.join(D[g]['uuid'] for g in sorted(ALLOWED))
def stop(job):
 try:os.killpg(job['proc'].pid,signal.SIGTERM)
 except ProcessLookupError:pass
 try:job['proc'].wait(timeout=12)
 except subprocess.TimeoutExpired:
  try:os.killpg(job['proc'].pid,signal.SIGKILL)
  except ProcessLookupError:pass
 job['log'].close();job['lock'].close()
def violation(job):
 r=subprocess.run(['nvidia-smi','-q','-x'],capture_output=True,text=True,check=True)
 bad=[];owned=[]
 for gpu in ET.fromstring(r.stdout).findall('gpu'):
  index=int(gpu.findtext('minor_number'))
  for proc in gpu.findall('./processes/process_info'):
   pid=int(proc.findtext('pid'))
   try:belongs=os.getpgid(pid)==job['proc'].pid
   except ProcessLookupError:continue
   if belongs:
    item=dict(gpu=index,pid=pid,memory=proc.findtext('used_memory'),type=proc.findtext('type'));owned.append(item)
    if index not in ALLOWED:bad.append(item)
 return bad,owned

def launch(g):
 lock=open(ROOT/f'runtime/locks/gpu{g}.lock','a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:lock.close();return False
 d=devices()[g]
 if d['free']<24000:lock.close();return False
 out=RUN/f'gpu{g}_seed{g-1}';out.mkdir(exist_ok=True)
 env=os.environ.copy();env.pop('LD_LIBRARY_PATH',None);env.update(CUDA_VISIBLE_DEVICES=D[g]['uuid'],I2O_ALLOWED_UUIDS=D[g]['uuid'],PINGANDOG_GPU=str(g),PINGANDOG_ROOT=str(ROOT),VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',VK_DRIVER_FILES='/usr/share/vulkan/icd.d/nvidia_icd.json',VK_ICD_FILE_NAMES='/usr/share/vulkan/icd.d/nvidia_icd.json',XDG_CACHE_HOME=str(ROOT/'runtime/cache62'),PYTHONUNBUFFERED='1')
 args=[str(ISAAC/'python.sh'),str(CODE/'expert_fix_gpu_probe.py'),str(g)] if a.probe else [str(ISAAC/'python.sh'),str(CODE/'microwave_expert_fix.py'),'--job-dir',str(out),'--gpu',str(g),'--seed',str(g-1)]
 sandbox=['/usr/bin/bwrap','--die-with-parent','--bind','/','/','--dev','/dev','--tmpfs','/dev/shm']
 for dev in ['nvidiactl','nvidia-uvm','nvidia-uvm-tools','nvidia-modeset',f'nvidia{g}','nvidia-caps',f'dri/card{g+1}',f'dri/renderD{128+g}']:
  if Path('/dev',dev).exists():sandbox+=['--dev-bind','/dev/'+dev,'/dev/'+dev]
 args=sandbox+args
 log=(out/('probe.log' if a.probe else 'process.log')).open('a');child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 jobs[g]=dict(proc=child,log=log,lock=lock,out=out,start=time.monotonic());attempts[g]+=1;event('launched',gpu=g,pid=child.pid,attempt=attempts[g],cuda_visible_devices=D[g]['uuid'],device_namespace=True,probe=a.probe)
 return True

def aggregate():
 summary={}
 for g in sorted(ALLOWED):
  out=RUN/f'gpu{g}_seed{g-1}';d={}
  for name in ['progress.json','gate.json','result.json']:
   f=out/name
   if f.exists():
    try:
     x=json.loads(f.read_text())
     if name=='result.json':
      x={**{k:v for k,v in x.items() if k not in ['baseline','final','best']},**{k:{n:v for n,v in x[k].items() if n!='results'} for k in ['baseline','final','best'] if k in x}}
     d[name]=x
    except json.JSONDecodeError:pass
  summary[str(g)]=d
 write('summary.json',dict(updated_utc=now().isoformat(),deadline=DEADLINE.isoformat(),allowed_gpus=sorted(ALLOWED),branches=summary,shared_il_initialization=True))
 lines=['# Microwave expert provenance correction','',f'Updated: {now().isoformat()}','Allowed physical GPUs: 2, 3, 4. Shared IL initialization; independent online seeds.','', '| GPU | Stage | Short baseline / 500 / 1000 | Final / best (100 each) |','|---|---|---|---|']
 for g,d in summary.items():
  gate=d.get('gate.json',{});res=d.get('result.json',{});pr=d.get('progress.json',{});lines.append(f"| {g} | {res.get('stage',pr.get('stage','waiting'))} | {gate.get('baseline','-')} / {gate.get('step500','-')} / {gate.get('step1000','-')} | {res.get('final',{}).get('successes','-')} / {res.get('best',{}).get('successes','-')} |")
 (RUN/'REPORT.md').write_text('\n'.join(lines)+'\n')

done=set();probe_ok=True;started=time.monotonic()
try:
 while now()<DEADLINE:
  for g in ([2] if a.probe else sorted(ALLOWED)):
   if g in done:continue
   out=RUN/f'gpu{g}_seed{g-1}'
   if not a.probe and (out/'result.json').exists():done.add(g);continue
   if g not in jobs:
    if attempts[g]>=3:done.add(g);event('retries_exhausted',gpu=g);continue
    launch(g);continue
   job=jobs[g];bad,owned=violation(job)
   write(f'gpu{g}_device_audit.json',dict(utc=now().isoformat(),pid=job['proc'].pid,processes=owned,violation=bad))
   if bad:
    stop(job);del jobs[g];done.add(g);probe_ok=False;event('forbidden_gpu_blocked',gpu=g,violations=bad);continue
   rc=job['proc'].poll()
   if rc is not None:
    job['log'].close();job['lock'].close();del jobs[g];event('exited',gpu=g,returncode=rc)
    if a.probe:done.add(g);probe_ok=probe_ok and rc==0
    elif (out/'result.json').exists():done.add(g)
    continue
   f=out/'progress.json'
   if not a.probe and f.exists() and time.time()-f.stat().st_mtime>1800:
    stop(job);del jobs[g];event('stalled_restarting',gpu=g);continue
   if a.probe and time.monotonic()-job['start']>180:
    stop(job);del jobs[g];done.add(g);probe_ok=False;event('probe_timeout',gpu=g)
  if not a.probe:aggregate()
  if done>=({2} if a.probe else ALLOWED):break
  time.sleep(5)
 if a.probe:write('device_probe_result.json',dict(passed=probe_ok,done=sorted(done),allowed=sorted(ALLOWED)))
 else:aggregate();event('supervisor_finished',completed_gpus=sorted(done))
finally:
 for g,j in list(jobs.items()):stop(j)
