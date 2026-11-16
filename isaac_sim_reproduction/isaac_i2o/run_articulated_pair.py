"""Two independent task processes with shared or separate project GPU leases.

Separate mode uses the existing idle-card launcher for each task.
Stage gates are enforced inside each learner. Bounded recovery, per-task PAUSE.
"""
import argparse,os,sys,json,signal,time,subprocess,datetime
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--tasks',nargs=2,default=['open_drawer','open_door'],choices=['open_drawer','open_door','open_microwave','put_books_on_bookshelf']);p.add_argument('--mode',choices=['probe','train'],required=True);p.add_argument('--ready-root',type=Path);p.add_argument('--scene',choices=['procedural','native_assets'],default='procedural');p.add_argument('--separate-gpus',action='store_true');p.add_argument('--root',type=Path,required=True);a=p.parse_args()
ROOT=Path('/home/ubuntu/pingandog');OUT=a.root.resolve();OUT.mkdir(parents=True,exist_ok=True)
GPU=-1 if a.separate_gpus else int(os.environ['PINGANDOG_GPU'])
if not a.separate_gpus:os.fstat(9)
children={};retries={t:0 for t in a.tasks};finished=set();stopping=False

def event(**data):
 data.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=GPU)
 with (OUT/'supervisor.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
def stop(proc):
 if proc.poll() is not None:return
 try:os.killpg(proc.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:proc.wait(timeout=30)
 except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
def shutdown(*_):
 global stopping
 stopping=True
 for proc in children.values():stop(proc)
signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
def launch(task):
 directory=OUT/task;directory.mkdir(exist_ok=True)
 if a.mode=='probe':cmd=[str(ROOT/'isaac_i2o/probe_articulated.py'),task,str(directory)]
 else:cmd=[str(ROOT/'isaac_i2o/aligned_night.py'),'--task',task,'--scene',a.scene,'--job-dir',str(directory),'--gate-successes','4','--max-warmup','5000','--deadline','not specified']
 log=open(directory/'process.log','a',buffering=1)
 launcher=ROOT/('scripts/run_isaac62.sh' if a.separate_gpus else 'runtime/isaac-sim-6.2.0/python.sh')
 proc=subprocess.Popen([str(launcher),*cmd],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,pass_fds=() if a.separate_gpus else (9,),start_new_session=True)
 log.close();children[task]=proc;event(event='launched',task=task,pid=proc.pid,retry=retries[task])
for task in retries:
 if a.ready_root is None or (a.ready_root/(task+'.json')).exists():launch(task)
while not stopping and len(finished)<2:
 for task in retries:
  if task in finished:continue
  if task not in children:
   if a.ready_root is None or (a.ready_root/(task+'.json')).exists():launch(task)
   continue
  proc=children[task]
  directory=OUT/task
  if (directory/'PAUSE').exists():stop(proc);finished.add(task);event(event='paused',task=task);continue
  f=directory/('results.json' if a.mode=='probe' else 'progress.json');d={}
  try:d=json.loads(f.read_text()) if f.exists() else {}
  except json.JSONDecodeError:continue
  complete=d.get('successes')==3 if a.mode=='probe' else d.get('stage')=='complete'
  if complete:
   stop(proc);finished.add(task);event(event='complete',task=task);continue
  if a.mode=='train' and (directory/'motion_alert.json').exists():
   (directory/'PAUSE').write_text('motion collapse; inspect before resuming');stop(proc);finished.add(task);event(event='motion_guard',task=task);continue
  failed_gate='startup gate not met' in d.get('error','')
  if failed_gate:stop(proc);finished.add(task);event(event='failed_learning_gate',task=task);continue
  if f.exists() and time.time()-f.stat().st_mtime>1800:stop(proc);event(event='stale_progress',task=task)
  if proc.poll() is not None:
   if a.mode=='probe' or retries[task]>=3:finished.add(task);event(event='failed',task=task,exitcode=proc.returncode)
   else:
    retries[task]+=1;time.sleep(10);launch(task)
 time.sleep(10)
event(event='supervisor_finished',mode=a.mode)
