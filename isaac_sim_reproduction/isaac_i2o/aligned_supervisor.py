"""Bounded recovery of only the dedicated aligned training service."""
import datetime,json,subprocess,time,zipfile
from pathlib import Path
ROOT=Path('/home/ubuntu/pingandog');OUT=ROOT/'outputs/aligned_night/job_20261113'
UNIT='pingandog-aligned-night-20261113.service'
def command(*args):return subprocess.run(args,capture_output=True,text=True)
def record(value):
 value['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (OUT/'supervisor.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
retries=0
while True:
 try:
  if (OUT/'PAUSE').exists():time.sleep(30);continue
  p=OUT/'progress.json';data=json.loads(p.read_text()) if p.exists() else {}
  if data.get('stage')=='complete':record({'event':'completed'});break
  error=data.get('error','')
  if 'startup gate not met' in error:
   record({'event':'stopped_for_failed_learning_gate','error':error});break
  if (OUT/'motion_alert.json').exists():
   record({'event':'stopped_for_motion_collapse','alert':json.loads((OUT/'motion_alert.json').read_text())['label']})
   (OUT/'PAUSE').write_text('motion collapse detected; inspect evaluation before resuming')
   command('systemctl','--user','stop',UNIT);break
  active=command('systemctl','--user','is-active',UNIT).stdout.strip() in ['active','activating']
  stale=p.exists() and time.time()-p.stat().st_mtime>1800
  if active and stale:
   record({'event':'stale_progress_stopping_own_service'});command('systemctl','--user','stop',UNIT);active=False
  if not active:
   if retries>=5:record({'event':'retry_limit_reached'});break
   latest=OUT/'latest.pt';previous=OUT/'previous.pt'
   if latest.exists() and not zipfile.is_zipfile(latest):
    if previous.exists() and zipfile.is_zipfile(previous):
     latest.rename(OUT/f'corrupt_{int(time.time())}.pt');latest.write_bytes(previous.read_bytes())
     record({'event':'restored_previous_checkpoint'})
    else:record({'event':'no_readable_checkpoint'});break
   time.sleep(30);command('systemctl','--user','reset-failed',UNIT)
   result=command('systemctl','--user','start',UNIT);retries+=1
   record({'event':'restart','attempt':retries,'returncode':result.returncode,'stderr':result.stderr})
  time.sleep(30)
 except Exception as exc:
  record({'event':'supervisor_error','error':repr(exc)});time.sleep(30)
