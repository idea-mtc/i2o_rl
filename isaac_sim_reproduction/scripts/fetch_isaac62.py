"""Resumable range download of the pinned NVIDIA standalone release."""
import concurrent.futures, hashlib, json, os, time, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
URL='https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-6.2.0-linux-x86_64.zip'
SIZE=12964788213
SHA='aac4f9065f36d56f56a6523931710fad3ea9c3e5eeb067b781fa53098fcf19f2'
P=ROOT/'runtime/isaac-sim-standalone-6.2.0-linux-x86_64.zip'
J=P.with_suffix('.download.json')
if J.exists():
    state=json.loads(J.read_text())
    assert state['url']==URL and state['size']==SIZE
else:
    prefix=P.stat().st_size if P.exists() else 0
    assert prefix <= SIZE
    state={'url':URL,'size':SIZE,'sha256':SHA,'prefix':prefix,'complete':[]}
    J.write_text(json.dumps(state))
fd=os.open(P,os.O_CREAT|os.O_RDWR,0o644)
os.ftruncate(fd,SIZE)
chunk=64*1024*1024
ranges=[(a,min(a+chunk,SIZE)-1) for a in range(state['prefix'],SIZE,chunk) if a not in state['complete']]
def fetch(pair):
    a,b=pair
    for attempt in range(3):
        try:
            part=P.with_suffix(f'.part.{a}')
            header=P.with_suffix(f'.header.{a}')
            subprocess.run(['curl','--fail','--location','--silent','--show-error',
                '--range',f'{a}-{b}','--max-time','90','--dump-header',str(header),
                URL,'-o',str(part)],check=True)
            assert f'content-range: bytes {a}-{b}/{SIZE}' in header.read_text().lower(), header.read_text()
            assert part.stat().st_size == b-a+1, part.stat().st_size
            pos=a
            with part.open('rb') as stream:
                while data:=stream.read(1024*1024):
                    view=memoryview(data)
                    while view:
                        count=os.pwrite(fd,view,pos);pos+=count;view=view[count:]
            part.unlink();header.unlink()
            return a
        except Exception as exc:
            print(f"RANGE_ERROR {a} {type(exc).__name__}: {exc}",flush=True)
            if attempt==2: raise
            time.sleep(1)
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for fut in concurrent.futures.as_completed([pool.submit(fetch,pair) for pair in ranges]):
            state['complete'].append(fut.result())
            temp=J.with_suffix('.tmp');temp.write_text(json.dumps(state));temp.replace(J)
            done=state['prefix']+sum(min(chunk,SIZE-a) for a in state['complete'])
            print(f'DOWNLOAD {done/SIZE:.1%} {done}/{SIZE}',flush=True)
    os.fsync(fd)
finally: os.close(fd)
h=hashlib.sha256()
with P.open('rb') as stream:
    while data:=stream.read(8*1024*1024): h.update(data)
assert h.hexdigest()==SHA, 'SHA256 mismatch; archive must not be extracted'
state['verified']=True;J.write_text(json.dumps(state,indent=2))
print('VERIFIED',SHA,flush=True)
