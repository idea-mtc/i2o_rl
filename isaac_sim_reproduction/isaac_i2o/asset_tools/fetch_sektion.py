"""Mirror the official cabinet and its USD/texture dependencies, no training changes."""
from urllib.request import urlopen,Request
from urllib.parse import urljoin,urlparse
from pathlib import Path
from pxr import Sdf,UsdUtils
import json,hashlib
BASE='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/'
OUT=Path('/home/ubuntu/pingandog/assets/nvidia_isaac_5');OUT.mkdir(parents=True,exist_ok=True)
queue=[BASE+'Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd'];seen=set();files=[]
while queue:
 url=queue.pop(0)
 if url in seen:continue
 seen.add(url);assert url.startswith(BASE),url
 relative=url[len(BASE):];path=OUT/relative;path.parent.mkdir(parents=True,exist_ok=True)
 with urlopen(Request(url,headers={'User-Agent':'curl/8.5.0'}),timeout=90) as r:b=r.read()
 path.write_bytes(b);files.append(dict(url=url,path=relative,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()));print('ASSET',relative,len(b),flush=True)
 if path.suffix.lower() in ['.usd','.usda','.usdc']:
  layer=Sdf.Layer.FindOrOpen(str(path));assert layer,path;assets=[]
  def collect(p):assets.append(p);return p
  UsdUtils.ModifyAssetPaths(layer,collect)
  for asset in assets:
   if not asset or asset.endswith('.mdl'):continue
   dep=urljoin(url,asset)
   if dep.startswith(BASE):queue.append(dep)
   else:raise RuntimeError('external dependency needs review '+dep)
(OUT/'provenance.json').write_text(json.dumps(dict(entry='Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd',source=BASE,files=files,license='NVIDIA Isaac Sim Additional Software and Materials License'),indent=2))
print('SEKTION_DEPENDENCIES_COMPLETE',len(files),flush=True)
