import json
from pathlib import Path
import numpy as np
p=Path('assets/rlbench_put_books_on_bookshelf/export');d=json.loads((p/'sensor_reference.json').read_text());s=json.loads((p/'scene.json').read_text());book=next(x for x in s['objects'] if x['name']=='book2');bb=np.array(book['bbox']);hb=(bb[3:]-bb[:3])/2;cb=(bb[3:]+bb[:3])/2
ha=np.array([.15,.05,.05]);ca=np.array([0.,0.,.05]);mismatches=[]
for sample in d['samples']:
 m=np.array(sample['relative_matrix']).reshape(3,4);B=m[:3,:3];delta=B@cb+m[:3,3]-ca;axes=[*np.eye(3),*B.T]+[np.cross(x,y) for x in np.eye(3) for y in B.T];inside=True
 for v in axes:
  if np.linalg.norm(v)>1e-7 and abs(delta@v)>abs(v)@ha+np.abs(B.T@v)@hb:inside=False;break
 if inside!=sample['detected']:mismatches.append(dict(center=sample['center'],obb=inside,native=sample['detected']))
report=dict(samples=len(d['samples']),matches=len(d['samples'])-len(mismatches),mismatches=mismatches,comparison='OBB intersection against sim.checkProximitySensor(original success, original book2)')
(p/'sensor_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
