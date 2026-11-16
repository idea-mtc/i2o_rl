import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,'isaac_i2o')
from fast_panda_ik import FastPandaIK
s=json.loads(Path('assets/rlbench_put_books_on_bookshelf/export/scene.json').read_text())
def mat(v):
 m=np.eye(4);m[:3]=np.array(v).reshape(3,4);return m
norm=np.linalg.inv(mat(s['root_matrix']));objs={o['name']:o for o in s['objects']}
for x in [.4,.45,.5,.55]:
 for z in [.04,.08,.12]:
  for flip in [False,True]:
   ik=FastPandaIK('assets/robot/panda/panda.urdf');q=np.array([0,-.5,0,-2,0,1.6,.7]);good=True;worst=1.
   for i in range(6):
    goal=norm@mat(objs['waypoint'+str(i)]['matrix']);goal[:3,3]+=[x,0,z]
    if flip:goal[:3,:3]=goal[:3,:3]@np.diag([-1.,-1.,1.])
    for step in range(160):
     actual,_=ik.fk_jacobian(q);err=goal[:3,3]-actual[:3,3];pos=actual[:3,3]+err*min(1.,.012/max(np.linalg.norm(err),1e-9));rot=Rotation.from_matrix(goal[:3,:3]@actual[:3,:3].T).as_rotvec();ik.rotation=Rotation.from_rotvec(rot*min(1.,.12/max(np.linalg.norm(rot),1e-9))).as_matrix()@actual[:3,:3]
     try:q=ik(pos,q)
     except RuntimeError:good=False;break
     worst=min(worst,float(np.min(np.minimum(q-ik.lower,ik.upper-q))))
     if np.linalg.norm(err)<.004 and np.linalg.norm(rot)<.03:break
    if not good:break
   print(x,z,flip,good,i,round(worst,3),flush=True)
