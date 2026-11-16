"""Analytic Panda geometric Jacobian with damped least-squares pose IK."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


class FastPandaIK:
    def __init__(self, urdf, lower=None, upper=None, fallback=None):
        tree=ET.parse(urdf);j={x.get('name'):x for x in tree.getroot().findall('joint')}
        chain=[j['panda_joint%d'%i] for i in range(1,9)]+[j['panda_hand_joint']]
        self.origins=[];limits=[]
        for joint in chain:
            org=joint.find('origin');t=np.eye(4)
            t[:3,3]=np.fromstring(org.get('xyz','0 0 0'),sep=' ')
            t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(org.get('rpy','0 0 0'),sep=' ')).as_matrix()
            self.origins.append(t)
            if joint.get('type')=='revolute':
                limit=joint.find('limit');limits.append([float(limit.get('lower')),float(limit.get('upper'))])
        self.lower=np.array(limits)[:,0]+.01 if lower is None else lower
        self.upper=np.array(limits)[:,1]-.01 if upper is None else upper
        self.rotation=Rotation.from_euler('xyz',[np.pi,0,0]).as_matrix()
        self.tip=np.eye(4);self.tip[2,3]=.1034
        self.fallback=fallback;self.fallback_count=0

    def fk_jacobian(self,q):
        t=np.eye(4);axes=[];positions=[]
        for i,origin in enumerate(self.origins):
            t=t@origin
            if i<7:
                axes.append(t[:3,2].copy());positions.append(t[:3,3].copy())
                c=np.cos(q[i]);s=np.sin(q[i]);rz=np.eye(4)
                rz[:2,:2]=[[c,-s],[s,c]];t=t@rz
        t=t@self.tip;axes=np.array(axes);positions=np.array(positions)
        jac=np.r_[np.cross(axes,t[:3,3]-positions).T,axes.T]
        return t,jac

    def __call__(self,xyz,seed):
        q=np.clip(np.array(seed,dtype=float),self.lower,self.upper)
        for _ in range(60):
            t,j=self.fk_jacobian(q)
            position_error=np.asarray(xyz)-t[:3,3]
            orientation_error=Rotation.from_matrix(self.rotation@t[:3,:3].T).as_rotvec()
            if np.linalg.norm(position_error)<.00005 and np.linalg.norm(orientation_error)<.0005:
                return q
            error=np.r_[5*position_error,orientation_error];j[:3]*=5
            dq=j.T@np.linalg.solve(j@j.T+np.eye(6)*1e-5,error)
            scale=max(1.,np.max(np.abs(dq))/.18)
            q=np.clip(q+dq/scale,self.lower,self.upper)
        if self.fallback is not None:
            self.fallback_count+=1
            return self.fallback(xyz,seed)
        raise RuntimeError('analytic IK failed to reach pose tolerance')
