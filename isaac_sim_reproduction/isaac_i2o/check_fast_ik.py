import time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from fast_panda_ik import FastPandaIK

ik=FastPandaIK('/home/ubuntu/pingandog/assets/robot/panda/panda.urdf')
q=np.array([0,-.5,0,-2,0,1.6,.7]);eps=1e-6
_,jac=ik.fk_jacobian(q)
for j in range(7):
    step=np.eye(7)[j]*eps
    a=ik.fk_jacobian(q+step)[0];b=ik.fk_jacobian(q-step)[0]
    column=np.r_[(a[:3,3]-b[:3,3])/(2*eps),Rotation.from_matrix(a[:3,:3]@b[:3,:3].T).as_rotvec()/(2*eps)]
    np.testing.assert_allclose(jac[:,j],column,atol=1e-7)
rng=np.random.default_rng(18);errors=[];start=time.monotonic()
for ep in range(20):
    xy=np.array([.46,0])+rng.uniform(-.02,.02,2)
    for z in [.482,.322,.522]:
        p=np.r_[xy,z];q=ik(p,q);t,_=ik.fk_jacobian(q)
        errors.append(float(np.linalg.norm(t[:3,3]-p)))
        assert np.linalg.norm(Rotation.from_matrix(ik.rotation@t[:3,:3].T).as_rotvec())<.0005
print('FAST_IK_FINITE_DIFFERENCE_AND_60_POSE_TEST_PASS', 'seconds',time.monotonic()-start,'max_position_error_m',max(errors))
