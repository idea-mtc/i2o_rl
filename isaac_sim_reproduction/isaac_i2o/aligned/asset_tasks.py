"""Asset-based tasks: official Sektion cabinet and converted pinned RLBench door.
No procedural replacement objects, scripted object motion or phase observations.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import UsdGeom,UsdPhysics,Gf
from aligned.environment import AlignedLidEnv
from aligned.articulated import ArticulatedEnv
from panda_env import ROOT

def mat(v):
 m=np.eye(4);m[:3]=np.asarray(v).reshape(3,4);return m
class AssetTaskEnv(ArticulatedEnv):
 def __init__(self,task,frame_stack=8,seed=1):
  self.task_name=task;self.unlocked=False
  AlignedLidEnv.__init__(self,frame_stack,100 if task=='open_drawer' else 125,seed)
  self.world.stop();stage=self.world.stage
  self.world.scene.remove_object('saucepan_lid');stage.RemovePrim('/World/Saucepan');self.world.scene.remove_object('table')
  from isaacsim.core.utils.stage import add_reference_to_stage
  from isaacsim.core.api.robots import Robot
  if task=='open_door':
   from isaacsim.core.api.objects import FixedCuboid
   self.world.scene.add(FixedCuboid(prim_path='/World/FixturePedestal',name='fixture_pedestal',position=np.array([.4,-.13,.10]),scale=np.array([.65,.5,.2]),color=np.array([.3,.33,.37])))
  calibration=json.loads((ROOT/'assets/rlbench_open_door/export/tip_calibration.json').read_text())
  assert calibration['max_matrix_variation']<.005,calibration
  # Keep the validated URDF TCP; Coppelia/URDF robot frames are not a rigid exact fit.
  self.policyC=np.eye(4);self.native_to_urdf=np.eye(4)
  if task=='open_door':
   self.asset=ROOT/'assets/rlbench_open_door/usd/open_door.usda';self.base_position=np.array([.36,-.13,.20]);self.base_rotation=np.eye(3)
   self.joint_names=['door_frame_joint','door_handle_joint'];self.success_threshold=np.deg2rad(25);body='frame';self.grip_body='handle';anchor='fixed_root'
   source=json.loads((ROOT/'assets/rlbench_open_door/export/scene.json').read_text());ri=np.linalg.inv(mat(source['root_matrix']))
   handle=next(o for o in source['objects'] if o['name']=='door_handle_joint');handle_pose=ri@mat(handle['matrix'])
   self.native_waypoints=json.loads((ROOT/'assets/rlbench_open_door/export/robot_and_waypoints.json').read_text())['waypoints']
   grip=mat(self.native_waypoints[1]['matrix'])[:3,3];self.grip_local=np.linalg.solve(handle_pose[:3,:3],grip-handle_pose[:3,3])
  else:
   self.asset=ROOT/'assets/nvidia_isaac_5/Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd';self.base_position=np.array([.95,0,.4842]);self.base_rotation=Rotation.from_euler('z',np.pi).as_matrix()
   self.joint_names=['drawer_bottom_joint'];self.success_threshold=.15;body='sektion';self.grip_body='drawer_handle_bottom';self.grip_local=np.array([.290,0,.01]);anchor='rootJoint'
  if task=='open_drawer':
   from isaacsim.core.api.objects import FixedCuboid
   self.world.scene.add(FixedCuboid(prim_path='/World/CabinetPlinth',name='cabinet_plinth',position=np.array([.95,0,.05]),scale=np.array([.63,.76,.1]),color=np.array([.3,.33,.37])))
  add_reference_to_stage(str(self.asset),'/World/Mechanism')
  root=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Mechanism'));root.ClearXformOpOrder();pose=np.eye(4);pose[:3,:3]=self.base_rotation;pose[:3,3]=self.base_position;root.AddTransformOp().Set(Gf.Matrix4d(*pose.T.flatten().tolist()))
  self.base_quat=Rotation.from_matrix(self.base_rotation).as_quat()[[3,0,1,2]]
  self.rootfix=UsdPhysics.FixedJoint.Get(stage,'/World/Mechanism/'+anchor)
  self.rootfix.GetLocalPos0Attr().Set(Gf.Vec3f(*self.base_position));self.rootfix.GetLocalRot0Attr().Set(Gf.Quatf(float(self.base_quat[0]),Gf.Vec3f(*self.base_quat[1:])))
  self.mechanism=self.world.scene.add(Robot(prim_path='/World/Mechanism/'+body if task=='open_door' else '/World/Mechanism',name='asset_mechanism'))
  self.world.reset();self.indices={n:i for i,n in enumerate(self.mechanism.dof_names)}
  print('ASSET_DOFS',self.mechanism.dof_names,flush=True)
  if task=='open_door':self.lock_drive=UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath('/World/Mechanism/door_frame_joint'),'angular')
  self.initial_joints=np.zeros(len(self.joint_names));self.shift=np.zeros(3)
  for name,eye in {'front':(-.35,-1.,1.05),'left_shoulder':(.18,1.15,1.15),'right_shoulder':(.18,-1.15,1.15)}.items():
   p=stage.GetPrimAtPath(self.camera_paths[name]);UsdGeom.Xformable(p).GetOrderedXformOps()[0].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(.57,0,.52),Gf.Vec3d(0,0,1)).GetInverse())
  self.config=dict(task=task,asset_source=str(self.asset),scene='native_assets',action='measured joint delta and binary gripper',state_dim=120,phase_information=False,
   reward='success1 otherwise0',horizon=self.horizon,grasp_attachment=False,selected_joint=self.joint_names[0],success_threshold=self.success_threshold,xy_randomization_m=.02,
   remaining_differences=['PhysX controller and collision solver','camera calibration and reset distribution','Sektion bottom drawer substitutes RLBench drawer' if task=='open_drawer' else 'pinned RLBench original door geometry; motor gains mapped to PhysX'])
 def observe(self):
  if not hasattr(self,'policyC'):return AlignedLidEnv.observe(self)
  raw=self.images();rgb=np.stack([raw[k].transpose(2,0,1) for k in self.camera_names]);q=np.asarray(self.robot.get_joint_positions());tcp=self.fk(q[self.arm])@self.policyC
  low=np.r_[q[self.arm],float(np.all(q[self.fingers]>.036)),tcp[:3,3],Rotation.from_matrix(tcp[:3,:3]).as_quat()].astype(np.float32)
  if not self.rgb_frames:
   for _ in range(self.stack):self.rgb_frames.append(rgb.copy());self.states.append(low.copy())
  else:self.rgb_frames.append(rgb);self.states.append(low)
  return np.concatenate(list(self.rgb_frames),axis=1),np.concatenate(list(self.states))
 def _drive(self,steps=6):
  AlignedLidEnv._drive(self,steps)
  if hasattr(self,'lock_drive') and self.task_name=='open_door' and not self.unlocked:
   if abs(self.joint_positions()[1])>np.deg2rad(25):self.unlocked=True;self.lock_drive.GetStiffnessAttr().Set(0.);self.lock_drive.GetDampingAttr().Set(.01);self.lock_drive.GetMaxForceAttr().Set(1.)
 def reset(self,*args,**kwargs):
  self.unlocked=False
  if self.task_name=='open_door':self.lock_drive.GetStiffnessAttr().Set(50000.);self.lock_drive.GetDampingAttr().Set(100.);self.lock_drive.GetTargetPositionAttr().Set(0.);self.lock_drive.GetMaxForceAttr().Set(1000.)
  self.world.reset();self.target=self.home.copy();self.robot.set_joint_positions(self.target);self.robot.set_joint_velocities(np.zeros_like(self.target))
  self.shift=np.r_[self.rng.uniform(-.02,.02,2),0.];position=self.base_position+self.shift
  self.rootfix.GetLocalPos0Attr().Set(Gf.Vec3f(*position));self.mechanism.set_world_pose(position=position,orientation=self.base_quat)
  self.mechanism.set_joint_positions(np.zeros(self.mechanism.num_dof));self.mechanism.set_joint_velocities(np.zeros(self.mechanism.num_dof))
  self.robot.get_articulation_controller().set_gains(kps=np.array([8000. if i in self.arm else 4000. for i in range(self.robot.num_dof)]),kds=np.array([150. if i in self.arm else 100. for i in range(self.robot.num_dof)]))
  self.initial_joints=np.zeros(len(self.joint_names));self._drive(30);self.mechanism.set_joint_positions(np.zeros(self.mechanism.num_dof));self.mechanism.set_joint_velocities(np.zeros(self.mechanism.num_dof));self._drive(120)
  assert abs(self.joint_positions()[0])<.003,('reset is not closed',self.joint_positions())
  self.t=0;self.done=False;self.attached=False;self.rgb_frames.clear();self.states.clear();self._render();return self.observe()
 def expert_raw_actions(self):
  from fast_panda_ik import FastPandaIK
  from scipy.optimize import least_squares
  solver=FastPandaIK(ROOT/'assets/robot/panda/panda.urdf',self.lower,self.upper)
  def fallback(position,seed):
   def residual(q):
    m=self.fk(q);return np.r_[(m[:3,3]-position)*5,Rotation.from_matrix(solver.rotation@m[:3,:3].T).as_rotvec()]
   r=least_squares(residual,seed,bounds=(self.lower,self.upper),max_nfev=300);assert np.linalg.norm(residual(r.x))<.01,('native expert IK unreachable',position.tolist(),seed.tolist(),r.x.tolist(),residual(r.x).tolist());return r.x
  solver.fallback=fallback
  def move(goal,opened,tolerance=.008):
   for _ in range(self.horizon):
    desired_pose=goal();q=np.asarray(self.robot.get_joint_positions())[self.arm];actual=self.fk(q);err=desired_pose[:3,3]-actual[:3,3]
    max_step=.005 if self.expert_phase=='pull' else .008
    position=actual[:3,3]+err*min(1.,max_step/max(float(np.linalg.norm(err)),1e-9));rotation_error=Rotation.from_matrix(desired_pose[:3,:3]@actual[:3,:3].T).as_rotvec();solver.rotation=Rotation.from_rotvec(rotation_error*min(1.,.10/max(float(np.linalg.norm(rotation_error)),1e-9))).as_matrix()@actual[:3,:3]
    desired=solver(position,q);delta=desired-q;delta*=min(1.,.18/max(float(np.max(abs(delta))),1e-9));yield np.r_[delta,opened].astype(np.float32)
    if self.done:return
    if self.expert_phase=='turn_handle' and self.unlocked:return
    actual=self.fk(np.asarray(self.robot.get_joint_positions())[self.arm])
    if np.linalg.norm(goal()[:3,3]-actual[:3,3])<tolerance and np.linalg.norm(Rotation.from_matrix(goal()[:3,:3]@actual[:3,:3].T).as_rotvec())<.04:return
  if self.task_name=='open_drawer':
   rotation=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
   grip=self.knob_position().copy()
   def pose(position):
    m=np.eye(4);m[:3,:3]=rotation;m[:3,3]=position;return m
   self.expert_phase='approach';yield from move(lambda:pose(grip-rotation[:,2]*.12),1.)
   self.expert_phase='grasp';yield from move(lambda:pose(grip),1.,.005)
   self.expert_phase='close';yield np.r_[np.zeros(7),0.].astype(np.float32)
   self.expert_phase='pull';yield from move(lambda:pose(grip+np.array([-.20,0,0])),0.,.005)
  else:
   placement=np.eye(4);placement[:3,:3]=self.base_rotation;placement[:3,3]=self.base_position+self.shift
   for i,w in enumerate(self.native_waypoints[:2]):
    self.expert_phase=w['name'];goal=placement@mat(w['matrix'])@self.native_to_urdf
    yield from move(lambda g=goal:g,1.,.006)
    if self.done:return
   self.expert_phase='close';yield np.r_[np.zeros(7),0.].astype(np.float32)
   # Preserve actual contact while following the original handle/hinge axes.
   q=np.asarray(self.robot.get_joint_positions())[self.arm];held=np.linalg.inv(self.pose('handle'))@self.fk(q)
   def turn_goal():
    angle=min(.10,max(0.,np.deg2rad(29)-self.joint_positions()[1]));r=np.eye(4);r[:3,:3]=Rotation.from_rotvec([0,0,angle]).as_matrix()
    return self.pose('handle')@r@held
   self.expert_phase='turn_handle';yield from move(turn_goal,0.,.001)
   if not self.unlocked or self.done:return
   q=np.asarray(self.robot.get_joint_positions())[self.arm];held=np.linalg.inv(self.pose('handle'))@self.fk(q)
   def open_goal():
    angle=min(.08,max(0.,np.deg2rad(29)-self.joint_positions()[0]));r=np.eye(4);r[:3,:3]=Rotation.from_rotvec([0,0,angle]).as_matrix();door=self.pose('door')
    return door@r@np.linalg.inv(door)@self.pose('handle')@held
   self.expert_phase='open_door';yield from move(open_goal,0.,.001)
