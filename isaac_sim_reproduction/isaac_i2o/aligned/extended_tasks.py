"""Pinned RLBench microwave and bookshelf assets in PhysX.

Variant zero (one book), original waypoint poses and joint/sensor thresholds.
Sensor overlap is an explicit OBB approximation to Coppelia proximity detection.
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import UsdGeom,UsdPhysics,Gf
from aligned.environment import AlignedLidEnv
from aligned.articulated import ArticulatedEnv
from aligned.asset_tasks import AssetTaskEnv,mat
from panda_env import ROOT

class ExtendedTaskEnv(ArticulatedEnv):
 def __init__(self,task,frame_stack=8,seed=1):
  assert task in ['open_microwave','put_books_on_bookshelf']
  self.task_name=task;self.unlocked=False;self.policyC=np.eye(4)
  AlignedLidEnv.__init__(self,frame_stack,125 if task=='open_microwave' else 175,seed)
  self.world.stop();stage=self.world.stage
  self.world.scene.remove_object('saucepan_lid');stage.RemovePrim('/World/Saucepan');self.world.scene.remove_object('table')
  from isaacsim.core.utils.stage import add_reference_to_stage
  from isaacsim.core.api.objects import FixedCuboid
  self.base=np.array([.45,0.,.30]) if task=='open_microwave' else np.array([.40,0.,.04])
  self.source=json.loads((ROOT/f'assets/rlbench_{task}/export/scene.json').read_text());self.objects={o['name']:o for o in self.source['objects']};self.norm=np.linalg.inv(mat(self.source['root_matrix']))
  self.asset=ROOT/f'assets/rlbench_{task}/usd/task.usda'
  add_reference_to_stage(str(self.asset),'/World/Mechanism')
  self.world.scene.add(FixedCuboid(prim_path='/World/TaskPlinth',name='task_plinth',position=np.r_[self.base[:2]+([.03,.33] if task=='open_microwave' else [.16,.05]),(self.base[2]-(.02 if task=='open_microwave' else .01))/2],scale=np.array([.60,.32,self.base[2]-.02]) if task=='open_microwave' else np.array([.45,.45,self.base[2]-.01]),color=np.array([.35,.38,.4])))
  self.anchor=UsdPhysics.FixedJoint.Get(stage,'/World/Mechanism/fixed_root')
  self.books={};self.grasped_book=None;self.grasp_path='/World/BookGrasp'
  if task=='open_microwave':
   from isaacsim.core.api.robots import Robot
   self.mechanism=self.world.scene.add(Robot(prim_path='/World/Mechanism/frame',name='microwave'))
  else:
   from isaacsim.core.prims import SingleRigidPrim
   self.frame=self.world.scene.add(SingleRigidPrim(prim_path='/World/Mechanism/frame',name='shelf_frame'))
   for n in ['book0','book1','book2']:self.books[n]=self.world.scene.add(SingleRigidPrim(prim_path='/World/Mechanism/'+n,name=n))
  if task=='open_microwave':
   drive=UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath('/World/Mechanism/microwave_door_joint'),'angular');drive.GetDampingAttr().Set(2000.);drive.CreateTargetVelocityAttr(0.)
  self.world.reset();self.shift=np.zeros(3);self.initial_joints=np.zeros(1)
  self.grip_body='door' if task=='open_microwave' else 'book2'
  self.grip_local=np.zeros(3)
  for name,eye in {'front':(-.30,-1.05,1.05),'left_shoulder':(.1,1.15,1.0),'right_shoulder':(.1,-1.15,1.0)}.items():
   UsdGeom.Xformable(stage.GetPrimAtPath(self.camera_paths[name])).GetOrderedXformOps()[0].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(.45,0,.43),Gf.Vec3d(0,0,1)).GetInverse())
  self.config=dict(task=task,scene='native_assets',asset_source=str(self.asset),source_commit='b80e51feb3694d9959cb8c0408cd385001b01382',variation=0,books_required=1 if self.books else None,horizon=self.horizon,state_dim=120,action='measured joint delta and binary gripper',phase_information=False,reward='success1 otherwise0',grasp_attachment=bool(self.books),grasp_detector='book OBB overlaps TCP box half-extents .025,.025,.035; Coppelia gripper sensor proxy',xy_randomization_m=.02,success='hinge displacement >80degrees' if not self.books else 'book2 overlaps original success sensor OBB; no release condition in source',remaining_differences=['PhysX controller/contact dynamics','camera/reset calibration','OBB sensor overlap approximates Coppelia pyramid detection','RLBench-style grasp attachment with calibrated TCP-box sensor proxy'])
 def placement(self):
  p=np.eye(4);p[:3,3]=self.base+self.shift;return p
 def native_pose(self,name):return self.placement()@self.norm@mat(self.objects[name]['matrix'])
 observe=AssetTaskEnv.observe
 def joint_positions(self):return np.asarray(self.mechanism.get_joint_positions())[:1] if not self.books else np.array([0.])
 def knob_position(self):
  if self.books:return self.pose('book2')[:3,3]
  # Original grasp waypoint follows the dynamic door body.
  door0=self.norm@mat(self.objects['microwave_door_joint']['matrix']);grip=self.norm@mat(self.objects['waypoint1']['matrix'])
  return (self.pose('door')@np.linalg.inv(door0)@grip)[:3,3]
 def sensor_overlap(self):
  # Original book collision shape is a box, tested against the exported sensor.
  sensor=self.native_pose('success');bbox=np.array(self.objects['success']['bbox']);lo=bbox[:3].copy();lo[2]=0.;hi=bbox[3:]
  book=self.pose('book2');bb=np.array(self.objects['book2']['bbox']);ca=(lo+hi)/2;ha=(hi-lo)/2;cb=(bb[:3]+bb[3:])/2;hb=(bb[3:]-bb[:3])/2
  A=sensor[:3,:3];B=book[:3,:3];a=A@ca+sensor[:3,3];b=B@cb+book[:3,3]
  axes=[*A.T,*B.T]+[np.cross(x,y) for x in A.T for y in B.T]
  for v in axes:
   if np.linalg.norm(v)<1e-7:continue
   if abs((b-a)@v)>np.abs(A.T@v)@ha+np.abs(B.T@v)@hb:return False
  return True
 def status(self):
  q=np.asarray(self.robot.get_joint_positions());tcp=self.fk(q[self.arm])[:3,3];j=self.joint_positions();success=self.sensor_overlap() if self.books else bool(abs(j[0])>np.deg2rad(80))
  return dict(task=self.task_name,step=self.t,tcp=tcp.tolist(),handle=self.knob_position().tolist(),joint_positions=j.tolist(),joint_displacements=j.tolist(),success=bool(success),failed=False,finger_positions_m=q[self.fingers].tolist(),unlocked=False,attached=self.grasped_book is not None,lift_m=0.,opening_m=None,opening_deg=None if self.books else float(np.rad2deg(abs(j[0]))),handle_tcp_distance_m=float(np.linalg.norm(self.knob_position()-tcp)),sim_time=float(self.world.current_time),detected=bool(success) if self.books else None,grasped_book=self.grasped_book)
 def reset(self,*args,**kwargs):
  if self.world.stage.GetPrimAtPath(self.grasp_path).IsValid():self.world.stage.RemovePrim(self.grasp_path)
  self.grasped_book=None
  self.world.reset();self.target=self.home.copy();self.robot.set_joint_positions(self.target);self.robot.set_joint_velocities(np.zeros_like(self.target))
  self.shift=np.r_[self.rng.uniform(-.02,.02,2),0.];placement=self.placement()
  # Fixed root pose and independent dynamic books are restored separately.
  position=placement[:3,3];self.anchor.GetLocalPos0Attr().Set(Gf.Vec3f(*position));self.anchor.GetLocalRot0Attr().Set(Gf.Quatf(1.))
  if self.books:
   self.frame.set_world_pose(position=position,orientation=np.array([1.,0,0,0]))
   for n,b in self.books.items():
    p=self.native_pose(n);b.set_world_pose(position=p[:3,3],orientation=Rotation.from_matrix(p[:3,:3]).as_quat()[[3,0,1,2]]);b.set_linear_velocity(np.zeros(3));b.set_angular_velocity(np.zeros(3))
  else:
   self.mechanism.set_world_pose(position=position,orientation=np.array([1.,0,0,0]));self.mechanism.set_joint_positions(np.zeros(self.mechanism.num_dof));self.mechanism.set_joint_velocities(np.zeros(self.mechanism.num_dof));self.mechanism.set_linear_velocity(np.zeros(3));self.mechanism.set_angular_velocity(np.zeros(3))
  self.robot.get_articulation_controller().set_gains(kps=np.array([8000. if i in self.arm else 4000. for i in range(self.robot.num_dof)]),kds=np.array([150. if i in self.arm else 100. for i in range(self.robot.num_dof)]))
  self.t=0;self.done=False;self.attached=False;self._drive(30);
  if not self.books:self.mechanism.set_joint_positions(np.zeros(self.mechanism.num_dof));self.mechanism.set_joint_velocities(np.zeros(self.mechanism.num_dof))
  self._drive(90);self.rgb_frames.clear();self.states.clear();self._render()
  assert not self.status()['success'],'reset already successful'
  return self.observe()
 def _drive(self,steps=6):
  if getattr(self,'books',{}) and self.grasped_book is None and np.all(self.target[self.fingers]<.01):self._attach_book()
  AlignedLidEnv._drive(self,steps)
  if hasattr(self,'task_name') and not np.isfinite(self.robot.get_joint_positions()).all():raise RuntimeError('nonfinite Panda joints in '+self.task_name+' reset/drive')
 def _release_book(self):
  if self.world.stage.GetPrimAtPath(self.grasp_path).IsValid():self.world.stage.RemovePrim(self.grasp_path)
  self.grasped_book=None
 def _attach_book(self):
  if self.grasped_book is not None:return
  tcp=self.fk(np.asarray(self.robot.get_joint_positions())[self.arm]);candidates=[]
  for name in self.books:
   book=self.pose(name);local=np.linalg.inv(tcp)@book;bb=np.array(self.objects[name]['bbox']);center=(bb[:3]+bb[3:])/2;half=(bb[3:]-bb[:3])/2
   c=local[:3,:3]@center+local[:3,3];extent=np.abs(local[:3,:3])@half
   if np.all(abs(c)<=extent+np.array([.025,.025,.035])):candidates.append((np.linalg.norm(c),name))
  if not candidates:return
  name=min(candidates)[1];cache=UsdGeom.XformCache();stage=self.world.stage
  hand=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(self.wrist_link_path));book=cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Mechanism/'+name));relative=book*hand.GetInverse();q=relative.ExtractRotationQuat()
  joint=UsdPhysics.FixedJoint.Define(stage,self.grasp_path);joint.CreateBody0Rel().SetTargets([self.wrist_link_path]);joint.CreateBody1Rel().SetTargets(['/World/Mechanism/'+name]);joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*relative.ExtractTranslation()));joint.CreateLocalRot0Attr().Set(Gf.Quatf(q.GetReal(),Gf.Vec3f(*q.GetImaginary())));joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0));joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.));joint.CreateCollisionEnabledAttr(False);self.grasped_book=name
 def step_raw(self,raw):
  if self.books and raw[7]>.5:self._release_book()
  # The inherited adapter completes arm motion and gripper actuation first.
  obs,reward,term,trunc,info=ArticulatedEnv.step_raw(self,raw)
  if self.books and raw[7]<=.5:
   self._attach_book();info['attached']=self.grasped_book is not None
  return obs,reward,term,trunc,info
 def expert_raw_actions(self):
  from fast_panda_ik import FastPandaIK
  from scipy.optimize import least_squares
  solver=FastPandaIK(ROOT/'assets/robot/panda/panda.urdf',self.lower,self.upper)
  def fallback(position,seed):
   def residual(q):
    pose=self.fk(q);return np.r_[5*(pose[:3,3]-position),Rotation.from_matrix(solver.rotation@pose[:3,:3].T).as_rotvec()]
   result=least_squares(residual,np.clip(seed,self.lower,self.upper),bounds=(self.lower,self.upper),max_nfev=300)
   assert np.linalg.norm(residual(result.x))<.01,('native expert IK unreachable',self.expert_phase,position.tolist())
   return result.x
  solver.fallback=fallback
  def move(goal,opened,tol=.005):
   for _ in range(self.horizon):
    desired=goal();q=np.asarray(self.robot.get_joint_positions())[self.arm];actual=self.fk(q);err=desired[:3,3]-actual[:3,3];position=actual[:3,3]+err*min(1.,(.012 if self.books else .014)/max(np.linalg.norm(err),1e-9));rot=Rotation.from_matrix(desired[:3,:3]@actual[:3,:3].T).as_rotvec();solver.rotation=Rotation.from_rotvec(rot*min(1.,.12/max(np.linalg.norm(rot),1e-9))).as_matrix()@actual[:3,:3]
    target=solver(position,q);delta=target-q;delta*=min(1.,.18/max(np.max(abs(delta)),1e-9));yield np.r_[delta,opened].astype(np.float32)
    if self.done:return
    actual=self.fk(np.asarray(self.robot.get_joint_positions())[self.arm]);desired=goal()
    if np.linalg.norm(desired[:3,3]-actual[:3,3])<tol and np.linalg.norm(Rotation.from_matrix(desired[:3,:3]@actual[:3,:3].T).as_rotvec())<.04:return
  opened=1.
  # Clear the book stack/cabinet before adopting the source grasp orientation.
  current=self.fk(np.asarray(self.robot.get_joint_positions())[self.arm]);safe=current.copy();safe[:3,3]=[.36,-.30,.70 if self.books else .60]
  self.expert_phase='clearance';yield from move(lambda:safe,1.)
  book_shift=self.pose('book2')[:3,3]-self.native_pose('book2')[:3,3] if self.books else np.zeros(3)
  for i in range(6 if self.books else 2):
   name='waypoint'+str(i);self.expert_phase=name;goal=self.native_pose(name)
   if self.books:
    calibration=json.loads((ROOT/'assets/rlbench_open_door/export/tip_calibration.json').read_text());goal=goal@np.linalg.inv(np.asarray(calibration['urdf_to_native_tip']))
   if self.books and i in [0,1,2]:goal[:3,3]+=book_shift
   yield from move(lambda g=goal:g,opened)
   if self.done:return
   ext=self.objects[name].get('extension') or ''
   if 'close_gripper' in ext:opened=0.;yield np.r_[np.zeros(7),opened].astype(np.float32)
   elif 'open_gripper' in ext:opened=1.;yield np.r_[np.zeros(7),opened].astype(np.float32)
  if not self.books:
   held=np.linalg.inv(self.pose('door'))@self.fk(np.asarray(self.robot.get_joint_positions())[self.arm])
   def goal():
    delta=min(.07,max(0.,np.deg2rad(85)-self.joint_positions()[0]));r=np.eye(4);r[:3,:3]=Rotation.from_rotvec([0,0,delta]).as_matrix();return self.pose('door')@r@held
   self.expert_phase='open_microwave';yield from move(goal,0.,.001)
