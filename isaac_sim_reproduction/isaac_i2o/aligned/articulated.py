"""PhysX task analogues with pinned RLBench joint success thresholds.

Procedural assets, not converted RLBench models. No scripted object motion,
no grasp attachment, no task state/phase in policy observations. Drawer has a
weak 20 N/m return spring; this is an explicit dynamics difference.
"""
import numpy as np
from pxr import Usd,UsdShade,UsdGeom,UsdPhysics,Gf,PhysxSchema
from scipy.spatial.transform import Rotation
from aligned.environment import AlignedLidEnv

class ArticulatedEnv(AlignedLidEnv):
 def __init__(self,task,frame_stack=8,seed=1):
  assert task in ['open_drawer','open_door'];self.task_name=task
  super().__init__(frame_stack,100 if task=='open_drawer' else 125,seed)
  self.world.stop();stage=self.world.stage
  self.world.scene.remove_object('saucepan_lid');stage.RemovePrim('/World/Saucepan')
  UsdGeom.Xform.Define(stage,'/World/Mechanism')
  self.base_position=np.array([.54,0,.30]) if task=='open_drawer' else np.array([.60,-.12,.22])
  base=self.body('Base',self.base_position,.5)
  UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
  rootfix=UsdPhysics.FixedJoint.Define(stage,'/World/Mechanism/WorldAnchor')
  rootfix.CreateBody1Rel().SetTargets([base.GetPath()]);rootfix.CreateLocalPos0Attr().Set(Gf.Vec3f(*self.base_position))
  rootfix.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0));self.rootfix=rootfix
  if task=='open_drawer':
   self.box('Base/Back',(.025,.29,.20),(.135,0,0),(.24,.28,.33))
   for side in [-1,1]:self.box('Base/Side'+('Left' if side<0 else 'Right'),(.29,.018,.20),(0,side*.145,0),(.3,.34,.4))
   self.box('Base/Bottom',(.29,.29,.018),(0,0,-.095),(.3,.34,.4))
   moving=self.body('Moving',self.base_position,.30)
   self.box('Moving/Tray',(.20,.24,.075),(0,0,0),(.45,.34,.22))
   self.box('Moving/Front',(.018,.27,.11),(-.11,0,0),(.60,.43,.25))
   self.grip_local=np.array([-.18,0,.045]);self.knob('Moving/Handle',self.grip_local)
   j=UsdPhysics.PrismaticJoint.Define(stage,'/World/Mechanism/Slide')
   self.link(j,'Base','Moving');j.CreateAxisAttr('X');j.CreateLowerLimitAttr(-.28);j.CreateUpperLimitAttr(0.)
   j.CreateLocalPos0Attr().Set(Gf.Vec3f(0,0,0));j.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
   drive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'linear');drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(20.);drive.CreateDampingAttr(5.);drive.CreateTargetPositionAttr(0.);drive.CreateMaxForceAttr(20.)
   self.grip_body='Moving';self.joint_names=['Slide'];self.success_threshold=.15
  else:
   self.box('Base/Post',(.035,.035,.40),(0,0,.18),(.22,.25,.28))
   moving=self.body('Moving',self.base_position,.4)
   self.box('Moving/Panel',(.018,.25,.36),(0,.125,.18),(.48,.30,.16))
   hinge=UsdPhysics.RevoluteJoint.Define(stage,'/World/Mechanism/Hinge');self.link(hinge,'Base','Moving')
   hinge.CreateAxisAttr('Z');hinge.CreateLowerLimitAttr(-85.);hinge.CreateUpperLimitAttr(0.)
   hinge.CreateLocalPos0Attr().Set(Gf.Vec3f(0,0,0));hinge.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
   self.lock_drive=UsdPhysics.DriveAPI.Apply(hinge.GetPrim(),'angular');self.lock_drive.CreateTypeAttr('force');self.lock_drive.CreateStiffnessAttr(10000.);self.lock_drive.CreateDampingAttr(100.);self.lock_drive.CreateTargetPositionAttr(0.);self.lock_drive.CreateMaxForceAttr(500.)
   self.handle_pivot=np.array([-.055,.175,.20]);handle=self.body('Handle',self.base_position+self.handle_pivot,.04)
   self.box('Handle/Lever',(.018,.075,.015),(0,.035,0),(.12,.13,.15))
   self.grip_local=np.array([0,.06,0]);self.knob('Handle/Knob',self.grip_local)
   j=UsdPhysics.RevoluteJoint.Define(stage,'/World/Mechanism/Lever');self.link(j,'Moving','Handle')
   j.CreateAxisAttr('X');j.CreateLowerLimitAttr(-65.);j.CreateUpperLimitAttr(5.)
   j.CreateLocalPos0Attr().Set(Gf.Vec3f(*self.handle_pivot));j.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
   d=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'angular');d.CreateTypeAttr('force');d.CreateStiffnessAttr(.01);d.CreateDampingAttr(.0005);d.CreateTargetPositionAttr(0.);d.CreateMaxForceAttr(2.)
   self.grip_body='Handle';self.joint_names=['Hinge','Lever'];self.success_threshold=np.deg2rad(25)
  material=UsdShade.Material.Get(stage,'/World/GraspMaterial')
  for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/Mechanism')):
   if prim.HasAPI(UsdPhysics.CollisionAPI):UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,UsdShade.Tokens.weakerThanDescendants,'physics')
  from isaacsim.core.api.robots import Robot
  self.mechanism=self.world.scene.add(Robot(prim_path='/World/Mechanism/Base',name='task_mechanism'))
  self.world.reset();self.indices={n:i for i,n in enumerate(self.mechanism.dof_names)}
  print('MECHANISM_DOFS',self.mechanism.dof_names,flush=True)
  self.unlocked=False;self.initial_joints=np.zeros(len(self.joint_names));self.shift=np.zeros(3)
  self.config=dict(task=task,asset_source='procedural PhysX analogue; not converted native RLBench asset',
   action='measured joint delta and discrete gripper',state_dim=120,phase_information=False,
   success=f'abs joint displacement > {self.success_threshold}; pinned RLBench JointCondition',
   door_unlock='abs lever displacement > 25 degrees before hinge released' if task=='open_door' else None,
   reward='success 1 otherwise 0',arm_pd_gains=[8000,150],finger_pd_gains=[4000,100],horizon=self.horizon,grasp_attachment=False,xy_randomization_m=.02,
   variation='bottom drawer analogue only' if task=='open_drawer' else 0,
   remaining_differences=['procedural geometry, inertia and friction','unmatched native cameras','XY +/-2cm, no native base rotation randomization','native asset variant correspondence unverified'])
 def body(self,name,position,mass):
  p=UsdGeom.Xform.Define(self.world.stage,'/World/Mechanism/'+name);p.AddTranslateOp().Set(Gf.Vec3d(*position))
  UsdPhysics.RigidBodyAPI.Apply(p.GetPrim());UsdPhysics.MassAPI.Apply(p.GetPrim()).CreateMassAttr(mass)
  return p
 def box(self,path,size,pos,color):return self._box('/World/Mechanism/'+path,size,pos,color)
 def knob(self,path,pos):return self._cylinder('/World/Mechanism/'+path,.022 if self.task_name=='open_drawer' else .014,.035,pos,(.85,.24,.06))
 def link(self,j,a,b):
  j.CreateBody0Rel().SetTargets(['/World/Mechanism/'+a]);j.CreateBody1Rel().SetTargets(['/World/Mechanism/'+b]);j.CreateCollisionEnabledAttr(False)
 def pose(self,body):
  m=UsdGeom.XformCache().GetLocalToWorldTransform(self.world.stage.GetPrimAtPath('/World/Mechanism/'+body))
  return np.array(m).T
 def knob_position(self):
  m=self.pose(self.grip_body);return m[:3,:3]@self.grip_local+m[:3,3]
 def joint_positions(self):return np.array([self.mechanism.get_joint_positions()[self.indices[n]] for n in self.joint_names])
 def _drive(self,steps=6):
  super()._drive(steps)
  if hasattr(self,'mechanism') and self.task_name=='open_door' and not self.unlocked:
   if abs(self.joint_positions()[1]-self.initial_joints[1])>np.deg2rad(25):
    self.unlocked=True;self.lock_drive.GetStiffnessAttr().Set(0.);self.lock_drive.GetDampingAttr().Set(.02)
 def reset(self,*args,**kwargs):
  self.unlocked=False
  if self.task_name=='open_door':self.lock_drive.GetStiffnessAttr().Set(10000.);self.lock_drive.GetDampingAttr().Set(100.)
  print('TASK_RESET_WORLD_BEGIN',self.task_name,flush=True)
  self.world.reset();print('TASK_RESET_WORLD_DONE',flush=True);self.target=self.home.copy();self.robot.set_joint_positions(self.target);self.robot.set_joint_velocities(np.zeros_like(self.target))
  self.shift=np.r_[self.rng.uniform(-.02,.02,2),0.];pos=self.base_position+self.shift
  self.rootfix.GetLocalPos0Attr().Set(Gf.Vec3f(*pos));self.mechanism.set_world_pose(position=pos,orientation=np.array([1.,0,0,0]))
  self.mechanism.set_joint_positions(np.zeros(self.mechanism.num_dof));self.mechanism.set_joint_velocities(np.zeros(self.mechanism.num_dof))
  # Passive mechanism joints; Panda alone receives motor commands.
  kp=np.array([8000. if i in self.arm else 4000. for i in range(self.robot.num_dof)]);kd=np.array([150. if i in self.arm else 100. for i in range(self.robot.num_dof)])
  self.robot.get_articulation_controller().set_gains(kps=kp,kds=kd)
  print('TASK_RESET_SETTLE',flush=True)
  self.initial_joints=np.zeros(len(self.joint_names));self._drive(30)
  self.mechanism.set_joint_positions(np.zeros(self.mechanism.num_dof));self.mechanism.set_joint_velocities(np.zeros(self.mechanism.num_dof));self._drive(120)
  print('TASK_RESET_SETTLED',self.joint_positions(),flush=True)
  assert abs(self.joint_positions()[0])<.003,'mechanism reset is not closed'
  self.initial_joints=np.zeros(len(self.joint_names));self.t=0;self.done=False;self.attached=False
  self.rgb_frames.clear();self.states.clear();self._render();return self.observe()
 def images(self,kind="rgb"):
  # A valid wrist view can be almost uniform when occluded by a drawer.
  # Validate actual render buffers (shape/alpha), not scene texture variance.
  from PIL import Image
  import json
  images={};buffers={}
  for _ in range(20):
   buffers={n:np.asarray(self.annotators[n]['preview'].get_data()).copy() for n in self.camera_names}
   valid=all(v.ndim==3 and v.shape[:2]==(self.preview_resolution[1],self.preview_resolution[0]) and v.shape[2]>=3 and
     (v.shape[2]<4 or np.any(v[:,:,3]>0)) for v in buffers.values())
   if valid and any(np.any(v[:,:,:3]) for v in buffers.values()):break
   self._render()
  else:raise RuntimeError('camera render buffer missing or all views uninitialized')
  for name,v in buffers.items():
   rgb=v[:,:,:3].astype(np.uint8);images[name]=rgb
   if rgb.std()<3 and hasattr(self,'diagnostics_dir'):
    directory=self.diagnostics_dir;directory.mkdir(exist_ok=True)
    index=getattr(self,'low_contrast_saved',0)
    if index<8:
     Image.fromarray(rgb).save(directory/f'{name}_{index}.png')
     (directory/f'{name}_{index}.json').write_text(json.dumps(dict(camera=name,mean=float(rgb.mean()),std=float(rgb.std()),minimum=int(rgb.min()),maximum=int(rgb.max()),alpha_max=int(v[:,:,3].max()) if v.shape[2]>3 else None,step=self.t)))
     self.low_contrast_saved=index+1
  if kind=='rgb':images={n:np.asarray(Image.fromarray(v).resize((84,84),Image.Resampling.BILINEAR)) for n,v in images.items()}
  return images
 def status(self):
  q=np.asarray(self.robot.get_joint_positions());tcp=self.fk(q[self.arm])[:3,3];j=self.joint_positions()
  delta=j-self.initial_joints;success=bool(abs(delta[0])>self.success_threshold)
  return dict(task=self.task_name,step=self.t,tcp=tcp.tolist(),handle=self.knob_position().tolist(),joint_positions=j.tolist(),joint_displacements=delta.tolist(),
   success=success,failed=False,finger_positions_m=q[self.fingers].tolist(),unlocked=self.unlocked,attached=False,lift_m=0.,opening_m=float(abs(delta[0])) if self.task_name=='open_drawer' else None,
   opening_deg=float(np.rad2deg(abs(delta[0]))) if self.task_name=='open_door' else None,
   handle_tcp_distance_m=float(np.linalg.norm(self.knob_position()-tcp)),sim_time=float(self.world.current_time))
 def step_raw(self,raw):
  from isaacsim.core.utils.types import ArticulationAction
  if self.done:raise RuntimeError('reset required')
  raw=np.asarray(raw,np.float32);assert raw.shape==(8,) and np.isfinite(raw).all()
  q=np.asarray(self.robot.get_joint_positions()).copy();before=q[self.arm].copy()
  self.target[self.arm]=np.clip(before+raw[:7],self.lower,self.upper);applied=self.target[self.arm]-before;self._drive(6)
  q=np.asarray(self.robot.get_joint_positions());self.target[self.arm]=q[self.arm]
  desired=bool(raw[7]>.5);opened=bool(np.all(q[self.fingers]>.036))
  target_open=bool(np.all(self.target[self.fingers]>.036))
  if desired!=opened or desired!=target_open:
   self.target[self.fingers]=.04 if desired else 0.;previous=q[self.fingers].copy();stationary=0
   for _ in range(40):
    self._drive(6);f=np.asarray(self.robot.get_joint_positions())[self.fingers];stationary=stationary+1 if np.max(abs(f-previous))<1e-5 else 0;previous=f.copy()
    if (desired and min(f)>.039) or (not desired and (max(f)<.001 or stationary>=3)):break
   if desired:self._drive(60)
  self.t+=1;self._render();info=self.status();self.done=info['success'] or self.t>=self.horizon
  info.update(applied_delta=applied.tolist(),measured_before=before.tolist(),requested_raw_action=raw.tolist())
  return self.observe(),float(info['success']),bool(info['success']),bool(self.t>=self.horizon and not info['success']),info
 def expert_raw_actions(self):
  from fast_panda_ik import FastPandaIK
  from panda_env import ROOT
  solver=FastPandaIK(ROOT/'assets/robot/panda/panda.urdf',self.lower,self.upper)
  down=Rotation.from_euler('xyz',[np.pi,0,0]).as_matrix()
  if self.task_name=='open_door':down=Rotation.from_euler('y',-np.pi/4).as_matrix()@down@Rotation.from_euler('z',-np.pi/2).as_matrix()
  def fallback(position,seed):
   from scipy.optimize import least_squares
   def residual(q):
    m=self.fk(q);return np.r_[(m[:3,3]-position)*5,Rotation.from_matrix(solver.rotation@m[:3,:3].T).as_rotvec()]
   result=least_squares(residual,seed,bounds=(self.lower,self.upper),max_nfev=300)
   assert np.linalg.norm(residual(result.x))<.01,'expert pose IK unreachable'
   return result.x
  solver.fallback=fallback
  def move(goal,rotation,opened,tolerance=.008):
   for _ in range(self.horizon):
    q=np.asarray(self.robot.get_joint_positions())[self.arm];actual=self.fk(q);err=goal()-actual[:3,3]
    position=actual[:3,3]+err*min(1.,.008/max(np.linalg.norm(err),1e-9));solver.rotation=rotation()
    desired=solver(position,q);delta=desired-q;delta*=min(1.,.18/max(np.max(abs(delta)),1e-9))
    yield np.r_[delta,opened].astype(np.float32)
    if self.done:return
    if self.expert_phase=='unlock' and self.unlocked:return
    if np.linalg.norm(goal()-self.fk(np.asarray(self.robot.get_joint_positions())[self.arm])[:3,3])<tolerance:return
  self.expert_phase='approach';grip=self.knob_position().copy()
  yield from move(lambda:grip+(-down[:,2]*.12 if self.task_name=='open_door' else np.array([0,0,.12])),lambda:down,1.)
  self.expert_phase='descend';yield from move(lambda:grip,lambda:down,1.,.006)
  self.expert_phase='close';yield np.r_[np.zeros(7),0.].astype(np.float32)
  if self.task_name=='open_drawer':
   self.expert_phase='pull';goal=grip+np.array([-.19,0,0]);yield from move(lambda:goal,lambda:down,0.,.005)
  else:
   self.expert_phase='unlock';theta=-np.deg2rad(40);pivot=self.base_position+self.shift+self.handle_pivot
   goal=pivot+Rotation.from_rotvec([theta,0,0]).apply(self.grip_local)
   yield from move(lambda:goal,lambda:Rotation.from_rotvec([theta,0,0]).as_matrix()@down,0.,.004)
   if not self.unlocked:return
   # Native unlock is latched: release the lever, then push the panel by contact.
   self.expert_phase='release_handle';yield np.r_[np.zeros(7),1.].astype(np.float32)
   hinge=self.base_position+self.shift
   def panel_rotation():return Rotation.from_rotvec([0,0,self.joint_positions()[0]]).as_matrix()
   self.expert_phase='push_approach'
   yield from move(lambda:hinge+panel_rotation()@np.array([-.08,.16,.18]),lambda:panel_rotation()@down,1.,.008)
   self.expert_phase='push'
   yield from move(lambda:hinge+panel_rotation()@np.array([.025,.16,.18]),lambda:panel_rotation()@down,1.,.004)
