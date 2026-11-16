"""Measured-joint delta and discrete gripper adapter, without task-phase inputs.

The procedural Isaac assets and sensor volumes remain proxies, not RLBench assets.
Import only after SimulationApp. Native reference: README-pinned RLBench commit.
"""
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation
from pxr import UsdGeom,UsdPhysics,Gf
from saucepan_env import SaucepanEnv

class AlignedLidEnv(SaucepanEnv):
    def __init__(self,frame_stack=8,horizon=100,seed=1):
        super().__init__(frame_stack,horizon,seed,.02,preview_resolution=(168,126))
        self.home[self.fingers]=.04
        self.codec=None;self.grasp_path='/World/AlignedGrasp';self.attached=False
        self.config.update(interface='measured_joint_delta_discrete_gripper_v1',
          arm_scene_dt_s=.05,arm_scene_physics_substeps=6,command_integrator=False,
          integral_bias=False,gravity_compensation='none; calibrated PD arm gains 4000/100',gripper_state='binary fully open',policy_phase_information=False,
          grasp='RLBench-style attachment PROXY, local TCP box half extents .025,.025,.035 m',
          success='PROXY: attached lid in box initial_lid + [0,0,.20], half extents [.05,.05,.05]',
          remaining_differences=['procedural assets','unmatched native camera calibration',
            'proxy grasp/success volumes','PhysX PD motor dynamics versus CoppeliaSim'])
    def _drive(self,steps=6):
        from isaacsim.core.utils.types import ArticulationAction
        for _ in range(steps):
            self.robot.apply_action(ArticulationAction(joint_positions=self.target))
            self.world.step(render=False)
    def observe(self):
        raw=self.images();rgb=np.stack([raw[k].transpose(2,0,1) for k in self.camera_names])
        q=np.asarray(self.robot.get_joint_positions());tcp=self.fk(q[self.arm])
        low=np.r_[q[self.arm],float(np.all(q[self.fingers]>.036)),tcp[:3,3],
                  Rotation.from_matrix(tcp[:3,:3]).as_quat()].astype(np.float32)
        if not self.rgb_frames:
            for _ in range(self.stack):self.rgb_frames.append(rgb.copy());self.states.append(low.copy())
        else:self.rgb_frames.append(rgb);self.states.append(low)
        return np.concatenate(list(self.rgb_frames),axis=1),np.concatenate(list(self.states))
    def _release(self):
        if self.world.stage.GetPrimAtPath(self.grasp_path).IsValid():self.world.stage.RemovePrim(self.grasp_path)
        self.attached=False
    def reset(self,*args,**kwargs):
        self._release();super().reset(*args,**kwargs)
        self.robot.get_articulation_controller().set_gains(
            kps=np.array([4000. if i in self.arm else 1000. for i in range(self.robot.num_dof)]),
            kds=np.array([100. if i in self.arm else 50. for i in range(self.robot.num_dof)]))
        self._drive(60);self._render();self.rgb_frames.clear();self.states.clear()
        return self.observe()
    def _grasp_in_volume(self):
        q=np.asarray(self.robot.get_joint_positions());tcp=self.fk(q[self.arm])
        local=tcp[:3,:3].T@(self.knob_position()-tcp[:3,3])
        return bool(np.all(np.abs(local)<=np.array([.025,.025,.035])))
    def _try_grasp(self):
        q=np.asarray(self.robot.get_joint_positions());tcp=self.fk(q[self.arm])
        local=tcp[:3,:3].T@(self.knob_position()-tcp[:3,3])
        if not self._grasp_in_volume():return False
        cache=UsdGeom.XformCache();stage=self.world.stage
        hand=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(self.wrist_link_path))
        lid=cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Lid'))
        relative=lid*hand.GetInverse();quat=relative.ExtractRotationQuat()
        joint=UsdPhysics.FixedJoint.Define(stage,self.grasp_path)
        joint.CreateBody0Rel().SetTargets([self.wrist_link_path]);joint.CreateBody1Rel().SetTargets(['/World/Lid'])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*relative.ExtractTranslation()))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(quat.GetReal(),Gf.Vec3f(*quat.GetImaginary())))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0));joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))
        joint.CreateCollisionEnabledAttr().Set(False);self.attached=True
        return True
    def step_raw(self,raw):
        if self.done:raise RuntimeError('reset required')
        raw=np.asarray(raw,np.float32);assert raw.shape==(8,) and np.isfinite(raw).all()
        q=np.asarray(self.robot.get_joint_positions()).copy()
        measured_before=q[self.arm].copy()
        requested=measured_before+raw[:7]
        self.target[self.arm]=np.clip(requested,self.lower,self.upper)
        applied_delta=self.target[self.arm]-q[self.arm]
        self._drive(6)
        # Match JointPosition.action_post_step: stop arm at newly measured position.
        q=np.asarray(self.robot.get_joint_positions()).copy();self.target[self.arm]=q[self.arm]
        desired_open=float(raw[-1]>.5);current_open=float(np.all(q[self.fingers]>.036))
        grasp_event=False
        if current_open!=desired_open:
            if not desired_open:grasp_event=self._try_grasp()
            else:self._release()
            self.target[self.fingers]=.04 if desired_open else 0.
            stationary=0;previous=q[self.fingers].copy()
            for _ in range(40):
                self._drive(6);f=np.asarray(self.robot.get_joint_positions())[self.fingers]
                stationary=stationary+1 if np.max(np.abs(f-previous))<1e-5 else 0
                previous=f.copy()
                if (desired_open and np.min(f)>.039) or (not desired_open and (np.max(f)<.001 or stationary>=3)):break
            if desired_open:self._drive(60)
        self._render();self.t+=1
        info=self.status();goal=self.initial_lid+np.array([0,0,.20])
        detected=bool(np.all(np.abs(np.array(info['lid'])-goal)<np.array([.05,.05,.05])))
        success=bool(self.attached and detected)
        failed=info['lid'][2]<.18 or info['xy_offset_m']>.25
        terminated=success or failed;truncated=self.t>=self.horizon and not terminated
        self.done=bool(terminated or truncated)
        info.update(success=success,failed=bool(failed),attached=self.attached,grasp_event=grasp_event,
          detected=detected,requested_raw_action=raw.tolist(),applied_delta=applied_delta.tolist(),
          measured_before=measured_before.tolist(),sim_time=float(self.world.current_time))
        return self.observe(),float(success),bool(terminated),bool(truncated),info
    def step(self,normalized):
        assert self.codec is not None
        return self.step_raw(self.codec.decode(normalized))
    def expert_raw_actions(self):
        # Native-style motion-planning teacher: IK generates a smooth TCP path,
        # but the learner and recorded actions remain measured-joint deltas.
        from fast_panda_ik import FastPandaIK
        from panda_env import ROOT
        solver=FastPandaIK(ROOT/'assets/robot/panda/panda.urdf',self.lower,self.upper,fallback=self.ik)
        knob=self.knob_position();grasp=knob+np.array([0,0,.003])
        for phase,position,opened in [('approach',grasp+np.array([0,0,.16]),1.),
          ('descend',grasp,1.),('close',grasp,0.),('lift',grasp+np.array([0,0,.20]),0.)]:
            for _ in range(self.horizon):
                self.expert_phase=phase
                q=np.asarray(self.robot.get_joint_positions())[self.arm]
                actual=self.fk(q)[:3,3]
                if phase=='close':delta=np.zeros(7)
                else:
                    error=position-actual
                    next_position=actual+error*min(1.,.008/max(np.linalg.norm(error),1e-9))
                    desired=solver(next_position,q)
                    delta=desired-q
                    delta*=min(1.,.18/max(np.max(np.abs(delta)),1e-9))
                yield np.r_[delta,opened].astype(np.float32)
                if self.done:return
                actual=self.fk(np.asarray(self.robot.get_joint_positions())[self.arm])[:3,3]
                if phase=='close':
                    if self.attached:break
                elif phase=='descend' and self._grasp_in_volume():break
                elif np.linalg.norm(actual-position)<.009:break
