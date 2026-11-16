"""RLBench's latched handle-unlock and hinge-success semantics for converted USD.
Motor dynamics are an explicit PhysX mapping, not a Coppelia controller replica.
"""
import numpy as np
from pxr import UsdPhysics
class NativeDoorLogic:
 def __init__(self,stage,mechanism,root):
  self.mechanism=mechanism;self.indices={n:i for i,n in enumerate(mechanism.dof_names)}
  self.drive=UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath(root+'/door_frame_joint'),'angular')
  self.reset()
 def reset(self):
  self.reference=self.positions().copy();self.unlocked=False
  self.drive.GetTargetPositionAttr().Set(0.);self.drive.GetStiffnessAttr().Set(50000.);self.drive.GetDampingAttr().Set(100.)
 def positions(self):
  q=self.mechanism.get_joint_positions();return np.array([q[self.indices[n]] for n in ['door_frame_joint','door_handle_joint']])
 def step(self):
  displacement=self.positions()-self.reference
  if not self.unlocked and abs(displacement[1])>np.deg2rad(25):
   self.unlocked=True;self.drive.GetStiffnessAttr().Set(0.);self.drive.GetDampingAttr().Set(.01)
  return dict(unlocked=self.unlocked,success=bool(abs(displacement[0])>np.deg2rad(25)),hinge_degrees=float(np.rad2deg(displacement[0])),handle_degrees=float(np.rad2deg(displacement[1])))
