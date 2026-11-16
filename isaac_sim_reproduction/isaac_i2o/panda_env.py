"""Isaac Sim Panda cube-lift environment. Import after SimulationApp startup."""
import os
import json
from pathlib import Path
from collections import deque
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from isaacsim.core.utils.types import ArticulationAction
ROOT = Path(os.environ.get("PINGANDOG_ROOT", "/home/ubuntu/pingandog"))

class PandaLiftEnv:
    def __init__(self, frame_stack=2, horizon=220, seed=1, randomization=0.02):
        import numpy as np
        import omni.kit.app
        from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, Gf
        from isaacsim.core.api import World
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.types import ArticulationAction
        tree = ET.parse(ROOT / "assets/robot/panda/panda.urdf")
        usd = json.loads((ROOT / "assets/robot/panda/import.json").read_text())["usd"]
        print("PANDA_USD", usd, flush=True)
        world = World(stage_units_in_meters=1.0, physics_dt=1/120, rendering_dt=1/30)
        stage = world.stage
        ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
        ground.CreatePointsAttr([(-3,-3,0),(3,-3,0),(3,3,0),(-3,3,0)])
        ground.CreateFaceVertexCountsAttr([4]); ground.CreateFaceVertexIndicesAttr([0,1,2,3])
        ground.CreateDisplayColorAttr([(0.18,0.22,0.28)])
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1200)
        add_reference_to_stage(usd, "/World/Panda")
        root = stage.GetPrimAtPath("/World/Panda")
        if root.GetVariantSet("Physics").GetVariantNames(): root.GetVariantSet("Physics").SetVariantSelection("physx")
        robot = world.scene.add(Robot(prim_path="/World/Panda", name="panda"))
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        UsdGeom.Xformable(camera).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.8,-1.8,1.3), Gf.Vec3d(0.2,0,0.5), Gf.Vec3d(0,0,1)).GetInverse())
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from pxr import UsdShade, PhysxSchema
        table = world.scene.add(FixedCuboid(prim_path="/World/Table", name="table", position=np.array([0.5,0,0.1]), scale=np.array([0.5,0.5,0.2]), color=np.array([0.35,0.4,0.45])))
        cube = world.scene.add(DynamicCuboid(prim_path="/World/Cube", name="cube", position=np.array([0.45,0,0.222]), scale=np.array([0.04]*3), mass=0.05, color=np.array([0.9,0.16,0.04])))
        material = UsdShade.Material.Define(stage, "/World/GraspMaterial")
        mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        mat.CreateStaticFrictionAttr(1.0); mat.CreateDynamicFrictionAttr(0.8); mat.CreateRestitutionAttr(0.0)
        for prim in list(Usd.PrimRange(root)) + [stage.GetPrimAtPath("/World/Cube")]:
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
        world.reset()
        from scipy.spatial.transform import Rotation
        from scipy.optimize import least_squares
        import omni.replicator.core as rep
        from PIL import Image
        names = list(robot.dof_names)
        arm = [names.index("panda_joint"+str(i)) for i in range(1,8)]
        fingers = [i for i,n in enumerate(names) if "finger_joint" in n]
        joints_by_name = {j.get("name"): j for j in tree.getroot().findall("joint")}
        chain = [joints_by_name["panda_joint%d"%i] for i in range(1,9)] + [joints_by_name["panda_hand_joint"]]
        origins=[]; limits=[]
        for j in chain:
            org=j.find("origin"); t=np.eye(4)
            t[:3,3]=np.fromstring(org.get("xyz","0 0 0"),sep=" ")
            t[:3,:3]=Rotation.from_euler("xyz",np.fromstring(org.get("rpy","0 0 0"),sep=" ")).as_matrix()
            origins.append(t)
            if j.get("type")=="revolute": limits.append([float(j.find("limit").get("lower")),float(j.find("limit").get("upper"))])
        def fk(q):
            t=np.eye(4)
            for i,org in enumerate(origins):
                t=t@org
                if i<7:
                    r=np.eye(4);r[:3,:3]=Rotation.from_rotvec([0,0,q[i]]).as_matrix();t=t@r
            tip=np.eye(4);tip[2,3]=0.1034
            return t@tip
        target_rot=Rotation.from_euler("xyz",[np.pi,0,0]).as_matrix()
        qhome=np.array([0,-0.5,0,-2,0,1.6,0.7])
        lower=np.array(limits)[:,0]+0.01; upper=np.array(limits)[:,1]-0.01
        def ik(pos,seed):
            def residual(q):
                t=fk(q)
                return np.r_[(t[:3,3]-pos)*5,Rotation.from_matrix(target_rot@t[:3,:3].T).as_rotvec()]
            sol=least_squares(residual,np.clip(seed,lower,upper),bounds=(lower,upper),max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            assert np.linalg.norm(fk(sol.x)[:3,3]-pos)<0.002, "IK unreachable"
            return sol.x
        q=np.zeros(robot.num_dof); q[arm]=qhome;q[fingers]=0.035
        robot.set_joint_positions(q)
        robot.get_articulation_controller().set_gains(kps=np.array([800. if i in arm else 1000. for i in range(robot.num_dof)]),kds=np.array([40. if i in arm else 50. for i in range(robot.num_dof)]))
        self.world, self.robot, self.cube = world, robot, cube
        self.arm, self.fingers = arm, fingers
        self.fk, self.ik = fk, ik
        self.lower, self.upper, self.home = lower, upper, q.copy()
        self.stack, self.horizon = frame_stack, horizon
        self.rng = np.random.default_rng(seed)
        self.randomization = randomization
        self.rgb_frames, self.states = deque(maxlen=frame_stack), deque(maxlen=frame_stack)
        self.rp = rep.create.render_product("/World/Camera", (84,84))
        self.ann = rep.AnnotatorRegistry.get_annotator("rgb")
        self.ann.attach(self.rp)
        self.delta = 0.04
        self.rgb_shape = (1,3*frame_stack,84,84)
        self.state_shape = (15*frame_stack,)
        self.action_shape = (8,)

    def _drive(self, steps=12):
        for k in range(steps):
            q = np.asarray(self.robot.get_joint_positions())
            self.bias = np.clip(self.bias+(self.target[self.arm]-q[self.arm])*0.003,-0.08,0.08)
            cmd = self.target.copy()
            cmd[self.arm] = np.clip(cmd[self.arm]+self.bias,self.lower,self.upper)
            self.robot.apply_action(ArticulationAction(joint_positions=cmd))
            self.world.step(render=(k==steps-1))

    def observe(self):
        rgb = np.asarray(self.ann.get_data())
        for _ in range(60):
            if rgb.shape[:2] == (84,84) and rgb.ndim == 3 and rgb[...,:3].std()>3:
                break
            self._drive(1)
            rgb = np.asarray(self.ann.get_data())
        if rgb.shape[:2] != (84,84) or rgb.ndim!=3 or rgb[...,:3].std()<=3:
            raise RuntimeError("Panda camera returned invalid or blank RGB")
        rgb = rgb[...,:3].astype(np.uint8).transpose(2,0,1).copy()
        q = np.asarray(self.robot.get_joint_positions())
        tcp = self.fk(q[self.arm])
        # No object pose enters the policy observation; it is used only by teacher/reward.
        low = np.r_[q[self.arm], q[self.fingers].mean()/0.04,
                    tcp[:3,3], Rotation.from_matrix(tcp[:3,:3]).as_quat()].astype(np.float32)
        if not self.rgb_frames:
            for _ in range(self.stack):
                self.rgb_frames.append(rgb.copy()); self.states.append(low.copy())
        else:
            self.rgb_frames.append(rgb); self.states.append(low)
        obs = (np.concatenate(list(self.rgb_frames),axis=0)[None],np.concatenate(list(self.states)))
        assert obs[0].shape == self.rgb_shape and obs[1].shape == self.state_shape
        assert np.isfinite(obs[1]).all()
        return obs

    def reset(self):
        self.world.reset()
        self.target=self.home.copy(); self.bias=np.zeros(7)
        self.robot.set_joint_positions(self.target)
        self.robot.set_joint_velocities(np.zeros_like(self.target))
        pos=np.array([0.45,0,0.222]);pos[:2]+=self.rng.uniform(-self.randomization,self.randomization,2)
        self.cube.set_world_pose(position=pos,orientation=np.array([1.,0,0,0]))
        self.cube.set_linear_velocity(np.zeros(3));self.cube.set_angular_velocity(np.zeros(3))
        self.robot.get_articulation_controller().set_gains(
            kps=np.array([800. if i in self.arm else 1000. for i in range(self.robot.num_dof)]),
            kds=np.array([40. if i in self.arm else 50. for i in range(self.robot.num_dof)]))
        self._drive(120)
        self.initial_cube=np.asarray(self.cube.get_world_pose()[0]).copy()
        self.t=0;self.held=0;self.done=False
        self.rgb_frames.clear();self.states.clear()
        return self.observe()

    def step(self,action):
        if self.done:raise RuntimeError("reset required after episode end")
        action=np.asarray(action,dtype=np.float32)
        assert action.shape==(8,) and np.isfinite(action).all()
        action=np.clip(action,-1,1)
        self.target[self.arm]=np.clip(self.target[self.arm]+self.delta*action[:7],self.lower,self.upper)
        self.target[self.fingers]=0.02*(action[7]+1)
        self._drive(12)
        self.t+=1
        obs=self.observe()
        pos=np.asarray(self.cube.get_world_pose()[0]);q=np.asarray(self.robot.get_joint_positions())
        tcp=self.fk(q[self.arm])[:3,3]
        lift=float(pos[2]-self.initial_cube[2])
        grasped=lift>0.12 and np.linalg.norm(pos-tcp)<0.06 and q[self.fingers].mean()<0.03
        self.held=self.held+1 if grasped else 0
        success=self.held>=20
        failed=pos[2]<0.1 or np.linalg.norm(pos[:2]-self.initial_cube[:2])>0.3
        terminated=bool(success or failed);truncated=self.t>=self.horizon and not terminated
        self.done=terminated or truncated
        info={"success":bool(success),"failed":bool(failed),"lift_m":lift,"held_steps":self.held,
              "cube":pos.tolist(),"tcp":tcp.tolist(),"step":self.t}
        return obs,float(success),terminated,bool(truncated),info

    def expert_actions(self):
        # Teacher alone may read simulator object pose and solve IK.
        p=self.initial_cube.copy();above=p+np.array([0,0,0.17]);grasp=p+np.array([0,0,0.005])
        pre=self.ik(above,self.home[self.arm]);down=self.ik(grasp,pre)
        up=self.ik(grasp+np.array([0,0,0.18]),down)
        waypoints=[(pre,0.035,30),(down,0.035,40),(down,0.,20),(up,0.,40),(up,0.,30)]
        previous=self.target.copy()
        for joints,opening,steps in waypoints:
            target=previous.copy();target[self.arm]=joints;target[self.fingers]=opening
            for k in range(steps):
                alpha=min(1.,(k+1)/(steps*0.65));alpha=alpha*alpha*(3-2*alpha)
                desired=previous+(target-previous)*alpha
                action=np.r_[(desired[self.arm]-self.target[self.arm])/self.delta,
                             desired[self.fingers].mean()/0.02-1]
                yield np.clip(action,-1,1).astype(np.float32)
            previous=target.copy()

    def save_image(self,path):
        from PIL import Image
        a=np.asarray(self.ann.get_data())[...,:3].astype(np.uint8)
        Image.fromarray(a).save(path)

    def close(self):
        self.ann.detach(self.rp)
