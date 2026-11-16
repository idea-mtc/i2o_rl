"""Procedural take-lid-off-saucepan scene with real wrist + three fixed cameras.

This is an Isaac Sim task analogue, not a conversion of the RLBench .ttm asset.
Import after SimulationApp startup. The previously validated Panda and controller
are reused without changing the cube environment.
"""
from pathlib import Path
from collections import deque
import json
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdLux, Gf
from isaacsim.core.prims import SingleRigidPrim
from panda_env import PandaLiftEnv, ROOT


class SaucepanEnv(PandaLiftEnv):
    camera_names = ("front", "wrist", "left_shoulder", "right_shoulder")

    def __init__(self, frame_stack=2, horizon=260, seed=1, randomization=0.02,
                 preview_resolution=(640, 480)):
        super().__init__(frame_stack, horizon, seed, randomization)
        # Build new rigid bodies while stopped; the base scene already reset once.
        self.world.stop()
        self.world.stage.SetDefaultPrim(self.world.stage.GetPrimAtPath("/World"))
        import omni.replicator.core as rep
        self.rep = rep
        self.ann.detach(self.rp)
        self.rp.destroy()
        self.world.scene.remove_object("cube")
        self.world.stage.RemovePrim("/World/Camera")
        stage = self.world.stage
        self.table_top = 0.2
        self.pot_height = 0.083
        self.lid_mass = 0.2
        self.pot_radius = 0.082
        self.pot_xy = np.array([0.46, 0.0])
        self.pot_root = UsdGeom.Xform.Define(stage, "/World/Saucepan")
        self.pot_translation = self.pot_root.AddTranslateOp()
        self.pot_rotation = self.pot_root.AddRotateZOp()
        self.pot_translation.Set(Gf.Vec3d(0.46, 0, self.table_top))
        self.pot_rotation.Set(0.)
        self._cylinder("/World/Saucepan/Base", 0.082, 0.008, (0,0,0.004), (0.28,0.32,0.37))
        # Hollow annular wall: separate convex boxes, no solid cylinder plug.
        for i in range(32):
            a = i*2*np.pi/32
            wall = self._box(f"/World/Saucepan/Wall_{i:02d}",
                (0.008, 2*np.pi*0.078/32*1.015, 0.075),
                (0.078*np.cos(a), 0.078*np.sin(a), 0.0455), (0.38,0.43,0.49))
            UsdGeom.Xformable(wall).AddRotateZOp().Set(float(np.degrees(a)))
            ops=UsdGeom.Xformable(wall).GetOrderedXformOps()
            UsdGeom.Xformable(wall).SetXformOpOrder([ops[0],ops[2],ops[1]])
        self._box("/World/Saucepan/Handle", (0.16,0.028,0.022),
                  (0.152,0,0.066), (0.07,0.08,0.09))
        self._box("/World/Saucepan/HandleMount", (0.035,0.035,0.024),
                  (0.083,0,0.066), (0.38,0.43,0.49))
        lid_root = UsdGeom.Xform.Define(stage, "/World/Lid")
        self._cylinder("/World/Lid/Disc", 0.084, 0.008, (0,0,0), (0.65,0.70,0.76))
        self._cylinder("/World/Lid/Stem", 0.008, 0.018, (0,0,0.013), (0.09,0.1,0.11))
        self._cylinder("/World/Lid/Knob", 0.018, 0.025, (0,0,0.032), (0.85,0.23,0.055))
        # All three shapes form one freely moving rigid body. No joint to gripper/pot.
        UsdPhysics.RigidBodyAPI.Apply(lid_root.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(lid_root.GetPrim())
        mass_api.CreateMassAttr(self.lid_mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0,0,0.004))
        # Explicit plausible compound inertia in SI (dominant thin disc + light knob).
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.00030,0.00030,0.00059))
        self.lid = self.world.scene.add(SingleRigidPrim(
            prim_path="/World/Lid", name="saucepan_lid", mass=self.lid_mass,
            position=np.array([0.46,0,self.table_top+self.pot_height+0.006])))
        self.cube = self.lid  # inherited control helper does not use object shape.
        mat = UsdShade.Material.Get(stage, "/World/GraspMaterial")
        # Bind the same material used by the already-tested physical cube grasp.
        for prim in Usd.PrimRange(lid_root.GetPrim()):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        for prim in stage.Traverse():
            if prim.GetName() in ("panda_leftfinger", "panda_rightfinger"):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    mat, UsdShade.Tokens.strongerThanDescendants, "physics")
        wrist = next(p for p in stage.Traverse() if p.GetName()=="panda_link7")
        self.wrist_link_path = str(wrist.GetPath())
        self.camera_paths = {}
        fixed = {
            "front": ((1.35,0,0.95), (0.43,0,0.30)),
            "left_shoulder": ((0.35,1.05,0.95),(0.43,0,0.30)),
            "right_shoulder": ((0.35,-1.05,0.95),(0.43,0,0.30)),
        }
        for name,(eye,target) in fixed.items():
            path=f"/World/Cameras/{name}"
            camera=UsdGeom.Camera.Define(stage,path)
            UsdGeom.Xformable(camera).AddTransformOp().Set(
                Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),
                                      Gf.Vec3d(0,0,1)).GetInverse())
            self._camera_optics(camera)
            self.camera_paths[name]=path
        path=self.wrist_link_path+"/WristCamera"
        camera=UsdGeom.Camera.Define(stage,path)
        # URDF link7 -> panda_hand: translate +0.107 Z, rotate -pi/4 Z.
        hand=np.eye(4);hand[:3,:3]=Rotation.from_euler("z",-np.pi/4).as_matrix()
        hand[2,3]=0.107
        local_camera=Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(-0.070,0,0.015),Gf.Vec3d(0,0,0.145),Gf.Vec3d(0,1,0)).GetInverse()
        # Gf uses row-vector transforms, numpy FK uses column-vector transforms.
        mount=Gf.Matrix4d(*hand.T.flatten().tolist())
        UsdGeom.Xformable(camera).AddTransformOp().Set(local_camera*mount)
        self._camera_optics(camera)
        self.camera_paths["wrist"]=path
        self.preview_resolution=preview_resolution
        self.products={};self.annotators={}
        for name in self.camera_names:
            self.products[name]={};self.annotators[name]={}
            # One native frame per camera: derive 84x84 observations from that
            # same image, avoiding duplicate-camera Hydra product invalidation.
            for kind,size in (("preview",preview_resolution),):
                product=rep.create.render_product(self.camera_paths[name],size,
                    force_new=True,name="saucepan_"+name+"_"+kind)
                annotator=rep.AnnotatorRegistry.get_annotator("rgb")
                annotator.attach(product)
                self.products[name][kind]=product;self.annotators[name][kind]=annotator
        self.rgb_frames=deque(maxlen=frame_stack);self.states=deque(maxlen=frame_stack)
        self.rgb_shape=(4,3*frame_stack,84,84)
        self.frame_counter=0
        self.config={"task":"take_lid_off_saucepan", "asset_source":"project procedural geometry, not RLBench ttm",
            "physics":"PhysX", "physics_hz":120, "control_hz":10,
            "camera_order":list(self.camera_names),"camera_paths":self.camera_paths,
            "rgb_resolution":[84,84],"preview_resolution":list(preview_resolution),
            "training_resize":f"bilinear from the same native {preview_resolution[0]}x{preview_resolution[1]} camera frame",
            "wrist_mount_link":self.wrist_link_path,"lid_mass_kg":self.lid_mass,
            "lid_radius_m":0.084,"lid_thickness_m":0.008,"knob_radius_m":0.018,
            "pot_outer_radius_m":self.pot_radius,"pot_height_m":self.pot_height,
            "table_top_m":self.table_top,"xy_randomization_m":randomization,
            "success":{"minimum_lift_m":0.15,"maximum_xy_offset_m":0.05,
                       "maximum_knob_tcp_distance_m":0.05,"hold_control_steps":30},
            "material_static_friction":1.,"material_dynamic_friction":0.8,
            "no_attachment_joint":True}

    def _camera_optics(self,camera):
        camera.CreateFocalLengthAttr(16.)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(15.2908)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.005,20.))

    def _cylinder(self,path,radius,height,position,color):
        shape=UsdGeom.Cylinder.Define(self.world.stage,path)
        shape.CreateRadiusAttr(radius);shape.CreateHeightAttr(height);shape.CreateAxisAttr("Z")
        shape.AddTranslateOp().Set(Gf.Vec3d(*position));shape.CreateDisplayColorAttr([color])
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        return shape

    def _box(self,path,size,position,color):
        shape=UsdGeom.Cube.Define(self.world.stage,path);shape.CreateSizeAttr(1.)
        shape.AddTranslateOp().Set(Gf.Vec3d(*position));shape.AddScaleOp().Set(Gf.Vec3f(*size))
        shape.CreateDisplayColorAttr([color]);UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        return shape

    def _render(self):
        # Render-only updates must not advance physics behind the controller.
        # Pausing is safe here because play() is explicit and always restored.
        self.world.pause()
        try:
            self.rep.orchestrator.step(rt_subframes=1,delta_time=0.0,pause_timeline=False)
        finally:
            self.world.play()
        self.frame_counter+=1

    def _drive(self,steps=12):
        from isaacsim.core.utils.types import ArticulationAction
        for _ in range(steps):
            q=np.asarray(self.robot.get_joint_positions())
            self.bias=np.clip(self.bias+(self.target[self.arm]-q[self.arm])*0.003,-0.08,0.08)
            cmd=self.target.copy()
            cmd[self.arm]=np.clip(cmd[self.arm]+self.bias,self.lower,self.upper)
            self.robot.apply_action(ArticulationAction(joint_positions=cmd))
            self.world.step(render=False)

    def images(self,kind="rgb"):
        from PIL import Image
        images={}
        expected=(self.preview_resolution[1],self.preview_resolution[0],3)
        for _ in range(20):
            images={name:np.asarray(self.annotators[name]["preview"].get_data())[...,:3].astype(np.uint8).copy()
                    for name in self.camera_names}
            if all(a.shape==expected and a.std()>3 for a in images.values()):break
            self._render()
        for name,a in images.items():
            if a.shape!=expected or not a.size or a.std()<3:
                raise RuntimeError(f"invalid camera {name}/{kind}: {a.shape}")
        if kind=="rgb":
            images={name:np.asarray(Image.fromarray(a).resize((84,84),Image.Resampling.BILINEAR))
                    for name,a in images.items()}
        return images

    def camera_poses(self):
        cache=UsdGeom.XformCache()
        return {name:np.array(cache.GetLocalToWorldTransform(
            self.world.stage.GetPrimAtPath(path))).tolist() for name,path in self.camera_paths.items()}

    def observe(self):
        raw=self.images()
        rgb=np.stack([raw[name].transpose(2,0,1) for name in self.camera_names])
        q=np.asarray(self.robot.get_joint_positions());tcp=self.fk(q[self.arm])
        low=np.r_[q[self.arm],q[self.fingers].mean()/0.04,tcp[:3,3],
                  Rotation.from_matrix(tcp[:3,:3]).as_quat()].astype(np.float32)
        if not self.rgb_frames:
            for _ in range(self.stack):self.rgb_frames.append(rgb.copy());self.states.append(low.copy())
        else:self.rgb_frames.append(rgb);self.states.append(low)
        obs=(np.concatenate(list(self.rgb_frames),axis=1),np.concatenate(list(self.states)))
        assert obs[0].shape==self.rgb_shape and np.isfinite(obs[1]).all()
        return obs

    def reset(self,nominal=False):
        self.world.reset()
        self.target=self.home.copy();self.bias=np.zeros(7)
        self.robot.set_joint_positions(self.target);self.robot.set_joint_velocities(np.zeros_like(self.target))
        self.pot_xy=np.array([0.46,0.])
        if not nominal:self.pot_xy+=self.rng.uniform(-self.randomization,self.randomization,2)
        self.pot_translation.Set(Gf.Vec3d(*self.pot_xy,self.table_top))
        self.pot_rotation.Set(0.)
        pos=np.r_[self.pot_xy,self.table_top+self.pot_height+0.006]
        self.lid.set_world_pose(position=pos,orientation=np.array([1.,0,0,0]))
        self.lid.set_linear_velocity(np.zeros(3));self.lid.set_angular_velocity(np.zeros(3))
        self.robot.get_articulation_controller().set_gains(
            kps=np.array([800. if i in self.arm else 1000. for i in range(self.robot.num_dof)]),
            kds=np.array([40. if i in self.arm else 50. for i in range(self.robot.num_dof)]))
        self._drive(180)
        self.initial_lid=np.asarray(self.lid.get_world_pose()[0]).copy()
        self.t=0;self.held=0;self.done=False;self.rgb_frames.clear();self.states.clear()
        for _ in range(3):self._render()
        return self.observe()

    def knob_position(self):
        p,q=self.lid.get_world_pose()
        q=np.asarray(q)
        return np.asarray(p)+Rotation.from_quat(q[[1,2,3,0]]).apply([0,0,0.032])

    def status(self):
        p,q=self.lid.get_world_pose();p=np.asarray(p);q=np.asarray(q)
        joints=np.asarray(self.robot.get_joint_positions());tcp=self.fk(joints[self.arm])[:3,3]
        knob=self.knob_position()
        return {"lid":p.tolist(),"lid_quaternion_wxyz":q.tolist(),"knob":knob.tolist(),"tcp":tcp.tolist(),
            "lift_m":float(p[2]-self.initial_lid[2]),
            "xy_offset_m":float(np.linalg.norm(p[:2]-self.initial_lid[:2])),
            "knob_tcp_distance_m":float(np.linalg.norm(knob-tcp)),
            "finger_positions_m":joints[self.fingers].tolist(),"q":joints.tolist(),"step":self.t}

    def step(self,action):
        if self.done:raise RuntimeError("reset required after termination")
        action=np.asarray(action,dtype=np.float32)
        assert action.shape==(8,) and np.isfinite(action).all()
        action=np.clip(action,-1,1)
        self.target[self.arm]=np.clip(self.target[self.arm]+self.delta*action[:7],self.lower,self.upper)
        self.target[self.fingers]=0.02*(action[7]+1)
        self._drive(12);self._render();self.t+=1
        info=self.status()
        valid=(info["lift_m"]>0.15 and info["xy_offset_m"]<0.05 and
               info["knob_tcp_distance_m"]<0.05 and np.mean(info["finger_positions_m"])<0.025)
        self.held=self.held+1 if valid else 0
        success=self.held>=30
        failed=info["lid"][2]<0.18 or info["xy_offset_m"]>0.25
        terminated=bool(success or failed);truncated=self.t>=self.horizon and not terminated
        self.done=terminated or truncated
        info.update(success=bool(success),failed=bool(failed),held_steps=self.held)
        return self.observe(),float(success),terminated,bool(truncated),info

    def expert_actions(self,close=True):
        knob=self.knob_position();grasp=knob+np.array([0,0,0.003])
        above=grasp+np.array([0,0,0.16])
        pre=self.ik(above,self.home[self.arm]);down=self.ik(grasp,pre)
        up=self.ik(grasp+np.array([0,0,0.20]),down)
        waypoints=[("approach",pre,0.035,35),("descend",down,0.035,40),
            ("close" if close else "keep_open",down,0.0 if close else 0.035,25),
            ("lift",up,0.0 if close else 0.035,45),("hold",up,0.0 if close else 0.035,35)]
        previous=self.target.copy()
        for phase,joints,opening,steps in waypoints:
            target=previous.copy();target[self.arm]=joints;target[self.fingers]=opening
            for k in range(steps):
                self.expert_phase=phase
                alpha=min(1.,(k+1)/(steps*0.65));alpha=alpha*alpha*(3-2*alpha)
                desired=previous+(target-previous)*alpha
                action=np.r_[(desired[self.arm]-self.target[self.arm])/self.delta,
                             desired[self.fingers].mean()/0.02-1]
                yield np.clip(action,-1,1).astype(np.float32)
            previous=target.copy()

    def save_image(self,path):
        from PIL import Image, ImageDraw
        images=self.images("preview")
        w,h=self.preview_resolution
        canvas=Image.new("RGB",(2*w,2*h));draw=ImageDraw.Draw(canvas)
        for i,name in enumerate(self.camera_names):
            x=(i%2)*w;y=(i//2)*h
            canvas.paste(Image.fromarray(images[name]),(x,y))
            draw.text((x+3,y+3),name,fill=(255,80,30))
        canvas.save(path)

    def close(self):
        for name in self.camera_names:
            for kind,ann in self.annotators[name].items():
                ann.detach(self.products[name][kind]);self.products[name][kind].destroy()
