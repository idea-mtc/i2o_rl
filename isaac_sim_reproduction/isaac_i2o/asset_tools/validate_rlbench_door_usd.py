"""Convert/load original asset and test joints/logic; NOT a learned-policy demo."""
import os,sys,json,traceback,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
ROOT=Path('/home/ubuntu/pingandog');ASSET=ROOT/'assets/rlbench_open_door';OUT=ASSET/'validation';OUT.mkdir(exist_ok=True)
GPU=int(os.environ['PINGANDOG_GPU'])
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RaytracedLighting','multi_gpu':False,'active_gpu':GPU,'physics_gpu':GPU,'extra_args':['--/physics/updateToUsd=true']})
writer=None
try:
 import numpy as np
 from pxr import Usd,UsdGeom,UsdPhysics,UsdLux,Gf
 from PIL import Image,ImageDraw
 from build_rlbench_door_usd import build,matrix
 from native_door_logic import NativeDoorLogic
 from isaacsim.core.api import World
 from isaacsim.core.api.robots import Robot
 from isaacsim.core.utils.stage import add_reference_to_stage
 import omni.replicator.core as rep
 report=build(ASSET/'export/scene.json',ASSET/'usd/open_door.usda')
 world=World(stage_units_in_meters=1.,physics_dt=1/240,rendering_dt=1/60);stage=world.stage
 add_reference_to_stage(str(ASSET/'usd/open_door.usda'),'/World/OriginalDoor')
 # Exact closed-geometry correspondence, before physics initialization.
 source=json.loads((ASSET/'export/scene.json').read_text());norm=np.linalg.inv(matrix(source['root_matrix']));cache=UsdGeom.XformCache();errors=[];allpoints=[]
 for obj in source['objects']:
  if 'shape' not in obj:continue
  for i,part in enumerate(obj['shape']['parts']):
   matches=[p for p in stage.Traverse() if p.GetName()==f'mesh_{obj["handle"]}_{i}'];assert len(matches)==1
   p=matches[0];m=np.asarray(cache.GetLocalToWorldTransform(p)).T
   points=np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get());actual=points@m[:3,:3].T+m[:3,3]
   m=norm@matrix(part['matrix']);original=np.asarray(part['vertices']).reshape(-1,3)@m[:3,:3].T+m[:3,3]
   errors.append(float(np.max(abs(actual-original))));allpoints.append(actual)
 assert max(errors)<1e-5,errors
 report['closed_geometry_max_error_m']=max(errors)
 xyz=np.concatenate(allpoints);low=xyz.min(0);high=xyz.max(0);center=(low+high)/2;scale=max(high-low)
 report['closed_bounds_m']=[low.tolist(),high.tolist()]
 light=UsdLux.DomeLight.Define(stage,'/World/Light');light.CreateIntensityAttr(1500.)
 floor=UsdGeom.Mesh.Define(stage,'/World/Ground');z=float(low[2]-.02)
 floor.CreatePointsAttr([(-3,-3,z),(3,-3,z),(3,3,z),(-3,3,z)]);floor.CreateFaceVertexCountsAttr([4]);floor.CreateFaceVertexIndicesAttr([0,1,2,3]);floor.CreateDisplayColorAttr([(.23,.25,.28)])
 camera=UsdGeom.Camera.Define(stage,'/World/Camera');eye=center+np.array([-1.45,1.6,.75])*scale
 camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*center),Gf.Vec3d(0,0,1)).GetInverse());camera.CreateFocalLengthAttr(28.)
 product=rep.create.render_product('/World/Camera',(960,720));ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach(product)
 robot=world.scene.add(Robot(prim_path='/World/OriginalDoor/frame',name='original_door'));world.reset()
 assert set(robot.dof_names)=={'door_frame_joint','door_handle_joint'},robot.dof_names
 robot.set_joint_positions(np.zeros(robot.num_dof));robot.set_joint_velocities(np.zeros(robot.num_dof))
 logic=NativeDoorLogic(stage,robot,'/World/OriginalDoor');lever=UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath('/World/OriginalDoor/door_handle_joint'),'angular');hinge=logic.drive
 writer=subprocess.Popen(['/usr/bin/ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x760','-r','15','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'joint_validation.mp4')],stdin=subprocess.PIPE)
 rows=[]
 def frame(label,state):
  world.render();raw=np.asarray(ann.get_data())
  assert raw.shape[:2]==(720,960),raw.shape
  canvas=Image.new('RGB',(960,760));canvas.paste(Image.fromarray(raw[:,:,:3].astype(np.uint8)),(0,0));draw=ImageDraw.Draw(canvas)
  draw.text((8,724),'Original RLBench door asset | PhysX joint validation | no robot policy',fill='white')
  draw.text((8,744),f"{label} | handle {state['handle_degrees']:.1f}deg hinge {state['hinge_degrees']:.1f}deg | unlocked {state['unlocked']} success {state['success']}",fill='white')
  writer.stdin.write(np.asarray(canvas).tobytes());return canvas
 for i in range(120):world.step(render=(i%4==0));state=logic.step()
 assert not state['unlocked'] and not state['success'],state
 frame('Closed',state).save(OUT/'closed.png');report['closed']=state
 for _ in range(15):frame('Closed',state)
 for i in range(240):
  lever.GetTargetPositionAttr().Set(min(28.,i*28/160));world.step(render=False);state=logic.step();rows.append(dict(phase='turn_handle',**state))
  if i%4==0:frame('Turning handle',state)
 assert state['unlocked'],state
 report['handle_unlocked']=state;frame('Unlocked',state).save(OUT/'unlocked.png')
 hinge.GetStiffnessAttr().Set(1.);hinge.GetDampingAttr().Set(.01)
 for i in range(300):
  hinge.GetTargetPositionAttr().Set(min(28.,i*28/180));world.step(render=False);state=logic.step();rows.append(dict(phase='open_hinge',**state))
  if i%4==0:frame('Opening door (joint test)',state)
 assert state['success'] and state['unlocked'],state
 report['opened']=state;frame('Open',state).save(OUT/'open.png')
 for _ in range(20):frame('Open',state)
 writer.stdin.close();assert writer.wait()==0;writer=None
 report['pass']=True;(OUT/'result.json').write_text(json.dumps(report,indent=2));(OUT/'trace.json').write_text(json.dumps(rows));print('NATIVE_DOOR_USD_VALIDATION_PASS',json.dumps(state),flush=True)
 ann.detach(product);product.destroy()
except Exception:
 (OUT/'error.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
finally:
 if writer:writer.stdin.close();writer.wait()
 app.close()
