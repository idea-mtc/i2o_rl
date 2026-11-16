"""Build USD from Coppelia's original scene export; no primitive replacements."""
import json,re,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf,Sdf

def matrix(values):
 m=np.eye(4);m[:3]=np.asarray(values).reshape(3,4);return m
def transform(prim,m):
 UsdGeom.Xformable(prim).AddTransformOp().Set(Gf.Matrix4d(*m.T.flatten().tolist()))
def quat(r):
 x,y,z,w=Rotation.from_matrix(r).as_quat();return Gf.Quatf(float(w),Gf.Vec3f(float(x),float(y),float(z)))
def build(source,output):
 source=Path(source);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 data=json.loads(source.read_text());objects={x['handle']:x for x in data['objects']};joints={x['name']:x for x in objects.values() if 'joint' in x}
 required=['door_frame_joint','door_handle_joint'];assert set(joints)==set(required),list(joints)
 world={h:matrix(o['matrix']) for h,o in objects.items()};root=matrix(data['root_matrix']);norm=np.linalg.inv(root)
 poses={'frame':root,'door':world[joints[required[0]]['handle']],'handle':world[joints[required[1]]['handle']]}
 def group(h):
  while h in objects:
   name=objects[h]['name']
   if name==required[1]:return 'handle'
   if name==required[0]:return 'door'
   h=objects[h]['parent']
  return 'frame'
 members={name:[o for h,o in objects.items() if 'shape' in o and group(h)==name] for name in poses}
 stage=Usd.Stage.CreateNew(str(output));UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
 model=UsdGeom.Xform.Define(stage,'/RLBenchDoor');stage.SetDefaultPrim(model.GetPrim())
 report={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_objects':len(objects),'bodies':{},'joints':{},'textured_parts':0,
  'texture_transfer':'original RGBA pixels and original face UVs',
  'notes':['Geometry, transforms, joint axes and limits read from original TTM.', 'Normals regenerated with45degree crease because legacy getShapeViz normals use inconsistent frames.', 'PhysX collision cooking and motor/solver dynamics differ from CoppeliaSim.', 'Unlock rule requires the companion runtime helper.'], 'mesh_count':0}
 for name,pose in poses.items():
  body=UsdGeom.Xform.Define(stage,f'/RLBenchDoor/{name}');transform(body.GetPrim(),norm@pose)
  UsdPhysics.RigidBodyAPI.Apply(body.GetPrim());massparts=[]
  for o in members[name]:
   shape=o['shape'];body_from_shape=np.linalg.inv(pose)@world[o['handle']]
   if not shape.get('static',1) and shape.get('mass',0)>0:
    m=shape['mass'];c=body_from_shape[:3,:3]@np.asarray(shape['com'])+body_from_shape[:3,3]
    r=body_from_shape[:3,:3];inertia=r@np.asarray(shape['inertia']).reshape(3,3)@r.T;massparts.append((m,c,inertia))
   for index,part in enumerate(shape['parts']):
    vertices=np.asarray(part['vertices'],float).reshape(-1,3);indices=np.asarray(part['indices'],int)
    local=np.linalg.inv(pose)@matrix(part['matrix']);points=vertices@local[:3,:3].T+local[:3,3]
    path=f'/RLBenchDoor/{name}/mesh_{o["handle"]}_{index}'
    mesh=UsdGeom.Mesh.Define(stage,path);mesh.CreatePointsAttr(points.tolist());mesh.CreateFaceVertexCountsAttr([3]*(len(indices)//3));mesh.CreateFaceVertexIndicesAttr(indices.tolist());mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
    # The legacy getShapeViz normals are not in the returned vertices' frame.
    # Regenerate corner normals from the exact original topology (45deg crease).
    faces=indices.reshape(-1,3);cross=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
    face_normals=cross/np.maximum(np.linalg.norm(cross,axis=1,keepdims=True),1e-12)
    adjacency=[[] for _ in range(len(points))]
    for fi,face in enumerate(faces):
     for vi in face:adjacency[vi].append(fi)
    normals=[]
    for fi,face in enumerate(faces):
     for vi in face:
      neighbors=np.asarray(adjacency[vi]);keep=neighbors[face_normals[neighbors]@face_normals[fi]>.70710678]
      normal=cross[keep].sum(0);normal/=max(float(np.linalg.norm(normal)),1e-12);normals.append(normal.tolist())
    mesh.CreateNormalsAttr(normals);mesh.SetNormalsInterpolation('faceVarying')
    visible=bool(o.get('visibility_layer',1)&255)
    if not visible:mesh.CreateVisibilityAttr('invisible')
    color=part.get('color') or [.6,.6,.6];color=color[:3]
    mesh.CreateDisplayColorAttr([color]);material=UsdShade.Material.Define(stage,path+'_material')
    shader=UsdShade.Shader.Define(stage,path+'_material/Surface');shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color));shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.55);material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    if part.get('texture'):
     from PIL import Image
     tex=part['texture'];width,height=tex['resolution'];texdir=output.parent/'textures';texdir.mkdir(exist_ok=True)
     texname=Path(tex['file']).stem+'.png';rgba=(source.parent/tex['file']).read_bytes();assert len(rgba)==width*height*4
     image=Image.frombytes('RGBA',(width,height),rgba).transpose(Image.Transpose.FLIP_TOP_BOTTOM);image.save(texdir/texname)
     uv=np.asarray(tex['coordinates']).reshape(-1,2);interp='faceVarying' if len(uv)==len(indices) else 'vertex'
     assert len(uv) in [len(indices),len(vertices)]
     UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,interp).Set(uv.tolist())
     reader=UsdShade.Shader.Define(stage,path+'_material/UV');reader.CreateIdAttr('UsdPrimvarReader_float2');reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
     texture=UsdShade.Shader.Define(stage,path+'_material/Texture');texture.CreateIdAttr('UsdUVTexture');texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('textures/'+texname));texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result');texture.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
     texture.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat' if tex['options']&1 else 'clamp');texture.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat' if tex['options']&2 else 'clamp')
     if tex['apply_mode']==0:texture.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*color,1.))
     texture.CreateOutput('rgb',Sdf.ValueTypeNames.Float3);shader.GetInput('diffuseColor').ConnectToSource(texture.ConnectableAPI(),'rgb')
    physical=UsdPhysics.MaterialAPI.Apply(material.GetPrim());friction=shape.get('bullet_body_friction')
    if friction is None:friction=.5
    physical.CreateStaticFrictionAttr(float(friction));physical.CreateDynamicFrictionAttr(float(friction));physical.CreateRestitutionAttr(float(shape.get('bullet_body_restitution') or 0))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material);UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,UsdShade.Tokens.weakerThanDescendants,'physics')
    if shape.get('respondable',0):
     UsdPhysics.CollisionAPI.Apply(mesh.GetPrim());approx='none' if name=='frame' else 'convexHull' if part.get('geom_flags',0)&4 else 'convexDecomposition'
     UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approx)
    report['mesh_count']+=1
    if part.get('texture_id',-1) is not None and part.get('texture_id',-1)>=0:report['textured_parts']+=1
  if massparts:
   mass=sum(x[0] for x in massparts);com=sum(m*c for m,c,i in massparts)/mass
   inertia=sum(i+m*(np.dot(c-com,c-com)*np.eye(3)-np.outer(c-com,c-com)) for m,c,i in massparts)
   vals,vecs=np.linalg.eigh(inertia)
   if np.linalg.det(vecs)<0:vecs[:,0]*=-1
   assert np.min(vals)>0,(name,vals)
  else:mass=1.;com=np.zeros(3);vals=np.ones(3)*.01;vecs=np.eye(3)
  api=UsdPhysics.MassAPI.Apply(body.GetPrim());api.CreateMassAttr(float(mass));api.CreateCenterOfMassAttr(Gf.Vec3f(*com));api.CreateDiagonalInertiaAttr(Gf.Vec3f(*vals));api.CreatePrincipalAxesAttr(quat(vecs))
  report['bodies'][name]={'source_shapes':[o['name'] for o in members[name]],'mass':float(mass),'mass_source':'dynamic original shapes' if massparts else 'fixed-root placeholder','com':com.tolist(),'principal_inertia':vals.tolist()}
 UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath('/RLBenchDoor/frame'))
 anchor=UsdPhysics.FixedJoint.Define(stage,'/RLBenchDoor/fixed_root');anchor.CreateBody1Rel().SetTargets(['/RLBenchDoor/frame'])
 for source_name,parent,child in [(required[0],'frame','door'),(required[1],'door','handle')]:
  o=joints[source_name];j=o['joint'];assert j['kind']==10,('expected revolute joint',source_name,j)
  joint=UsdPhysics.RevoluteJoint.Define(stage,'/RLBenchDoor/'+source_name);joint.CreateBody0Rel().SetTargets(['/RLBenchDoor/'+parent]);joint.CreateBody1Rel().SetTargets(['/RLBenchDoor/'+child]);joint.CreateAxisAttr('Z');joint.CreateCollisionEnabledAttr(False)
  f0=np.linalg.inv(poses[parent])@world[o['handle']];f1=np.linalg.inv(poses[child])@world[o['handle']]
  for number,frame in [(0,f0),(1,f1)]:getattr(joint,f'CreateLocalPos{number}Attr')().Set(Gf.Vec3f(*frame[:3,3]));getattr(joint,f'CreateLocalRot{number}Attr')().Set(quat(frame[:3,:3]))
  low,span=j['interval'];q0=j['position'];joint.CreateLowerLimitAttr(float(np.rad2deg(low-q0)));joint.CreateUpperLimitAttr(float(np.rad2deg(low+span-q0)))
  drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'angular');drive.CreateTypeAttr('force');drive.CreateTargetPositionAttr(0.);drive.CreateStiffnessAttr(50000. if source_name==required[0] else 1.);drive.CreateDampingAttr(100. if source_name==required[0] else .01);drive.CreateMaxForceAttr(float(j.get('max_force') or 1.))
  report['joints'][source_name]={'source':j,'usd_limits_deg':[float(np.rad2deg(low-q0)),float(np.rad2deg(low+span-q0))],'axis':'Z in exported original joint frame','local_frame0':f0.tolist(),'local_frame1':f1.tolist()}
 model.GetPrim().SetCustomDataByKey('rlbench_source_commit','b80e51feb3694d9959cb8c0408cd385001b01382');model.GetPrim().SetCustomDataByKey('unlock_handle_displacement_degrees',25.);model.GetPrim().SetCustomDataByKey('success_hinge_displacement_degrees',25.)
 stage.GetRootLayer().Save();report['usd']=str(output);(output.parent/'conversion_report.json').write_text(json.dumps(report,indent=2));return report
