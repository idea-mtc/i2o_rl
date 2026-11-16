-- Read original RLBench model through CoppeliaSim's public scene API.
-- No dynamics simulation, object scaling or scene source modification.
sim.getInt32Param=sim.getInt32Param or sim.getInt32Parameter
sim.getObjectInt32Param=sim.getObjectInt32Param or sim.getObjectInt32Parameter
sim.getEngineFloatParam=sim.getEngineFloatParam or sim.getEngineFloatParameter
local TASK=os.getenv('RLBENCH_EXPORT_TASK');assert(TASK)
local BASE='/home/ubuntu/pingandog/assets/candidates/rlbench_'..TASK..'_original/'
local OUT='/home/ubuntu/pingandog/assets/rlbench_'..TASK..'/export/'
local function json(x)
 local t=type(x)
 if t=='nil' then return 'null' elseif t=='boolean' or t=='number' then return tostring(x)
 elseif t=='string' then return '"'..x:gsub('\\','\\\\'):gsub('"','\\"'):gsub('\n','\\n'):gsub('\r','\\r')..'"'
 elseif t=='table' then
  local out={};local n=0;local arr=true
  for k,_ in pairs(x) do n=n+1;if type(k)~='number' then arr=false end end
  if arr and n>0 then for i=1,#x do out[#out+1]=json(x[i]) end;return '['..table.concat(out,',')..']' end
  for k,v in pairs(x) do out[#out+1]=json(tostring(k))..':'..json(v) end
  return '{'..table.concat(out,',')..'}'
 end
 error('unsupported JSON type '..t)
end
local function call(f,...)
 if not f then return nil end
 local ok,a,b,c=pcall(f,...);if ok then return a,b,c end;return nil
end
local function parameter(h,name)
 if not sim[name] then return nil end
 local a,b=call(sim.getObjectInt32Param,h,sim[name]);if b~=nil then return b end;return a
end
local function export()
 local root=sim.loadModel(BASE..TASK..'.ttm')
 local objects=sim.getObjectsInTree(root,sim.handle_all,0)
 local result={format='original_rlbench_scene_v1',root=root,root_matrix=sim.getObjectMatrix(root,-1),version=sim.getInt32Param(sim.intparam_program_version),objects={}}
 for _,h in ipairs(objects) do
  local typ=sim.getObjectType(h)
  local item={handle=h,name=sim.getObjectName(h),parent=sim.getObjectParent(h),type=typ,matrix=sim.getObjectMatrix(h,-1),visibility_layer=parameter(h,'objintparam_visibility_layer')}
  item.extension=call(sim.getExtensionString,h,-1,nil)
  item.bbox={}
  for _,key in ipairs({'objbbox_min_x','objbbox_min_y','objbbox_min_z','objbbox_max_x','objbbox_max_y','objbbox_max_z'}) do
   local a,b=call(sim.getObjectFloatParameter,h,sim['objfloatparam_'..key]);item.bbox[#item.bbox+1]=b or a
  end
  if typ==sim.object_proximitysensor_type then
   item.sensor_type=parameter(h,'proxintparam_volume_type')
  elseif typ==sim.object_path_type then
   item.path={}
   for j=0,40 do
    local xyz=sim.getPositionOnPath(h,j/40);local rpy=sim.getOrientationOnPath(h,j/40)
    item.path[#item.path+1]=sim.multiplyMatrices(item.matrix,sim.buildMatrix(xyz,rpy))
   end
  end
  if typ==sim.object_joint_type then
   local cyclic,interval=sim.getJointInterval(h)
   item.joint={kind=sim.getJointType(h),position=sim.getJointPosition(h),cyclic=cyclic,interval=interval,
    mode=call(sim.getJointMode,h),target_velocity=call(sim.getJointTargetVelocity,h),target_position=call(sim.getJointTargetPosition,h),max_force=call(sim.getJointMaxForce,h),
    motor_enabled=parameter(h,'jointintparam_motor_enabled'),control_enabled=parameter(h,'jointintparam_ctrl_enabled'),locked_at_zero_velocity=parameter(h,'jointintparam_velocity_lock')}
  elseif typ==sim.object_shape_type then
   local mass,inertia,com=call(sim.getShapeMassAndInertia,h,item.matrix)
   item.shape={static=parameter(h,'shapeintparam_static'),respondable=parameter(h,'shapeintparam_respondable'),respondable_mask=parameter(h,'shapeintparam_respondable_mask'),mass=mass,inertia=inertia,com=com,parts={}}
   for _,name in ipairs({'bullet_body_friction','bullet_body_restitution','bullet_body_lineardamping','bullet_body_angulardamping','ode_body_friction'}) do
    if sim[name] then item.shape[name]=call(sim.getEngineFloatParam,sim[name],h) end
   end
   -- Clone only for component extraction; original object transforms/physics above retained.
   local copies=sim.copyPasteObjects({h},0);local copy=copies[1]
   local parts={copy};local flags=sim.getShapeGeomInfo(copy)
   if flags and flags%2==1 then parts=sim.ungroupShape(copy) end
   for _,p in ipairs(parts) do
    local vertices,indices,normals=sim.getShapeMesh(p)
    local a,b=call(sim.getShapeColor,p,nil,sim.colorcomponent_ambient_diffuse);local color=b or a
    local viz=call(sim.getShapeViz,p,0)
    local geomflags,primitive,dimensions=sim.getShapeGeomInfo(p)
    local part={matrix=sim.getObjectMatrix(p,-1),vertices=vertices,indices=indices,normals=normals,color=color,geom_flags=geomflags,primitive=primitive,dimensions=dimensions,
      texture_id=call(sim.getShapeTextureId,p)}
    if viz then
     part.vertices=viz.vertices;part.indices=viz.indices;part.normals=viz.normals
     if viz.colors then part.color={viz.colors[1],viz.colors[2],viz.colors[3]};part.specular={viz.colors[4],viz.colors[5],viz.colors[6]} end
     if viz.texture then
      local texture=viz.texture;local filename='texture_'..texture.id..'.rgba'
      local f=assert(io.open(OUT..filename,'wb'));f:write(texture.texture);f:close()
      part.texture={file=filename,resolution=texture.resolution,coordinates=texture.coordinates,apply_mode=texture.applyMode,options=texture.options}
     end
    end
    item.shape.parts[#item.shape.parts+1]=part
   end
   -- Remove temporary parts; do not touch original hierarchy.
   for _,p in ipairs(parts) do sim.removeObject(p) end
  end
  result.objects[#result.objects+1]=item
 end
 local f=assert(io.open(OUT..'scene.json','w'));f:write(json(result));f:close()
 print('RLBENCH_EXPORT_COMPLETE '..OUT..'scene.json')
end
function sysCall_init()
 local ok,err=xpcall(export,debug.traceback)
 if not ok then local f=io.open(OUT..'error.txt','w');if f then f:write(err);f:close() end;print(err) end
 sim.quitSimulator(true)
end
