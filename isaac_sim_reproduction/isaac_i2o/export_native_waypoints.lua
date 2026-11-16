-- Read original RLBench model through CoppeliaSim's public scene API.
-- No dynamics simulation, object scaling or scene source modification.
sim.getInt32Param=sim.getInt32Param or sim.getInt32Parameter
sim.getObjectInt32Param=sim.getObjectInt32Param or sim.getObjectInt32Parameter
sim.getEngineFloatParam=sim.getEngineFloatParam or sim.getEngineFloatParameter
local BASE='/home/ubuntu/pingandog/assets/candidates/rlbench_open_door_original/'
local OUT='/home/ubuntu/pingandog/assets/rlbench_open_door/export/'
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
 sim.loadScene(BASE..'task_design.ttt')
 local base=sim.getObjectHandle('Panda');local tip=sim.getObjectHandle('Panda_tip');local joints={};local initial={}
 for i=1,7 do joints[i]=sim.getObjectHandle('Panda_joint'..i);initial[i]=sim.getJointPosition(joints[i]) end
 local result={robot={}}
 for k=0,2 do
  local q={};for i=1,7 do q[i]=initial[i]+k*.025;sim.setJointPosition(joints[i],q[i]) end
  result.robot[#result.robot+1]={q=q,base=sim.getObjectMatrix(base,-1),joint1=sim.getObjectMatrix(joints[1],-1),tip=sim.getObjectMatrix(tip,-1)}
 end
 local root=sim.loadModel(BASE..'open_door.ttm');result.waypoints={}
 for i=0,3 do
  local h=sim.getObjectHandle('waypoint'..i);local v={name='waypoint'..i,matrix=sim.getObjectMatrix(h,root),extension=call(sim.getExtensionString,h,-1,nil)}
  if sim.getObjectType(h)==sim.object_path_type then
   v.path={};local m=sim.getObjectMatrix(h,root)
   for j=0,40 do
    local xyz=sim.getPositionOnPath(h,j/40);local rpy=sim.getOrientationOnPath(h,j/40);local t=sim.buildMatrix(xyz,rpy);v.path[#v.path+1]=sim.multiplyMatrices(m,t)
   end
  end
  result.waypoints[#result.waypoints+1]=v
 end
 local f=assert(io.open(OUT..'robot_and_waypoints.json','w'));f:write(json(result));f:close()
end
function sysCall_init()
 local ok,err=xpcall(export,debug.traceback);if not ok then print(err);local f=io.open(OUT..'calibration_error.txt','w');f:write(err);f:close() end
 sim.quitSimulator(true)
end
