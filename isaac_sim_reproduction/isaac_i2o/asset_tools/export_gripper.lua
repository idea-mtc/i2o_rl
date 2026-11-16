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
 sim.loadScene('/home/ubuntu/pingandog/assets/candidates/rlbench_open_door_original/task_design.ttt')
 local tip=sim.getObjectHandle('Panda_tip');local result={objects={}}
 for _,h in ipairs(sim.getObjectsInTree(sim.handle_scene,sim.handle_all,0)) do
  local name=sim.getObjectName(h)
  if name:find('finger') or name:find('Prox') or name:find('prox') then
   result.objects[#result.objects+1]={name=name,type=sim.getObjectType(h),matrix=sim.getObjectMatrix(h,tip),respondable=parameter(h,'shapeintparam_respondable'),mask=parameter(h,'shapeintparam_respondable_mask')}
  end
 end
 local f=assert(io.open(OUT..'original_gripper.json','w'));f:write(json(result));f:close()
end
function sysCall_init()
 local ok,err=xpcall(export,debug.traceback)
 if not ok then local f=io.open(OUT..'error.txt','w');if f then f:write(err);f:close() end;print(err) end
 sim.quitSimulator(true)
end
