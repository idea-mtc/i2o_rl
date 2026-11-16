import os,sys,time,json
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES']==os.environ['I2O_ALLOWED_UUIDS']
GPU=int(sys.argv[1]);assert GPU in [2,3,4]
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'multi_gpu':False,'active_gpu':0,'physics_gpu':0,'extra_args':['--/renderer/multiGpu/enabled=false','--/renderer/multiGpu/autoEnable=false']})
try:
 import torch
 assert torch.cuda.device_count()==1
 torch.cuda.set_device(0);x=torch.ones(100,device='cuda:0')
 print('PROBE_READY',os.getpid(),GPU,torch.cuda.device_count(),flush=True)
 for _ in range(20):app.update();time.sleep(.5)
finally:app.close()
