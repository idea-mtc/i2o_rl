import sys,time,json
from pathlib import Path
import numpy as np,torch
from aligned.contract import ActionCodec
from aligned.gpu_replay import GPUReplay
from aligned.source_loader import load_original
from aligned.paper_update import update
from vendor.i2o_rl import I2OAgent
root=Path('/home/ubuntu/pingandog');job=root/'outputs/aligned_night/job_20261113'
gpu=int(json.loads((job/'progress.json').read_text())['gpu']);device=f'cuda:{gpu}'
torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
path=next((job/'dataset').glob('*.npz'))
with np.load(path) as f:d={k:f[k] for k in f.files}
c=ActionCodec.fit([d]);r=GPUReplay(c,device,capacity=256);r.add(d,True,path)
kwargs=dict(rgb_obs_shape=(4,24,84,84),low_dim_obs_shape=(120,),action_shape=(8,),device=device,lr=5e-5,weight_decay=.1,feature_dim=128,hidden_dim=512,levels=3,bins=5,atoms=51,v_min=-2.,v_max=2.,bc_lambda=1.,bc_margin=.01,critic_lambda=.1,critic_target_tau=.02,num_expl_steps=0,update_every_steps=1,stddev_schedule='.01',use_logger=True)
a=I2OAgent(**kwargs,action_mode='soft',softargmax_beta=1.,soft_endpoint_rescale=False,phase_heads=1)
t=time.monotonic()
for i in range(3):m=update(a,r.sample(256),rl=i>0)
torch.cuda.synchronize(gpu);result={'paper':m,'three_updates_s':time.monotonic()-t}
del a;torch.cuda.empty_cache()
o=load_original(root/'outputs/repro_audit/original').I2OAgent(**kwargs)
for net in o.encoder.conv_nets:net.register_forward_hook(lambda module,inputs,output:output.contiguous())
it=r.batches(256)
for i in range(2):m=o.update(it,i)
result['original']=m;result['peak_reserved_gib']=torch.cuda.max_memory_reserved(gpu)/2**30
(job/'gpu_check.json').write_text(json.dumps(result,indent=2));print('REAL_GPU_ALIGNED_CHECK_PASS',json.dumps(result),flush=True)
