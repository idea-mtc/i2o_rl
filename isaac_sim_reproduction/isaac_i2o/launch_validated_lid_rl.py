"""Launch the changed experiment only after independent 20/20 physical validation."""
import argparse,datetime,hashlib,json,subprocess
from pathlib import Path
from validate_imitation_gate import validate

p=argparse.ArgumentParser();p.add_argument('imitation_run',type=Path);a=p.parse_args()
root=Path('/home/ubuntu/pingandog');gate=validate(a.imitation_run)
assert gate['pass_gate'] and gate['successes']==20
assert hashlib.sha256(Path(gate['checkpoint']).read_bytes()).hexdigest()==gate['checkpoint_sha256']
repair=json.loads((a.imitation_run/'config.json').read_text());beta=repair['args']['beta']
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');out=root/f'outputs/i2o_panda/run_repaired_{stamp}'
unit=f'pingandog-i2o-repaired-{stamp}';log=root/f'logs/i2o_repaired_{stamp}.log'
cmd=[str(root/'scripts/run_isaac62.sh'),'isaac_i2o/train.py',
 '--task','take_lid_off_saucepan','--demo-dir',repair['demo_source'],'--demos','100',
 '--batch-size','256','--frame-stack','8','--steps','30250','--pretrain-updates','0',
 '--pretrain-mode','mse','--capacity','50000','--seed','1','--action-mode','soft',
 '--softargmax-beta',str(beta),'--soft-endpoint-rescale','--phase-clock',
 '--checkpoint-every','1000','--eval-every','5000','--eval-at','1000','--periodic-eval-episodes','10',
 '--eval-episodes','100','--eval-seed','100000','--resume',gate['checkpoint'],
 '--output-dir',str(out),'--run-label','clock_conditioned_lid_repaired_seed1']
if repair['args'].get('demo_phase_goals'):
 cmd.extend(['--demo-phase-goals','--demo-bc-only'])
cmd.extend(['--phase-heads',str(repair['args'].get('phase_heads',1))])
if repair['args'].get('cartesian_goals'):
 cmd.extend(['--cartesian-goals','--exploration-std','0.002','--augmentation-pad','0'])
if repair['args'].get('absolute_joints'):
 cmd.extend(['--absolute-joints','--exploration-std','0.002'])
# User-requested stop of the old run must have completed before launching a replacement.
base=root/'outputs/i2o_panda';pointer=base/'active_formal_saucepan.json'
old=json.loads(pointer.read_text())
r=subprocess.run(['systemctl','--user','is-active',old['service']],capture_output=True,text=True)
assert r.stdout.strip() not in ('active','activating'), 'old training is still active'
subprocess.run(['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+str(root),
 '--property=StandardOutput=append:'+str(log),'--property=StandardError=append:'+str(log),*cmd],check=True)
new=dict(run=str(out),log=str(log),service=unit,original_run=str(a.imitation_run),
         resume_checkpoint=gate['checkpoint'],imitation_gate=str(a.imitation_run/'physical_gate.json'),
         observation_variant='phase_clock_126',softargmax_beta=beta,soft_endpoint_rescale=True,absolute_joints=repair['args'].get('absolute_joints',False))
(base/f'pre_repair_pointer_{stamp}.json').write_text(json.dumps(old,indent=2))
if (base/'watchdog_state.json').exists():
 (base/'watchdog_state.json').rename(base/f'watchdog_state_before_repair_{stamp}.json')
pointer.write_text(json.dumps(new,indent=2))
print(json.dumps(new,indent=2))
# Keep WATCHDOG_PAUSE until initialization has actually been verified by the caller.
