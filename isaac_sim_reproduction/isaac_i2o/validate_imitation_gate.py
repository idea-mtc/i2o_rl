"""Independently recompute the 20-rollout physical gate before resuming RL."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def validate(directory):
    directory=Path(directory)
    result=json.loads((directory/'result.json').read_text())
    assert result['pass_gate'], 'imitation training has not passed its success gate'
    update=result['updates'];evaluations=json.loads((directory/f'eval_{update}.json').read_text())
    config=json.loads((directory/'config.json').read_text()) if (directory/'config.json').exists() else {}
    seeds=config.get('gate_seeds',list(range(300000,300020)))
    assert len(evaluations)==20 and {r['seed'] for r in evaluations}==set(seeds)
    checks=[]
    for e in evaluations:
        rows=json.loads((directory/f"trace_{update}_{e['seed']}.json").read_text())
        assert e['final']['success'] and len(rows)>=30
        hold=rows[-30:]
        assert all(x['lift_m']>.15 and x['xy_offset_m']<.05 and
                   x['knob_tcp_distance_m']<.05 and np.mean(x['finger_positions_m'])<.025 for x in hold)
        relative=np.array([np.array(x['knob'])-np.array(x['tcp']) for x in hold])
        slip=float(np.ptp(relative,axis=0).max());assert slip<.015,(e['seed'],slip)
        checks.append(dict(seed=e['seed'],min_lift_m=min(x['lift_m'] for x in hold),
                           relative_grasp_motion_m=slip,steps=len(rows)))
    ck=directory/'validated_imitation.pt';assert ck.exists()
    report=dict(pass_gate=True,episodes=20,successes=20,update=update,checks=checks,
                checkpoint=str(ck),checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),
                scope='fixed-base task, XY +/-2cm; clock-conditioned learned actions, no teacher execution')
    (directory/'physical_gate.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    print(json.dumps(validate(a.directory),indent=2))
