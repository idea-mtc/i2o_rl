import json,tempfile,unittest
from pathlib import Path
from validate_imitation_gate import validate


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'result.json').write_text(json.dumps({'pass_gate':True,'updates':500}))
        episodes=[]
        for seed in range(300000,300020):
            episodes.append({'seed':seed,'final':{'success':True}})
            rows=[{'lift_m':.2,'xy_offset_m':.001,'knob_tcp_distance_m':.003,
                   'finger_positions_m':[.018,.018],'knob':[.46,0,.52],'tcp':[.46,0,.523]} for _ in range(30)]
            (self.root/f'trace_500_{seed}.json').write_text(json.dumps(rows))
        (self.root/'eval_500.json').write_text(json.dumps(episodes))
        (self.root/'validated_imitation.pt').write_bytes(b'fixture, not a model')

    def test_full_physical_recompute(self):
        self.assertEqual(validate(self.root)['successes'],20)

    def test_success_flag_cannot_hide_dropped_lid(self):
        p=self.root/'trace_500_300000.json';rows=json.loads(p.read_text());rows[-1]['lift_m']=.01
        p.write_text(json.dumps(rows))
        with self.assertRaises(AssertionError):validate(self.root)

    def test_slip_rejected(self):
        p=self.root/'trace_500_300000.json';rows=json.loads(p.read_text());rows[-1]['knob'][0]+=.02
        p.write_text(json.dumps(rows))
        with self.assertRaises(AssertionError):validate(self.root)

    def test_missing_unseen_seed_rejected(self):
        p=self.root/'eval_500.json';episodes=json.loads(p.read_text());episodes[-1]['seed']=300000
        p.write_text(json.dumps(episodes))
        with self.assertRaises(AssertionError):validate(self.root)


if __name__=='__main__':unittest.main()
