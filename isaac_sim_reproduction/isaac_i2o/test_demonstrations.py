"""Dataset failure tests: reject false success, time drift, corruption and leakage."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from demonstrations import validate_episode,stacked_observations,audit_dataset,file_hash,FORMAT


def fixture():
    n=30
    pattern=(np.indices((84,84)).sum(0)%2*100).astype(np.uint8)
    rgb=np.stack([np.broadcast_to(pattern+t,(4,3,84,84)).copy() for t in range(n+1)])
    position=np.tile([0.46,0.,0.48],(n+1,1)).astype(np.float32);position[0,2]=0.28
    knob=position+np.array([0,0,0.032],np.float32)
    state=np.zeros((n+1,15),np.float32);state[:,7]=0.4
    return {"rgb":rgb,"state":state,"actions":np.zeros((n,8),np.float32),
        "rewards":np.r_[np.zeros(n-1),1.].astype(np.float32),
        "discounts":np.r_[np.full(n-1,0.99),0.].astype(np.float32),
        "terminated":np.arange(n)==n-1,"truncated":np.zeros(n,bool),
        "object_positions":position,"knob_positions":knob,"tcp_positions":knob.copy(),
        "sim_time":np.arange(n+1)*0.1}


class DataTests(unittest.TestCase):
    def test_false_success_rejected(self):
        data=fixture();validate_episode(data,"take_lid_off_saucepan")
        data["object_positions"][-1,2]=0.28
        with self.assertRaises(AssertionError):validate_episode(data,"take_lid_off_saucepan")

    def test_clock_drift_rejected(self):
        data=fixture();data["sim_time"]=np.arange(31)*0.125
        with self.assertRaises(AssertionError):validate_episode(data,"take_lid_off_saucepan")

    def test_episode_stack_starts_with_own_reset_frame(self):
        data=fixture();other=fixture();other["rgb"]=255-other["rgb"]
        a=stacked_observations(data,8);b=stacked_observations(other,8)
        self.assertEqual(b[0][0].shape,(4,24,84,84))
        for i in range(8):np.testing.assert_array_equal(b[0][0][:,3*i:3*i+3],other["rgb"][0])
        self.assertFalse(np.array_equal(b[0][0][:,:3],a[-1][0][:,-3:]))

    def test_corrupted_file_rejected_before_loading(self):
        data=fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/"episode.npz";np.savez_compressed(path,**data)
            manifest={"format":FORMAT,"task":"take_lid_off_saucepan","status":"complete","target_episodes":1,
                "episodes":[{"file":"episode.npz","sha256":file_hash(path),"seed":1,
                             "checks":validate_episode(data,"take_lid_off_saucepan")}]}
            (root/"manifest.json").write_text(json.dumps(manifest))
            audit_dataset(root)
            with path.open("ab") as stream:stream.write(b"corruption")
            with self.assertRaises(AssertionError):audit_dataset(root)


if __name__=="__main__":unittest.main()
