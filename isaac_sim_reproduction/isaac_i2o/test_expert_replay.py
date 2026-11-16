"""CPU tests for success/provenance independence and checkpoint migration."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from aligned.expert_replay import ExpertReplay, migrate_replay_entries


class Codec:
    def encode(self, x):
        return x


def episode():
    return dict(rgb=np.zeros((3,4,3,84,84),np.uint8),
                state=np.zeros((3,15),np.float32),
                raw_actions=np.zeros((2,8),np.float32),
                rewards=np.array([0,1],np.float32),terminated=np.array([False,True]))


class Tests(unittest.TestCase):
    def test_successful_online_is_not_expert(self):
        replay=ExpertReplay(Codec(),'cpu',capacity=12)
        replay.add(episode(),True,'/expert.npz',is_expert=True)
        replay.add(episode(),True,'/online.npz',is_expert=False)
        replay.add(episode(),False,'/failed.npz',is_expert=False)
        self.assertEqual(replay.success_ids,[0,1])
        self.assertEqual(replay.expert_ids,[0])
        self.assertEqual(replay.demo[[0,3,6],0].tolist(),[1.,0.,0.])
        self.assertTrue(bool((replay.sample(16,True)[11]==1).all()))
        self.assertEqual(replay.success_ids,[0,1])
        self.assertEqual([e['is_expert'] for e in replay.files()],[True,False,False])

    def test_provenance_required(self):
        replay=ExpertReplay(Codec(),'cpu',capacity=3)
        with self.assertRaises(TypeError):replay.add(episode(),True,'/online.npz')

    def test_migration_and_restore(self):
        entries=[dict(file='/expert.npz',successful=True),dict(file='/online.npz',successful=True)]
        migrated=migrate_replay_entries(entries,['/expert.npz'])
        self.assertEqual([e['is_expert'] for e in migrated],[True,False])
        self.assertEqual(migrate_replay_entries(migrated,['/expert.npz']),migrated)
        migrated[1]['is_expert']=True
        with self.assertRaises(ValueError):migrate_replay_entries(migrated,['/expert.npz'])


if __name__=='__main__':unittest.main()
