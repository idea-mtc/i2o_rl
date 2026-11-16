import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import watch_training as w


class WatchdogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root/'outputs/i2o_panda'
        self.run = self.base/'run_original'
        self.run.mkdir(parents=True)
        w.write(self.base/'active_formal_saucepan.json',
                {'run':str(self.run), 'service':'test-training'})
        w.write(self.run/'config.json', {'args':{'steps':30250, 'pretrain_updates':1000,
                                               'batch_size':256, 'frame_stack':8}})
        w.write(self.run/'progress.json', {'stage':'online_training','online_step':4256})
        self.ck = self.run/'checkpoint.pt'
        with zipfile.ZipFile(self.ck,'w') as z:
            z.writestr('checkpoint/data.pkl', b'test')

    def test_live_process_untouched(self):
        with patch.object(w,'service_active',return_value=True), patch.object(w.subprocess,'run') as launch:
            self.assertEqual(w.tick(self.root)['status'],'running')
            launch.assert_not_called()

    def test_failed_process_restores_original_config_and_checkpoint(self):
        with patch.object(w,'service_active',return_value=False), patch.object(w.subprocess,'run') as launch:
            s=w.tick(self.root)
            self.assertEqual(s['status'],'restarted')
            cmd=launch.call_args.args[0]
            self.assertEqual(cmd[cmd.index('--resume')+1], str(self.ck))
            self.assertEqual(cmd[cmd.index('--pretrain-updates')+1], '0')
            self.assertEqual(cmd[cmd.index('--steps')+1], '30250')
            self.assertEqual(cmd[cmd.index('--batch-size')+1], '256')
            self.assertEqual(w.tick(self.root)['status'],'restart_backoff')
            self.assertEqual(launch.call_count,1)

    def test_completion_does_not_restart(self):
        w.write(self.run/'progress.json',{'stage':'complete'})
        with patch.object(w,'service_active',return_value=False), patch.object(w.subprocess,'run') as launch:
            self.assertEqual(w.tick(self.root)['status'],'complete')
            launch.assert_not_called()

    def test_corruption_limit_and_pause(self):
        self.ck.write_bytes(b'broken checkpoint')
        with patch.object(w,'service_active',return_value=False):
            self.assertEqual(w.tick(self.root)['status'],'no_valid_checkpoint')
            w.write(self.base/'watchdog_state.json',{'restarts':5})
            self.assertEqual(w.tick(self.root)['status'],'restart_limit_reached')
            (self.base/'WATCHDOG_PAUSE').touch()
            self.assertEqual(w.tick(self.root)['status'],'paused_by_user')


if __name__=='__main__':
    unittest.main()
