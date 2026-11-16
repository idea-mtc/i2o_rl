# Training watchdog

`watch_training.py` checks the experiment identified by
`outputs/i2o_panda/active_formal_saucepan.json` every 60 seconds. It adopts the
current training service without restarting it. On service exit before a
`complete` progress marker, it selects the newest intact atomic PyTorch ZIP
checkpoint from this experiment's recovery ancestry and launches a separate
systemd user service. Each attempt gets its own output directory and log.
Saved training arguments are preserved, except resume path, output directory,
run label, and zero repeated pretraining. Model/target/optimizers and step/update
counters restore. Final-step checkpoints can resume final evaluation only.

The enabled user unit is `pingandog-i2o-watchdog.service`. It is independent of
SSH; existing `Linger=yes` allows it to survive logout. A maximum of five
training restarts and a minimum 120-second restart interval limit failure loops.
No complete marker means even a clean process exit is treated as interrupted.
Corrupt ZIP archives are skipped. If no checkpoint passes integrity checks or
restart attempts are exhausted, the supervisor records the condition and exits.
ZIP checks do not replace tensor/config validation, which the trainer performs.

A live process with progress older than one hour is flagged in watchdog state;
it is not killed automatically because rendering, evaluation and disk writes
can legitimately take time. This detects exits, not all possible GPU deadlocks.
No external notifications are sent.

Remote status:

```bash
systemctl --user status pingandog-i2o-watchdog.service
cat /home/ubuntu/pingandog/outputs/i2o_panda/watchdog_state.json
cat /home/ubuntu/pingandog/outputs/i2o_panda/active_formal_saucepan.json
tail /home/ubuntu/pingandog/logs/i2o_watchdog.log
```

Before intentionally stopping training, pause automatic recovery:

```bash
touch /home/ubuntu/pingandog/outputs/i2o_panda/WATCHDOG_PAUSE
```

Remove that marker to allow recovery again. To disable the supervisor entirely:

```bash
systemctl --user disable --now pingandog-i2o-watchdog.service
```

Recovery is from the last checkpoint, not the exact interrupted instruction.
Current checkpoint cadence is 1000 online steps. Old checkpoints do not persist
online replay, RNG, or physical simulator state. These restart; incomplete
progress since the last checkpoint is repeated. If interruption occurs during
final evaluation, evaluation is repeated from its beginning.

Validation: four unit tests on both local machine and server cover live-process
noninterference, abnormal-exit recovery/config preservation, restart backoff,
completion, checkpoint corruption, restart limit and intentional pause. Deployment
was checked against the real running training service without killing it.
