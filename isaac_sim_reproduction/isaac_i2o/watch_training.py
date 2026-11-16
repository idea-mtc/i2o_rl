"""Supervise the project's I2O service without depending on an SSH session."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import time
import zipfile


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def checkpoint(paths):
    """Reject incomplete/corrupted atomic torch ZIP archives before a restart."""
    for p in sorted(set(paths), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with zipfile.ZipFile(p) as z:
                if any(n.endswith('/data.pkl') for n in z.namelist()) and z.testzip() is None:
                    return p
        except (OSError, zipfile.BadZipFile):
            pass
    return None


def service_active(unit):
    r = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '--value'],
                       capture_output=True, text=True)
    return r.stdout.strip() in ('active', 'activating', 'deactivating', 'reloading')


def tick(root, max_restarts=5):
    base = root/'outputs/i2o_panda'
    pointer = base/'active_formal_saucepan.json'
    state_path = base/'watchdog_state.json'
    state = read(state_path)
    a = read(pointer)
    if not a.get('run') or not a.get('service'):
        raise RuntimeError('active training pointer must identify run and service')
    run = Path(a['run'])
    progress = read(run/'progress.json')
    state.update(checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(), run=str(run),
                 online_step=progress.get('online_step'), stage=progress.get('stage'))
    if (base/'WATCHDOG_PAUSE').exists():
        state['status'] = 'paused_by_user'
    elif service_active(a['service']):
        # Never kill a live process for a long evaluation, startup, or disk write.
        state['status'] = 'running'
        if (run/'progress.json').exists():
            age = time.time()-(run/'progress.json').stat().st_mtime
            state['progress_age_seconds'] = age
            if age > 3600:
                state['status'] = 'live_process_stale_progress_needs_review'
    elif progress.get('stage') == 'complete':
        state['status'] = 'complete'
    elif state.get('restarts', 0) >= max_restarts:
        state['status'] = 'restart_limit_reached'
    elif time.time()-state.get('last_restart_epoch', 0) < 120:
        state['status'] = 'restart_backoff'
    else:
        # Restrict candidates to this experiment's explicit recovery ancestry.
        runs = {run, Path(a.get('original_run', str(run)))}
        runs.update(Path(x['run']) for x in state.get('history', []))
        ck = checkpoint([p for r in runs for p in r.glob('checkpoint*.pt')])
        if ck is None:
            state['status'] = 'no_valid_checkpoint'
        else:
            config = read(run/'config.json')
            args = config.get('args')
            if not args:
                raise RuntimeError('missing saved training arguments; refusing guessed configuration')
            number = state.get('restarts', 0)+1
            stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
            out = base/f'run_watchdog_{stamp}_{number}'
            unit = f'pingandog-i2o-auto-{stamp}-{number}'
            log = root/f'logs/i2o_auto_{stamp}_{number}.log'
            command = [str(root/'scripts/run_isaac62.sh'), 'isaac_i2o/train.py']
            args = dict(args, resume=str(ck), pretrain_updates=0,
                        output_dir=str(out), run_label='formal_i2o_saucepan_seed1_auto_recovery')
            for k, v in args.items():
                if v is None or v is False:
                    continue
                command.append('--'+k.replace('_', '-'))
                if v is not True:
                    command.append(str(v))
            subprocess.run(['systemd-run', '--user', '--unit='+unit,
                            '--property=WorkingDirectory='+str(root),
                            '--property=StandardOutput=append:'+str(log),
                            '--property=StandardError=append:'+str(log), *command], check=True)
            # Seed configuration for recovery even if SimApp fails before initialization.
            out.mkdir(exist_ok=True)
            write(out/'config.json', dict(config, args=args))
            history = state.setdefault('history', [])
            history.append(dict(run=str(run), service=a['service'], checkpoint=str(ck),
                                last_progress=progress, restarted_utc=stamp))
            write(pointer, dict(run=str(out), log=str(log), service=unit,
                                original_run=a.get('original_run', str(run)),
                                resume_checkpoint=str(ck), online_replay_restored=False))
            state.update(status='restarted', restarts=number, last_restart_epoch=time.time(),
                         new_run=str(out), checkpoint=str(ck))
    write(state_path, state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/home/ubuntu/pingandog'))
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--max-restarts', type=int, default=5)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    import fcntl
    lock = open(args.root/'outputs/i2o_panda/watchdog.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        try:
            state = tick(args.root, args.max_restarts)
            print(json.dumps(state), flush=True)
            if state['status'] in ('complete', 'restart_limit_reached', 'no_valid_checkpoint'):
                break
        except Exception as e:
            print(json.dumps({'watchdog_error':repr(e)}), flush=True)
        if args.once:
            break
        time.sleep(args.interval)
