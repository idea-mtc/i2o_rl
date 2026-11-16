#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ubuntu/pingandog
cd "$ROOT"
python3 - <<'PY'
import json,hashlib
from pathlib import Path
root=Path('/home/ubuntu/pingandog')
c=json.loads((root/'isaac_i2o/configs/formal_saucepan_seed1_20261112.json').read_text())
b=json.loads((root/c['benchmark']).read_text())
assert b['pass'] and b['args']['batch_size']==256 and b['args']['frame_stack']==8
assert b['cuda_peak_reserved_gib']<70
manifest=root/c['dataset']/'manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==c['dataset_manifest_sha256']
assert len(json.loads(manifest.read_text())['episodes'])==100
print('FORMAL_PREFLIGHT_PASS: benchmark, dataset and requested configuration verified',flush=True)
PY
exec scripts/run_isaac62.sh isaac_i2o/train.py \
  --task take_lid_off_saucepan \
  --demo-dir datasets/take_lid_off_saucepan/fixed_100_20261112 \
  --demos 100 --batch-size 256 --frame-stack 8 \
  --steps 30250 --pretrain-updates 1000 --pretrain-mode mse \
  --capacity 50000 --seed 1 --action-mode soft --softargmax-beta 1 \
  --checkpoint-every 1000 --eval-every 5000 --periodic-eval-episodes 10 \
  --eval-episodes 100 --eval-seed 100000 --run-label formal_i2o_saucepan_seed1
