#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/pingandog
exec scripts/run_isaac62.sh isaac_i2o/train.py \
  --task take_lid_off_saucepan \
  --demo-dir datasets/take_lid_off_saucepan/fixed_100_20261112 \
  --demos 100 --batch-size 256 --frame-stack 8 \
  --steps 30250 --pretrain-updates 0 --pretrain-mode mse \
  --capacity 50000 --seed 1 --action-mode soft --softargmax-beta 1 \
  --checkpoint-every 1000 --eval-every 5000 --periodic-eval-episodes 10 \
  --eval-episodes 100 --eval-seed 100000 --resume outputs/i2o_panda/run_20261112_075738/checkpoint.pt --run-label formal_i2o_saucepan_seed1_recovery
