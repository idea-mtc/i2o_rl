# Fixed saucepan demonstrations and training task selection

The saucepan task now uses the same task factory for collection and training.
`train.py --task take_lid_off_saucepan --demo-dir DIRECTORY` loads saved episodes
instead of running the expert again. Loading checks task, environment/camera
contract, success, episode boundaries and SHA256 checksums before use.

## Collect

```bash
cd /home/ubuntu/pingandog
scripts/run_isaac62.sh isaac_i2o/collect_demos.py \
  --episodes 10 --pilot 10 --seed 12000
```

The printed `DEMO_DIRECTORY` is the fixed dataset path. Collection has its own
manifest, per-attempt seed and outcome records, pilot audit and final audit.
Failed expert attempts are counted and rejected, never relabeled as success.
Successful episodes are atomically stored and never overwritten.

To extend an unchanged collection using the same source version/config:

```bash
scripts/run_isaac62.sh isaac_i2o/collect_demos.py \
  --resume datasets/take_lid_off_saucepan/EXISTING_COLLECTION \
  --episodes 100 --pilot 10 --seed 12000
```

Resume refuses changed task/config/source versions or duplicate file writes.
Separate completed batches with disjoint seed blocks can be combined with
`merge_demos.py`, which re-audits all inputs, requires identical collection code
and configuration, copies all records and preserves source lineage. It refuses
an existing output directory and overlapping seeds. All attempts, not only
accepted episodes, are retained in the merged manifest.

## Data format

`manifest.json` identifies `isaac_i2o_raw_episode_v1`; each `episodes/*.npz` has:

| Field | Shape / meaning |
|---|---|
| rgb | `[T+1,4,3,84,84]`, uint8, raw unstacked RGB |
| state | `[T+1,15]`, seven joints, normalized opening, TCP XYZ + XYZW |
| actions | `[T,8]`, executed normalized joint increments and gripper command |
| rewards, discounts | `[T]`; sparse terminal success and bootstrap discount |
| terminated, truncated | `[T]`; true task ending versus time limit |
| object_positions, knob_positions, tcp_positions | `[T+1,3]`, physical audit only |
| sim_time | `[T+1]`, actual simulator clock, exactly 0.1 s between actions |
| phase | `[T]`, expert stage, metadata only |

Initial observation precedes the first action. Each action links observation t
to t+1. Object pose and expert phase do not enter learner observations.
Frames are not stored eight times: episode-local stacking happens when loaded.
At resets, missing history is padded only with that episode's own first frame.

Camera order is front, wrist, left_shoulder, right_shoulder. Training/collection
render native 168x126 (same 4:3 aspect and mounts as the 640x480 scene preview),
then bilinear resize to 84x84. Configuration records this explicitly; datasets
must match the training environment. No synthetic image generation is involved.

Success is independently recomputed on stored object/knob/TCP/state arrays:
at least 0.15 m lift, XY <0.05 m, knob/TCP <0.05 m and finger opening <0.025 m
for the last 30 control steps, plus relative slip <0.015 m. Nonblank/changing
four-view images, finite states, bounded actions, terminal rewards and actual
physics timestamps are also checked. Collection aborts if the pilot cannot be
obtained within 20 attempts. Larger runs have a bounded attempt budget.

## Train with saved data

```bash
scripts/run_isaac62.sh isaac_i2o/train.py \
  --task take_lid_off_saucepan --demo-dir DATASET_DIRECTORY \
  --steps 8 --batch-size 8 --pretrain-updates 4 --eval-episodes 0
```

Without `--demos`, all episodes in the fixed dataset are loaded. `--demos 10`
limits loading to ten; it does not regenerate them. Without `--demo-dir`, the
old live-collection path remains available (default two demos). The default
cube task is retained. Saucepan defaults to a 260-control-step horizon.

This small command checks task selection, loaded demonstrations, four-camera
network input, replay, finite losses, online actions, weights and checkpoint
reload. It is not a long training run or a success-rate experiment. The existing
`--pretrain-updates` uses the combined I2O losses, not MSE-only pretraining.
The prior one-camera cube checkpoint is not compatible with four-camera shapes.

## Clock correction

The initial collection audit detected 0.125 s per action despite the intended
0.1 s: live rendering was adding physics steps. Those attempts were rejected.
The saucepan controller now advances exactly 12 physics steps without rendering,
pauses physics only for image rendering, then explicitly resumes it. Recorded
sim_time verifies 0.1 s/action. This is not the unsafe pause-without-resume case
described in the project's earlier notes. Cube/Go2 controllers were not changed.

## What remains outside this step

These are expert demonstrations, not a trained learner. The assets are the
project's procedural Isaac Sim task, not original RLBench meshes. Data uses
random XY +/-2 cm; it does not establish broad object/appearance generalization.
Full batch-256 GPU memory, long training, held-out evaluation and VLM rewards
are separate work. `audit_demos.py` checks a batch of 256 on CPU and eight-frame
padding only, without claiming GPU training has been validated at that batch.

## Measured 2026-11-12 deliverable

Fixed dataset: `/home/ubuntu/pingandog/datasets/take_lid_off_saucepan/fixed_100_20261112`.
100 successful episodes, 100 unique seeds, 14,885 transitions, 478,964,550 bytes
compressed episode data (about 457 MiB). Five independently completed batches
(10+23+23+22+22) used disjoint seed blocks; all post-clock-fix collection attempts
were accepted. Earlier timing-debug runs are separate and excluded. Every batch
passed its pilot audit. Full data was reread, physical success recomputed, hashes
verified, and all 100 episodes loaded into replay. CPU batch 256 shape checks and
eight-frame reset padding passed. These are data checks, not GPU memory tests.

Manifest SHA256:
`c9c7f39f38700d9565f8e119b7ac72afc5fadfbc7b337c5eac86897ef0d071ce`.
Reports are `audit.json` and `loader_audit.json` in the fixed dataset. The merged
dataset is a frozen artifact and is not intended as a resumable collection run.
The post-collection auditor was strengthened to independently verify timestamps;
its hash is recorded separately from the historical collection-source hashes.

Training integration test: `outputs/i2o_panda/run_20261112_072620`, loaded the
10-episode pilot from disk (no recollection), ran 8 network-controlled steps and
12 total updates, and passed finite-loss, encoder/critic-change and checkpoint
reload checks. A real-time four-view tensor-stride compatibility issue was fixed
by using reshape rather than view in CNN feature flattening. No task success or
long training is claimed.

Example short check against the final fixed data:

```bash
scripts/run_isaac62.sh isaac_i2o/train.py \
  --task take_lid_off_saucepan \
  --demo-dir datasets/take_lid_off_saucepan/fixed_100_20261112 \
  --steps 8 --batch-size 8 --pretrain-updates 4 --eval-episodes 0
```

This example loads all 100 episodes by default. The GPU integration smoke run
used the 10-episode pilot; final all-100 loading was separately verified on CPU.
Long training has not been started.
