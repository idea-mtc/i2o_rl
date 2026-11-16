# I2O-RL on fixed-base Panda / Isaac Sim

This adapter connects the supplied `i2o_rl-master` I2O agent to fixed-base
Panda cube and saucepan-lid scenes in `/home/ubuntu/pingandog`. It does not
involve Go2, ReKep, RLBench runtime, or a language model. Shared Isaac Sim files and the original I2O
repository are not modified.

## Run on jiangxing

From `/home/ubuntu/pingandog`:

```bash
scripts/run_isaac62.sh isaac_i2o/train.py \
  --steps 64 --demos 2 --batch-size 8 --pretrain-updates 8 --eval-episodes 1
```

Use the existing launcher: it selects an idle GPU, holds the project GPU lock,
and supplies the matching CUDA/Vulkan device. The verified runtime is the
project-local Isaac Sim 6.2.0 release with PyTorch 2.11.0+cu128 on RTX 6000D.
Do not install the upstream conda environment into this runtime.

For a longer experiment (success/convergence not guaranteed):

```bash
scripts/run_isaac62.sh isaac_i2o/train.py \
  --steps 20000 --demos 20 --batch-size 32 --pretrain-updates 1000 \
  --eval-episodes 10 --capacity 10000 --checkpoint-every 1000
```

The process runs until completion; it is not automatically started by this README.
Logs are emitted by the launcher and outputs are under `outputs/i2o_panda/run_*/`.
`latest` points to the latest finished run, including failures; read `result.json`.

Continue model and optimizer training:

```bash
scripts/run_isaac62.sh isaac_i2o/train.py \
  --resume /absolute/path/to/checkpoint.pt \
  --steps 1000 --demos 10 --batch-size 32
```

Resume restores encoder, critic, target critic and available optimizer states.
With `--demo-dir`, fixed data is reloaded; otherwise new demos are collected.
Replay, random streams and local counters are fresh. This is
not an exact replay/RNG continuation. Frame-stack dimensions must match.

## Environment contract

- Fixed-base Panda, PhysX, 120 Hz physics, 12 physical steps per action (10 Hz).
- One world-fixed RGB view at 84x84. Default 2 frames stacked into channels,
  `[views=1, channels=6, height=84, width=84]`.
- Each 15-element state: 7 measured arm joints, normalized finger opening,
  TCP position and XYZW quaternion. Default stacked state has 30 elements.
- Policy observations contain no simulator cube coordinates. The teacher and
  reward function may use them.
- Eight normalized actions: 7 commanded-joint increments (max 0.04 rad per
  control step) and one absolute finger opening (`-1` closed, `+1` = 0.04 m
  per finger). Both mimic-coupled fingers receive the same opening target.
- Joint targets obey URDF limits. The same bounded joint-position bias controller
  as the physical grasp test compensates static tracking error, for both teacher
  and learner. It is not a Cartesian IK controller during learner execution.
- Cube: 4 cm, 50 g, XY randomized +/-2 cm by default. Static table top at 0.2 m.
- Success: lift above settled height by >0.12 m, cube-to-TCP distance <0.06 m,
  finger opening <0.03 m, maintained for 20 consecutive control steps (2 s).
- Sparse reward 1 on success, otherwise 0. Termination on success or cube
  falling/leaving the local workspace; 220-step time limit by default. Discount
  is 0 on true termination and 0.99 on continuing/time-limited transitions.
- Demonstrations use IK only in the teacher, then pass through exactly the same
  normalized action/environment step interface. Only successful complete episodes
  enter demo replay; failed attempts are reported. At most 3 attempts per requested
  demo; collection aborts if the target number is not reached.

This scene is a small single-object integration task. No full collision-aware
trajectory planning or deployment safety controller is implemented.

## Learning and outputs

`vendor/i2o_rl.py` retains the supplied CNN, cross-attention fusion, 3 levels /
5 bins / 51 atoms and distributional Q loss. As of 2026-11-12 it also has a
differentiable SoftArgmax action branch and active demo action MSE. Relative
imports avoid Isaac Sim's unrelated `utils` module. Default inference is soft;
`--action-mode hard` selects the original quantized greedy rollout. See
[SOFTARGMAX.md](SOFTARGMAX.md) for the algorithm choices and measured validation.
Source/deployed hashes are in `vendor/provenance.json`. This is an integration
of the supplied implementation, not a claim of reproducing paper results.

`replay.py` mixes 50% demo and 50% online transitions once online data exists.
Its 16-field batch preserves the original update API; unused task/ACT auxiliary
fields are zero placeholders and are not used by the active loss. There is no
successful-trajectory relabeling in this new replay adapter.

Outputs include `demonstrations.npz`, `metrics.jsonl`, `policy_actions.json`,
`checkpoint.pt`, observation PNGs and `result.json`. Demonstration data includes
RGB/state/action/reward/discount/next observation. The checkpoint includes model
and optimizer states plus environment shapes and run configuration.

`result.json: pass=true` means the integration checks passed: successful expert
data, bounded policy actions, finite losses, encoder/critic weight changes and
checkpoint encoder reload equality. It does **not** mean the learned policy has
passed the grasp task. `eval_results` and `eval_success_rate` report learner
performance separately.

## First measured result

`run_20261111_132359`: 2 successful demos / 250 transitions, 64 online policy
steps, 8 demo-only + 64 mixed updates. All integration checks passed; critic
maximum parameter change 0.002009, encoder change 0.000955. Independent learner
evaluation: **0/1 success**. This is a short smoke run, not a trained grasp policy.

The current action/state/camera configuration differs from the original RLBench
experiment (one view and two stacked frames instead of four views/eight frames).
Long training, repeated evaluation, held-out starts and multiple seeds remain
necessary to establish task performance.

## Saucepan task and fixed demonstrations

See [DEMOS.md](DEMOS.md) for independent collection, episode format and auditing.
`--task take_lid_off_saucepan` selects the four-camera task.
`--demo-dir DIRECTORY` loads fixed data without recollection; absent `--demos`,
all stored episodes are used. [SAUCEPAN.md](SAUCEPAN.md) describes scene geometry
and camera mounts. The clock synchronization correction is recorded in DEMOS.md.


## 2026-11-13 articulated task pair

`aligned/articulated.py` adds procedural PhysX open_drawer and open_door analogues. Success matches the pinned RLBench Python JointCondition: absolute slider displacement >0.15 m; absolute door hinge displacement >25 degrees, with the hinge locked until handle rotation exceeds25 degrees. The drawer corresponds only to a bottom-drawer analogue; native three-variation geometry is not imported. No gripper attachment is used. Native task sources are preserved in tmp/i2o_audit/upstream_open_drawer.py, upstream_open_door.py and upstream_conditions.py.

Both experts, no-contact negative controls and command/observed-gripper consistency passed in remote outputs/articulated_probe/pair_12, three independent resets per task. Drawer includes a weak20N/m return spring to prevent uncommanded sliding; cabinet geometry/inertia/friction, cameras and XY +/-2cm reset distribution are explicit migration differences. Panda arm PD gains are8000/150 and finger gains4000/100. The discrete gripper checks both observed state and lingering actuator target to avoid preserving an open motor target after a close command. This is not native asset conversion or a strict paper benchmark. Door expert turns/unlocks the handle, releases it and pushes the panel by physical contact; no object poses, task flags or phase clocks enter the learner.

run_articulated_pair.py supports two independent processes under a shared GPU lease or, with --separate-gpus, uses run_isaac62.sh for one project-exclusive GPU lease per task. The active run uses SEPARATE GPUs. Each RTX6000D has83GiB; each task retains its own GPU raw-frame cache, model, replay, dataset and checkpoints. No other user's GPU process is stopped. Parallel execution saves wall time only to the extent spare rendering/compute capacity permits; it does not promise2x throughput.

Each task uses100 newly collected successful demonstrations, fresh model initialization, original state120/action8 interface, shared coarse-to-fine head, beta1, no phase heads, sparse0/1 reward, batch256 and planned30250 online steps. Door horizon125 and drawer100 follow downloaded task configs. Paper-text loss reconstruction is retained from the lid experiment, distinct from the unmodified repository's hard/C51/FOSD objective. Periodic evaluation5/1000 steps and final100; single seed only. Imitation gate requires4/5 successful independent resets and TCP motion>3cm before RL, at most5000 warmup updates.

Per-task latest.pt/previous.pt restore model, optimizer and replay metadata. Physical episodes reset and partial episodes are truncated on recovery. The pair supervisor bounds retries, watches stale progress, and stops a task on failed learning gate or motion-collapse alert while allowing the other task to continue. Per-task PAUSE marker stops that task.

Active pair: remote outputs/articulated_training/run_20261113_v2/{open_drawer,open_door}. Service pingandog-articulated-training.service (enabled persistent user service). Earlier run_20261113 was stopped during collection (drawer16 accepted; door command/state assertion correctly prevented training); it is not the active training run and its demonstrations are not reused. Active pointer: outputs/articulated_training/active.json. Each task recollects100 demonstrations. Rejected gripper command/state examples are recorded in manifest.rejections and never enter replay.

## 2026-11-13 native asset training

The separate new-asset run is remote `outputs/native_asset_training/run_20261113`, managed by `pingandog-native-asset-training.service`. It does not reuse the procedural baseline datasets or checkpoints. `--scene native_assets --separate-gpus` launches each task with its own GPU flock; existing unrelated GPU allocations are neither reserved nor stopped. Launch is gated by `ready/<task>.json`, written only after successful expert validation. The training gates, 100 demonstrations per task, 30250 online steps, recovery and evaluation settings above also apply here.

Drawer uses the NVIDIA Isaac Sektion cabinet, bottom drawer, >0.15m success, a 0.1m plinth, root pose (0.95,0,0.4842) with pi yaw and handle-local TCP offset (0.290,0,0.01). The offset is 15mm deeper than Isaac Lab's 0.305m frame to suit the imported Panda finger geometry. Door uses the pinned original RLBench meshes/materials and joint limits converted to USD, on a 0.2m pedestal, root pose (0.36,-0.13,0.20). The root is 6cm closer than the first probe; retaining the original tool orientation rather than the attempted symmetric 180-degree roll avoids the wrist/joint-limit failure during opening. Handle >25 degrees unlocks the hinge, hinge >25 degrees succeeds. The locked hinge motor is mapped to PhysX (max force1000 while locked, restored to1 after unlock).

Experts use physical finger contact without attachment or direct object-joint commands. Door approaches the original first two waypoints (with the original tool orientation), then follows the actual handle/hinge axes with contact feedback. This replaces the source expert's final fixed waypoint/path; it is an adapted expert, not identical RLBench demonstrations. Standard URDF TCP is retained: the Coppelia-to-URDF calibration varied across postures and its fitted transform is not applied. Four 84x84 RGB views, eight stacked frames, state120/action8 and one shared action head are retained, without phase observations. Placement/randomization, camera calibration, PhysX and the paper-text reconstructed objective remain differences from an exact paper reproduction.

On 2026-11-13 the native drawer job was migrated from GPU4 to GPU7 using its saved checkpoint. A per-job `gpu.txt` containing `7` is read by `scripts/run_isaac62.sh`, which passes `PINGANDOG_REQUIRED_GPU` to the existing idle-device/project-flock launcher. Recovery keeps this assignment. Other jobs without the marker retain automatic device selection. Native door stays on GPU0; unrelated jobs are not stopped. Device pins do not reserve memory against unrelated users.

## 2026-11-14 original microwave and bookshelf task pair

New run: remote `outputs/extended_asset_training/run_20261114`, enabled user service `pingandog-extended-asset-training.service`. `run_articulated_pair.py --tasks open_microwave put_books_on_bookshelf --scene native_assets --separate-gpus` uses per-task readiness markers and GPU pins0/2. Each task starts only after its expert and no-contact negative control pass. The previous lid/drawer/door experiments and artifacts are unchanged.

Both models and Python task definitions come from RLBench commit b80e51feb3694d9959cb8c0408cd385001b01382, with download hashes/provenance and license in `assets/candidates/rlbench_<task>_original`. Original meshes, material pixels/UVs, dynamic book masses/inertias, microwave joint axis/limits and waypoint poses are exported through CoppeliaSim4.1's public API and mapped into USD. `asset_tools/build_rlbench_task_usd.py` builds the meshes; `tune_contact_offsets.py` maps thin-object contact offsets to1mm, rest offset0,16/4 solver iterations, retaining box-shaped book convex hulls. No scaling of original objects is applied.

Microwave: source JointCondition is absolute hinge displacement >80degrees; source limits0..90degrees; horizon125. Root placement(0.45,0,0.30), support top0.28m. The original zero-target-velocity motor and max torque2.073600292Nm are mapped to a zero-stiffness, damped PhysX angular drive (damping2000, force limited to the original torque). Expert approaches the original two grasp waypoints then follows the original hinge using actual contact, no attachment or direct joint actuation. Safe approach interpolation replaces native collision-aware motion planning.

Bookshelf: source variation0 requires only `book2` in the original success sensor and stops the original expert at waypoint5. Source variants1/2 require2/3 books; they are not silently combined here. Variation0 follows the downloaded wrapper's default variation and the README's one-variation collection setting; the paper does not identify its exact variation index. Horizon175. Root placement(0.40,0,0.04), support top0.03m. All3 original books remain dynamic and graspable. Expert visits original waypoints0..5 with a safe clearance approach and initial settled-book translation compensation. Original gripper/TCP-to-imported-URDF calibration is applied to teacher target poses only; its max matrix variation across3 calibration poses is0.0034168, so it is an approximation, not an exact robot identity.

RLBench books are registered graspable and use gripper attachment. The adapter uses a runtime fixed joint when a closing gripper's TCP sensing box intersects a book, releases on open, and never changes object poses during policy rollout. This approximates the native gripper proximity sensor: TCP half-extents(.025,.025,.035)m, OBB projection test; attachment may occur at the start of closing rather than after Coppelia actuation completes. The policy sees only the shared state120/action8 contract, with no phase or attachment flag. This differs from the purely physical drawer/door grasps. The original bookshelf success condition does NOT require release. Sensor success uses original oriented rectangular volume and book collision box SAT overlap, not center-only distance. Cross-validation against `sim.checkProximitySensor(original success, original book2)` matches175/175 sampled positions (`export/sensor_reference.json`, `sensor_validation.json`); this finite check is not a proof for all poses.

Unmatched elements remain explicit: PhysX contact/control vs Coppelia, approximate imported Panda/TCP/gripper sensing, camera calibration, original global placement/yaw sampling (current reset onlyXY +/-2cm, yaw fixed), and the previously documented paper-text reconstructed loss rather than unchanged downloaded C51/FOSD code. Both tasks use4 RGB84x84 views,8 frames, state120/action8, a single head, beta1, sparse0/1/noVLP,100 fresh accepted demonstrations,4/5 imitation gate with>3cm motion,5000 warmup cap,30250 online steps, periodic5 and final100 evaluations. Single training seed; not a complete three-seed paper reproduction. Known expert IK-unreachable episodes are rejected with seed/reason and counted against the200-attempt cap, never admitted as successes; other control/data assertions remain fatal. Checkpoints restore model/optimizers/replay/RNG and reset a physical partial episode on recovery.
