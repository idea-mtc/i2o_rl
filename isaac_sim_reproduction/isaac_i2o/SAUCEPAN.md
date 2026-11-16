# Panda saucepan-lid scene and four-camera expert validation

Implemented and physically tested on jiangxing, 2026-11-12, Isaac Sim 6.2 / PhysX.
This is a project-authored procedural analogue of the take-lid-off-saucepan
task, not the original RLBench .ttm model or an exact paper benchmark replica.

## Reproduce

```bash
cd /home/ubuntu/pingandog
scripts/run_isaac62.sh isaac_i2o/verify_saucepan.py --episodes 3 --seed 12
```

Use the existing launcher for its idle-GPU selection and project GPU lock.
Results: `outputs/panda_saucepan/run_*/`; `latest` points to the latest completed
run, including failures. Always check result.json. The verified run is
`run_20261112_065954`, pass=true, three successful expert trials.

## Scene

- Fixed-base Panda and the same bounded joint-bias position controller as the
  previously verified physical grasp scene. Physics 120 Hz, commands 10 Hz.
- Static table top at 0.2 m. Hollow saucepan: radius 0.082 m, bottom 0.008 m,
  annular wall made of 32 convex boxes, rim height 0.083 m, long side handle.
- Independent dynamic lid: 0.2 kg, disc radius 0.084 m, thickness 0.008 m,
  stem and grasp knob radius 0.018 m. Disc, stem and knob are one compound
  rigid body with explicitly authored mass/inertia. It rests on the pot rim.
- Grasp material static/dynamic friction 1.0/0.8. Grasp is contact/friction,
  with no robot-lid attachment, weld, suction or in-episode object teleportation.
- Pot/lid initial XY is nominal [0.46,0] or randomized +/-0.02 m per axis.
  Initial state setters are used only by reset before physical execution.

Saved scene: `/home/ubuntu/pingandog/assets/tasks/take_lid_off_saucepan/scene.usda`.
Configuration: same folder `config.json`. Scene references the already imported
project-local Panda; no online asset download is required. This exported USD
contains the nominal initial scene. Expert control and camera stream setup are
provided by the Python entry, not embedded as autonomous behavior in the USD.

## Cameras

Order: **front, wrist, left_shoulder, right_shoulder**. Three cameras are fixed
to the world. The wrist camera is parented to the actual Panda link7 with a
constant transform accounting for the URDF hand offset; it is not a world-fixed
camera renamed as wrist. Exact paths, optics and positions are in the source
and config. Native RGB is 640x480; bilinear resizing of the same frame produces
84x84 policy RGB. One render product per camera avoids duplicated-stream issues.

Raw RGB tensor: `[4,3,84,84]`. With default two-frame channel stacking:
`[4,6,84,84]`; state `[30]`; normalized action `[8]` (7 joint deltas + gripper).
The wrist near clipping plane is 0.005 m to retain near-field handle visibility.
Preview videos are labeled 2x2 composites at 1280x960, 10 fps simulated time.

## Expert and success rule

Expert uses object pose plus URDF numerical IK: approach above knob, descend,
close jaws, lift by a commanded 0.20 m and continue holding. Actions pass through
the same normalized joint-delta/absolute-gripper interface exposed to a future
learner. No learned policy or I2O training is used in this validation.

Success requires measured lid lift >0.15 m, XY deviation <0.05 m, knob-to-TCP
distance <0.05 m and measured finger opening <0.025 m for 30 consecutive control
steps (3 simulated seconds). The raised interval may include the end of ascent;
this is not a claim of 3 seconds motionless at the final height. The task stops
when its success condition is met. An additional expert check requires relative
knob/TCP drift <0.015 m during that interval and base drift <0.0001 m.

These explicit geometric thresholds are our Isaac Sim acceptance criteria.
RLBench uses grasp state and a proximity-success detector; original asset poses,
sensor volume, collision meshes and control timing have not been matched exactly.
Do not compare this expert success rate directly to the paper's learner results.

## Measured acceptance

`outputs/panda_saucepan/run_20261112_065954/`:

- One nominal + two random XY trials: **3/3 passed**.
- Nominal trial: final lift 0.20024 m, minimum lift over last 3 s 0.15672 m,
  relative knob/TCP range 0.0000227 m, base drift 0.0000020 m.
- Open-jaw negative control followed the same approach/lift route and did not
  lift the lid (maximum height change ~6e-8 m).
- No physics joints involving the lid were found in the stage.
- All four streams valid/nonblank and all four images changed during the task.
- Fixed camera matrix drift 0; wrist local mount drift 0; nominal wrist camera
  translation ~0.0634 m, proving its world pose changes with the arm.
- Initial and lifted four-view images were visually inspected; the lid is visibly
  separate from the hollow pot and held at its knob. MP4 format/duration checked.

Each episode contains phase PNGs for all cameras at native and 84x84 resolutions,
four-view composites and `expert_four_views.mp4`. Other outputs: `result.json`,
`config.json`, `trace.json`, `camera_snapshots.json`, `scene_initial.usda`.

Initial failed setup runs are preserved separately. Fixes included building the
new rigid body while physics is stopped, avoiding defaultPrim changes during
live physics, and using one native render product per view. No shared runtime,
Go2 asset or existing cube/SoftArgmax environment source was modified.

## Future training

`saucepan_env.SaucepanEnv` exposes reset/step/observe/expert_actions. The existing
`train.py` still selects the cube environment; a task selector and demo collector
must be connected before saucepan training. This work validates scene, cameras,
and scripted expert only. Three trials do not establish broad robustness or
provide the paper's 100 demonstrations.

## Fixed-data collection and training entry

The task is now selectable in `train.py` using `--task take_lid_off_saucepan`,
and the independent collector saves raw four-camera episodes. See DEMOS.md.
Collection/training native images are 168x126 at the same aspect/mounts, resized
to 84x84. A subsequent clock audit found that live rendering in the initial scene
verification added physics steps; its stated 10 Hz was nominal. The current
saucepan controller renders while paused and explicitly resumes; collected data
now verifies an actual 0.1 s action interval. Prior measured geometry/success
reports are preserved, and are not relabeled as clock-validated data.
