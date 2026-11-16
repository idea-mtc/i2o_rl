# Lid imitation repair

The old RL run and its watchdog were intentionally stopped on 2026-11-12 at
approximately online step 6720 (latest periodic checkpoint may be earlier).
`outputs/i2o_panda/WATCHDOG_PAUSE` blocks automatic recovery of that experiment.

The diagnostic showed two independent problems: bounded C51 Q scores with beta1
soft action extraction shrink amplitudes; a time-scripted expert has ambiguous
stationary observations at phase boundaries. Hard action extraction reduced
teacher-forced error but still stalled before descent. A new closed-loop imitation
experiment must pass before resuming online RL.

The repair is an explicit engineering variant, not the original paper setup:

- `phase_clock.py` appends five phase indicators and local phase progress, computed
  solely from the observable control step and expert schedule 35/40/25/45/35.
  State shape changes from 120 to 126. No object coordinates, IK result or expert
  action is supplied to the policy. The network still predicts all eight actions.
  This schedule is specific to the fixed-duration task; performance under altered
  speed, disturbances or delays requires additional validation.
- SoftArgmax beta increases to 5. Optional differentiable endpoint rescaling maps
  outermost refined bin centers from +/-0.992 to the normalized actuator range.
  Existing models retain default beta1/no rescaling unless explicitly configured.
  Rescaling preserves differentiability and bounds; it does not clip gradients.
- `repair_imitation.py` initializes from the step5000 visual/Q weights, zero-extends
  the state projection for the new clock features, and uses fresh optimizers.
  Training is pure MSE, phase-balanced, batch128, lr5e-5. Existing 100 episodes are
  reused without relabeling executed actions. Both training and inference apply
  the same soft-action decoding. Target critic is synchronized before evaluation.
- Every 500 updates, evaluate three new scenes. Once 3/3 succeed, evaluate 20
  unseen seeds 300000..300019 and record four-view videos and per-step actions.
  `validate_imitation_gate.py` independently checks the final 30 control steps
  of all 20 successful trajectories: lift>.15m, XY offset<.05m, knob/TCP<.05m,
  fingers<.025m and relative grasp motion<.015m. It hashes the validated checkpoint.

Passing 20/20 is a sample-based engineering gate, not a guarantee of universal
robustness. No RL restart is authorized by loss reduction alone. The old run's
statistics must not be merged with this changed observation/action variant.

Validation includes local phase boundary/stacking tests, original data-integrity
tests, and actual CUDA finite-difference gradient and actuator endpoint checks in
the repair process. Empirical success must be read from `physical_gate.json`.

## Closed-loop correction data

The first repaired policy (500 MSE updates) succeeded in most validation scenes
but failed seed300006; it did not pass the stability gate and RL remained stopped.
`repair_imitation.py --resume-repair ... --dagger 6` continues those imitation
weights/optimizers and gathers corrective labels on learner-visited states.
Rollout actions remain network outputs. The expert generator computes the desired
time-indexed trajectory using the initial scene geometry, then labels corrections
relative to the current command target. Raw RGB, robot/clock state, expert labels
and separately executed learner actions are saved in `dagger_*.npz`. Half of each
MSE batch comes from correction data, half from original demonstrations, balanced
by phase. Corrective observations are not marked as successful RL demonstrations.

Debug validation now includes the failed scene300006; final acceptance uses seeds
400000..400019, separate from those debugging scenes. A failed final rollout stops
that acceptance pass early; it cannot produce a validated checkpoint.
`validate_imitation_gate.py` reads the required seed list from the run config and
recomputes the same physical checks. Its synthetic tests verify that false success
flags, slip and duplicate scenes cannot pass. Server tests run with the Isaac
bundled Python, which supplies NumPy; the watchdog itself uses only the standard
library. The 15 current unit tests pass in that environment.

A validated model may launch a new RL experiment using
`runtime/isaac-sim-6.2.0/python.sh isaac_i2o/launch_validated_lid_rl.py <run>`.
The launcher independently validates the gate and checkpoint hash, checks the old
training service is stopped, preserves the old pointer, and starts 30,250 new
online steps with the changed observation/action configuration. An additional
10-episode evaluation at online step1000 checks for early degradation. The old
6,000-step checkpoint is not mixed into the new experiment's step count.

## Absolute targets and phase heads

Incremental-output trials could grasp but still drifted during lift/hold; they
were not accepted. The absolute-target contract in `absolute_joints.py` maps the
network's first seven normalized outputs to joint-limit-bounded position goals.
The regular controller tracks them with at most 0.04 rad target movement per
control step. Repeating a fixed action cannot integrate a bias indefinitely.
Original recorded actions are converted by integrating the *commanded* deltas
from the known home target, not by substituting measured positions. Tests show
converted demonstrations reproduce the same command trajectories. Gripper
semantics remain unchanged. Pure absolute-target imitation uses new critic
weights and only transfers the old visual encoder.

The shared-head absolute model could still ignore the phase condition on rollout.
The `--phase-heads 5` variant selects separately learned dueling value/advantage
output heads using the five observable clock phase indicators. All eight actuator
actions remain learned predictions; the clock selects a neural head, not an IK
command. The ordinary I2O architecture remains the default with `phase_heads=1`.
Soft-action gradients remain intact within each phase. A runtime test verifies
phase selection and that a sample only updates its chosen output head.

A validated absolute-target RL run uses normalized exploration std0.002 rather
than0.01, since the same normalized noise now spans a full joint range rather
than a 0.04-rad increment. Initial evaluation occurs at1000 online steps to check
for degradation. These explicit engineering changes mean it is not a strict
reproduction of the paper's original observation/action configuration.

## Phase goals, transition semantics and Cartesian control

The accepted candidate must also avoid teacher-forced identity shortcuts: raw
instantaneous position targets can be approximated by copying current pose. The
`--demo-phase-goals` loader extracts the recorded final command of each expert
phase and uses that as a constant goal label within the phase. This changes the
BC target objective; it does not alter the immutable original recordings.

Such relabeled goals are not the one-step actions recorded with the old next
observations. Therefore `--demo-phase-goals` requires `--demo-bc-only`: demo rows
contribute imitation/FOSD/margin objectives but are excluded from TD loss. Only
new factual online transitions contribute TD. A runtime test of the actual
critic update confirms that changing a demo's reward does not affect TD loss or
gradients, while changing an online reward does. All-demo batches have zero TD
loss, with a safe denominator. This is another explicit departure from the paper.

Joint-target trials still exhibited grasp-position error. The Cartesian candidate
uses `--cartesian-goals`: four learned actions (TCP XYZ, gripper), fixed downward
tool orientation, normalized workspace [0.38,-0.08,0.30]..[0.54,0.08,0.58] meters.
Ordinary robot IK transforms network-selected TCP goals into joint targets; it
does not access object geometry or choose task goals. Per-step joint target slew
is still limited to0.04rad. Expert BC targets are TCP positions obtained by FK of
the original commanded phase-end joint targets, not measured object pose passed
to the policy. This reduces redundant-joint regression sensitivity and gives a
more useful physical scale to the coarse-to-fine action range.

The final applicable configuration must be read from the validated run; intermediate
incremental, absolute-joint, and instantaneous-target trials were not accepted.
The final video and physical gate must correspond to that same checkpoint.
Current unit suite:22 passing tests, plus runtime phase-head/TD-exclusion tests.

## Validating the frozen Cartesian candidate

The Cartesian candidate completed500 MSE updates from the old visual encoder
with new four-action, five-phase critic heads. A frozen step500 checkpoint is
being evaluated in run `outputs/imitation_repair/run_20261112_123759`; the three
debug scenes passed before the20 new acceptance scenes began. Acceptance status
must still be read from the physical gate, not inferred from these debug results.

`fast_panda_ik.py` implements the exact imported URDF chain with an analytic
geometric Jacobian and damped least-squares pose IK. Startup asserts its FK agrees
with the existing environment FK at the home configuration to1e-10. Runtime
solver checks positional error<0.05mm and orientation error<0.0005rad; failures
fall back to the original SciPy IK. A finite-difference Jacobian check and60
reachable pose tests passed in about0.1sec, maximum pose error0.04974mm. The
controller seeds with the previous goal solution, caches goals within0.25mm,
and clears that cache at reset. Those changes affect ordinary robot kinematics,
not learned target selection. All23 current unit tests pass.

## Acceptance and resumed RL (verified 2026-11-12)

Run `outputs/imitation_repair/run_20261112_123759` passed the independent physical
gate with20/20 successful unseen scenes and frozen step500 imitation weights.
Final lift0.20147..0.20321m, worst final XY drift0.00483m, worst final knob/TCP
distance0.00293m, worst last30-step grasp slip0.000148m. Validated checkpoint
SHA256=e5b3c5bb19630a7dfa43091b05c319c56e7d2afbd3d7466e377a38bb74ec3e0e.
The acceptance model uses clock conditioning, five output heads, four Cartesian
goal/gripper actions, fixed downward orientation, phase-end BC labels, beta5
and soft endpoint rescaling. No DAgger data was used in this final candidate.

RL started in `outputs/i2o_panda/run_repaired_20261112_125239`, service
`pingandog-i2o-repaired-20261112_125239.service`, and actually reached160 online
updates with finite losses. Peak reserved CUDA memory5.73GiB. Demo rows are BC
only; TD uses real online transitions. Exploration std0.002, augmentation pad0
keeps the validated calibrated-camera input distribution. Additional evaluation
at1000 steps is pending. No post-RL success rate has yet been established.
The watchdog was resumed only after online updates were verified and now tracks
this new run. Old training remains stopped; its step count is not merged.

Local user-facing evidence: `panda_validation/imitation_repair/RESULTS.md`,
`physical_gate.json`, `validated_grasp.mp4`, and `repair_start_status.json`.
