# SoftArgmax imitation implementation and acceptance

Implemented 2026-11-12 for the existing fixed-base Panda/Isaac Sim adapter.
This adds an explicit differentiable action path; it does not claim complete
paper reproduction or a successful learned grasp policy.

## Changes

`vendor/cqn_utils.py:soft_zoom_in` calculates weights `softmax(beta * Q)`
over the **five action bins**, then a floating expected bin index. For bounds
`low, high` and `width=(high-low)/bins`, the next bounds are:

```text
index = sum(softmax(beta*Q)[i] * i)
next_low = low + width * index
next_high = next_low + width
```

At the last level the action is `(low+high)/2`. The index lies in `[0,bins-1]`,
so all bounds remain nested and actions remain within [-1,1] without clipping.
`C2FCritic.get_soft_action` preserves gradients through all three levels and
their previous-action conditioning. Q is the expected value of the existing
51-atom return distribution; softmax is taken over action bins, not atom indices.
No hard argmax, floor, integer casts or detach are on the imitation action path.
This smooth expected-interval refinement is an explicit implementation choice:
the supplied paper does not fully specify every interval-update detail.

`I2OAgent.imitation_update` performs **MSE-only** updates, without TD, margin,
FOSD, reward or augmentation. This isolates the claimed action-regression
gradient. The existing combined `update` now also uses differentiable MSE on
demo rows instead of the old nondifferentiable L1 branch. Its inherited C51,
FOSD and value-margin losses remain. MSE weight is `bc_lambda` (formerly the
inactive L1 branch had a fixed 0.5 factor). These retained distributional losses
and replay are not identical to the paper's simplified equations.

`act` defaults to continuous soft inference; `--action-mode hard` preserves
quantized greedy rollout for comparison. Both modes use soft imitation in the
loss. Hard greedy action selection is retained for the existing Q backup.
`--softargmax-beta` defaults to 1.0. Shape-compatible old weights can load, but
inference semantics change unless hard mode is explicitly chosen.

At beta=1, bounded return support [-2,2] limits how concentrated soft weights
can become; exact extreme actions +/-1 need not be attainable. This can impose
an endpoint regression floor and must be revisited with reward/support design
for a full paper-level experiment, rather than hidden by changing evaluation.

## Reproduce the isolated check

From `/home/ubuntu/pingandog`:

```bash
scripts/run_isaac62.sh isaac_i2o/validate_softargmax.py \
  --data outputs/i2o_panda/run_20261111_132359/demonstrations.npz \
  --updates 1000 --batch-size 32 --fit-samples 64 --beta 1 --eval-episodes 2
```

The launcher selects an idle GPU with its existing project lock. Shared runtime,
Go2 and original `i2o_rl-master` sources are unchanged. Old deployed modules are
backed up in `isaac_i2o/backups/pre_softargmax_20261112/` on the server.

## Acceptance result

Run: `outputs/softargmax_panda/run_20261112_035910/`, pass=true for mechanism
verification, **not** task success.

- Three-level double-precision finite-difference gradcheck: passed.
- Old hard-path action MSE requires_grad=false: confirmed.
- New output-head and all three Q-level gradients nonzero at initialization.
- Upstream heads start at exactly zero, so encoder/trunk gradients are initially
  zero. After 20 MSE-only updates, CNN, visual/state projections, fusion, Q trunk,
  Q head and all three levels have finite nonzero gradients. This is a module-level
  gradient test, not a claim that every parameter receives a gradient.
- Fit 64 sampled transitions from first successful episode, batch 32, 1000
  updates, lr 5e-5; second episode (125 transitions) held out entirely.
- Train MSE: 0.11372009 -> 0.000449401 (99.60% reduction).
- Train arm MSE: 0.03115148 -> 0.0000115403.
- Train gripper MSE: 0.69170034 -> 0.00351443.
- Held-out MSE: 0.11557116 -> 0.04063938 (64.84% reduction).
- CNN and action-head weights changed. A combined C51/margin/FOSD/MSE update
  also executed with finite metrics and nonzero CNN gradients. It was restored
  away before rollout, so the evaluated checkpoint remains MSE-only.
- Loaded checkpoint drove two 220-step Isaac Sim episodes using only network
  actions, no teacher calls: **0/2 successful grasps**. Action values were finite
  and bounded. Small-data fitting has not established closed-loop generalization.

`result.json`, `metrics.jsonl`, `predictions.npz`, `policy_trace.json`, evaluation
PNGs and `soft_imitation.pt` are saved in that run. The data split is saved by
index; teacher rewards were not used for fitting. Training sample fit does not
substitute for episode success or independent multi-seed testing.

## Continue joint I2O training from these weights

```bash
scripts/run_isaac62.sh isaac_i2o/train.py \
  --resume outputs/softargmax_panda/run_20261112_035910/soft_imitation.pt \
  --steps 1000 --demos 10 --batch-size 32 --action-mode soft
```

The pure imitation checkpoint exports model weights only. The training entry
reports `model_only_new_optimizers` and initializes new optimizers in this case.
Ordinary training checkpoints retain their optimizer states. Replay, demo
collection and counters are fresh as documented in README. The command above
is an example; no long job is automatically started.

Joint-training continuation was measured in `outputs/i2o_panda/run_20261112_040128`: 16 policy steps, 4 demo pre-updates plus 16 online combined updates, fresh optimizers, all integration checks passed. This is not another grasp-success evaluation.

Joint-training continuation was measured in `outputs/i2o_panda/run_20261112_040128`: 16 policy environment steps, 4 demo pre-updates plus 16 online combined updates, fresh optimizers from MSE-only weights, all integration checks passed. This is not an additional grasp-success evaluation.
