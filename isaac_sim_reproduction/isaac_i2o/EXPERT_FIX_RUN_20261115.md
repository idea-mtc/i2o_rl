# Microwave expert provenance correction, 2026-11-15

Authorized: use only physical GPUs 2, 3, 4. Target final results by 2026-11-16
09:00 Asia/Shanghai. The deadline is a runtime limit, not a promise of success.

Server run: `/home/ubuntu/pingandog/outputs/microwave_expert_fix/run_20261115`.
Service: `pingandog-microwave-expert-fix-20261115.service`.

## Intervention

`aligned/expert_replay.py` derives from the historical replay implementation.
Success still selects successful-pool membership. An explicit `is_expert` flag
selects action MSE and expert Q-margin, through the existing loss mask slot.
All sampled transitions remain eligible for TD with unchanged weight 0.1.
Old replay entries are migrated by matching original manifest paths, never by
copying successful/demo flags. Checkpoints persist and validate provenance.
The historical training path remains unchanged for comparison.

## Experiment

Three branches use the identical immutable source IL checkpoint
`outputs/extended_asset_training/run_20261114/open_microwave/validated_start.pt`.
They differ in online RNG seeds 1, 2, 3 on physical GPUs 2, 3, 4 respectively.
These are repeated fine-tuning runs, not independent from-scratch training.
The source100 demos are shared read-only; models and source snapshots are
isolated under this run directory.

Short collection is uninterrupted to1000. Checkpoints500 and1000 are evaluated
on seeds900000+5*i for i in0..19, after collecting1000 transitions. Baseline is
reevaluated per worker. A branch enters full training only if baseline>=14/20,
500>=12/20, and1000>=max(14,baseline-2)/20, with finite updates. Historical
uncorrected B1000=3/20 is a reference, not a matched-stream intervention.

Passing branches continue to30250 total online updates, with20-episode
intermediate evaluations each5000 steps. Best and latest checkpoints are saved
separately. The source IL, final30250 and best-development model are each
measured on independent held-out seeds910000..910099. Final seeds are never
used for checkpoint selection. Failed short branches stop and retain results.

## Device and recovery controls

CUDA visibility alone was insufficient: an initial device probe created a6MiB
GPU0 graphics context and was immediately stopped before training. The final
launcher uses bubblewrap with a private /dev exposing only each assigned NVIDIA
and DRM device. Rendering, physics and PyTorch use local device0 mapped to the
assigned physical GPU. A second probe passed and all actual workers were
verified solely on their assigned GPUs. The host supervisor additionally audits
all child GPU allocations and terminates only its own process group on a
violation. It never selects a GPU outside2,3,4 or stops unrelated processes.

A project lock is held per GPU. Launch requires24,000MiB free. Checkpoints are
atomic every250 updates; a partial physical episode is truncated on resume.
Each branch allows up to3 process launches and restarts after30min of unchanged
progress. Less than8GiB free disk or reaching the morning deadline stops work.
Systemd user service and user lingering keep it independent of SSH sessions.

Live outputs: `summary.json`, `REPORT.md`, `supervisor.jsonl`, per-branch
`progress.json`, `gate.json`, `result.json`, and `gpuN_device_audit.json`.
Three CPU provenance tests passed inside each initialized Isaac worker.
