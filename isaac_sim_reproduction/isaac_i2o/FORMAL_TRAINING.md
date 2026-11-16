# Saucepan I2O formal run, seed 1, 2026-11-12

The user authorized a full batch memory/speed test and, on success, starting a
formal single-task run on jiangxing. This run is I2O-only, not full VL-I2O.

## Measured batch-256 acceptance

`outputs/i2o_panda/run_20261112_075253/benchmark.json` passed all checks:

- All 100 fixed demos loaded, four views, 84x84, eight stacked frames.
- Batch 256, float32; actual MSE forward/backward/AdamW updates and combined
  C51/FOSD/margin/MSE updates executed, no OOM and finite losses.
- 8 MSE warmup updates + 16 online combined updates; periodic evaluation after
  step 8, resumed training, final evaluation and checkpoint reload passed.
- PyTorch peak allocated: 4.825 GiB; peak reserved: 5.588 GiB.
- Entire GPU used at benchmark end: 8779 MiB / 85651 MiB. This last value is
  an end-of-test device reading, not a sampled whole-device peak.
- Mean online step: 0.9580 s, post-warmup median: 0.9385 s, including interaction,
  replay sampling, CUDA transfers and optimizer updates, excluding evaluation.
- 260-step evaluation episode: approximately 46.3 s.
- Data/scene load ~27.7 s; 8 MSE updates ~4.10 s.

Single-seed estimate: ~8.05 hours online updates, ~8.5 minutes imitation warmup,
~2.06 hours for up to 160 evaluation episodes, plus I/O. Plan **10–12 hours**
under similar load; earlier task success can shorten evaluation. This is an
estimate from a short benchmark, not a completion guarantee.

## Started formal run

Run directory: `/home/ubuntu/pingandog/outputs/i2o_panda/run_20261112_075738`.
GPU selected by existing launcher: 0.
Console log: `/home/ubuntu/pingandog/logs/saucepan_formal_seed1_20261112.log`.

Configuration: `configs/formal_saucepan_seed1_20261112.json`.
Launch script: `bash isaac_i2o/run_formal_saucepan.sh` (do not launch it a second
time while the existing run is active).

| Setting | Formal run |
|---|---|
| Task/robot | Procedural saucepan lid / fixed-base Panda |
| Demos | Fixed, audited 100 episodes, 14,885 transitions |
| RGB | Four cameras, 84x84; eight-frame channel stacking |
| Input shapes | RGB [4,24,84,84], state [120], action [8] |
| Batch | 256, half demo/half online after online data exists |
| Optimizer | AdamW, lr 5e-5, weight decay 0.1 |
| Value/action network | 3 levels, 5 bins, 51 return atoms, SoftArgmax beta=1 |
| Loss weights | Action imitation 1, value margin 1 with margin 0.01, TD 0.1; inherited FOSD retained |
| Discount/target update | 0.99 / tau 0.02 |
| Initialization | Fresh model; benchmark checkpoint is not reused |
| Imitation warmup | 1000 MSE-only updates, then target critic synchronization |
| Online training | 30,250 environment steps, one combined update per step |
| Replay | Capacity 50,000 online transitions, plus fixed demos |
| Checkpoints | Atomic latest every 1000 steps, numbered every 5000; warmup every 250 |
| Periodic evaluation | Every 5000 steps, 10 episodes |
| Final evaluation | 100 episodes, seed stream 100000 |
| Training seed | 1; no additional seeds started |

Optimized parameters are the four CNN encoders, robot-state projection,
cross-attention fusion and hierarchical Q/value/action computation network.
The target critic follows via soft updates during joint RL; it is synchronized
after MSE-only warmup. No VLM is trained.

## Monitoring and artifacts

`progress.json` is atomically updated throughout startup, pretraining, online
training, evaluation and completion/failure. `metrics.jsonl` contains loss data;
`episodes.jsonl` contains train outcomes; `evaluation_history.json` contains
periodic held-out-initialization results. Evaluation never enters replay and
RNG streams are restored afterward. Because the physical environment is shared,
a periodic evaluation explicitly cuts the current partial training episode as a
time limit (bootstrapping retained) and starts a fresh training episode afterward.

The run preserves a copy of Python sources, dataset hash and resolved network
configuration. `checkpoint.pt` is the latest model/optimizer checkpoint;
`pretrain_checkpoint.pt` is saved during warmup. Optimizer/model continuation is
supported. Online resume restores checkpoint step/update counters and requires
`--pretrain-updates 0`; it preserves the saved target network. Online replay and
RNG/environment state were not saved, so this is not an exact whole-run continuation.

## Paper alignment and limitations

Known paper settings matched: Panda, four RGB views at 84x84, 100 demos,
batch 256, AdamW lr5e-5/wd0.1, discount0.99, coarse-to-fine levels/bins and beta1.
Eight-frame stacking follows repository configuration. Paper training-step units
and phase schedule are incompletely specified: this run explicitly counts
30,250 online steps/updates **plus** 1000 MSE warmup updates. Warmup length,
evaluation cadence and replay capacity are engineering choices, not claimed as
paper-reported values. It retains repository C51/FOSD losses and sparse +1
success rewards, not the full VLP-generated shaped rewards or the paper's
simplified squared Bellman objective. Geometry is the project Isaac Sim analogue,
not original RLBench assets. A first seed is running; the paper's three-run
averages have not been reproduced. Completion and grasp success remain unproven
until the running experiment finishes and evaluation results are checked.

## Recovery on 2026-11-12

Original run stopped around 09:17 UTC; its last progress record was step 4256.
The log ends with `Terminated`, without a Python traceback or CUDA OOM. Host
uptime rules out a server reboot. Accessible kernel journal has no entries;
dmesg is permission restricted. SSH session 78203 closed at 09:17:16 UTC, near
the interruption, but available records do not prove the signal sender.
The workspace isolation service only maintains a VS Code extension bootstrap;
its coincident execution is not evidence that it killed training.

Recovery uses `run_resume_saucepan.sh` and the original atomic `checkpoint.pt`,
in a separate run directory. A transient systemd user service
`pingandog-i2o-saucepan-recovery-20261112.service` owns the process independently
of SSH login scopes; user lingering was already enabled. Logs go to
`logs/saucepan_recovery_20261112.log`. No training hyperparameters were reduced.
The online replay restarts and the environment resets; this must be reported
when comparing results to uninterrupted training.

Verified recovery checkpoint: online step 4000, total updates 5000; all model
state tensors passed finite-value checks, both optimizer states loaded, and
new finite online updates were observed after reconnecting over SSH. Original
last reported step 4256 implies approximately 256 uncheckpointed steps replayed.
Recovery run: `outputs/i2o_panda/run_20261112_103403` on GPU 0.
