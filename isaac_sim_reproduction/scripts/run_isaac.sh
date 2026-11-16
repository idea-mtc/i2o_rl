#!/usr/bin/env bash
# pingandog per-process runtime environment for Isaac Sim on this host.
# Follows the current-driver approach of the proven 2026-10-28 Navforge
# launcher ("run_isaac_current.sh", read-only reference):
#
#   Kernel and system user libraries are now BOTH 595.99.02 - the old 595.84
#   compatibility bundle must NOT be loaded (unset LD_LIBRARY_PATH), the
#   system Vulkan ICD is used directly, and plain nvidia-smi works.
#
# pingandog additions on top:
# - holds a project-local flock per GPU so two pingandog runs never share a card
# - waits (up to 30 min) for an idle card instead of crowding a busy one
# - persistent RTX shader cache under runtime/cache (first compile ~9 min)
# - tees all output to logs/isaac_run_<ts>_<pid>.log
#
# Usage: scripts/run_isaac.sh <script.py> [args...]
set -euo pipefail

ROOT="/home/ubuntu/pingandog"
ISAAC="${PINGANDOG_ISAAC_ROOT:-/home/ubuntu/Navforge Forge/runtime/isaac-sim-6.0.1}"

[ -d "$ISAAC" ] || { echo "run_isaac.sh: Isaac not installed at $ISAAC" >&2; exit 1; }
[ -f "$1" ] || { echo "usage: run_isaac.sh <script.py> [args...]" >&2; exit 1; }

mkdir -p "$ROOT/runtime/locks" "$ROOT/runtime/cache" "$ROOT/logs"

# Never inherit environment overrides that break device pairing or load stale
# driver userspace (595.84 compat bundle era is over; kernel/user = 595.99.02).
unset LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES 2>/dev/null || true

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_DRIVER_FILES="$VK_ICD_FILENAMES"
export VK_ICD_FILE_NAMES="$VK_ICD_FILENAMES"
export GFX_DISABLE_EXCLUSIVE_FULLSCREEN=1

# --- pick idle GPU: require util==0 AND project lock; wait instead of crowding ---
GPU=""
for attempt in $(seq 1 60); do
  GPU_JSON=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits)
  for CAND in $(echo "$GPU_JSON" | sort -t, -k2,2n -k3,3n | awk -F', *' '$2==0 {print $1}'); do
    if [ -n "${PINGANDOG_REQUIRED_GPU:-}" ] && [ "$CAND" != "$PINGANDOG_REQUIRED_GPU" ]; then
      continue
    fi
    exec 9>>"$ROOT/runtime/locks/gpu$CAND.lock"
    if flock -n 9; then
      GPU=$CAND
      break
    fi
    exec 9>&-   # locked by another pingandog run; try next card
  done
  [ -n "$GPU" ] && break
  echo "$(date -Is) all GPUs busy or locked, waiting 30s (attempt $attempt)" >> "$ROOT/logs/gpu_selection.log"
  sleep 30
done
if [ -z "$GPU" ]; then
  echo "run_isaac.sh: no idle GPU after 30 minutes; aborting instead of crowding a busy card" >&2
  exit 4
fi
echo "$(date -Is) run_pid=$$ picked GPU $GPU from: [$(echo "$GPU_JSON" | tr '\n' ';')]" >> "$ROOT/logs/gpu_selection.log"

export PINGANDOG_GPU="$GPU"
export PINGANDOG_ROOT="$ROOT"
export XDG_CACHE_HOME="${PINGANDOG_CACHE_ROOT:-$ROOT/runtime/cache}"

SCRIPT="$1"; shift
cd "$ROOT"
RUN_LOG="$ROOT/logs/isaac_run_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "run_isaac.sh: GPU=$GPU script=$SCRIPT log=$RUN_LOG"
exec > >(tee -a "$RUN_LOG") 2>&1
exec "$ISAAC/python.sh" "$SCRIPT" "$@"
