#!/usr/bin/env bash
set -euo pipefail
export PINGANDOG_ISAAC_ROOT=/home/ubuntu/pingandog/runtime/isaac-sim-6.2.0
export PINGANDOG_CACHE_ROOT=/home/ubuntu/pingandog/runtime/cache62
# Optional persistent per-job GPU pin, also honored by supervisor recovery.
args=("$@")
for ((i=0; i<${#args[@]}-1; i++)); do
  if [[ "${args[$i]}" == "--job-dir" && -f "${args[$((i+1))]}/gpu.txt" ]]; then
    IFS= read -r requested_gpu < "${args[$((i+1))]}/gpu.txt"
    [[ "$requested_gpu" =~ ^[0-7]$ ]] || { echo "Invalid per-job GPU pin" >&2; exit 2; }
    export PINGANDOG_REQUIRED_GPU="$requested_gpu"
  fi
done
exec /home/ubuntu/pingandog/scripts/run_isaac.sh "$@"
