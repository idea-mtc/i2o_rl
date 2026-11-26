#!/usr/bin/env bash
# Run one command on an explicitly selected idle jiangxing physical GPU.
set -eu
gpu=${1:?Pass physical GPU 2, 3 or 4, followed by a command}
shift
case "$gpu" in
  2|3|4) ;;
  *) echo "Allowed physical GPUs: 2, 3, 4" >&2; exit 2 ;;
esac
uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)
util=$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ -n "$pids" || "$util" != 0 || "$memory" -ge 100 ]]; then
  echo "Physical GPU $gpu is busy; wait and retry." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$uuid"
exec "$@"
