#!/usr/bin/env bash
# Multi-GPU WMP training launcher for IsaacLab.
#
# Important:
#   Do NOT set CUDA_VISIBLE_DEVICES. Omniverse/PhysX GPU enumeration breaks
#   when visible devices are remapped (see IsaacLab #2756 / multi-GPU docs).
#   Pick physical GPUs with GPU_IDS / --gpu_ids instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Examples:
#   GPU_IDS=4,5 NPROC=2 ./scripts/launch_wmp_ddp.sh
#   GPU_IDS=0,1,2,3 NPROC=4 ./scripts/launch_wmp_ddp.sh
GPU_IDS="${GPU_IDS:-}"
if [[ -n "${GPU_IDS}" ]]; then
  # Infer nproc from GPU_IDS unless NPROC is explicitly set.
  IFS=',' read -r -a _GPU_ARR <<< "${GPU_IDS}"
  DEFAULT_NPROC="${#_GPU_ARR[@]}"
else
  DEFAULT_NPROC=4
fi
NPROC="${NPROC:-$DEFAULT_NPROC}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERS="${MAX_ITERS:-20000}"
LOG_DIR="${LOG_DIR:-logs/go2_amp_lab_ddp}"
MASTER_PORT="${MASTER_PORT:-29501}"

# NCCL workarounds recommended by IsaacLab for multi-GPU stability.
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
# Optional if hangs persist:
# export NCCL_P2P_DISABLE=1

# Prefer native GPU order; do not remap with CUDA_VISIBLE_DEVICES.
unset CUDA_VISIBLE_DEVICES || true

EXTRA_ARGS=()
if [[ -n "${GPU_IDS}" ]]; then
  EXTRA_ARGS+=(--gpu_ids "${GPU_IDS}")
fi

echo "[launch] nproc=${NPROC} gpu_ids=${GPU_IDS:-0..$((NPROC-1))} num_envs=${NUM_ENVS} max_iters=${MAX_ITERS}"
echo "[launch] NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE} NCCL_IB_DISABLE=${NCCL_IB_DISABLE} NCCL_ALGO=${NCCL_ALGO} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
echo "[launch] CUDA_VISIBLE_DEVICES is unset (required for IsaacLab multi-GPU)"

exec python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT}" \
  scripts/train_wmp.py \
  --task WMP-Go2-AMP-v0 \
  --num_envs "${NUM_ENVS}" \
  --headless \
  --distributed \
  --max_iterations "${MAX_ITERS}" \
  --log_dir "${LOG_DIR}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
