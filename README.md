# wmp_lab

IsaacLab migration of the Go2 WMP training code.

## Quick checks

```bash
cd /home/hongjiacheng/codes/robot_lab/wmp_lab
conda activate wmp_lab
python -m py_compile scripts/smoke_test.py scripts/train_ppo.py scripts/train_wmp.py
python scripts/smoke_test.py --num_envs 2 --device cuda:0
```

## Short training (single process)

```bash
cd /home/hongjiacheng/codes/robot_lab/wmp_lab
conda activate wmp_lab
python scripts/train_ppo.py --num_envs 2 --device cuda:0 --max_iterations 1
python scripts/train_wmp.py --num_envs 2 --device cuda:0 --wm_device cuda:0 --max_iterations 1
```

## Multi-GPU training (torchrun / DDP-style)

IsaacLab-style: one process per GPU, independent rollouts, gradient sync via `all_reduce`.

**Do not set `CUDA_VISIBLE_DEVICES`.** It remaps Omniverse/PhysX GPU IDs and commonly
triggers `omni.physx ... no suitable CUDA GPU was found!` (IsaacLab #2756).

```bash
cd /home/hongjiacheng/codes/robot_lab/wmp_lab
conda activate wmp_lab

# Recommended launcher (unsets CUDA_VISIBLE_DEVICES + sets NCCL workarounds)
chmod +x scripts/launch_wmp_ddp.sh

# Use physical GPUs 0-3:
GPU_IDS=0,1,2,3 NUM_ENVS=1024 MAX_ITERS=20000 ./scripts/launch_wmp_ddp.sh

# Use physical GPUs 4 and 5 only:
GPU_IDS=4,5 NUM_ENVS=1024 MAX_ITERS=20000 ./scripts/launch_wmp_ddp.sh

# Or manually (GPUs 4,5):
unset CUDA_VISIBLE_DEVICES
export NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_ALGO=Ring
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 \
  scripts/train_wmp.py \
  --task WMP-Go2-AMP-v0 \
  --num_envs 1024 \
  --headless \
  --distributed \
  --gpu_ids 4,5 \
  --max_iterations 20000 \
  --log_dir logs/go2_amp_lab_ddp
```

`--gpu_ids` maps `LOCAL_RANK -> physical GPU` without remapping CUDA device
enumeration, which is what breaks Omniverse/PhysX.

Notes:

- Effective parallel envs ≈ `num_envs × nproc_per_node`.
- World model runs on the same GPU as the env for each rank (`--wm_device` is overridden).
- After each WM/depth update, rank0 weights are broadcast so all ranks share the same features.
- AMP normalizer is updated on rank0 and broadcast; advantages use global mean/std.
- Only rank 0 writes TensorBoard / checkpoints.
- `torchrun` alone is enough: `--distributed` is auto-enabled when `WORLD_SIZE > 1`.

The migrated task IDs are:

- `WMP-Go2-Rough-v0`
- `WMP-Go2-AMP-v0`

## Play / evaluation

Opens the Isaac Sim viewer by default. Add `--headless` only if you do not need a window.

```bash
cd /home/hongjiacheng/codes/robot_lab/wmp_lab
conda activate wmp_lab
unset CUDA_VISIBLE_DEVICES

# GUI play (default), climb(=pit) terrain
python scripts/play_wmp.py \
  --device cuda:0 \
  --num_envs 10 \
  --log_dir logs/go2_amp_lab_ddp \
  --checkpoint -1 \
  --terrain climb \
  --num_episodes 1

# Headless (no window)
python scripts/play_wmp.py \
  --device cuda:0 \
  --headless \
  --checkpoint logs/go2_amp_lab_ddp/model_20000.pt \
  --terrain gap
```

Terrain presets: `slope`, `stair`, `gap`, `climb` (pit), `crawl`, `tilt`, `plane`.
