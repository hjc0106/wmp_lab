from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def writable_dir(preferred: Path, fallback: Path) -> str:
    try:
        os.makedirs(preferred, exist_ok=True)
        probe = preferred / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return str(preferred)
    except OSError:
        os.makedirs(fallback, exist_ok=True)
        return str(fallback)


def bootstrap_paths():
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    # Redirect the IsaacLab logger temp dir to a user-writable location (the
    # system default /tmp/isaaclab may be owned by another user on shared hosts).
    tmpdir = writable_dir(ROOT / "logs" / "tmp", Path("/tmp/wmp_lab_logs/tmp"))
    cfgdir = writable_dir(ROOT / "logs" / "config", Path("/tmp/wmp_lab_logs/config"))
    cachedir = writable_dir(ROOT / "logs" / "cache", Path("/tmp/wmp_lab_logs/cache"))
    os.environ["TMPDIR"] = tmpdir
    os.environ.setdefault("MPLCONFIGDIR", cfgdir)
    os.environ.setdefault("XDG_CONFIG_HOME", cfgdir)
    os.environ.setdefault("XDG_CACHE_HOME", cachedir)
    tempfile.tempdir = tmpdir


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--task", default="WMP-Go2-AMP-v0")
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wm_device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--gpu_ids",
        default=None,
        help="Physical GPU ids for ranks, e.g. '4,5'. Avoids CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Enable multi-GPU training via torchrun (IsaacLab-style).",
    )


def is_distributed() -> bool:
    return int(os.getenv("WORLD_SIZE", "1")) > 1


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "0"))


def get_global_rank() -> int:
    return int(os.getenv("RANK", "0"))


def is_main_process() -> bool:
    return get_global_rank() == 0


def resolve_distributed_flag(args):
    """Treat torchrun (WORLD_SIZE>1) as distributed even if --distributed was omitted."""
    args.distributed = bool(getattr(args, "distributed", False) or is_distributed())
    return args


def parse_gpu_ids(gpu_ids: str | None) -> list[int] | None:
    if not gpu_ids:
        return None
    ids = [int(x.strip()) for x in str(gpu_ids).split(",") if x.strip() != ""]
    if not ids:
        raise ValueError(f"Invalid --gpu_ids: {gpu_ids!r}")
    return ids


def resolve_physical_gpu(args, local_rank: int | None = None) -> int:
    """Map torchrun LOCAL_RANK -> physical GPU id.

    Default: physical == local_rank (uses GPUs 0..N-1).
    With --gpu_ids 4,5: rank0->4, rank1->5.
    """
    if local_rank is None:
        local_rank = get_local_rank()
    ids = parse_gpu_ids(getattr(args, "gpu_ids", None))
    if ids is None:
        if args.distributed:
            return local_rank
        if isinstance(args.device, str) and args.device.startswith("cuda:"):
            return int(args.device.split(":")[-1])
        return 0
    if local_rank >= len(ids):
        raise ValueError(
            f"LOCAL_RANK={local_rank} out of range for --gpu_ids={ids} "
            f"(need nproc_per_node={len(ids)})"
        )
    return ids[local_rank]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(
    task: str,
    num_envs: int,
    device: str,
    headless: bool = True,
    seed: int | None = None,
    cfg_overrides=None,
):
    import wmp_lab  # noqa: F401
    from wmp_lab.legacy_compat import LegacyRslRlWrapper, load_entry_point

    spec = gym.spec(task)
    cfg_cls = load_entry_point(spec.kwargs["env_cfg_entry_point"])
    cfg = cfg_cls()
    cfg.scene.num_envs = num_envs
    cfg.env.num_envs = num_envs
    cfg.depth.camera_num_envs = min(cfg.depth.camera_num_envs, num_envs)
    cfg.sim.device = device
    if seed is not None:
        if hasattr(cfg, "seed"):
            cfg.seed = seed
        if hasattr(cfg.scene, "seed"):
            cfg.scene.seed = seed
    if cfg_overrides is not None:
        cfg_overrides(cfg)
    terrain_generator = getattr(cfg.terrain, "terrain_generator", None)
    if seed is not None and terrain_generator is not None and terrain_generator.seed is None:
        terrain_generator.seed = seed
    cfg.depth.camera_num_envs = min(int(cfg.depth.camera_num_envs), int(cfg.scene.num_envs))
    log_dir = writable_dir(ROOT / "logs" / "isaaclab", Path("/tmp/wmp_lab_logs/isaaclab"))
    cfg.log_dir = log_dir
    cfg.sim.log_dir = log_dir
    env = gym.make(task, cfg=cfg, render_mode=None)
    wrapped = LegacyRslRlWrapper(env, clip_actions=env.unwrapped.cfg.normalization.clip_actions)
    return wrapped


def resolve_checkpoint(checkpoint: str | None, log_dir: str | None = None) -> Path:
    """Resolve a checkpoint path. ``checkpoint=-1`` picks the latest ``model_*.pt`` in log_dir."""
    if checkpoint is None or str(checkpoint) == "-1":
        search = Path(log_dir) if log_dir else (ROOT / "logs" / "go2_amp_lab_ddp")
        models = sorted(search.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
        if not models:
            raise FileNotFoundError(f"No model_*.pt under {search}")
        return models[-1]
    path = Path(checkpoint)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_agent_cfg(task: str):
    import gymnasium as gym
    from wmp_lab.legacy_compat import class_to_dict, load_entry_point
    import wmp_lab  # noqa: F401

    spec = gym.spec(task)
    key = spec.kwargs["rsl_rl_cfg_entry_point"]
    cfg_cls = load_entry_point(key)
    return class_to_dict(cfg_cls())


def configure_argv_for_world_model(args):
    sys.argv = [
        str(ROOT / "scripts" / "train_wmp.py"),
        "--headless",
        "--sim_device",
        args.device,
        "--wm_device",
        args.wm_device,
    ]


def launch_isaac_app(args):
    from isaaclab.app import AppLauncher

    resolve_distributed_flag(args)
    local_rank = get_local_rank()
    physical_id = resolve_physical_gpu(args, local_rank=local_rank)
    device = f"cuda:{physical_id}"
    args.device = device
    args.wm_device = device
    os.environ["WMP_LAB_CUDA_DEVICE"] = device
    if torch.cuda.is_available():
        torch.cuda.set_device(physical_id)

    # AppLauncher's distributed mode forces device_id=LOCAL_RANK (0..N-1).
    # When --gpu_ids remaps to physical GPUs (e.g. 4,5), disable that override
    # and bind physics/render explicitly via device=cuda:{physical_id}.
    use_app_distributed = bool(args.distributed) and parse_gpu_ids(getattr(args, "gpu_ids", None)) is None
    launcher_args = {
        "headless": args.headless,
        "device": device,
        "distributed": use_app_distributed,
        "multi_gpu": False,
    }
    if args.distributed and not use_app_distributed:
        # Mirror AppLauncher CPU throttling for multi-process Isaac Sim.
        num_cpu_cores = os.cpu_count() or 1
        num_threads = max(1, num_cpu_cores // int(os.getenv("WORLD_SIZE", "1")))
        os.environ["PXR_WORK_THREAD_LIMIT"] = str(num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
        sys.argv.append(f"--/plugins/carb.tasking.plugin/threadCount={num_threads}")

    launcher = AppLauncher(launcher_args)
    launcher.local_rank = local_rank
    launcher.physical_gpu = physical_id
    print(f"[INFO] rank={get_global_rank()} local_rank={local_rank} -> {device}", flush=True)
    return launcher


def apply_distributed_device(args, app_launcher):
    """Bind sim / RL / world-model to the physical GPU for this rank."""
    resolve_distributed_flag(args)
    local_rank = getattr(app_launcher, "local_rank", get_local_rank())
    physical_id = getattr(app_launcher, "physical_gpu", None)
    if physical_id is None:
        physical_id = resolve_physical_gpu(args, local_rank=local_rank)
    args.device = f"cuda:{physical_id}"
    args.wm_device = args.device
    os.environ["WMP_LAB_CUDA_DEVICE"] = args.device
    return args


def prepare_seed(args, agent_cfg, local_rank: int = 0) -> int:
    """Resolve per-rank seed before env/runner creation."""
    base = args.seed if args.seed is not None else int(agent_cfg.get("seed", 1))
    seed = base + (local_rank if args.distributed else 0)
    agent_cfg["seed"] = seed
    set_seed(seed)
    return seed


def cleanup_distributed():
    """Best-effort NCCL teardown. Avoid barrier: a dead peer makes barrier hang forever."""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] destroy_process_group failed: {exc}", flush=True)


def ensure_log_dir(default_name: str) -> str:
    return writable_dir(ROOT / "logs" / default_name, Path("/tmp/wmp_lab_logs") / default_name)
