from __future__ import annotations

import argparse
import faulthandler
import sys
import torch

from _common import add_common_args, bootstrap_paths, configure_argv_for_world_model, launch_isaac_app, load_agent_cfg, make_env


def check_env(task: str, num_envs: int, device: str):
    print(f"[smoke] creating {task} num_envs={num_envs} device={device}", flush=True)
    env = make_env(task, num_envs, device)
    print(f"[smoke] reset {task}", flush=True)
    obs, privileged = env.reset()
    assert obs.shape == (num_envs, env.num_obs), (obs.shape, env.num_obs)
    assert privileged.shape == (num_envs, env.num_privileged_obs), privileged.shape
    assert torch.isfinite(obs).all()
    actions = torch.zeros(num_envs, env.num_actions, device=env.device)
    print(f"[smoke] step {task}", flush=True)
    out = env.step(actions)
    assert len(out) == 7
    obs, privileged, rewards, dones, infos, reset_env_ids, terminal_amp = out
    assert obs.shape == (num_envs, env.num_obs)
    assert privileged.shape == (num_envs, env.num_privileged_obs)
    assert rewards.shape == (num_envs,)
    assert dones.shape == (num_envs,)
    assert torch.isfinite(obs).all()
    assert torch.isfinite(rewards).all()
    if task.endswith("AMP-v0"):
        assert env.get_amp_observations().shape[1] == 30
        assert env.get_forward_map().shape[1] == env.cfg.env.forward_height_dim
    env.close()
    print(f"[smoke] closed {task}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--runner_init", action="store_true", default=False)
    args = parser.parse_args()
    faulthandler.enable(file=sys.stderr)
    faulthandler.dump_traceback_later(180, repeat=True, file=sys.stderr)
    bootstrap_paths()
    print("[smoke] launching Isaac app", flush=True)
    app_launcher = launch_isaac_app(args)
    simulation_app = app_launcher.app
    print("[smoke] Isaac app launched", flush=True)
    configure_argv_for_world_model(args)

    check_env(args.task, args.num_envs, args.device)

    if args.runner_init:
        from rsl_rl.runners import OnPolicyRunner, WMPRunner

        env = make_env(args.task, args.num_envs, args.device)
        runner_cls = WMPRunner if args.task.endswith("AMP-v0") else OnPolicyRunner
        runner_cls(env, load_agent_cfg(args.task), log_dir=None, device=args.device)
        env.close()
    print("smoke ok")
    faulthandler.cancel_dump_traceback_later()
    simulation_app.close()


if __name__ == "__main__":
    main()
