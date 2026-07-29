from __future__ import annotations

import argparse

from _common import (
    add_common_args,
    apply_distributed_device,
    bootstrap_paths,
    cleanup_distributed,
    ensure_log_dir,
    get_local_rank,
    is_main_process,
    launch_isaac_app,
    load_agent_cfg,
    make_env,
    prepare_seed,
    resolve_distributed_flag,
)


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(task="WMP-Go2-Flat-PPO-v0")
    args = parser.parse_args()
    bootstrap_paths()
    resolve_distributed_flag(args)

    app_launcher = launch_isaac_app(args)
    simulation_app = app_launcher.app
    env = None
    try:
        args = apply_distributed_device(args, app_launcher)

        from rsl_rl.runners import OnPolicyRunner

        cfg = load_agent_cfg(args.task)
        if args.max_iterations is not None:
            cfg["runner"]["max_iterations"] = args.max_iterations
        local_rank = getattr(app_launcher, "local_rank", get_local_rank())
        seed = prepare_seed(args, cfg, local_rank=local_rank)

        env = make_env(args.task, args.num_envs, args.device, args.headless, seed=seed)

        log_dir = None
        if is_main_process():
            log_dir = args.log_dir or ensure_log_dir(cfg["runner"]["experiment_name"])

        runner = OnPolicyRunner(env, cfg, log_dir=log_dir, device=args.device)
        runner.learn(num_learning_iterations=cfg["runner"]["max_iterations"], init_at_random_ep_len=True)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
