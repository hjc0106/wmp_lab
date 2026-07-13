from __future__ import annotations

import argparse
import faulthandler
import sys
import traceback

from _common import (
    add_common_args,
    apply_distributed_device,
    bootstrap_paths,
    cleanup_distributed,
    configure_argv_for_world_model,
    ensure_log_dir,
    get_local_rank,
    get_global_rank,
    is_main_process,
    launch_isaac_app,
    load_agent_cfg,
    make_env,
    prepare_seed,
    resolve_distributed_flag,
)


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(task="WMP-Go2-AMP-v0")
    args = parser.parse_args()
    bootstrap_paths()
    resolve_distributed_flag(args)

    app_launcher = launch_isaac_app(args)
    simulation_app = app_launcher.app
    env = None
    try:
        args = apply_distributed_device(args, app_launcher)
        configure_argv_for_world_model(args)

        from rsl_rl.runners import WMPRunner

        cfg = load_agent_cfg(args.task)
        if args.max_iterations is not None:
            cfg["runner"]["max_iterations"] = args.max_iterations
        local_rank = getattr(app_launcher, "local_rank", get_local_rank())
        seed = prepare_seed(args, cfg, local_rank=local_rank)

        env = make_env(args.task, args.num_envs, args.device, args.headless, seed=seed)

        log_dir = None
        if is_main_process():
            log_dir = args.log_dir or ensure_log_dir(cfg["runner"]["experiment_name"])

        max_iters = int(cfg["runner"]["max_iterations"])
        print(
            f"[rank{get_global_rank()}] starting learn(max_iterations={max_iters})",
            flush=True,
        )
        runner = WMPRunner(env, cfg, log_dir=log_dir, device=args.device)
        try:
            runner.learn(num_learning_iterations=max_iters, init_at_random_ep_len=True)
            print(f"[rank{get_global_rank()}] learn() returned normally", flush=True)
        except BaseException:
            print(
                f"[rank{get_global_rank()}] learn() crashed:\n{traceback.format_exc()}",
                file=sys.stderr,
                flush=True,
            )
            raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
