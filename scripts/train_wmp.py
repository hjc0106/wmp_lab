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
    resolve_checkpoint,
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

        runner = WMPRunner(env, cfg, log_dir=log_dir, device=args.device)
        start_iter = 0
        if args.resume or args.checkpoint:
            ckpt = resolve_checkpoint(args.checkpoint, log_dir=log_dir)
            runner.load(
                str(ckpt),
                load_optimizer=args.load_optimizer,
                load_wm_optimizer=args.load_wm_optimizer,
            )
            start_iter = runner.current_learning_iteration
            print(
                f"[rank{get_global_rank()}] resumed from {ckpt} at iter {start_iter}",
                flush=True,
            )

        max_iters = int(cfg["runner"]["max_iterations"])
        if args.resume or args.checkpoint:
            # With resume, --max_iterations is the total target iteration (inclusive).
            num_iters = max(0, max_iters - start_iter)
        else:
            num_iters = max_iters

        print(
            f"[rank{get_global_rank()}] starting learn("
            f"from_iter={start_iter}, additional_iterations={num_iters}, target_iter={start_iter + num_iters})",
            flush=True,
        )
        try:
            runner.learn(num_learning_iterations=num_iters, init_at_random_ep_len=True)
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
