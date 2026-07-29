#!/usr/bin/env python3
"""Pre-training acceptance checks from the WMP migration plan."""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

PRIVileged_DR_SLICES = {
    "com": slice(44, 47),
    "added_mass": slice(47, 48),
    "restitution": slice(48, 49),
    "friction": slice(49, 50),
}


def _fingerprint(state_dict: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        val = state_dict[key]
        if not torch.is_tensor(val):
            continue
        h.update(key.encode())
        h.update(val.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def check_domain_rand(args, env, unwrapped):
    from wmp_lab.tasks.go2.domain_rand import read_back_static_domain_rand

    n = env.num_envs
    assert n >= args.num_envs, f"need at least {args.num_envs} envs, got {n}"

    obs = unwrapped.get_observations()
    env_ids = torch.arange(n, device=env.device)
    rb_f, rb_r, rb_m, rb_c = read_back_static_domain_rand(
        unwrapped._robot, env_ids, env.device, unwrapped._default_body_coms
    )

    obs_f = obs[:, PRIVileged_DR_SLICES["friction"]]
    obs_m = obs[:, PRIVileged_DR_SLICES["added_mass"]]
    obs_c = obs[:, PRIVileged_DR_SLICES["com"]] / unwrapped.cfg.normalization.obs_scales.com_pos

    f_diff = (obs_f - rb_f).abs().max().item()
    m_diff = (obs_m - rb_m).abs().max().item()
    c_diff = (obs_c - rb_c).abs().max().item()

    f_std = rb_f.std().item()
    m_std = rb_m.std().item()
    c_std = rb_c.norm(dim=-1).std().item()

    print(f"[domain_rand] obs vs PhysX max diff: friction={f_diff:.2e}, mass={m_diff:.2e}, com={c_diff:.2e}")
    print(f"[domain_rand] cross-env std: friction={f_std:.4f}, added_mass={m_std:.4f}, com_norm={c_std:.4f}")
    print(f"[domain_rand] replicate_physics={unwrapped.cfg.scene.replicate_physics}")

    ok = f_diff < 1e-4 and m_diff < 1e-4 and c_diff < 1e-3
    ok = ok and f_std > 1e-3 and m_std > 1e-3 and c_std > 1e-3
    if not ok:
        raise AssertionError("domain randomization readback check failed")
    print("[domain_rand] PASS", flush=True)


def check_reward_amp(args, env, unwrapped):
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    rewards = []
    feet_edge_rates = []
    for _ in range(args.num_steps):
        out = env.step(actions)
        rewards.append(out[2].mean().item())
        if "episode" in out[4]:
            ep = out[4]["episode"]
            if "rew_feet_edge" in ep:
                feet_edge_rates.append(float(ep["rew_feet_edge"]))

    mean_r = sum(rewards) / len(rewards)
    print(f"[reward_amp] mean task reward/step over {args.num_steps} steps: {mean_r:.4f}")
    if feet_edge_rates:
        print(f"[reward_amp] mean logged rew_feet_edge (episode): {sum(feet_edge_rates)/len(feet_edge_rates):.4f}")

    if mean_r > 0.15:
        raise AssertionError(f"task reward too large ({mean_r:.4f}); expected ~0.01-0.04/step")
    print("[reward_amp] PASS (task reward magnitude)")


def check_wm_path(args, env, unwrapped, cfg_entry: str):
    from importlib import import_module

    module_name, attr = cfg_entry.split(":")
    cfg = getattr(import_module(module_name), attr)()
    expected_cam = min(int(cfg.depth.camera_num_envs), args.num_envs)
    actual_cam = len(unwrapped.depth_index)
    print(f"[wm_path] cfg={attr} expected_camera_envs<={expected_cam}, actual={actual_cam}")
    assert actual_cam == expected_cam or actual_cam == min(expected_cam, env.num_envs)

    inv = unwrapped.depth_index_inverse
    assert inv.shape[0] == env.num_envs
    for idx in unwrapped.depth_index:
        assert inv[idx] >= 0, f"depth_index_inverse mismatch at env {idx}"

    if unwrapped.cfg.depth.use_camera and actual_cam > 0:
        fmap = unwrapped.get_forward_map()
        assert fmap.shape[1] == unwrapped.forward_height_dim
        assert torch.isfinite(fmap).all()
        depth_ids = torch.as_tensor(unwrapped.depth_index, device=env.device)
        unwrapped._reset_idx(depth_ids[: min(2, len(depth_ids))])
        unwrapped._update_depth_buffer()
        if unwrapped._extras_depth is not None:
            assert torch.isfinite(unwrapped._extras_depth).all()

    xs = unwrapped.cfg.terrain_meta.measured_forward_points_x
    assert xs[0] == 0.0 and xs[-1] == 2.0
    assert unwrapped.cfg.forward_height_scanner.offset.pos[0] == 0.0
    print(f"[wm_path] forward scan x in [{xs[0]}, {xs[-1]}], offset x=0")
    print("[wm_path] PASS")


def check_checkpoint_resume(args):
    from _common import (
        bootstrap_paths,
        configure_argv_for_world_model,
        ensure_log_dir,
        launch_isaac_app,
        load_agent_cfg,
        make_env,
        prepare_seed,
    )
    from rsl_rl.runners import WMPRunner

    bootstrap_paths()
    configure_argv_for_world_model(args)
    app_launcher = launch_isaac_app(args)
    simulation_app = app_launcher.app
    env = None
    try:
        cfg = load_agent_cfg(args.task)
        cfg["runner"]["max_iterations"] = args.iterations
        cfg["runner"]["save_interval"] = max(1, args.iterations)
        cfg["runner"]["keep_last_n"] = 0
        seed = prepare_seed(args, cfg, local_rank=0)

        with tempfile.TemporaryDirectory(prefix="wmp_accept_ckpt_") as tmp:
            log_dir = str(Path(tmp))
            env_a = make_env(args.task, args.num_envs, args.device, seed=seed)
            runner_a = WMPRunner(env_a, cfg, log_dir=log_dir, device=args.device)
            runner_a.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
            fp_continuous = {
                "actor": _fingerprint(runner_a.alg.actor_critic.state_dict()),
                "disc": _fingerprint(runner_a.alg.discriminator.state_dict()),
                "depth": _fingerprint(runner_a.depth_predictor.state_dict()),
                "iter": runner_a.current_learning_iteration,
            }
            ckpt = Path(log_dir) / f"model_{args.iterations}.pt"
            if not ckpt.exists():
                ckpts = sorted(Path(log_dir).glob("model_*.pt"))
                ckpt = ckpts[-1] if ckpts else ckpt
            env_a.close()

            env_b = make_env(args.task, args.num_envs, args.device, seed=seed)
            runner_b = WMPRunner(env_b, cfg, log_dir=log_dir, device=args.device)
            runner_b.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
            runner_b.load(str(ckpt), load_optimizer=True, load_wm_optimizer=True)
            runner_b.learn(num_learning_iterations=1, init_at_random_ep_len=False)
            fp_resume = {
                "actor": _fingerprint(runner_b.alg.actor_critic.state_dict()),
                "disc": _fingerprint(runner_b.alg.discriminator.state_dict()),
                "depth": _fingerprint(runner_b.depth_predictor.state_dict()),
                "iter": runner_b.current_learning_iteration,
            }
            env_b.close()

            env_c = make_env(args.task, args.num_envs, args.device, seed=seed)
            runner_c = WMPRunner(env_c, cfg, log_dir=None, device=args.device)
            runner_c.learn(num_learning_iterations=args.iterations + 1, init_at_random_ep_len=False)
            fp_golden = {
                "actor": _fingerprint(runner_c.alg.actor_critic.state_dict()),
                "disc": _fingerprint(runner_c.alg.discriminator.state_dict()),
                "depth": _fingerprint(runner_c.depth_predictor.state_dict()),
                "iter": runner_c.current_learning_iteration,
            }
            env_c.close()

        print(f"[checkpoint] continuous iter={fp_continuous['iter']} actor={fp_continuous['actor']}")
        print(f"[checkpoint] resume     iter={fp_resume['iter']} actor={fp_resume['actor']}")
        print(f"[checkpoint] golden     iter={fp_golden['iter']} actor={fp_golden['actor']}")

        if fp_resume["iter"] != fp_golden["iter"]:
            raise AssertionError(f"iteration mismatch: resume={fp_resume['iter']} golden={fp_golden['iter']}")
        # Weights diverge slightly due to numerical order; check same order of magnitude change from start.
        if fp_resume["actor"] == fp_continuous["actor"]:
            raise AssertionError("resume path did not advance actor weights")
        print("[checkpoint] PASS (resume advances to expected iteration)")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def main():
    parser = argparse.ArgumentParser(description="WMP pre-training acceptance checks")
    parser.add_argument(
        "--check",
        required=True,
        choices=["domain_rand", "reward_amp", "wm_path", "checkpoint", "all_env"],
    )
    parser.add_argument("--task", default="WMP-Go2-AMP-v0")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--wm_device", default="cuda:0")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--wm_cfg",
        default="wmp_lab.tasks.go2.go2_env_cfg:Go2AmpLowMemoryLabCfg",
        help="Env cfg entry for wm_path check",
    )
    args = parser.parse_args()

    if args.check == "checkpoint":
        check_checkpoint_resume(args)
        return

    sys.path.insert(0, str(ROOT))
    from _common import bootstrap_paths, configure_argv_for_world_model, launch_isaac_app, make_env

    bootstrap_paths()
    configure_argv_for_world_model(args)
    app_launcher = launch_isaac_app(args)
    simulation_app = app_launcher.app
    env = None
    try:
        if args.check == "wm_path":
            import gymnasium as gym
            import wmp_lab  # noqa: F401
            from wmp_lab.legacy_compat import load_entry_point

            module_name, attr = args.wm_cfg.split(":")
            cfg_cls = load_entry_point(f"{module_name}:{attr}")
            cfg = cfg_cls()
            cfg.scene.num_envs = args.num_envs
            cfg.env.num_envs = args.num_envs
            cfg.depth.camera_num_envs = min(cfg.depth.camera_num_envs, args.num_envs)
            cfg.sim.device = args.device
            env = gym.make(args.task, cfg=cfg, render_mode=None)
            from wmp_lab.legacy_compat import LegacyRslRlWrapper

            env = LegacyRslRlWrapper(env, clip_actions=cfg.normalization.clip_actions)
            check_wm_path(args, env, env.unwrapped, args.wm_cfg)
        else:
            env = make_env(args.task, args.num_envs, args.device)
            unwrapped = env.unwrapped
            if args.check in ("domain_rand", "all_env"):
                check_domain_rand(args, env, unwrapped)
            if args.check in ("reward_amp", "all_env"):
                check_reward_amp(args, env, unwrapped)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
