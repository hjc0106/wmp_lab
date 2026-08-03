"""Play / evaluate a trained WMP-Go2 policy in Isaac Lab.

Mirrors ``legged_gym/scripts/play.py`` from the IsaacGym WMP repo.
"""
from __future__ import annotations

import argparse

import torch

from _common import (
    add_common_args,
    apply_distributed_device,
    bootstrap_paths,
    configure_argv_for_world_model,
    launch_isaac_app,
    load_agent_cfg,
    make_env,
    prepare_seed,
    resolve_checkpoint,
    resolve_distributed_flag,
)


def _apply_play_overrides(cfg, args):
    from wmp_lab.tasks.go2.go2_env_cfg import (
        GO2_TERRAIN_NAMES,
        PLAY_TERRAIN_PRESETS,
        _build_go2_terrain_generator,
    )

    num_envs = int(args.num_envs)
    cfg.scene.num_envs = num_envs
    cfg.env.num_envs = num_envs
    cfg.noise.add_noise = False

    dr = cfg.domain_rand
    dr.randomize_friction = False
    dr.randomize_restitution = False
    dr.randomize_base_mass = False
    dr.randomize_link_mass = False
    dr.randomize_com_pos = False
    dr.randomize_action_latency = False
    dr.push_robots = False
    dr.randomize_gains = True
    dr.randomize_link_mass = False
    dr.randomize_motor_strength = False
    dr.stiffness_multiplier_range = [1.0, 1.0]
    dr.damping_multiplier_range = [1.0, 1.0]

    r = cfg.commands.ranges
    r.lin_vel_x = [args.lin_vel_x, args.lin_vel_x]
    r.lin_vel_y = [0.0, 0.0]
    r.ang_vel_yaw = [0.0, 0.0]
    r.heading = [0.0, 0.0]
    r.flat_lin_vel_x = [args.lin_vel_x, args.lin_vel_x]
    r.flat_lin_vel_y = [0.0, 0.0]
    r.flat_ang_vel_yaw = [0.0, 0.0]
    r.flat_heading = [0.0, 0.0]
    cfg.commands.heading_command = False

    cfg.depth.use_camera = True
    cfg.depth.camera_num_envs = num_envs

    terrain_key = args.terrain
    if terrain_key not in PLAY_TERRAIN_PRESETS:
        print(
            f"[play] unknown terrain={terrain_key!r}; "
            f"expected one of {sorted(PLAY_TERRAIN_PRESETS)}; defaulting to climb"
        )
        terrain_key = "climb"

    terrain_name = PLAY_TERRAIN_PRESETS[terrain_key]
    terrain_seed = getattr(args, "terrain_seed", None)
    if terrain_seed is None:
        terrain_seed = getattr(args, "seed", None)

    if terrain_name is None:
        cfg.terrain_meta.mesh_type = "plane"
        cfg.terrain_meta.curriculum = False
        cfg.terrain.terrain_type = "plane"
        cfg.terrain.terrain_generator = None
        cfg.terrain.use_terrain_origins = False
        cfg.rewards.scales.feet_edge = 0.0
        cfg.rewards.scales.feet_stumble = 0.0
        cfg.terrain_meta.terrain_proportions = [0.0] * 9 + [1.0]
    else:
        max_difficulty = 1.0
        if terrain_name == "climb":
            # Climb adds up to 5 cm of roughness on top of its 0.6*difficulty
            # step. Keep the resulting highest point within the requested Go2
            # standing height (40 cm by default).
            roughness_height = 0.05
            max_difficulty = max(
                0.0,
                min(1.0, (args.max_climb_height - roughness_height) / 0.6),
            )

        proportions = {name: 0.0 for name in GO2_TERRAIN_NAMES}
        proportions[terrain_name] = 1.0
        cfg.terrain_meta.mesh_type = "trimesh"
        cfg.terrain_meta.curriculum = False
        cfg.terrain_meta.num_rows = max(1, int(args.terrain_rows))
        cfg.terrain_meta.num_cols = max(1, int(args.terrain_cols))
        cfg.terrain_meta.terrain_proportions = [proportions[n] for n in GO2_TERRAIN_NAMES]
        cfg.terrain_meta.max_init_terrain_level = cfg.terrain_meta.num_rows - 1
        cfg.terrain.terrain_type = "generator"
        cfg.terrain.terrain_generator = _build_go2_terrain_generator(
            num_rows=cfg.terrain_meta.num_rows,
            num_cols=cfg.terrain_meta.num_cols,
            curriculum=False,
            proportions=proportions,
            seed=terrain_seed,
            compat_mode=args.terrain_compat,
            slope_direction=args.slope_direction,
            max_difficulty=max_difficulty,
        )
        cfg.terrain.terrain_generator.border_width = max(0.0, args.terrain_border_width)
        cfg.terrain.use_terrain_origins = True
        cfg.terrain.max_init_terrain_level = cfg.terrain_meta.max_init_terrain_level


def play(args):
    from rsl_rl.runners import WMPRunner

    ckpt = resolve_checkpoint(args.checkpoint, args.log_dir)
    print(f"[play] checkpoint: {ckpt}", flush=True)

    cfg = load_agent_cfg(args.task)
    cfg["runner"]["amp_num_preload_transitions"] = 1
    seed = prepare_seed(args, cfg, local_rank=0)

    env = make_env(
        args.task,
        args.num_envs,
        args.device,
        headless=args.headless,
        seed=seed,
        cfg_overrides=lambda c: _apply_play_overrides(c, args),
    )
    env.reset()
    obs = env.get_observations()

    runner = WMPRunner(env, cfg, log_dir=None, device=args.device)
    runner.load(str(ckpt), load_optimizer=False, load_wm_optimizer=False, allow_inference_only=True)
    policy = runner.get_inference_policy(device=env.device)
    depth_predictor = runner.depth_predictor
    world_model = runner._world_model
    world_model.eval()
    depth_predictor.eval()

    history_length = 5
    trajectory_history = torch.zeros(
        (env.num_envs, history_length, env.num_obs - env.privileged_dim - env.height_dim - 3),
        device=env.device,
    )
    obs_without_command = torch.concat(
        (
            obs[:, env.privileged_dim : env.privileged_dim + 6],
            obs[:, env.privileged_dim + 9 : -env.height_dim],
        ),
        dim=1,
    )
    trajectory_history = torch.concat(
        (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1
    )

    wm_latent = wm_action = None
    wm_is_first = torch.ones(env.num_envs, device=world_model.device)
    wm_update_interval = env.cfg.depth.update_interval
    wm_action_history = torch.zeros(
        (env.num_envs, wm_update_interval, env.num_actions),
        device=world_model.device,
    )
    wm_obs = {
        "prop": obs[:, env.privileged_dim : env.privileged_dim + env.cfg.env.prop_dim].to(
            world_model.device
        ),
        "is_first": wm_is_first,
    }
    if env.cfg.depth.use_camera:
        wm_obs["image"] = torch.zeros(
            (env.num_envs,) + env.cfg.depth.resized + (1,),
            device=world_model.device,
        )
    wm_feature = torch.zeros((env.num_envs, runner.wm_feature_dim), device=env.device)
    infos = {"depth": None}

    total_reward = torch.zeros(env.num_envs, device=env.device)
    alive = torch.ones(env.num_envs, device=env.device)
    max_steps = int(args.num_episodes * env.max_episode_length) + 3
    print(
        f"[play] envs={env.num_envs} terrain={args.terrain} "
        f"steps={max_steps} headless={args.headless}",
        flush=True,
    )

    gait_stats = None
    if args.diagnose_gait:
        joint_names = list(env._robot.joint_names[: env.num_actions])
        print(f"[gait] joint_order={joint_names}", flush=True)
        gait_stats = {
            "samples": 0,
            "action_abs": torch.zeros(env.num_actions, device=env.device),
            "torque_abs": torch.zeros(env.num_actions, device=env.device),
            "torque_sat": torch.zeros(env.num_actions, device=env.device),
            "joint_min": torch.full((env.num_actions,), torch.inf, device=env.device),
            "joint_max": torch.full((env.num_actions,), -torch.inf, device=env.device),
            "foot_contact": torch.zeros(len(env.feet_indices), device=env.device),
            "foot_force": torch.zeros(len(env.feet_indices), device=env.device),
        }

    with torch.inference_mode():
        for step in range(max_steps):
            try:
                if env.global_counter % wm_update_interval == 0:
                    if env.cfg.depth.use_camera:
                        forward_heightmap = env.get_forward_map().to(world_model.device)
                        pred_depth = depth_predictor(forward_heightmap, wm_obs["prop"])
                        wm_obs["image"] = pred_depth
                        depth = infos.get("depth")
                        if depth is not None and len(env.depth_index) > 0:
                            wm_obs["image"][env.depth_index] = depth.unsqueeze(-1).to(
                                world_model.device
                            )

                    wm_embed = world_model.encoder(wm_obs)
                    # Match training: do not force sample=True
                    wm_latent, _ = world_model.dynamics.obs_step(
                        wm_latent, wm_action, wm_embed, wm_obs["is_first"]
                    )
                    wm_feature = world_model.dynamics.get_deter_feat(wm_latent).to(env.device)
                    wm_is_first[:] = 0

                history = trajectory_history.flatten(1)
                actions = policy(obs.detach(), history.detach(), wm_feature.detach())
                obs, _, rews, dones, infos, reset_env_ids, _ = env.step(actions.detach())
            except Exception:
                import traceback

                print(f"[play] crashed at step={step}:\n{traceback.format_exc()}", flush=True)
                raise

            alive = alive * (1.0 - dones.float())
            total_reward += rews * alive

            if gait_stats is not None:
                robot = env._robot.data
                joint_pos = robot.joint_pos[:, : env.num_actions]
                torque = robot.applied_torque[:, : env.num_actions]
                limits = env.torque_limits.unsqueeze(0)
                foot_force = env._contact_sensor.data.net_forces_w[:, env.feet_indices, :]
                foot_force_norm = torch.linalg.vector_norm(foot_force, dim=-1)
                gait_stats["samples"] += env.num_envs
                gait_stats["action_abs"] += actions.abs().sum(0)
                gait_stats["torque_abs"] += torque.abs().sum(0)
                gait_stats["torque_sat"] += (torque.abs() >= 0.99 * limits).sum(0)
                gait_stats["joint_min"] = torch.minimum(gait_stats["joint_min"], joint_pos.amin(0))
                gait_stats["joint_max"] = torch.maximum(gait_stats["joint_max"], joint_pos.amax(0))
                gait_stats["foot_contact"] += (foot_force_norm > 1.0).sum(0)
                gait_stats["foot_force"] += foot_force_norm.sum(0)

            wm_action_history = torch.concat(
                (wm_action_history[:, 1:], actions.unsqueeze(1).to(world_model.device)),
                dim=1,
            )
            wm_obs = {
                "prop": obs[
                    :, env.privileged_dim : env.privileged_dim + env.cfg.env.prop_dim
                ].to(world_model.device),
                "is_first": wm_is_first,
            }
            if env.cfg.depth.use_camera:
                wm_obs["image"] = torch.zeros(
                    (env.num_envs,) + env.cfg.depth.resized + (1,),
                    device=world_model.device,
                )

            if len(reset_env_ids) > 0:
                reset_ids = reset_env_ids.cpu().numpy()
                wm_action_history[reset_ids, :] = 0
                wm_is_first[reset_ids] = 1
                alive[reset_env_ids] = 1.0
                total_reward[reset_env_ids] = 0.0

            wm_action = wm_action_history.flatten(1)

            env_ids = dones.nonzero(as_tuple=False).flatten()
            trajectory_history[env_ids] = 0
            obs_without_command = torch.concat(
                (
                    obs[:, env.privileged_dim : env.privileged_dim + 6],
                    obs[:, env.privileged_dim + 9 : -env.height_dim],
                ),
                dim=1,
            )
            trajectory_history = torch.concat(
                (trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1
            )
            if step == 0 or (step + 1) % 25 == 0:
                print(
                    f"[play] step={step + 1}/{max_steps} "
                    f"mean_return={total_reward.mean().item():.2f}",
                    flush=True,
                )

    if gait_stats is not None:
        count = max(1, gait_stats["samples"])
        joint_names = env._robot.joint_names[: env.num_actions]
        for leg in ("FL", "FR", "RL", "RR"):
            ids = [i for i, name in enumerate(joint_names) if name.startswith(f"{leg}_")]
            values = lambda key: [round(v, 4) for v in gait_stats[key][ids].tolist()]
            mean_values = lambda key: [round(v / count, 4) for v in gait_stats[key][ids].tolist()]
            print(
                f"[gait] {leg} mean_abs_action={mean_values('action_abs')} "
                f"mean_abs_torque={mean_values('torque_abs')} "
                f"torque_sat_rate={mean_values('torque_sat')} "
                f"joint_range={list(zip(values('joint_min'), values('joint_max')))}",
                flush=True,
            )
        foot_sensor_names = [env._contact_sensor.body_names[i] for i in env.feet_indices.tolist()]
        foot_robot_names = [env._robot.body_names[i] for i in env.feet_indices.tolist()]
        mapped_foot_names = [env._robot.body_names[i] for i in env.feet_body_indices.tolist()]
        print(
            f"[gait] contact_sensor_feet={foot_sensor_names} "
            f"same_indices_in_articulation={foot_robot_names} "
            f"mapped_articulation_feet={mapped_foot_names} "
            f"contact_duty={(gait_stats['foot_contact'] / count).tolist()} "
            f"mean_force={(gait_stats['foot_force'] / count).tolist()}",
            flush=True,
        )
    print(f"[play] finished. last mean cumulative reward: {total_reward.mean().item():.3f}", flush=True)
    return env


def main():
    parser = argparse.ArgumentParser(description="Play a trained WMP-Go2 policy")
    add_common_args(parser)
    # Play opens a viewer by default. add_common_args sets headless=True for training;
    # override that here. Pass --headless to run without a window.
    parser.set_defaults(
        task="WMP-Go2-AMP-v0",
        num_envs=10,
        log_dir="logs/go2_amp_lab_ddp",
        checkpoint="-1",
        headless=False,
    )
    parser.add_argument(
        "--terrain",
        default="climb",
        choices=["slope", "stair", "gap", "climb", "crawl", "tilt", "plane"],
        help="Single-terrain play preset (legacy play.py compatible)",
    )
    parser.add_argument("--terrain_rows", type=int, default=10)
    parser.add_argument(
        "--terrain_cols",
        type=int,
        default=1,
        help="Number of 8 m terrain lanes; one keeps the play scene narrow",
    )
    parser.add_argument(
        "--terrain_border_width",
        type=float,
        default=5.0,
        help="Flat border around generated play terrain in meters",
    )
    parser.add_argument(
        "--max_climb_height",
        type=float,
        default=0.4,
        help="Maximum climb surface height including roughness, in meters",
    )
    parser.add_argument("--terrain_seed", type=int, default=None)
    parser.add_argument(
        "--terrain_compat",
        default="legacy_exact",
        choices=["legacy_exact", "legacy_fixed"],
        help="legacy_exact reproduces old crawl overlap; legacy_fixed symmetric crawl bars",
    )
    parser.add_argument(
        "--slope_direction",
        default="legacy",
        choices=["legacy", "up", "down"],
        help="Slope sign override (legacy splits selected slope columns by direction)",
    )
    parser.add_argument("--lin_vel_x", type=float, default=0.6)
    parser.add_argument("--num_episodes", type=float, default=1.0)
    parser.add_argument(
        "--diagnose_gait",
        action="store_true",
        help="Print per-leg action, torque, joint-range, and contact diagnostics",
    )
    args = parser.parse_args()
    bootstrap_paths()
    resolve_distributed_flag(args)
    args.distributed = False

    app_launcher = launch_isaac_app(args)
    simulation_app = app_launcher.app
    env = None
    try:
        args = apply_distributed_device(args, app_launcher)
        configure_argv_for_world_model(args)
        env = play(args)
    except BaseException:
        import traceback

        print(f"[play] fatal:\n{traceback.format_exc()}", flush=True)
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
