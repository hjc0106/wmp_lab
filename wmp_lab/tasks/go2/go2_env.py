from __future__ import annotations

import importlib
import math
from collections.abc import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCamera

from .go2_env_cfg import Go2AmpLabCfg, Go2LabCfg, Go2RoughLabCfg
from .legacy_terrain_layout import LEGACY_TERRAIN_NAMES, column_category_names


def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angles + math.pi, 2 * math.pi) - math.pi


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by quaternion ``quat`` (w,x,y,z), batched (N,4)/(N,3)->(N,3)."""
    q = quat
    v = vec
    qvec = q[:, 1:]
    uv = torch.linalg.cross(qvec, v, dim=-1)
    uuv = torch.linalg.cross(qvec, uv, dim=-1)
    return v + 2 * (q[:, 0:1] * uv + uuv)


class Go2WmpLabEnv(DirectRLEnv):
    """IsaacLab Go2 environment exposing the legacy WMP / RSL-RL tensor contract.

    The observation tensor layout (285) exactly matches the unchanged
    ``rsl_rl.WMPRunner`` slicing:

    ``[contact_flag(8), contact_force_feet(12), d_rel(12), p_rel(12), com(3),
       added_mass(1), restitution(1), friction(1), base_lin_vel(3),
       base_ang_vel(3), gravity(3), commands(3), dof_pos-default(12),
       dof_vel(12), actions(12), heights(187)]``

    so that ``obs[:, privileged_dim-3:privileged_dim]`` is ``base_lin_vel``
    (the vel-predict target) and ``obs[:, privileged_dim+6:privileged_dim+9]``
    is the commanded velocity seen by the actor.
    """

    cfg: Go2LabCfg | Go2RoughLabCfg | Go2AmpLabCfg

    def __init__(
        self,
        cfg: Go2LabCfg | None = None,
        render_mode: str | None = None,
        env_cfg_entry_point: str | None = None,
        **kwargs,
    ):
        kwargs.pop("rsl_rl_cfg_entry_point", None)
        kwargs.pop("skrl_cfg_entry_point", None)
        if cfg is None:
            if env_cfg_entry_point is None:
                raise ValueError("Go2WmpLabEnv requires either cfg or env_cfg_entry_point.")
            module_name, attr_name = env_cfg_entry_point.split(":")
            cfg = getattr(importlib.import_module(module_name), attr_name)()
        super().__init__(cfg, render_mode, **kwargs)

        # ---- legacy tensor contract scalars the runner reads ----
        self.num_obs = self.cfg.env.num_observations
        self.num_privileged_obs = self.cfg.env.num_privileged_obs
        self.num_actions = self.cfg.env.num_actions
        self.privileged_dim = self.cfg.env.privileged_dim
        self.height_dim = self.cfg.env.height_dim
        self.forward_height_dim = self.cfg.env.forward_height_dim
        self.include_history_steps = self.cfg.env.include_history_steps
        self.dt = self.step_dt

        self._actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self._last_actions = torch.zeros_like(self._actions)
        self._last_last_actions = torch.zeros_like(self._actions)
        self._last_dof_vel = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self._last_torques = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self._processed_actions = torch.zeros_like(self._actions)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, device=self.device)
        self.commands_scale = torch.tensor([1.0, 1.0, 0.25], device=self.device)
        self.global_counter = 0
        self._substep = 0

        act = self.num_actions
        self.default_dof_pos = self._robot.data.default_joint_pos[:, :act].clone()
        self.p_gains = torch.full((act,), self.cfg.control.stiffness["joint"], device=self.device)
        self.d_gains = torch.full((act,), self.cfg.control.damping["joint"], device=self.device)
        self.randomized_p_gains = (self.p_gains.unsqueeze(0).repeat(self.num_envs, 1).clone())
        self.randomized_d_gains = (self.d_gains.unsqueeze(0).repeat(self.num_envs, 1).clone())
        self.randomized_frictions = torch.ones(self.num_envs, 1, device=self.device)
        self.randomized_restitutions = torch.zeros(self.num_envs, 1, device=self.device)
        self.randomized_added_masses = torch.zeros(self.num_envs, 1, device=self.device)
        self.randomized_com_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._motor_strength = torch.ones(self.num_envs, act, device=self.device)
        self._action_latency = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._latency_range = [
            max(0, int((self.cfg.domain_rand.latency_range[0] + 1e-8) / self.sim.get_physics_dt())),
            max(0, int((self.cfg.domain_rand.latency_range[1] - 1e-8) / self.sim.get_physics_dt())),
        ]
        self._push_interval_steps = max(1, int(self.cfg.domain_rand.push_interval_s / self.dt))

        soft = self._robot.data.soft_joint_pos_limits[0, :act]
        self.dof_pos_limits = soft.clone()
        self.dof_vel_limits = self._robot.data.joint_vel_limits[0, :act].clone()
        self.torque_limits = self._robot.data.joint_effort_limits[0, :act].clone()

        self._init_contact_body_ids()
        self._compute_terrain_category_masks()
        self._init_depth_indices()

        self.depth_buffer = torch.zeros(
            len(self.depth_index),
            self.cfg.depth.buffer_len,
            self.cfg.depth.resized[0],
            self.cfg.depth.resized[1],
            device=self.device,
        )

        self._policy_obs = torch.zeros(self.num_envs, self.num_obs, device=self.device)
        self._critic_obs = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device)
        self.obs_buf = self._policy_obs
        self.privileged_obs_buf = self._critic_obs
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)

        self.extras = {}

        self.last_reset_env_ids = torch.zeros(0, dtype=torch.long, device=self.device)
        self.terminal_amp_states = torch.zeros(0, 30, device=self.device)

        # episode logging sums
        self.episode_sums = {
            key: torch.zeros(self.num_envs, device=self.device)
            for key in [
                "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "torques",
                "dof_acc", "action_rate", "dof_error", "collision",
                "feet_air_time", "feet_stumble", "feet_edge", "cheat", "stuck",
            ]
        }

        self.reward_curriculum_coef = [s[2] for s in self.cfg.rewards.reward_curriculum_schedule] if self.cfg.rewards.reward_curriculum else []

        self._reward_scales = self._prepare_reward_scales()
        self._contact_filt = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self._default_body_masses = None
        self._default_body_inertia = None
        self._default_body_coms = None
        self._link_mass_scales = torch.ones(self.num_envs, 1, device=self.device)

        from .domain_rand import validate_domain_rand_config

        self._init_edge_mask()
        self._capture_default_body_properties()
        self._sample_static_domain_rand(torch.arange(self.num_envs, device=self.device))
        self._apply_static_domain_rand(torch.arange(self.num_envs, device=self.device))
        self._resample_commands(torch.arange(self.num_envs, device=self.device))
        self._update_heading_command()

        self._policy_obs = self._build_observations()
        self._critic_obs = self._policy_obs.clone()
        self.obs_buf = self._policy_obs
        self.privileged_obs_buf = self._critic_obs

    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        self._forward_height_scanner = RayCaster(self.cfg.forward_height_scanner)
        self.scene.sensors["forward_height_scanner"] = self._forward_height_scanner
        if self.cfg.depth.use_camera:
            self._depth_camera = RayCasterCamera(self.cfg.depth_camera)
            self.scene.sensors["depth_camera"] = self._depth_camera

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _prepare_reward_scales(self) -> dict[str, float]:
        """Scale reward coefficients by step_dt once (matches IsaacGym _prepare_reward_function)."""
        scales = {}
        cfg_scales = self.cfg.rewards.scales
        for key in [
            "tracking_lin_vel", "tracking_ang_vel", "lin_vel_z", "torques", "dof_acc",
            "action_rate", "collision", "feet_air_time", "feet_stumble", "feet_edge",
            "dof_error", "cheat", "stuck",
        ]:
            scale = float(getattr(cfg_scales, key, 0.0))
            scales[key] = scale * self.step_dt if scale != 0.0 else 0.0
        return scales

    def _init_edge_mask(self) -> None:
        self._x_edge_mask = None
        self._edge_mask_origin = None
        self._edge_mask_scale = self.cfg.terrain_meta.horizontal_scale
        terrain = getattr(self, "_terrain", None)
        if terrain is not None and getattr(terrain, "x_edge_mask", None) is not None:
            self._x_edge_mask = torch.tensor(terrain.x_edge_mask, device=self.device, dtype=torch.bool)
            self._edge_mask_origin = torch.tensor(
                terrain.edge_mask_world_origin, device=self.device, dtype=torch.float32
            )
            self._edge_mask_scale = float(terrain.edge_mask_horizontal_scale)

    def query_edge_mask(self, world_xy: torch.Tensor) -> torch.Tensor:
        if self._x_edge_mask is None or self._edge_mask_origin is None:
            return torch.zeros(world_xy.shape[:-1], device=self.device, dtype=torch.bool)
        idx = ((world_xy - self._edge_mask_origin) / self._edge_mask_scale).round().long()
        idx[..., 0].clamp_(0, self._x_edge_mask.shape[0] - 1)
        idx[..., 1].clamp_(0, self._x_edge_mask.shape[1] - 1)
        return self._x_edge_mask[idx[..., 0], idx[..., 1]]

    # ------------------------------------------------------------------ #
    def _init_contact_body_ids(self):
        def find(pattern):
            ids, _ = self._contact_sensor.find_bodies(pattern)
            return torch.tensor(ids, dtype=torch.long, device=self.device)

        self.base_contact_ids = find("base")
        feet_ids, feet_names = self._contact_sensor.find_bodies(".*foot")
        self.feet_indices = torch.tensor(feet_ids, dtype=torch.long, device=self.device)
        if len(self.feet_indices) == 0:
            feet_ids, feet_names = self._contact_sensor.find_bodies(".*_foot")
            self.feet_indices = torch.tensor(feet_ids, dtype=torch.long, device=self.device)
        # ContactSensor and Articulation maintain independent body orderings.
        # Keep sensor indices for contact tensors and map the same names into
        # articulation indices for body poses (used by the feet-edge reward).
        robot_body_id = {name: idx for idx, name in enumerate(self._robot.body_names)}
        missing = [name for name in feet_names if name not in robot_body_id]
        if missing:
            raise RuntimeError(f"Contact-sensor feet missing from articulation: {missing}")
        self.feet_body_indices = torch.tensor(
            [robot_body_id[name] for name in feet_names], dtype=torch.long, device=self.device
        )
        self.penalised_contact_indices = find(".*thigh|.*calf")
        self.termination_contact_indices = self.base_contact_ids

    def _init_depth_indices(self):
        camera_n = int(min(self.cfg.depth.camera_num_envs, self.num_envs)) if self.cfg.depth.use_camera else 0
        if camera_n <= 0:
            self.depth_index = torch.zeros(0, dtype=torch.long, device="cpu").numpy()
            self.depth_index_without_crawl_tilt = torch.zeros(0, dtype=torch.long, device="cpu").numpy()
            self.depth_index_inverse = -torch.ones(self.num_envs, dtype=torch.long, device="cpu").numpy()
            return
        idx = torch.arange(self.num_envs, device=self.device)
        # Prefer non-tilt/crawl envs for depth cameras so DepthPredictor has
        # training samples (matches original IsaacGym Go2 sampling).
        tc_mask = self._cat_mask(("tilt", "crawl"))
        ntc_envs = idx[~tc_mask]
        tc_envs = idx[tc_mask]
        chosen_ntc = ntc_envs[:camera_n]
        remaining = max(0, camera_n - len(chosen_ntc))
        chosen_tc = tc_envs[:remaining] if remaining > 0 and len(tc_envs) > 0 else idx[:0]
        depth_index = torch.cat([chosen_ntc, chosen_tc]).sort().values
        self.depth_index = depth_index.cpu().numpy().astype(int)
        ntc_positions = torch.isin(depth_index, tc_envs, assume_unique=False) if len(tc_envs) > 0 else \
            torch.zeros(len(depth_index), dtype=torch.bool, device=self.device)
        self.depth_index_without_crawl_tilt = depth_index[~ntc_positions].cpu().numpy().astype(int)
        inverse = -torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        inverse[depth_index] = torch.arange(len(depth_index), device=self.device)
        self.depth_index_inverse = inverse.cpu().numpy().astype(int)

    def _compute_terrain_category_masks(self):
        """Per-env bool masks for each curriculum terrain category (plane => all False)."""
        n = self.num_envs
        device = self.device
        self._cat_masks = {k: torch.zeros(n, dtype=torch.bool, device=device) for k in LEGACY_TERRAIN_NAMES}
        if self.cfg.terrain_meta.mesh_type != "trimesh":
            # plane: everything is treated as "rough_flat"
            self._cat_masks["rough_flat"][:] = True
            self._terrain_levels = torch.zeros(n, dtype=torch.long, device=device)
            return
        # rebuild column -> category using the shared legacy mapping
        cols = self.cfg.terrain_meta.num_cols
        col_names = column_category_names(cols, self.cfg.terrain_meta.terrain_proportions)
        ttypes = self._terrain.terrain_types if hasattr(self, "_terrain") else \
                 torch.div(torch.arange(n, device=device), max(1, n / cols), rounding_mode="floor").to(torch.long)
        for ci, name in enumerate(col_names):
            self._cat_masks[name] |= ttypes == ci
        self._terrain_levels = self._terrain.terrain_levels.clone()

    @staticmethod
    def _cat_name_to_idx(name):
        return LEGACY_TERRAIN_NAMES.index(name)

    def _cat_mask(self, cats):
        m = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for c in cats:
            m |= self._cat_masks[c]
        return m

    # ------------------------------------------------------------------ #
    def _pre_physics_step(self, actions: torch.Tensor):
        clip = self.cfg.normalization.clip_actions
        self._actions = torch.clamp(actions, -clip, clip)
        self._processed_actions = self.cfg.control.action_scale * self._actions + self.default_dof_pos
        self.global_counter += 1
        self._substep = 0
        dr = self.cfg.domain_rand
        if dr.randomize_action_latency:
            lo, hi = self._latency_range
            self._action_latency = torch.randint(lo, hi + 1, (self.num_envs,), device=self.device)
        # push robots
        if dr.push_robots and self.global_counter % self._push_interval_steps == 0:
            maxv = dr.max_push_vel_xy
            # write_root_velocity_to_sim expects (N, 6) = lin(3) + ang(3)
            vel = self._robot.data.root_com_vel_w.clone()
            vel[:, 0] = torch.empty(self.num_envs, device=self.device).uniform_(-maxv, maxv)
            vel[:, 1] = torch.empty(self.num_envs, device=self.device).uniform_(-maxv, maxv)
            self._robot.write_root_velocity_to_sim(vel)

    def _apply_action(self):
        dr = self.cfg.domain_rand
        # action latency: use last action for the first ``latency`` sub-steps
        use_last = self._action_latency > self._substep  # (N,)
        active = torch.where(use_last.unsqueeze(1), self._last_actions, self._actions)
        target = self.cfg.control.action_scale * active + self.default_dof_pos

        dof_pos = self._robot.data.joint_pos[:, :self.num_actions]
        dof_vel = self._robot.data.joint_vel[:, :self.num_actions]
        p = self.randomized_p_gains
        d = self.randomized_d_gains
        torques = p * (target - dof_pos) - d * dof_vel
        if dr.randomize_motor_strength:
            torques = torques * self._motor_strength
        torques = torch.clamp(torques, -self.torque_limits.unsqueeze(0), self.torque_limits.unsqueeze(0))
        self._robot.set_joint_effort_target(torques)
        self._substep += 1

    # ------------------------------------------------------------------ #
    def _get_observations(self) -> dict:
        self._policy_obs = self._build_observations()
        self._critic_obs = self._policy_obs.clone()
        self._update_depth_buffer()
        self.extras["depth"] = getattr(self, "_extras_depth", None)

        self._last_last_actions[:] = self._last_actions
        self._last_actions[:] = self._actions
        self._last_dof_vel[:] = self._robot.data.joint_vel[:, :self.num_actions]
        self._last_torques[:] = self._robot.data.applied_torque[:, :self.num_actions]

        self.obs_buf = self._policy_obs
        self.privileged_obs_buf = self._critic_obs
        return {"policy": self._policy_obs, "critic": self._critic_obs}

    def _get_rewards(self) -> torch.Tensor:
        return self._compute_rewards()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # post-physics bookkeeping that must happen before rewards
        self._post_physics_update()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)

        # Match original IsaacGym Go2 termination (legged_robot.check_termination).
        # Do NOT use absolute root_z < 0.18 — that kills robots on climb/low tiles every step.
        if len(self.termination_contact_indices) > 0:
            forces = self._contact_sensor.data.net_forces_w[:, self.termination_contact_indices, :]
            terminated |= torch.any(torch.norm(forces, dim=-1) > 1.0, dim=1)

        vel_err = self._robot.data.root_lin_vel_b[:, 0] - self.commands[:, 0]
        violate = ((vel_err > 1.5) & (self.commands[:, 0] < 0)) | ((vel_err < -1.5) & (self.commands[:, 0] > 0))
        violate &= (self._terrain_levels > 3)
        terminated |= violate

        # Gym root_states[:, 9] is world-frame linear vz (not angular vz).
        lin_vel_z = self._robot.data.root_lin_vel_w[:, 2]
        fall = (lin_vel_z < -3.0) | (self._robot.data.projected_gravity_b[:, 2] > 0.0)
        terminated |= fall

        reset_env_ids = (terminated | time_out).nonzero(as_tuple=False).flatten()
        self.last_reset_env_ids = reset_env_ids
        if len(reset_env_ids) > 0:
            self.terminal_amp_states = self.get_amp_observations()[reset_env_ids]
        else:
            self.terminal_amp_states = torch.zeros(0, 30, device=self.device)
        return terminated, time_out

    # ------------------------------------------------------------------ #
    def _post_physics_update(self):
        # resample commands at episode resampling_time, then heading-from-pos
        resample_steps = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % resample_steps == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._update_heading_command()

    def _update_heading_command(self):
        if not self.cfg.commands.heading_command:
            return
        forward = _quat_apply(self._robot.data.root_quat_w, torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1))
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        # obstacle (non rough_flat) envs steer toward their heading target
        obs_mask = ~self._cat_mask(("rough_flat",))
        self.commands[obs_mask, 2] = torch.clip(0.5 * _wrap_to_pi(self.commands[obs_mask, 3] - heading[obs_mask]), -1.0, 1.0)
        # tilt/climb envs use no heading target
        zero_mask = self._cat_mask(("tilt", "climb"))
        self.commands[zero_mask, 3] = 0.0
        self.commands[zero_mask, 2] = 0.0

    def _resample_commands(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        r = self.cfg.commands.ranges
        self.commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*r.lin_vel_x)
        self.commands[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(*r.lin_vel_y)
        self.commands[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(*r.ang_vel_yaw)
        self.commands[env_ids, 3] = torch.empty(len(env_ids), device=self.device).uniform_(*r.heading)
        # rough_flat uses the "flat" command ranges
        flat = env_ids[self._cat_mask(("rough_flat",))[env_ids]]
        if len(flat) > 0:
            self.commands[flat, 0] = torch.empty(len(flat), device=self.device).uniform_(*r.flat_lin_vel_x)
            self.commands[flat, 1] = torch.empty(len(flat), device=self.device).uniform_(*r.flat_lin_vel_y)
            self.commands[flat, 2] = torch.empty(len(flat), device=self.device).uniform_(*r.flat_ang_vel_yaw)
            self.commands[flat, 3] = torch.empty(len(flat), device=self.device).uniform_(*r.flat_heading)
        # zero small commands
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    # ------------------------------------------------------------------ #
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        # terrain curriculum (must run before super resets the robot)
        if self.cfg.terrain_meta.mesh_type == "trimesh" and self.cfg.terrain_meta.curriculum and len(env_ids) > 0:
            roots_xy = self._robot.data.root_pos_w[env_ids, :2]
            origins = self._terrain.env_origins[env_ids, :2]
            distance = torch.norm(roots_xy - origins, dim=1)
            move_up = distance > (self.cfg.terrain_meta.terrain_length / 2)
            move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length * self.dt * 0.5) & ~move_up
            self._terrain.update_env_origins(env_ids, move_up, move_down)
            self._terrain_levels[env_ids] = self._terrain.terrain_levels[env_ids]

        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        # robot state
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, :2] += torch.empty((len(env_ids), 2), device=self.device).uniform_(-1.0, 1.0)
        default_root_state[:, 7:13] = torch.empty((len(env_ids), 6), device=self.device).uniform_(-0.5, 0.5)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        joint_pos = self.default_dof_pos[env_ids] * torch.empty((len(env_ids), self.num_actions), device=self.device).uniform_(0.5, 1.5)
        joint_vel = torch.zeros((len(env_ids), self.num_actions), device=self.device)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._resample_commands(env_ids)
        if self.cfg.domain_rand.randomize_gains:
            self._sample_gains(env_ids)

        # reset buffers
        self._last_actions[env_ids] = 0.0
        self._last_last_actions[env_ids] = 0.0
        self._last_dof_vel[env_ids] = 0.0
        self._last_torques[env_ids] = 0.0
        self._contact_filt[env_ids] = False
        if len(self.depth_index) > 0:
            depth_ids = torch.as_tensor(self.depth_index, dtype=torch.long, device=self.device)
            in_reset = torch.isin(depth_ids, env_ids)
            self.depth_buffer[in_reset] = 0.0

        # episode logging
        ep = {}
        ep_len_s = self.cfg.env.episode_length_s
        for key in self.episode_sums:
            ep["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / ep_len_s
            self.episode_sums[key][env_ids] = 0.0
        if self.cfg.terrain_meta.curriculum:
            ep["terrain_level"] = torch.mean(self._terrain_levels.float())
        self.extras["episode"] = ep

    # ------------------------------------------------------------------ #
    def _sample_gains(self, env_ids):
        if len(env_ids) == 0:
            return
        dr = self.cfg.domain_rand
        sm = torch.empty((len(env_ids), self.num_actions), device=self.device).uniform_(*dr.stiffness_multiplier_range)
        dm = torch.empty((len(env_ids), self.num_actions), device=self.device).uniform_(*dr.damping_multiplier_range)
        self.randomized_p_gains[env_ids] = self.p_gains.unsqueeze(0) * sm
        self.randomized_d_gains[env_ids] = self.d_gains.unsqueeze(0) * dm
        if dr.randomize_motor_strength:
            self._motor_strength[env_ids] = torch.empty((len(env_ids), self.num_actions), device=self.device).uniform_(*dr.motor_strength_range)

    def _capture_default_body_properties(self) -> None:
        view = self._robot.root_physx_view
        self._default_body_masses = view.get_masses().clone()
        self._default_body_inertia = view.get_inertias().clone()
        self._default_body_coms = view.get_coms().clone()

    def _apply_static_domain_rand(self, env_ids: torch.Tensor) -> None:
        from .domain_rand import apply_static_domain_rand

        if self._default_body_masses is None:
            return
        apply_static_domain_rand(
            self._robot,
            env_ids,
            self.cfg.domain_rand,
            frictions=self.randomized_frictions,
            restitutions=self.randomized_restitutions,
            added_masses=self.randomized_added_masses,
            com_offsets=self.randomized_com_pos,
            link_mass_scales=self._link_mass_scales,
            default_masses=self._default_body_masses,
            default_inertia=self._default_body_inertia,
        )
        self._sync_privileged_from_physx(env_ids)

    def _sync_privileged_from_physx(self, env_ids: torch.Tensor) -> None:
        from .domain_rand import read_back_static_domain_rand

        if len(env_ids) == 0:
            return
        f, r, m, c = read_back_static_domain_rand(
            self._robot, env_ids, self.device, self._default_body_coms
        )
        self.randomized_frictions[env_ids] = f
        self.randomized_restitutions[env_ids] = r
        self.randomized_added_masses[env_ids] = m
        self.randomized_com_pos[env_ids] = c

    def _sample_static_domain_rand(self, env_ids):
        if len(env_ids) == 0:
            return
        dr = self.cfg.domain_rand
        if dr.randomize_friction:
            self.randomized_frictions[env_ids] = torch.empty((len(env_ids), 1), device=self.device).uniform_(*dr.friction_range)
        else:
            self.randomized_frictions[env_ids] = 1.0
        if dr.randomize_restitution:
            self.randomized_restitutions[env_ids] = torch.empty((len(env_ids), 1), device=self.device).uniform_(*dr.restitution_range)
        else:
            self.randomized_restitutions[env_ids] = 0.0
        if dr.randomize_base_mass:
            self.randomized_added_masses[env_ids] = torch.empty((len(env_ids), 1), device=self.device).uniform_(*dr.added_mass_range)
        else:
            self.randomized_added_masses[env_ids] = 0.0
        if dr.randomize_com_pos:
            for i, axis in enumerate(("com_x_pos_range", "com_y_pos_range", "com_z_pos_range")):
                self.randomized_com_pos[env_ids, i] = torch.empty((len(env_ids),), device=self.device).uniform_(*getattr(dr, axis))
        else:
            self.randomized_com_pos[env_ids] = 0.0
        if dr.randomize_link_mass:
            n_links = max(1, self._robot.num_bodies - 1)
            if self._link_mass_scales.shape[1] != n_links:
                self._link_mass_scales = torch.ones(self.num_envs, n_links, device=self.device)
            self._link_mass_scales[env_ids] = torch.empty(
                (len(env_ids), n_links), device=self.device
            ).uniform_(*dr.link_mass_range)
        else:
            self._link_mass_scales[env_ids] = 1.0

    # ------------------------------------------------------------------ #
    def _height_scan(self) -> torch.Tensor:
        sn = self._height_scanner
        base_z = self._robot.data.root_pos_w[:, 2].unsqueeze(1)
        heights = base_z - sn.data.ray_hits_w[..., 2]
        heights = (heights - self.cfg.normalization.base_height).clip(-1.0, 1.0)
        if heights.shape[1] < self.height_dim:
            heights = torch.nn.functional.pad(heights, (0, self.height_dim - heights.shape[1]))
        return heights[:, :self.height_dim] * self.cfg.normalization.obs_scales.height_measurements

    def get_forward_map(self) -> torch.Tensor:
        sn = self._forward_height_scanner
        base_z = self._robot.data.root_pos_w[:, 2].unsqueeze(1)
        heights = base_z - sn.data.ray_hits_w[..., 2]
        heights = (heights - self.cfg.normalization.base_height).clip(-1.0, 1.0)
        if heights.shape[1] < self.forward_height_dim:
            heights = torch.nn.functional.pad(heights, (0, self.forward_height_dim - heights.shape[1]))
        return heights[:, :self.forward_height_dim] * self.cfg.normalization.obs_scales.height_measurements

    def _build_observations(self) -> torch.Tensor:
        n = self.num_envs
        device = self.device
        cs = self._contact_sensor.data
        na = self.num_actions

        # contact_flag (penalised thigh+calf) and contact_force (feet xyz)
        contact_flag = torch.zeros(n, len(self.penalised_contact_indices), device=device)
        contact_force = torch.zeros(n, 12, device=device)
        if len(self.penalised_contact_indices) > 0:
            pf = cs.net_forces_w[:, self.penalised_contact_indices, :]
            contact_flag = (torch.norm(pf, dim=-1) > 0.1).float()
        if len(self.feet_indices) > 0:
            ff = cs.net_forces_w[:, self.feet_indices, :]
            contact_force = ff.reshape(n, -1)[:, :12]

        p_rel = (self.randomized_p_gains[:, :na] / self.p_gains.unsqueeze(0)[:, :na] - 1.0) * self.cfg.normalization.obs_scales.pd_gains
        d_rel = (self.randomized_d_gains[:, :na] / self.d_gains.unsqueeze(0)[:, :na] - 1.0) * self.cfg.normalization.obs_scales.pd_gains

        privileged = torch.cat(
            (
                contact_flag,  # 8
                contact_force * self.cfg.normalization.obs_scales.contact_force,  # 12
                d_rel,  # 12
                p_rel,  # 12
                self.randomized_com_pos * self.cfg.normalization.obs_scales.com_pos,  # 3
                self.randomized_added_masses,  # 1
                self.randomized_restitutions,  # 1
                self.randomized_frictions,  # 1
                self._robot.data.root_lin_vel_b * self.cfg.normalization.obs_scales.lin_vel,  # 3
            ),
            dim=-1,
        )

        prop = torch.cat(
            (
                self._robot.data.root_ang_vel_b * self.cfg.normalization.obs_scales.ang_vel,  # 3
                self._robot.data.projected_gravity_b,  # 3
                self.commands[:, :3] * self.commands_scale,  # 3
                (self._robot.data.joint_pos[:, :na] - self.default_dof_pos) * self.cfg.normalization.obs_scales.dof_pos,  # 12
                self._robot.data.joint_vel[:, :na] * self.cfg.normalization.obs_scales.dof_vel,  # 12
            ),
            dim=-1,
        )

        obs = torch.cat((privileged, prop, self._actions, self._height_scan()), dim=-1)
        obs = torch.clamp(obs[:, :self.num_obs], -self.cfg.normalization.clip_observations, self.cfg.normalization.clip_observations)
        return obs

    def _update_depth_buffer(self):
        self._extras_depth = None
        if not self.cfg.depth.use_camera or len(self.depth_index) == 0:
            return
        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        cam = self._depth_camera
        depth = cam.data.output["distance_to_image_plane"].squeeze(-1)  # (N,H,W)
        depth = depth[torch.as_tensor(self.depth_index, device=self.device)]
        depth = torch.clamp(depth, self.cfg.depth.near_clip, self.cfg.depth.far_clip)
        depth = (depth - self.cfg.depth.near_clip) / (self.cfg.depth.far_clip - self.cfg.depth.near_clip) - 0.5
        init = (self.episode_length_buf[torch.as_tensor(self.depth_index, device=self.device)] <= 1).float().view(-1, 1, 1)
        new_frame = depth
        # fill whole buffer on init step
        filled = init * new_frame + (1 - init) * self.depth_buffer[:, -1]
        self.depth_buffer = torch.cat((self.depth_buffer[:, 1:], filled.unsqueeze(1)), dim=1)
        self._extras_depth = self.depth_buffer[:, -1]

    # ------------------------------------------------------------------ #
    def get_amp_observations(self) -> torch.Tensor:
        return torch.cat(
            (
                self._robot.data.joint_pos[:, :self.num_actions],
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.joint_vel[:, :self.num_actions],
            ),
            dim=-1,
        )

    def get_observations(self) -> torch.Tensor:
        return self._policy_obs

    def get_privileged_observations(self) -> torch.Tensor:
        return self._critic_obs

    def update_reward_curriculum(self, current_iter: int):
        if not self.cfg.rewards.reward_curriculum:
            return
        for idx, sched in enumerate(self.cfg.rewards.reward_curriculum_schedule):
            ratio = (current_iter - sched[0]) / max(sched[1] - sched[0], 1)
            ratio = max(0.0, min(1.0, ratio))
            self.reward_curriculum_coef[idx] = (1 - ratio) * sched[2] + ratio * sched[3]

    # ------------------------------------------------------------------ #
    @property
    def dof_pos(self):
        return self._robot.data.joint_pos[:, :self.num_actions]

    @property
    def dof_vel(self):
        return self._robot.data.joint_vel[:, :self.num_actions]

    # ------------------------------------------------------------------ #
    def _compute_rewards(self) -> torch.Tensor:
        s = self._reward_scales
        robot = self._robot.data
        dof_pos = robot.joint_pos[:, :self.num_actions]
        dof_vel = robot.joint_vel[:, :self.num_actions]
        torques = robot.applied_torque[:, :self.num_actions]
        lin_vel_b = robot.root_lin_vel_b
        ang_vel_b = robot.root_ang_vel_b
        commands = self.commands

        total = torch.zeros(self.num_envs, device=self.device)
        logs = {}

        def add(name, term, scale):
            nonlocal total
            if scale == 0.0:
                logs[name] = torch.zeros_like(total)
                return
            # reward curriculum
            coef = 1.0
            if self.cfg.rewards.reward_curriculum:
                for j, t in enumerate(self.cfg.rewards.reward_curriculum_term):
                    if t == name:
                        coef = self.reward_curriculum_coef[j]
                        break
            r = term * coef
            total = total + scale * r
            self.episode_sums[name] += scale * r
            logs[name] = torch.mean(r).detach()

        # ----- individual reward terms -----
        add("tracking_lin_vel", self._rew_tracking_lin_vel(lin_vel_b, commands), s["tracking_lin_vel"])
        add("tracking_ang_vel", self._rew_tracking_ang_vel(ang_vel_b, commands), s["tracking_ang_vel"])
        add("lin_vel_z", torch.square(lin_vel_b[:, 2]), s["lin_vel_z"])
        add("torques", torch.sum(torch.square(torques), dim=1), s["torques"])
        add("dof_acc", torch.sum(torch.square((self._last_dof_vel - dof_vel) / self.dt), dim=1), s["dof_acc"])
        add("action_rate", torch.sum(torch.square(self._last_actions - self._actions), dim=1), s["action_rate"])
        add("dof_error", torch.sum(torch.square(dof_pos - self.default_dof_pos), dim=1), s["dof_error"])
        add("collision", self._rew_collision(), s["collision"])
        add("feet_air_time", self._rew_feet_air_time(commands), s["feet_air_time"])
        add("feet_stumble", self._rew_feet_stumble(), s["feet_stumble"])
        add("feet_edge", self._rew_feet_edge(), s["feet_edge"])
        add("cheat", self._rew_cheat(), s["cheat"])
        add("stuck", self._rew_stuck(lin_vel_b, commands), s["stuck"])

        total = torch.clamp(total, min=0.0)
        # minimal per-step episode record; _reset_idx fills the real episode dict
        if "episode" not in self.extras:
            self.extras["episode"] = {}
        self.extras["episode"]["rew_total"] = torch.mean(total).detach()
        return total

    # ----- reward term implementations -----
    def _rew_tracking_lin_vel(self, lin_vel_b, commands):
        lin_vel = lin_vel_b[:, :2].clone()
        clip = self.cfg.rewards.lin_vel_clip
        upper = torch.where(commands[:, :2] < 0, torch.full_like(commands[:, :2], 1e5), commands[:, :2] + clip)
        lower = torch.where(commands[:, :2] > 0, torch.full_like(commands[:, :2], -1e5), commands[:, :2] - clip)
        clamped = torch.clip(lin_vel, lower, upper)
        err = torch.sum(torch.square(commands[:, :2] - clamped), dim=1)
        return torch.exp(-err / self.cfg.rewards.tracking_sigma)

    def _rew_tracking_ang_vel(self, ang_vel_b, commands):
        err = torch.square(commands[:, 2] - ang_vel_b[:, 2])
        return torch.exp(-err / self.cfg.rewards.tracking_sigma)

    def _rew_collision(self):
        if len(self.penalised_contact_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        f = self._contact_sensor.data.net_forces_w[:, self.penalised_contact_indices, :]
        return torch.sum((torch.norm(f, dim=-1) > 0.1).float(), dim=1)

    def _rew_feet_air_time(self, commands):
        if len(self.feet_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        first = self._contact_sensor.compute_first_contact(self.step_dt)[:, self.feet_indices]
        last_air = self._contact_sensor.data.last_air_time[:, self.feet_indices]
        contact = self._contact_sensor.data.net_forces_w[:, self.feet_indices, 2] > 1.0
        self._contact_filt = contact | self._contact_filt
        rew = torch.sum((last_air - 0.5) * first, dim=1)
        rew = rew * (torch.norm(commands[:, :2], dim=1) > 0.1).float()
        return rew

    def _rew_feet_stumble(self):
        if len(self.feet_indices) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        f = self._contact_sensor.data.net_forces_w[:, self.feet_indices, :]
        xy = torch.norm(f[:, :, :2], dim=-1)
        z = torch.abs(f[:, :, 2])
        stumbled = torch.any(xy > 4 * z, dim=1).float()
        stumbled = stumbled * (self._terrain_levels > 3).float()
        out = torch.zeros(self.num_envs, device=self.device)
        mask = self._cat_mask(("gap", "climb"))
        out[mask] = stumbled[mask]
        return out

    def _rew_feet_edge(self):
        out = torch.zeros(self.num_envs, device=self.device)
        if len(self.feet_indices) == 0 or self._x_edge_mask is None:
            return out
        feet_xy = self._robot.data.body_pos_w[:, self.feet_body_indices, :2]
        feet_at_edge = self.query_edge_mask(feet_xy)
        edge_contact = self._contact_filt & feet_at_edge
        rew = (self._terrain_levels > 3).float() * torch.sum(edge_contact.float(), dim=-1)
        mask = self._cat_mask(("gap", "climb"))
        out[mask] = rew[mask]
        return out

    def _rew_cheat(self):
        forward = _quat_apply(self._robot.data.root_quat_w, torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1))
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        cheat = ((heading > 1.0) | (heading < -1.0)).float()
        out = torch.zeros(self.num_envs, device=self.device)
        obs_mask = ~self._cat_mask(("rough_flat",))
        out[obs_mask] = cheat[obs_mask]
        return out

    def _rew_stuck(self, lin_vel_b, commands):
        return ((torch.abs(lin_vel_b[:, 0]) < 0.1) & (torch.abs(commands[:, 0]) > 0.1)).float()


def register_tasks():
    """Register the Go2 tasks via gymnasium (idempotent)."""
    import gymnasium as gym
    from .agents import (  # noqa: F401
        ppo_cfg,
        wmp_cfg,
    )

    gym.register(
        id="WMP-Go2-Flat-PPO-v0",
        entry_point=f"{__name__}:Go2WmpLabEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}_cfg:Go2RoughLabCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.ppo_cfg:Go2RoughRunnerCfg",
        },
    )
    gym.register(
        id="WMP-Go2-AMP-v0",
        entry_point=f"{__name__}:Go2WmpLabEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}_cfg:Go2AmpLabCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.wmp_cfg:Go2AmpWMPRunnerCfg",
        },
    )
