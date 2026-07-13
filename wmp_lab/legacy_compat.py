from __future__ import annotations

from typing import Any

import torch


def class_to_dict(obj: Any) -> dict:
    """Recursively convert legacy config classes/configclass instances to dictionaries."""
    if isinstance(obj, dict):
        return {key: class_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [class_to_dict(value) for value in obj]
    if not hasattr(obj, "__dict__") and not isinstance(obj, type):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        try:
            value = getattr(obj, key)
        except AttributeError:
            continue
        if callable(value) and not isinstance(value, type):
            continue
        result[key] = class_to_dict(value)
    return result


def load_entry_point(entry_point: str):
    import importlib

    module_name, attr_name = entry_point.split(":")
    return getattr(importlib.import_module(module_name), attr_name)


class LegacyRslRlWrapper:
    """Adapter from IsaacLab DirectRLEnv to the legacy RSL-RL VecEnv contract."""

    def __init__(self, env, clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions
        self.unwrapped = env.unwrapped
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_obs = self.unwrapped.num_obs
        self.num_privileged_obs = self.unwrapped.num_privileged_obs
        self.num_actions = self.unwrapped.num_actions
        self.privileged_dim = self.unwrapped.privileged_dim
        self.height_dim = self.unwrapped.height_dim
        self.include_history_steps = self.unwrapped.include_history_steps
        self.cfg = self.unwrapped.cfg

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value

    def reset(self):
        obs_dict, _ = self.env.reset()
        self.unwrapped.obs_buf = obs_dict["policy"]
        self.unwrapped.privileged_obs_buf = obs_dict.get("critic", obs_dict["policy"])
        return self.unwrapped.obs_buf, self.unwrapped.privileged_obs_buf

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not getattr(self.unwrapped.cfg, "is_finite_horizon", False):
            extras["time_outs"] = truncated
        obs = obs_dict["policy"]
        privileged_obs = obs_dict.get("critic", obs)
        reset_env_ids = self.unwrapped.last_reset_env_ids
        terminal_amp_states = self.unwrapped.terminal_amp_states
        return obs, privileged_obs, rewards, dones, extras, reset_env_ids, terminal_amp_states

    def get_observations(self):
        return self.unwrapped.get_observations()

    def get_privileged_observations(self):
        return self.unwrapped.get_privileged_observations()

    def get_amp_observations(self):
        return self.unwrapped.get_amp_observations()

    def get_forward_map(self):
        return self.unwrapped.get_forward_map()

    def update_reward_curriculum(self, current_iter: int):
        return self.unwrapped.update_reward_curriculum(current_iter)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.unwrapped, name)
