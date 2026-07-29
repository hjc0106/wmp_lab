"""Go2 task registrations."""

import gymnasium as gym

from . import agents

gym.register(
    id="WMP-Go2-Flat-PPO-v0",
    entry_point="wmp_lab.tasks.go2.go2_env:Go2WmpLabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "wmp_lab.tasks.go2.go2_env_cfg:Go2RoughLabCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:Go2RoughRunnerCfg",
    },
)

# NOTE: WMP-Go2-Rough-v0 (full Gym rough terrain) is not implemented yet.
# Use WMP-Go2-Flat-PPO-v0 for plane PPO smoke tests until the rough task lands.

gym.register(
    id="WMP-Go2-AMP-v0",
    entry_point="wmp_lab.tasks.go2.go2_env:Go2WmpLabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "wmp_lab.tasks.go2.go2_env_cfg:Go2AmpLabCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.wmp_cfg:Go2AmpWMPRunnerCfg",
    },
)
