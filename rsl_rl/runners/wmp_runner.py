# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

# This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
# All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

import time
import os
import shutil
from collections import deque
import statistics

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import AMPPPO, PPO
from rsl_rl.modules import ActorCritic, ActorCriticWMP, ActorCriticRecurrent
from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer
from rsl_rl.modules import DepthPredictor
import torch.optim as optim

from dreamer.models import *
import ruamel.yaml as yaml
import argparse
import pathlib
import sys
import collections
from dreamer import tools
import datetime
import uuid
class WMPRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 history_length=5,
                 ):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.depth_predictor_cfg = train_cfg["depth_predictor"]
        self.device = device
        self.env = env
        self.history_length = history_length
        self._configure_multi_gpu()
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        if self.env.include_history_steps is not None:
            num_actor_obs = self.env.num_obs * self.env.include_history_steps
        else:
            num_actor_obs = self.env.num_obs



        # build world model
        self._build_world_model()

        # build depth predictor
        self.depth_predictor = DepthPredictor().to(self._world_model.device)
        self.depth_predictor_opt = optim.Adam(self.depth_predictor.parameters(), lr=self.depth_predictor_cfg["lr"],
                                              weight_decay=self.depth_predictor_cfg["weight_decay"])

        self.history_dim = history_length * (self.env.num_obs - self.env.privileged_dim - self.env.height_dim-3) #exclude command
        actor_critic = ActorCriticWMP(num_actor_obs=num_actor_obs,
                                          num_critic_obs=num_critic_obs,
                                          num_actions=self.env.num_actions,
                                          height_dim=self.env.height_dim,
                                          privileged_dim=self.env.privileged_dim,
                                          history_dim=self.history_dim,
                                          wm_feature_dim=self.wm_feature_dim,
                                          **self.policy_cfg).to(self.device)

        amp_data = AMPLoader(
            device, time_between_frames=self.env.dt, preload_transitions=True,
            num_preload_transitions=train_cfg['runner']['amp_num_preload_transitions'],
            motion_files=self.cfg["amp_motion_files"])
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            train_cfg['runner']['amp_reward_coef'],
            train_cfg['runner']['amp_discr_hidden_dims'], device,
            train_cfg['runner']['amp_task_reward_lerp']).to(self.device)

        # self.discr: AMPDiscriminator = AMPDiscriminator()
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        min_std = (
                torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
                (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0])))
        self.alg: PPO = alg_class(actor_critic, discriminator, amp_data, amp_normalizer, device=self.device,
                                  min_std=min_std, multi_gpu_cfg=self.cfg.get("multi_gpu"), **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs],
                              [self.env.num_privileged_obs], [self.env.num_actions], self.history_dim, self.wm_feature_dim)

        # Log
        self.log_dir = log_dir if self.gpu_global_rank == 0 else None
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()


    def _configure_multi_gpu(self):
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }
        expected = os.getenv("WMP_LAB_CUDA_DEVICE", f"cuda:{self.gpu_local_rank}")
        if self.device != expected:
            raise ValueError(
                f"Device '{self.device}' does not match expected device '{expected}' "
                f"(local_rank={self.gpu_local_rank})."
            )
        # NCCL must see the rank-local device before process-group creation.
        # Otherwise all ranks may initialize on cuda:0 and later hang/fail in collectives.
        device_id = int(str(self.device).split(":")[-1]) if "cuda:" in str(self.device) else self.gpu_local_rank
        if "cuda:" in str(self.device):
            torch.cuda.set_device(device_id)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
            )

    def _broadcast_world_model_state(self):
        """Keep world-model / depth-predictor weights identical across ranks (rank0 source).

        Broadcast tensors in-place. Do NOT use broadcast_object_list on CUDA
        state_dicts: pickling large GPU tensors often OOMs / kills rank0 and
        then other ranks die with ncclRemoteError on BROADCAST.
        """
        if not self.is_distributed:
            return
        if "cuda:" in str(self.device):
            torch.cuda.set_device(int(str(self.device).split(":")[-1]))
        for module in (self._world_model, self.depth_predictor):
            for param in module.parameters():
                torch.distributed.broadcast(param.data, src=0)
            for buffer in module.buffers():
                torch.distributed.broadcast(buffer.data, src=0)


    def _build_world_model(self):
        # world model
        print('Begin construct world model')
        root_dir = pathlib.Path(__file__).resolve().parents[2]
        yaml_loader = yaml.YAML(typ="safe", pure=True)
        configs = yaml_loader.load((root_dir / "dreamer/configs.yaml").read_text())

        def recursive_update(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base:
                    recursive_update(base[key], value)
                else:
                    base[key] = value

        name_list = ["defaults"]
        defaults = {}
        for name in name_list:
            recursive_update(defaults, configs[name])
        parser = argparse.ArgumentParser()
        parser.add_argument("--headless", action="store_true", default=False)
        parser.add_argument("--sim_device", default='cuda:0')
        parser.add_argument("--wm_device", default='None')
        parser.add_argument("--terrain", default='climb')
        for key, value in sorted(defaults.items(), key=lambda x: x[0]):
            arg_type = tools.args_type(value)
            parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
        self.wm_config = parser.parse_args()
        # allow world model and rl env on different device
        if (self.wm_config.wm_device != 'None'):
            self.wm_config.device = self.wm_config.wm_device
        self.wm_config.num_actions = self.wm_config.num_actions * self.env.cfg.depth.update_interval
        prop_dim = self.env.num_obs - self.env.privileged_dim - self.env.height_dim - self.env.num_actions
        image_shape = self.env.cfg.depth.resized + (1,)
        obs_shape = {'prop': (prop_dim,), 'image': image_shape,}

        self._world_model = WorldModel(self.wm_config, obs_shape, use_camera=self.env.cfg.depth.use_camera)
        self._world_model = self._world_model.to(self._world_model.device)
        print('Finish construct world model')
        self.wm_feature_dim = self.wm_config.dyn_deter #+ self.wm_config.dyn_stoch * self.wm_config.dyn_discrete


    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if self.is_distributed:
            self.alg.broadcast_parameters()
            self._broadcast_world_model_state()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        # process trajectory history
        self.trajectory_history = torch.zeros(size=(self.env.num_envs, self.history_length, self.env.num_obs -
                                                    self.env.privileged_dim - self.env.height_dim - 3),
                                              device=self.device)
        obs_without_command = torch.concat((obs[:, self.env.privileged_dim:self.env.privileged_dim + 6],
                                            obs[:, self.env.privileged_dim + 9:-self.env.height_dim]), dim=1)
        self.trajectory_history = torch.concat((self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)),
                                               dim=1)

        # init world model input
        sum_wm_dataset_size = 0
        wm_latent = wm_action = None
        wm_is_first = torch.ones(self.env.num_envs, device=self._world_model.device)
        wm_obs = {
            "prop": obs[:, self.env.privileged_dim: self.env.privileged_dim + self.env.cfg.env.prop_dim].to(self._world_model.device),
            "is_first": wm_is_first,
        }

        if(self.env.cfg.depth.use_camera):
            wm_obs["image"] = torch.zeros(((self.env.num_envs,) + self.env.cfg.depth.resized + (1,)), device=self._world_model.device)

        wm_metrics = None
        self.wm_update_interval = self.env.cfg.depth.update_interval
        wm_action_history = torch.zeros(size=(self.env.num_envs, self.wm_update_interval, self.env.num_actions),
                                        device=self._world_model.device)
        wm_reward = torch.zeros(self.env.num_envs, device=self._world_model.device)
        wm_feature = torch.zeros((self.env.num_envs, self.wm_feature_dim))

        self.init_wm_dataset()


        for it in range(self.current_learning_iteration, tot_iter):
            if self.gpu_global_rank == 0:
                print(f"[wmp] begin iteration {it}/{tot_iter}", flush=True)
            if (self.env.cfg.rewards.reward_curriculum):
                self.env.update_reward_curriculum(it)
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if (self.env.global_counter % self.wm_update_interval == 0):
                        # world model obs step
                        wm_embed = self._world_model.encoder(wm_obs)
                        wm_latent, _ = self._world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed,
                                                                           wm_obs["is_first"])
                        wm_feature = self._world_model.dynamics.get_deter_feat(wm_latent)
                        wm_is_first[:] = 0

                    history = self.trajectory_history.flatten(1).to(self.device)
                    actions = self.alg.act(obs, critic_obs, amp_obs, history, wm_feature.to(self.env.device))
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(
                        actions)
                    next_amp_obs = self.env.get_amp_observations()

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_amp_obs, rewards, dones = obs.to(self.device), critic_obs.to(
                        self.device), next_amp_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # update world model input
                    wm_action_history = torch.concat(
                        (wm_action_history[:, 1:], actions.unsqueeze(1).to(self._world_model.device)), dim=1)
                    wm_obs = {
                        "prop": obs[:, self.env.privileged_dim: self.env.privileged_dim + self.env.cfg.env.prop_dim].to(self._world_model.device),
                        "is_first": wm_is_first,
                    }

                    # store the data in buffer into the dataset before reset
                    reset_env_ids = reset_env_ids.cpu().numpy()
                    if (len(reset_env_ids) > 0):
                        for k, v in self.wm_dataset.items():
                            if(k == "image"):
                                for id in reset_env_ids:
                                    idx_in_buffer = np.where(self.env.depth_index == id)[0]
                                    if(len(idx_in_buffer) > 0):
                                        v[idx_in_buffer, :] = self.wm_buffer[k][idx_in_buffer].to(self._world_model.device)
                            else:
                                v[reset_env_ids, :] = self.wm_buffer[k][reset_env_ids].to(self._world_model.device)

                        self.wm_dataset_size[reset_env_ids] = self.wm_buffer_index[reset_env_ids]
                        self.wm_buffer_index[reset_env_ids] = 0
                        sum_wm_dataset_size = np.sum(self.wm_dataset_size)

                        wm_action_history[reset_env_ids, :] = 0
                        wm_is_first[reset_env_ids] = 1

                    wm_action = wm_action_history.flatten(1)
                    wm_reward += rewards.to(self._world_model.device)

                    # store current step into buffer
                    if (self.env.global_counter % self.wm_update_interval == 0):
                        if (self.env.cfg.depth.use_camera):
                            forward_heightmap = self.env.get_forward_map().to(self._world_model.device)
                            pred_depth_image = self.depth_predictor(forward_heightmap, wm_obs["prop"])
                            wm_obs["image"] = pred_depth_image
                            self.wm_buffer["forward_height_map"][range(self.env.num_envs), self.wm_buffer_index,:] = forward_heightmap[:].to('cpu')
                            wm_obs["image"][self.env.depth_index] = infos["depth"].unsqueeze(-1).to(self._world_model.device)
                            self.wm_buffer["image"][range(self.env.cfg.depth.camera_num_envs),
                            self.wm_buffer_index[self.env.depth_index], :] = wm_obs["image"][self.env.depth_index].to(
                                'cpu')
                        # not_reset_env_ids = (~dones).nonzero(as_tuple=False).flatten().cpu().numpy()
                        not_reset_env_ids = (1 - wm_is_first).nonzero(as_tuple=False).flatten().cpu().numpy()
                        if (len(not_reset_env_ids) > 0):
                            for k, v in wm_obs.items():
                                if(k != "is_first" and k != "image"):
                                    self.wm_buffer[k][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = v[not_reset_env_ids].to('cpu')
                            self.wm_buffer["action"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids], :] = \
                                wm_action[not_reset_env_ids, :].to('cpu')
                            self.wm_buffer["reward"][not_reset_env_ids, self.wm_buffer_index[not_reset_env_ids]] = \
                                wm_reward[not_reset_env_ids].to('cpu')
                            self.wm_buffer_index[not_reset_env_ids] += 1
                            # Prevent silent CUDA OOB if an env exceeds the preallocated horizon.
                            max_buf = self.wm_buffer["prop"].shape[1] - 1
                            overflow = self.wm_buffer_index >= self.wm_buffer["prop"].shape[1]
                            if np.any(overflow):
                                self.wm_buffer_index[overflow] = max_buf

                        wm_reward[:] = 0

                    # Account for terminal states.
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)[0]
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)

                    # process trajectory history
                    env_ids = dones.nonzero(as_tuple=False).flatten()
                    self.trajectory_history[env_ids] = 0
                    obs_without_command = torch.concat((obs[:, self.env.privileged_dim:self.env.privileged_dim + 6],
                                                        obs[:, self.env.privileged_dim + 9:-self.env.height_dim]),
                                                       dim=1)
                    self.trajectory_history = torch.concat(
                        (self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, wm_feature.to(self.env.device))
            mean_value_loss, mean_surrogate_loss, mean_vel_predict_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if self.log_dir is not None and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()


            start_time = time.time()
            # Only rank0 trains the WM/depth modules, so rank0 must own the
            # readiness decision. Broadcast it so all ranks enter the same
            # collectives after the optional rank0 update.
            do_wm = 1 if sum_wm_dataset_size > self.wm_config.train_start_steps else 0
            if self.is_distributed:
                flag = torch.tensor(
                    [do_wm if self.gpu_global_rank == 0 else 0],
                    device=self.device,
                    dtype=torch.int32,
                )
                torch.distributed.broadcast(flag, src=0)
                do_wm = int(flag.item())

            if do_wm:
                # Train only on rank0, then broadcast weights so all ranks stay aligned.
                if self.gpu_global_rank == 0:
                    if (it % self.depth_predictor_cfg["training_interval"] == 0):
                        depth_mse_loss = self.train_depth_predictor()
                        if self.writer is not None:
                            self.writer.add_scalar('DepthPredictor/loss', depth_mse_loss, it)

                    wm_metrics = self.train_world_model()
                    if self.writer is not None:
                        for name, values in wm_metrics.items():
                            self.writer.add_scalar('World_model/' + name, float(np.mean(values)), it)
                    # Surface async CUDA errors from WM/depth before the next rollout.
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                if self.is_distributed:
                    self._broadcast_world_model_state()

            if self.gpu_global_rank == 0:
                print('training world model time:', time.time() - start_time, flush=True)

            # copy the migrated IsaacLab task config for reproducibility
            if self.log_dir is not None and it == 0:
                config_path = pathlib.Path(__file__).resolve().parents[2] / "wmp_lab/tasks/go2/go2_env_cfg.py"
                if config_path.exists():
                    shutil.copy2(config_path, self.log_dir)

        self.current_learning_iteration += num_learning_iterations
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def init_wm_dataset(self):
        self.wm_dataset = {
            "prop": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, self.env.cfg.env.prop_dim),
                                device=self._world_model.device),
            "action": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   self.env.num_actions * self.wm_update_interval), device=self._world_model.device),
            "reward": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,),
                                  device=self._world_model.device),
        }
        if(self.env.cfg.depth.use_camera):
            self.wm_dataset["image"] = torch.zeros(((self.env.cfg.depth.camera_num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,)
                                               + self.env.cfg.depth.resized + (1,)), device=self._world_model.device)
            self.wm_dataset["forward_height_map"] = torch.zeros(
                (self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                 self.env.cfg.env.forward_height_dim), device=self._world_model.device)

        self.wm_dataset_size = np.zeros(self.env.num_envs)

        self.wm_buffer = {
            "prop": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3, self.env.cfg.env.prop_dim),
                                device='cpu'),
            "action": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                                   self.env.num_actions * self.wm_update_interval), device='cpu'),
            "reward": torch.zeros((self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,),
                                  device='cpu'),
        }
        if(self.env.cfg.depth.use_camera):
            self.wm_buffer["image"] = torch.zeros(((self.env.cfg.depth.camera_num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,)
                                               + self.env.cfg.depth.resized + (1,)), device='cpu')
            self.wm_buffer["forward_height_map"] = torch.zeros(
                (self.env.num_envs, int(self.env.max_episode_length / self.wm_update_interval) + 3,
                 self.env.cfg.env.forward_height_dim), device='cpu')

        self.wm_buffer_index = np.zeros(self.env.num_envs)

    def train_depth_predictor(self):
        total_mse_loss = 0
        ready_env_ids = [
            idx for idx in self.env.depth_index_without_crawl_tilt
            if int(self.wm_dataset_size[idx]) > 0
        ]
        if len(ready_env_ids) == 0:
            return 0.0
        for _ in range(self.depth_predictor_cfg["training_iters"]):
            batch_idx = np.random.choice(ready_env_ids, self.depth_predictor_cfg["batch_size"], replace=True)
            # Valid indices are [0, size); size itself is out of range.
            time_index = [np.random.randint(0, int(self.wm_dataset_size[idx])) for idx in batch_idx]
            forward_heightmap = self.wm_dataset["forward_height_map"][batch_idx, time_index]
            prop = self.wm_dataset["prop"][batch_idx, time_index]
            depth_image = self.wm_dataset["image"][self.env.depth_index_inverse[batch_idx], time_index]

            predict_depth_image = self.depth_predictor(forward_heightmap, prop)
            depth_predict_loss = (depth_image - predict_depth_image).pow(2).mean() * self.depth_predictor_cfg[
                "loss_scale"]
            # Gradient step
            self.depth_predictor_opt.zero_grad()
            depth_predict_loss.backward()
            nn.utils.clip_grad_norm_(self.depth_predictor.parameters(), 1)
            self.depth_predictor_opt.step()
            total_mse_loss += depth_predict_loss.detach() / self.depth_predictor_cfg["loss_scale"]
        return float(total_mse_loss / self.depth_predictor_cfg["training_iters"])

    def train_world_model(self):
        wm_metrics = {}
        mets = {}
        dataset_size_sum = float(np.sum(self.wm_dataset_size))
        if dataset_size_sum <= 0:
            return wm_metrics
        for i in range(self.wm_config.train_steps_per_iter):
            p = self.wm_dataset_size / dataset_size_sum
            batch_idx = np.random.choice(range(self.env.num_envs), self.wm_config.batch_size, replace=True,
                                         p=p)
            batch_length = min(int(self.wm_dataset_size[batch_idx].min()), self.wm_config.batch_length)
            if (batch_length <= 1):
                continue  # an error occur about the predict loss if batch_length < 1
            batch_end_idx = [
                np.random.randint(batch_length, int(self.wm_dataset_size[idx]) + 1)
                for idx in batch_idx
            ]
            batch_data = {}
            for k, v in self.wm_dataset.items():
                if (k == "forward_height_map"):
                    continue
                value = []
                for idx, end_idx in zip(batch_idx, batch_end_idx):
                    if (k == "image"):
                        idx_in_buffer = np.where(self.env.depth_index == idx)[0]
                        if (len(idx_in_buffer) == 0):
                            # not in the buffer, use the predicted ones
                            tmp_forward_heightmap = self.wm_dataset["forward_height_map"][idx,
                                                    end_idx - batch_length: end_idx]
                            tmp_prop = self.wm_dataset["prop"][idx, end_idx - batch_length: end_idx]
                            pred_depth_image = self.depth_predictor(tmp_forward_heightmap, tmp_prop)
                            value.append(pred_depth_image)
                        else:
                            value.append(v[idx_in_buffer[0], end_idx - batch_length: end_idx])
                    else:
                        value.append(v[idx, end_idx - batch_length: end_idx])
                value = torch.stack(value)
                batch_data[k] = value
            is_first = torch.zeros((self.wm_config.batch_size, batch_length))
            is_first[:, 0] = 1
            batch_data["is_first"] = is_first
            post, context, mets = self._world_model._train(batch_data)
        wm_metrics.update(mets)
        return wm_metrics

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            episode_keys = sorted({key for ep_info in locs['ep_infos'] for key in ep_info})
            for key in episode_keys:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if key not in ep_info:
                        continue
                    # handle scalar and zero dimensional tensor infos
                    value = ep_info[key]
                    if not isinstance(value, torch.Tensor):
                        value = torch.Tensor([value])
                    if len(value.shape) == 0:
                        value = value.unsqueeze(0)
                    infotensor = torch.cat((infotensor, value.to(self.device)))
                if len(infotensor) == 0:
                    continue
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs * self.gpu_world_size / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/vel_predict', locs['mean_vel_predict_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/AMP_mean_policy_pred', locs['mean_policy_pred'], locs['it'])
        self.writer.add_scalar('Loss/AMP_mean_expert_pred', locs['mean_expert_pred'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Vel predict loss:':>{pad}} {locs['mean_vel_predict_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'world_model_dict': self._world_model.state_dict(),
            'wm_optimizer_state_dict': self._world_model._model_opt._opt.state_dict(),
            'depth_predictor': self.depth_predictor.state_dict(),
            # 'discriminator_state_dict': self.alg.discriminator.state_dict(),
            # 'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True, load_wm_optimizer = False):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'], strict=False)
        self._world_model.load_state_dict(loaded_dict['world_model_dict'], strict=False)
        if loaded_dict.get('depth_predictor') is not None:
            self.depth_predictor.load_state_dict(loaded_dict['depth_predictor'], strict=False)
        if(load_wm_optimizer):
            self._world_model._model_opt._opt.load_state_dict(loaded_dict['wm_optimizer_state_dict'])
        # self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'], strict=False)
        # self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
