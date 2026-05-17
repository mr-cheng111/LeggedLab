# -*- coding: utf-8 -*-
"""AMP-PPO runner that keeps rsl_rl 5 core untouched."""

from __future__ import annotations

import glob
import os
import time

import torch
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger

from legged_lab.algorithms import AMPPPO
from legged_lab.amp import AMPDiscriminator, AMPMotionDataset, Normalizer


class AMPPPORunner:
    alg: AMPPPO

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self._configure_multi_gpu()

        obs = self.env.get_observations()
        self.cfg["algorithm"]["class_name"] = "legged_lab.algorithms.amp_ppo:AMPPPO"
        self.alg = AMPPPO.construct_algorithm(obs, self.env, self.cfg, self.device)
        self._build_amp()

        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )
        self.current_learning_iteration = 0

    def _build_amp(self) -> None:
        amp_cfg = self.cfg.get("amp", {})
        motion_files = amp_cfg.get("motion_files") or sorted(glob.glob("datasets/wmp_mocap_motions/*.txt"))
        retarget_cfg = amp_cfg.get("retarget_adapter", {}) or {}
        retarget_class_path = retarget_cfg.get("class_path", "legged_lab.amp.retarget:NoOpRetargetAdapter")
        retarget_kwargs = {k: v for k, v in retarget_cfg.items() if k != "class_path"}
        retarget_adapter = resolve_callable(retarget_class_path)(
            canonical_obs_dim=int(amp_cfg.get("canonical_obs_dim", 30)),
            **retarget_kwargs,
        )
        amp_data = AMPMotionDataset(
            self.device,
            time_between_frames=self.env.step_dt,
            motion_files=motion_files,
            retarget_adapter=retarget_adapter,
            preload_transitions=True,
            num_preload_transitions=amp_cfg.get("num_preload_transitions", 100000),
        )
        canonical_dim = int(amp_cfg.get("canonical_obs_dim", 30))
        if amp_data.observation_dim != canonical_dim:
            raise ValueError(f"AMP expert dim={amp_data.observation_dim}, expected canonical_obs_dim={canonical_dim}.")
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            amp_cfg.get("reward_coef", 0.01),
            amp_cfg.get("discriminator_hidden_dims", [1024, 512]),
            self.device,
            task_reward_lerp=0.0,
        ).to(self.device)
        normalizer = Normalizer(amp_data.observation_dim, device=self.device)
        self.amp_reward_weight = float(amp_cfg.get("amp_reward_weight", 1.0))
        self.task_reward_weight = float(amp_cfg.get("task_reward_weight", 1.0))
        self.alg.attach_amp(
            discriminator,
            amp_data,
            normalizer,
            amp_cfg.get("replay_buffer_size", 1000000),
            amp_cfg.get("grad_penalty_coef", 1.0),
        )
        print(
            f"[INFO] AMP-PPO enabled: expert_dim={amp_data.observation_dim}, "
            f"task_weight={self.task_reward_weight}, amp_weight={self.amp_reward_weight}"
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    amp_obs = self.env.get_amp_observations().to(self.device)
                    actions = self.alg.act(obs, amp_obs=amp_obs)
                    next_obs, task_rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    next_amp_obs = self.env.get_amp_observations().to(self.device)
                    if self.cfg.get("check_for_nan", True):
                        check_nan(next_obs, task_rewards, dones)
                    next_obs = next_obs.to(self.device)
                    task_rewards = task_rewards.to(self.device)
                    dones = dones.to(self.device)

                    reset_env_ids = extras.get("reset_env_ids")
                    terminal_amp_states = extras.get("terminal_amp_states")
                    next_amp_obs_with_term = next_amp_obs.clone()
                    if reset_env_ids is not None and terminal_amp_states is not None and len(reset_env_ids) > 0:
                        next_amp_obs_with_term[reset_env_ids.to(self.device)] = terminal_amp_states.to(self.device)

                    amp_rewards, _ = self.alg.discriminator.predict_amp_reward(
                        amp_obs,
                        next_amp_obs_with_term,
                        task_rewards,
                        normalizer=self.alg.amp_normalizer,
                    )
                    rewards = self.task_reward_weight * task_rewards + self.amp_reward_weight * amp_rewards
                    self.alg.process_env_step(
                        next_obs,
                        rewards,
                        dones,
                        extras,
                        next_amp_obs=next_amp_obs_with_term,
                        task_rewards=task_rewards,
                        amp_rewards=amp_rewards,
                    )
                    self.logger.process_env_step(rewards, dones, extras, None)
                    obs = next_obs

                collect_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            learn_time = time.time() - start
            self.current_learning_iteration = it
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        self.alg.eval_mode()
        return self.alg.get_policy().to(device)

    def _configure_multi_gpu(self) -> None:
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
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'.")
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)
