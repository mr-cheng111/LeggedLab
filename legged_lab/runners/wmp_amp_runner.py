# -*- coding: utf-8 -*-
"""WMP + AMP-PPO runner for rsl_rl 5。"""

from __future__ import annotations

import glob
import os
import time

import torch
from rsl_rl.storage import RolloutStorage
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.utils import check_nan, resolve_callable, resolve_obs_groups
from rsl_rl.utils.logger import Logger
from tensordict import TensorDict

from legged_lab.algorithms import WMPAMPPPO
from legged_lab.amp import AMPDiscriminator, AMPLoader, Normalizer
from legged_lab.world_models.wmp import WMPReplayBuffer, WorldModel, depth_to_wmp_image, make_default_wmp_config


class WMPAMPRunner:
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.log_dir = log_dir
        self.current_learning_iteration = 0
        self.is_distributed = False
        self.gpu_world_size = 1
        self.gpu_global_rank = 0
        self.cfg["multi_gpu"] = None

        self._build_world_model()
        obs = self._augment_obs(
            self.env.get_observations().to(self.device),
            torch.zeros(self.env.num_envs, self.wm_feature_dim, device=self.device),
        )
        self.alg = self._build_algorithm(obs)
        self._build_amp()
        self.wm_replay = WMPReplayBuffer(self.cfg["wmp"].get("replay_capacity", 50000), self.wm_device)
        self.logger = Logger(log_dir, self.cfg, self.env.cfg, self.env.num_envs, False, 1, 0, self.device)

    def _feature_dim_from_cfg(self):
        wmp_cfg = self.cfg.get("wmp", {})
        return 1536 if wmp_cfg.get("feature_type", "deter") == "full" else 512

    def _build_world_model(self):
        wmp_cfg = self.cfg.get("wmp", {})
        self.wm_device = wmp_cfg.get("device", self.device)
        prop_dim = self.env.get_wmp_proprioception().shape[-1]
        wm_config = make_default_wmp_config(device=self.wm_device, num_actions=self.env.num_actions)
        for key, value in wmp_cfg.items():
            if hasattr(wm_config, key):
                setattr(wm_config, key, value)
        self.wm_config = wm_config
        self.world_model = WorldModel(wm_config, {"prop": (prop_dim,), "image": (64, 64, 1)}, use_camera=True).to(self.wm_device)
        self.wm_feature_dim = self.world_model.feature_dim if wm_config.feature_type == "full" else self.world_model.deter_dim
        print(f"[INFO] WMP prop_dim={prop_dim}, feature_dim={self.wm_feature_dim}, device={self.wm_device}")

    def _build_algorithm(self, obs):
        cfg = self.cfg
        cfg["algorithm"]["class_name"] = "legged_lab.algorithms.wmp_amp_ppo:WMPAMPPPO"
        alg_class: type[WMPAMPPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], self.env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], self.env)
        actor = actor_class(obs, cfg["obs_groups"], "actor", self.env.num_actions, **cfg["actor"]).to(self.device)
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(self.device)
        storage = RolloutStorage("rl", self.env.num_envs, cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device)
        return alg_class(actor, critic, storage, device=self.device, **cfg["algorithm"], multi_gpu_cfg=None)

    def _build_amp(self):
        amp_cfg = self.cfg.get("amp", {})
        motion_files = amp_cfg.get("motion_files") or sorted(glob.glob("datasets/wmp_mocap_motions/*.txt"))
        amp_data = AMPLoader(
            self.device,
            time_between_frames=self.env.step_dt,
            motion_files=motion_files,
            preload_transitions=True,
            num_preload_transitions=amp_cfg.get("num_preload_transitions", 100000),
        )
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            amp_cfg.get("reward_coef", 2.0),
            amp_cfg.get("discriminator_hidden_dims", [1024, 512]),
            self.device,
            amp_cfg.get("task_reward_lerp", 0.0),
        ).to(self.device)
        normalizer = Normalizer(amp_data.observation_dim, device=self.device)
        self.alg.attach_amp(
            discriminator,
            amp_data,
            normalizer,
            amp_cfg.get("replay_buffer_size", 100000),
            amp_cfg.get("grad_penalty_coef", 1.0),
        )

    def _augment_obs(self, obs: TensorDict, wm_feature: torch.Tensor):
        obs = obs.to(self.device)
        obs["wmp"] = wm_feature.to(self.device)
        return obs

    def _read_wm_obs(self, is_first):
        depth = self.env.get_depth_observations()
        image = depth_to_wmp_image(
            depth,
            near=self.env.cfg.scene.gemini2_camera.depth_near,
            far=self.env.cfg.scene.gemini2_camera.depth_far,
        ).to(self.wm_device)
        prop = self.env.get_wmp_proprioception().to(self.wm_device)
        return {"prop": prop, "image": image, "is_first": is_first.to(self.wm_device)}

    def _wm_feature(self, latent):
        if self.wm_config.feature_type == "full":
            return self.world_model.dynamics.get_feat(latent)
        return self.world_model.dynamics.get_deter_feat(latent)

    def _train_world_model(self):
        metrics = {}
        if not self.wm_replay.can_sample(self.wm_config.batch_size, self.wm_config.batch_length):
            return metrics
        for _ in range(self.wm_config.train_steps_per_iter):
            batch = self.wm_replay.sample(self.wm_config.batch_size, self.wm_config.batch_length)
            _, _, metrics = self.world_model._train(batch)
        return {f"wm_{k}": v for k, v in metrics.items()}

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        self.alg.train_mode()
        self.logger.init_logging_writer()
        wm_latent = None
        wm_action = torch.zeros(self.env.num_envs, self.env.num_actions, device=self.wm_device)
        is_first = torch.ones(self.env.num_envs, device=self.wm_device)
        wm_feature = torch.zeros(self.env.num_envs, self.wm_feature_dim, device=self.device)
        obs = self._augment_obs(self.env.get_observations().to(self.device), wm_feature)
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    wm_obs = self._read_wm_obs(is_first)
                    wm_embed = self.world_model.encoder(wm_obs)
                    wm_latent, _ = self.world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_obs["is_first"])
                    wm_feature = self._wm_feature(wm_latent).to(self.device)
                    obs = self._augment_obs(obs, wm_feature)
                    amp_obs = self.env.get_amp_observations().to(self.device)
                    actions = self.alg.act(obs, amp_obs=amp_obs)
                    next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    next_amp_obs = self.env.get_amp_observations().to(self.device)
                    if self.cfg.get("check_for_nan", True):
                        check_nan(next_obs, rewards, dones)
                    self.wm_replay.add(
                        {
                            "prop": wm_obs["prop"],
                            "image": wm_obs["image"],
                            "action": actions.to(self.wm_device),
                            "reward": rewards.to(self.wm_device).unsqueeze(-1),
                            "is_first": is_first.to(self.wm_device),
                            "is_terminal": dones.to(self.wm_device).float(),
                        }
                    )
                    is_first = dones.to(self.wm_device).float()
                    wm_action = actions.to(self.wm_device)
                    next_obs = self._augment_obs(next_obs.to(self.device), wm_feature)
                    task_rewards = rewards.to(self.device)
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
                    self.alg.process_env_step(
                        next_obs,
                        amp_rewards,
                        dones,
                        extras,
                        next_amp_obs=next_amp_obs_with_term,
                        task_rewards=task_rewards,
                    )
                    self.logger.process_env_step(amp_rewards, dones, extras)
                    obs = next_obs
                collect_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            if it * self.cfg["num_steps_per_env"] * self.env.num_envs >= self.wm_config.train_start_steps:
                loss_dict.update(self._train_world_model())
            learn_time = time.time() - start
            self.current_learning_iteration = it
            self.logger.log(
                it,
                start_it,
                total_it,
                collect_time,
                learn_time,
                loss_dict,
                self.alg.learning_rate,
                self.alg.get_policy().output_std,
                None,
            )
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None):
        saved = self.alg.save()
        saved["world_model_state_dict"] = self.world_model.state_dict()
        saved["world_model_optimizer_state_dict"] = self.world_model.model_opt.state_dict()
        saved["iter"] = self.current_learning_iteration
        saved["infos"] = infos
        torch.save(saved, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
        loaded = torch.load(path, weights_only=False, map_location=map_location)
        if self.alg.load(loaded, load_cfg, strict):
            self.current_learning_iteration = loaded["iter"]
        if "world_model_state_dict" in loaded:
            self.world_model.load_state_dict(loaded["world_model_state_dict"], strict=strict)
        if load_cfg is None or load_cfg.get("optimizer", True):
            if "world_model_optimizer_state_dict" in loaded:
                self.world_model.model_opt.load_state_dict(loaded["world_model_optimizer_state_dict"])
        return loaded.get("infos")

    def get_inference_policy(self, device: str | None = None):
        self.alg.eval_mode()
        return self.alg.get_policy().to(device)
