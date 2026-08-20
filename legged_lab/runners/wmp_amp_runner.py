# -*- coding: utf-8 -*-
"""WMP + AMP-PPO runner for rsl_rl 5。"""

from __future__ import annotations

import glob
import os
import resource
import statistics
import time

import torch
from rsl_rl.storage import RolloutStorage
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.utils import check_nan, resolve_callable, resolve_obs_groups
from rsl_rl.utils.logger import Logger
from tensordict import TensorDict

from legged_lab.algorithms import WMPAMPPPO
from legged_lab.amp import AMPDiscriminator, AMPLoader, Normalizer
from legged_lab.world_models.wmp import WMPTrainingController, WorldModel, make_default_wmp_config

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional.
    wandb = None


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
        self.logger = Logger(log_dir, self.cfg, self.env.cfg, self.env.num_envs, False, 1, 0, self.device)

    def _feature_dim_from_cfg(self):
        wmp_cfg = self.cfg.get("wmp", {})
        return 1536 if wmp_cfg.get("feature_type", "deter") == "full" else 512

    def _build_world_model(self):
        wmp_cfg = self.cfg.get("wmp", {})
        self.wm_device = wmp_cfg.get("device", self.device)
        prop_dim = self.env.get_wmp_proprioception().shape[-1]
        update_interval = int(wmp_cfg.get("update_interval", wmp_cfg.get("wmp_update_interval", 5)))
        wm_config = make_default_wmp_config(device=self.wm_device, num_actions=self.env.num_actions * update_interval)
        wm_config.env_num_actions = self.env.num_actions
        wm_config.action_dim = self.env.num_actions * update_interval
        wm_config.update_interval = update_interval
        wm_config.prop_dim = prop_dim
        for key, value in wmp_cfg.items():
            if key == "depth_predictor" and isinstance(value, dict):
                for dp_key, dp_value in value.items():
                    if hasattr(wm_config.depth_predictor, dp_key):
                        setattr(wm_config.depth_predictor, dp_key, dp_value)
                continue
            if key == "wmp_update_interval":
                key = "update_interval"
            if hasattr(wm_config, key):
                setattr(wm_config, key, value)
        self.wm_config = wm_config
        self.world_model = WorldModel(wm_config, {"prop": (prop_dim,), "image": (64, 64, 1)}, use_camera=True).to(self.wm_device)
        self.wm_feature_dim = self.world_model.feature_dim if wm_config.feature_type == "full" else self.world_model.deter_dim
        self.wmp_controller = WMPTrainingController(self.env, wm_config, self.world_model)
        print(
            f"[INFO] WMP prop_dim={prop_dim}, action_dim={wm_config.action_dim}, "
            f"update_interval={wm_config.update_interval}, feature_dim={self.wm_feature_dim}, "
            f"camera_envs={int(self.wmp_controller.camera_env_ids.numel())}/{self.env.num_envs}, "
            f"replay_device={wm_config.replay_device}, device={self.wm_device}"
        )

    def _build_algorithm(self, obs):
        cfg = self.cfg
        cfg["algorithm"]["class_name"] = "legged_lab.algorithms.wmp_amp_ppo:WMPAMPPPO"
        alg_class: type[WMPAMPPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))
        # IsaacLab's adapter config includes this construction-only field, but
        # the locally installed rsl_rl PPO constructor does not accept it.
        cfg["algorithm"].pop("share_cnn_encoders", None)
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
        canonical_dim = int(amp_cfg.get("canonical_obs_dim", 30))
        retarget_adapter = self._make_amp_retarget_adapter(amp_cfg)
        amp_data = AMPLoader(
            self.device,
            time_between_frames=self.env.step_dt,
            motion_files=motion_files,
            retarget_adapter=retarget_adapter,
            preload_transitions=True,
            num_preload_transitions=amp_cfg.get("num_preload_transitions", 100000),
        )
        if amp_data.observation_dim != canonical_dim:
            raise ValueError(f"AMP expert dim={amp_data.observation_dim}, expected canonical_obs_dim={canonical_dim}.")
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            amp_cfg.get("reward_coef", 2.0),
            amp_cfg.get("discriminator_hidden_dims", [1024, 512]),
            self.device,
            amp_cfg.get("task_reward_lerp", 0.0),
        ).to(self.device)
        normalizer = Normalizer(amp_data.observation_dim, device=self.device)
        if amp_cfg.get("preload_normalizer", True):
            expert_state, expert_next_state = amp_data.get_preloaded_transitions()
            normalizer.update(expert_state)
            normalizer.update(expert_next_state)
        self.alg.attach_amp(
            discriminator,
            amp_data,
            normalizer,
            amp_cfg.get("replay_buffer_size", 100000),
            amp_cfg.get("grad_penalty_coef", 1.0),
        )
        min_std = self._make_min_action_std(amp_cfg.get("min_normalized_std"))
        if min_std is not None and hasattr(self.alg, "set_min_action_std"):
            self.alg.set_min_action_std(min_std)
            print(f"[INFO] WMP-AMP min action std clamp: {min_std.detach().cpu().tolist()}")
        retarget_name = retarget_adapter.__class__.__name__
        print(
            f"[INFO] WMP-AMP enabled: expert_dim={amp_data.observation_dim}, "
            f"retarget={retarget_name}, preload_normalizer={amp_cfg.get('preload_normalizer', True)}, "
            f"reward_coef={amp_cfg.get('reward_coef', 2.0)}, task_reward_lerp={amp_cfg.get('task_reward_lerp', 0.0)}"
        )
        self._log_amp_joint_stats(amp_data)

    def _make_amp_retarget_adapter(self, amp_cfg: dict):
        retarget_cfg = amp_cfg.get("retarget_adapter", {}) or {}
        retarget_class_path = retarget_cfg.get("class_path", "legged_lab.amp.retarget:NoOpRetargetAdapter")
        retarget_kwargs = {k: v for k, v in retarget_cfg.items() if k != "class_path"}
        if retarget_kwargs.get("target_joint_order") == "env":
            retarget_kwargs["target_joint_order"] = list(self.env.robot.joint_names)
        return resolve_callable(retarget_class_path)(
            canonical_obs_dim=int(amp_cfg.get("canonical_obs_dim", 30)),
            **retarget_kwargs,
        )

    def _make_min_action_std(self, min_normalized_std) -> torch.Tensor | None:
        if min_normalized_std is None:
            return None
        min_normalized_std = torch.as_tensor(min_normalized_std, device=self.device, dtype=torch.float32)
        if min_normalized_std.numel() != self.env.num_actions:
            raise ValueError(
                "amp.min_normalized_std must have one value per action: "
                f"got {min_normalized_std.numel()}, expected {self.env.num_actions}."
            )
        limits = getattr(self.env.robot.data, "soft_joint_pos_limits", None)
        if limits is None:
            limits = getattr(self.env.robot.data, "joint_pos_limits", None)
        if limits is None:
            raise RuntimeError("Cannot compute WMP min action std: robot joint position limits are unavailable.")
        limits = limits[0].to(self.device)
        action_range = limits[:, 1] - limits[:, 0]
        if action_range.numel() != self.env.num_actions:
            raise RuntimeError(
                "Cannot compute WMP min action std: joint limit dim does not match action dim, "
                f"limits={action_range.numel()}, actions={self.env.num_actions}."
            )
        # 原版公式: min_std = min_normalized_std * (dof_upper_limit - dof_lower_limit)。
        return min_normalized_std * action_range

    def _log_amp_joint_stats(self, amp_data: AMPLoader):
        if not hasattr(amp_data, "preloaded_s"):
            return
        expert_joint_pos = amp_data.preloaded_s[:, :12].detach()
        policy_joint_pos = self.env.get_amp_observations()[:, :12].detach().to(expert_joint_pos.device)
        expert_mean = expert_joint_pos.mean(dim=0)
        expert_min = expert_joint_pos.amin(dim=0)
        expert_max = expert_joint_pos.amax(dim=0)
        policy_mean = policy_joint_pos.mean(dim=0)
        joint_names = list(getattr(self.env, "amp_joint_names", self.env.robot.joint_names))
        print("[INFO] WMP-AMP joint stats after expert retarget (expert_mean[min,max] vs current_env_mean):")
        for idx, joint_name in enumerate(joint_names[:12]):
            print(
                f"[INFO]   {joint_name}: "
                f"{expert_mean[idx].item():+.3f}[{expert_min[idx].item():+.3f},{expert_max[idx].item():+.3f}] "
                f"vs {policy_mean[idx].item():+.3f}"
            )

    def _augment_obs(self, obs: TensorDict, wm_feature: torch.Tensor):
        obs = obs.to(self.device)
        obs["wmp"] = wm_feature.to(self.device)
        return obs

    def _read_wm_obs(self, is_first):
        del is_first
        wm_obs, _ = self.wmp_controller.observe_before_policy()
        return wm_obs

    def _wm_feature(self, latent):
        if self.wm_config.feature_type == "full":
            return self.world_model.dynamics.get_feat(latent)
        return self.world_model.dynamics.get_deter_feat(latent)

    def _train_world_model(self):
        return self.wmp_controller.train_if_ready(self.current_learning_iteration, self.total_env_steps)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        self.alg.train_mode()
        self.logger.init_logging_writer()
        self.total_env_steps = self.current_learning_iteration * self.cfg["num_steps_per_env"] * self.env.num_envs
        wm_feature = torch.zeros(self.env.num_envs, self.wm_feature_dim, device=self.device)
        obs = self._augment_obs(self.env.get_observations().to(self.device), wm_feature)
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        for it in range(start_it, total_it):
            curriculum_logs = {}
            if hasattr(self.env, "update_training_curriculum"):
                curriculum_logs = self.env.update_training_curriculum(it)
            elif hasattr(self.env, "update_reward_curriculum"):
                curriculum_logs = self.env.update_reward_curriculum(it)
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    wm_obs, wm_feature = self.wmp_controller.observe_before_policy()
                    wm_feature = wm_feature.to(self.device)
                    obs = self._augment_obs(obs, wm_feature)
                    amp_obs = self.env.get_amp_observations().to(self.device)
                    actions = self.alg.act(obs, amp_obs=amp_obs)
                    next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    next_amp_obs = self.env.get_amp_observations().to(self.device)
                    if self.cfg.get("check_for_nan", True):
                        check_nan(next_obs, rewards, dones)
                    self.wmp_controller.after_env_step(actions, rewards, dones, wm_obs)
                    self.total_env_steps += self.env.num_envs
                    next_obs = self._augment_obs(next_obs.to(self.device), wm_feature)
                    task_rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    reset_env_ids = extras.get("reset_env_ids")
                    terminal_amp_states = extras.get("terminal_amp_states")
                    next_amp_obs_with_term = next_amp_obs.clone()
                    if reset_env_ids is not None and terminal_amp_states is not None and len(reset_env_ids) > 0:
                        next_amp_obs_with_term[reset_env_ids.to(self.device)] = terminal_amp_states.to(self.device)
                    amp_total_rewards, _, amp_style_rewards = self.alg.discriminator.predict_amp_reward(
                        amp_obs,
                        next_amp_obs_with_term,
                        task_rewards,
                        normalizer=self.alg.amp_normalizer,
                        return_details=True,
                    )
                    rewards = amp_total_rewards
                    self.alg.process_env_step(
                        next_obs,
                        rewards,
                        dones,
                        extras,
                        next_amp_obs=next_amp_obs_with_term,
                        task_rewards=task_rewards,
                        amp_rewards=amp_style_rewards,
                    )
                    self.logger.process_env_step(rewards, dones, extras)
                    obs = next_obs
                collect_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(obs)

            update_start = time.time()
            loss_dict = self.alg.update()
            loss_dict["time/ppo_update"] = time.time() - update_start
            loss_dict.update(curriculum_logs)
            loss_dict.update(self.wmp_controller.train_if_ready(it, self.total_env_steps))
            loss_dict.update(self.wmp_controller.replay_stats())
            loss_dict.update(self._resource_stats())
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
            self._log_wandb_history(it, collect_time, learn_time, loss_dict)
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None):
        saved = self.alg.save()
        saved["world_model_state_dict"] = self.world_model.state_dict()
        saved["world_model_optimizer_state_dict"] = self.world_model.model_opt.state_dict()
        saved.update(self.wmp_controller.state_dict())
        saved["iter"] = self.current_learning_iteration
        saved["infos"] = infos
        torch.save(saved, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def _resource_stats(self) -> dict[str, float]:
        stats = {"cpu_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)}
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            device = torch.device(self.device)
            stats["gpu_alloc_gb"] = torch.cuda.memory_allocated(device) / (1024.0**3)
            stats["gpu_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024.0**3)
        return stats

    def _log_wandb_history(self, it: int, collect_time: float, learn_time: float, loss_dict: dict):
        if self.cfg.get("logger") != "wandb" or wandb is None or wandb.run is None:
            return
        collection_size = self.cfg["num_steps_per_env"] * self.env.num_envs * self.gpu_world_size
        payload = {
            "Perf/total_fps_batched": int(collection_size / max(collect_time + learn_time, 1.0e-8)),
            "Perf/collection_time_batched": collect_time,
            "Perf/learning_time_batched": learn_time,
            "Policy/mean_std_batched": self.alg.get_policy().output_std.mean().item(),
        }
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()
            payload[f"Loss/{key}_batched"] = float(value)
        if len(getattr(self.logger, "rewbuffer", [])) > 0:
            payload["Train/mean_reward_batched"] = statistics.mean(self.logger.rewbuffer)
            payload["Train/mean_episode_length_batched"] = statistics.mean(self.logger.lenbuffer)
        wandb.log(payload, step=it, commit=True)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
        loaded = torch.load(path, weights_only=False, map_location=map_location)
        if self.alg.load(loaded, load_cfg, strict):
            self.current_learning_iteration = loaded["iter"]
        if "world_model_state_dict" in loaded:
            self.world_model.load_state_dict(loaded["world_model_state_dict"], strict=strict)
        if load_cfg is None or load_cfg.get("optimizer", True):
            if "world_model_optimizer_state_dict" in loaded:
                self.world_model.model_opt.load_state_dict(loaded["world_model_optimizer_state_dict"])
        self.wmp_controller.load_state_dict(loaded, strict=strict, load_optimizer=(load_cfg is None or load_cfg.get("optimizer", True)))
        return loaded.get("infos")

    def get_inference_policy(self, device: str | None = None):
        self.alg.eval_mode()
        return self.alg.get_policy().to(device)


class WMPRunner(WMPAMPRunner):
    """WMP-PPO runner for robots without compatible AMP motion data."""

    def _build_algorithm(self, obs):
        cfg = self.cfg
        cfg["algorithm"]["class_name"] = "rsl_rl.algorithms:PPO"
        alg_class = resolve_callable(cfg["algorithm"]["class_name"])
        return alg_class.construct_algorithm(obs, self.env, cfg, self.device)

    def _build_amp(self):
        print("[INFO] WMP-PPO enabled without AMP expert motions.")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        self.alg.train_mode()
        self.logger.init_logging_writer()
        self.total_env_steps = self.current_learning_iteration * self.cfg["num_steps_per_env"] * self.env.num_envs
        wm_feature = torch.zeros(self.env.num_envs, self.wm_feature_dim, device=self.device)
        obs = self._augment_obs(self.env.get_observations().to(self.device), wm_feature)
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        for it in range(start_it, total_it):
            curriculum_logs = {}
            if hasattr(self.env, "update_training_curriculum"):
                curriculum_logs = self.env.update_training_curriculum(it)
            elif hasattr(self.env, "update_reward_curriculum"):
                curriculum_logs = self.env.update_reward_curriculum(it)
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    wm_obs, wm_feature = self.wmp_controller.observe_before_policy()
                    obs = self._augment_obs(obs, wm_feature.to(self.device))
                    actions = self.alg.act(obs)
                    next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(next_obs, rewards, dones)
                    self.wmp_controller.after_env_step(actions, rewards, dones, wm_obs)
                    self.total_env_steps += self.env.num_envs
                    next_obs = self._augment_obs(next_obs.to(self.device), wm_feature)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(next_obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras)
                    obs = next_obs
                collect_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(obs)

            update_start = time.time()
            loss_dict = self.alg.update()
            loss_dict["time/ppo_update"] = time.time() - update_start
            loss_dict.update(curriculum_logs)
            loss_dict.update(self.wmp_controller.train_if_ready(it, self.total_env_steps))
            loss_dict.update(self.wmp_controller.replay_stats())
            loss_dict.update(self._resource_stats())
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
            self._log_wandb_history(it, collect_time, learn_time, loss_dict)
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()
