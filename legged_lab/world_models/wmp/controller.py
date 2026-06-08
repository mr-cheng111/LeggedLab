# -*- coding: utf-8 -*-
"""WMP 训练控制器。

该层承接 bytedance/WMP runner 中与 world model 相关的逻辑，但保持在
LeggedLab 本地模块中，避免修改 rsl_rl5：

1. 每 `update_interval` 个环境步更新一次 RSSM。
2. 将最近 k 步动作展平为 world model action，保证时间尺度一致：
       a_wm[t] = [a[t-k+1], ..., a[t]]
3. 使用真实 depth 训练 DepthPredictor，再用预测/真实 depth 训练 RSSM。
"""

from __future__ import annotations

import time

import torch
from torch import nn

from .depth_predictor import DepthPredictor
from .models import WMPFixedEpisodeReplayBuffer, WorldModel
from .preprocess import depth_to_wmp_image


class WMPTrainingController:
    def __init__(self, env, config, world_model: WorldModel):
        self.env = env
        self.config = config
        self.world_model = world_model
        self.device = torch.device(config.device)
        self.update_interval = int(config.update_interval)
        self.action_dim = int(config.action_dim or config.num_actions * self.update_interval)
        self.env_action_dim = int(config.env_num_actions or config.num_actions)
        self.feature_dim = world_model.feature_dim if config.feature_type == "full" else world_model.deter_dim
        self.depth_predictor = DepthPredictor(
            forward_heightmap_dim=config.forward_heightmap_dim,
            prop_dim=config.prop_dim,
            depth_image_dims=config.depth_predictor.depth_image_dims,
            encoder_hidden_dims=config.depth_predictor.encoder_hidden_dims,
            depth=config.depth_predictor.depth,
            act=config.depth_predictor.act,
            norm=config.depth_predictor.norm,
            kernel_size=config.depth_predictor.kernel_size,
            minres=config.depth_predictor.minres,
            outscale=config.depth_predictor.outscale,
            cnn_sigmoid=config.depth_predictor.cnn_sigmoid,
        ).to(self.device)
        self.depth_predictor_opt = torch.optim.Adam(
            self.depth_predictor.parameters(),
            lr=config.depth_predictor.lr,
            weight_decay=config.depth_predictor.weight_decay,
        )
        self.depth_index = self._select_camera_env_ids()
        self.camera_env_ids = self.depth_index
        self.depth_index_without_crawl_tilt = self._select_depth_index_without_crawl_tilt()
        self.depth_index_inverse = self._select_depth_index_inverse()
        self.camera_env_mask = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        if self.camera_env_ids.numel() > 0:
            self.camera_env_mask[self.camera_env_ids] = True
        self.depth_predictor_excluded_env_ids = self._tilt_crawl_env_ids()
        max_episode_steps = int(getattr(self.env, "max_episode_length", 1000) // self.update_interval + 3)
        self.replay = WMPFixedEpisodeReplayBuffer(
            self.env.num_envs,
            max_episode_steps,
            self.depth_index,
            config.replay_device,
            depth_index_without_crawl_tilt=self.depth_index_without_crawl_tilt,
            depth_index_inverse=self.depth_index_inverse,
            depth_predictor_excluded_env_ids=self.depth_predictor_excluded_env_ids,
        )
        self.reset_state()

    def reset_state(self):
        self.latent = None
        self.prior_latent = None
        self.is_first = torch.ones(self.env.num_envs, device=self.device)
        self.pending_obs: dict[str, torch.Tensor] | None = None
        self.step_counter = 0
        self.action_history = torch.zeros(
            self.env.num_envs,
            self.update_interval,
            self.env_action_dim,
            device=self.device,
        )
        self.reward_accumulator = torch.zeros(self.env.num_envs, device=self.device)
        self.feature = torch.zeros(self.env.num_envs, self.feature_dim, device=self.device)

    def feature_for_policy(self) -> torch.Tensor:
        return self.feature

    def observe_before_policy(self) -> tuple[dict[str, torch.Tensor] | None, torch.Tensor]:
        """在策略动作前按 WMP 时间尺度更新 RSSM feature。

        原版 WMP 在 policy step 前消费上一帧 WMP obs；真实相机 depth 则在
        env.step() 后、`global_counter % update_interval == 0` 时才刷新。
        """
        if self.step_counter % self.update_interval != 0:
            return None, self.feature
        if self.pending_obs is None:
            self.pending_obs = self._make_initial_wm_obs()
        wm_obs = self.pending_obs
        with torch.inference_mode():
            embed = self.world_model.encoder(wm_obs)
            wm_action = self.action_history.reshape(self.env.num_envs, -1)
            self.latent, self.prior_latent = self.world_model.dynamics.obs_step(
                self.latent,
                wm_action,
                embed,
                wm_obs["is_first"],
            )
            self.feature = self._wm_feature(self.latent)
            self.is_first.zero_()
        return wm_obs, self.feature

    def predict_depth_image(self, use_prior: bool = True) -> torch.Tensor | None:
        """从 RSSM latent 解码 depth image。

        `use_prior=True` 对应 Dreamer/WMP 的一步预测：
            prior_t = RSSM(s_{t-1}, a_{t-1})
            image_pred_t = Decoder(prior_t)
        用它和同一 WMP tick 的真实 image_t 对比，能看出世界模型是否真的在预测。
        """
        latent = self.prior_latent if use_prior else self.latent
        if latent is None:
            return None
        with torch.inference_mode():
            feat = self.world_model.dynamics.get_feat(latent)
            pred = self.world_model.decode(feat).get("image")
            if pred is None:
                return None
            image = pred.mode() if callable(getattr(pred, "mode", None)) else pred.mean()
        return image

    def after_env_step(
        self,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        wm_obs: dict[str, torch.Tensor] | None = None,
    ):
        del wm_obs
        actions = actions.detach().to(self.device)
        rewards = rewards.detach().to(self.device)
        dones = dones.detach().to(self.device).bool()
        self.action_history = torch.cat((self.action_history[:, 1:], actions.unsqueeze(1)), dim=1)
        self.reward_accumulator += rewards

        if torch.any(dones):
            reset_ids = dones.nonzero(as_tuple=False).flatten()
            for env_id in reset_ids.tolist():
                self.replay.finish_episode(env_id)
            self.action_history[reset_ids] = 0.0
            self.reward_accumulator[reset_ids] = 0.0
            self.is_first[reset_ids] = 1.0

        next_step_counter = self.step_counter + 1
        if next_step_counter % self.update_interval == 0:
            next_wm_obs = self._read_wm_obs()
            wm_action = self.action_history.reshape(self.env.num_envs, -1)
            store_ids = (1.0 - self.is_first).nonzero(as_tuple=False).flatten()
            for env_id in store_ids.tolist():
                self.replay.add_step(
                    env_id,
                    {
                        "prop": next_wm_obs["prop"][env_id],
                        "image": next_wm_obs["image"][env_id],
                        "forward_height_map": next_wm_obs["forward_height_map"][env_id],
                        "action": wm_action[env_id],
                        "reward": self.reward_accumulator[env_id].view(1),
                        "is_first": next_wm_obs["is_first"][env_id].view(1),
                        "is_terminal": dones[env_id].float().view(1),
                        "has_real_depth": next_wm_obs["has_real_depth"][env_id].view(1),
                    },
                )
            self.reward_accumulator.zero_()
            self.pending_obs = next_wm_obs

        self.step_counter = next_step_counter

    def train_if_ready(self, iteration: int, total_env_steps: int) -> dict[str, float]:
        del total_env_steps
        metrics = {}
        if getattr(self.replay, "num_steps", 0) <= self.config.train_start_steps:
            return metrics
        train_interval = max(1, int(getattr(self.config, "train_interval", 1)))
        if iteration % train_interval != 0:
            return metrics
        if self.config.use_depth_predictor and iteration % self.config.depth_predictor.training_interval == 0:
            start = time.time()
            depth_loss = self._train_depth_predictor()
            metrics["time/depth_predictor"] = time.time() - start
            if depth_loss is not None:
                metrics["wm_depth_predictor_loss"] = depth_loss
        if self.replay.can_sample(self.config.batch_size, self.config.batch_length):
            start = time.time()
            wm_metrics = self._train_world_model()
            metrics["time/world_model"] = time.time() - start
            metrics.update({f"wm_{key}": value for key, value in wm_metrics.items()})
        return metrics

    def replay_stats(self) -> dict[str, float]:
        stats = {
            "wm_replay_episodes": float(len(self.replay)),
            "wm_replay_steps": float(getattr(self.replay, "num_steps", 0)),
            "wm_depth_predictor_excluded_envs": float(self.depth_predictor_excluded_env_ids.numel()),
            "wm_depth_index_count": float(self.depth_index.numel()),
            "wm_depth_index_without_crawl_tilt_count": float(self.depth_index_without_crawl_tilt.numel()),
            "wm_tilt_crawl_depth_count": float(self.depth_index.numel() - self.depth_index_without_crawl_tilt.numel()),
            "wm_real_depth_ratio": float(self.depth_index.numel()) / max(float(self.env.num_envs), 1.0),
        }
        if hasattr(self.replay, "max_episode_steps"):
            stats["wm_replay_max_episode_steps"] = float(self.replay.max_episode_steps)
        return stats

    def state_dict(self) -> dict:
        return {
            "depth_predictor_state_dict": self.depth_predictor.state_dict(),
            "depth_predictor_optimizer_state_dict": self.depth_predictor_opt.state_dict(),
        }

    def load_state_dict(self, state: dict, strict: bool = True, load_optimizer: bool = True):
        if "depth_predictor_state_dict" in state:
            self.depth_predictor.load_state_dict(state["depth_predictor_state_dict"], strict=strict)
        if load_optimizer and "depth_predictor_optimizer_state_dict" in state:
            self.depth_predictor_opt.load_state_dict(state["depth_predictor_optimizer_state_dict"])

    def _read_wm_obs(self) -> dict[str, torch.Tensor]:
        # 调用点在 `after_env_step()` 中，且只在 env.step 完成后的 WMP tick 读取。
        # A1 WMP 配置里 camera.update_period 与 sim.render_interval 都等于
        # step_dt * update_interval，因此这里的 prop / forward_height_map / depth
        # 对应同一个完成后的机器人状态。
        prop = self.env.get_wmp_proprioception().to(self.device)
        forward_height_map = self.env.get_wmp_forward_height_map().to(self.device)
        image, has_real_depth = self._current_depth_image(prop, forward_height_map)
        return {
            "prop": prop,
            "image": image,
            "forward_height_map": forward_height_map,
            "is_first": self.is_first.clone(),
            "has_real_depth": has_real_depth,
        }

    def _make_initial_wm_obs(self) -> dict[str, torch.Tensor]:
        prop = self.env.get_wmp_proprioception().to(self.device)
        forward_height_map = self.env.get_wmp_forward_height_map().to(self.device)
        height, width = self.config.depth_predictor.depth_image_dims
        image = torch.zeros(self.env.num_envs, height, width, 1, device=self.device)
        return {
            "prop": prop,
            "image": image,
            "forward_height_map": forward_height_map,
            "is_first": self.is_first.clone(),
            "has_real_depth": torch.zeros(self.env.num_envs, device=self.device),
        }

    def _current_depth_image(self, prop: torch.Tensor, forward_height_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """生成 WMP image：先预测全 env，再用真实相机子集覆盖。

        这对应原版 WMP 的公式：
            image_all = DepthPredictor(forward_height_map, prop)
            image_all[depth_index] = real_depth
        """
        with torch.inference_mode():
            image = self.depth_predictor(forward_height_map, prop)
        has_real_depth = torch.zeros(self.env.num_envs, device=self.device)
        if hasattr(self.env, "get_wmp_depth_observations") and getattr(self.env, "wmp_depth_buffer", None) is not None:
            real_image = self.env.get_wmp_depth_observations().to(self.device)
            real_env_ids = self._real_depth_env_ids(real_image.shape[0])
            if real_env_ids.numel() > 0:
                image = image.clone()
                image[real_env_ids] = real_image[: real_env_ids.numel()]
                has_real_depth[real_env_ids] = 1.0
            return image, has_real_depth
        if hasattr(self.env, "get_depth_observations"):
            try:
                depth = self.env.get_depth_observations().to(self.device)
                real_image = depth_to_wmp_image(
                    depth,
                    near=self.env.cfg.scene.rgbd_camera.depth_near,
                    far=self.env.cfg.scene.rgbd_camera.depth_far,
                )
                real_env_ids = self._real_depth_env_ids(real_image.shape[0])
                if real_env_ids.numel() > 0:
                    image = image.clone()
                    if real_image.shape[0] == self.env.num_envs:
                        image[real_env_ids] = real_image[real_env_ids]
                    else:
                        image[real_env_ids] = real_image[: real_env_ids.numel()]
                    has_real_depth[real_env_ids] = 1.0
            except Exception:
                pass
        return image, has_real_depth

    def _real_depth_env_ids(self, depth_batch_size: int) -> torch.Tensor:
        if hasattr(self.env, "get_depth_index"):
            env_ids = self.env.get_depth_index().to(self.device).long()
            if depth_batch_size == self.env.num_envs:
                return env_ids[(env_ids >= 0) & (env_ids < self.env.num_envs)]
            return env_ids[: min(depth_batch_size, env_ids.numel())]
        if hasattr(self.env, "get_depth_camera_env_ids"):
            env_ids = self.env.get_depth_camera_env_ids().to(self.device).long()
            if depth_batch_size == self.env.num_envs:
                return env_ids[(env_ids >= 0) & (env_ids < self.env.num_envs)]
            return env_ids[: min(depth_batch_size, env_ids.numel())]
        if depth_batch_size == self.env.num_envs:
            return self.camera_env_ids
        return self.camera_env_ids[: min(depth_batch_size, self.camera_env_ids.numel())]

    def _select_camera_env_ids(self) -> torch.Tensor:
        if self.config.camera_env_ids is not None:
            ids = torch.as_tensor(self.config.camera_env_ids, device=self.device, dtype=torch.long)
            return self._ordered_unique_valid(ids)

        if hasattr(self.env, "get_depth_index"):
            ids = self.env.get_depth_index().to(self.device).long()
            if ids.numel() > 0 and ids.numel() <= self.env.num_envs:
                return self._ordered_unique_valid(ids)

        if hasattr(self.env, "get_depth_camera_env_ids"):
            ids = self.env.get_depth_camera_env_ids().to(self.device).long()
            if ids.numel() > 0 and ids.numel() < self.env.num_envs:
                return self._ordered_unique_valid(ids)

        if self.config.camera_sample_all_envs:
            return torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)

        camera_num_envs = self.config.camera_num_envs
        if camera_num_envs is None:
            camera_num_envs = 1024
        camera_num_envs = min(int(camera_num_envs), self.env.num_envs)
        camera_num_envs = max(camera_num_envs, 0)

        forced = self._forced_camera_env_ids()
        if forced.numel() > camera_num_envs:
            print(
                "[WARN] WMP depth_index expands camera_num_envs to include all tilt/crawl envs: "
                f"requested={camera_num_envs}, tilt_crawl={int(forced.numel())}"
            )
            camera_num_envs = int(forced.numel())
        remaining = camera_num_envs - int(forced.numel())
        if remaining <= 0:
            return self._ordered_unique_valid(forced)

        pool_mask = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
        if forced.numel() > 0:
            pool_mask[forced] = False
        pool = pool_mask.nonzero(as_tuple=False).flatten()
        if pool.numel() <= remaining:
            sampled = pool
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(self.config.camera_env_seed))
            perm = torch.randperm(pool.numel(), generator=generator, device="cpu")[:remaining].to(self.device)
            sampled = pool[perm]
        return self._ordered_unique_valid(torch.cat((sampled, forced), dim=0))

    def _select_depth_index_without_crawl_tilt(self) -> torch.Tensor:
        if hasattr(self.env, "get_depth_index_without_crawl_tilt"):
            ids = self.env.get_depth_index_without_crawl_tilt().to(self.device).long()
            if ids.numel() > 0:
                return self._ordered_unique_valid(ids)
        if self.depth_index.numel() == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)
        tilt_crawl = self._tilt_crawl_env_ids()
        if tilt_crawl.numel() == 0:
            return self.depth_index
        keep_mask = ~torch.isin(self.depth_index, tilt_crawl)
        return self.depth_index[keep_mask]

    def _select_depth_index_inverse(self) -> torch.Tensor:
        if hasattr(self.env, "get_depth_index_inverse"):
            inverse = self.env.get_depth_index_inverse().to(self.device).long()
            if inverse.numel() == self.env.num_envs:
                return inverse
        inverse = torch.full((self.env.num_envs,), -1, device=self.device, dtype=torch.long)
        if self.depth_index.numel() > 0:
            inverse[self.depth_index] = torch.arange(self.depth_index.numel(), device=self.device, dtype=torch.long)
        return inverse

    def _ordered_unique_valid(self, ids: torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long).flatten()
        keep = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.bool)
        out = []
        for env_id in ids.tolist():
            if 0 <= int(env_id) < self.env.num_envs and not bool(keep[int(env_id)].item()):
                keep[int(env_id)] = True
                out.append(int(env_id))
        return torch.tensor(out, device=self.device, dtype=torch.long)

    def _forced_camera_env_ids(self) -> torch.Tensor:
        if not self.config.camera_force_tilt_crawl:
            return torch.empty(0, device=self.device, dtype=torch.long)
        return self._tilt_crawl_env_ids()

    def _tilt_crawl_env_ids(self) -> torch.Tensor:
        terrain_types = getattr(self.env, "terrain_types", None)
        if terrain_types is None:
            return torch.empty(0, device=self.device, dtype=torch.long)

        cols_by_kind = getattr(self.env, "wmp_terrain_cols_by_kind", {}) or {}
        tilt_crawl_cols = list(cols_by_kind.get("tilt", ())) + list(cols_by_kind.get("crawl", ()))
        if not tilt_crawl_cols:
            terrain_generator = getattr(self.env.cfg.scene, "terrain_generator", None)
            if terrain_generator is None or len(terrain_generator.sub_terrains) <= 1:
                return torch.empty(0, device=self.device, dtype=torch.long)
            keys = list(terrain_generator.sub_terrains.keys())
            tilt_kind_ids = [idx for idx, key in enumerate(keys) if key.removeprefix("wmp_") in ("tilt", "crawl")]
            if not tilt_kind_ids:
                return torch.empty(0, device=self.device, dtype=torch.long)
            proportions = torch.tensor(
                [float(terrain_generator.sub_terrains[key].proportion) for key in keys],
                device=self.device,
            )
            proportions = proportions / proportions.sum().clamp_min(1.0e-6)
            cumulative = torch.cumsum(proportions, dim=0)
            num_cols = int(terrain_generator.num_cols)
            col_kind = []
            for col in range(num_cols):
                kind = int(
                    torch.searchsorted(
                        cumulative,
                        torch.tensor(col / num_cols + 0.001, device=self.device),
                        right=False,
                    ).item()
                )
                col_kind.append(kind)
            tilt_crawl_cols = [col for col, kind in enumerate(col_kind) if kind in tilt_kind_ids]
        if not tilt_crawl_cols:
            return torch.empty(0, device=self.device, dtype=torch.long)

        terrain_types = terrain_types.to(self.device)
        col_tensor = torch.tensor(tilt_crawl_cols, device=self.device, dtype=terrain_types.dtype)
        return torch.isin(terrain_types, col_tensor).nonzero(as_tuple=False).flatten()

    def _wm_feature(self, latent: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.config.feature_type == "full":
            return self.world_model.dynamics.get_feat(latent)
        return self.world_model.dynamics.get_deter_feat(latent)

    def _train_depth_predictor(self) -> float | None:
        if not self.replay.can_sample_real_depth(self.config.depth_predictor.batch_size):
            return None
        total_loss = 0.0
        used_iters = 0
        iters = int(self.config.depth_predictor.training_iters)
        batch_size = int(self.config.depth_predictor.batch_size)
        for _ in range(iters):
            batch = self.replay.sample_real_depth(batch_size)
            forward_height_map = batch["forward_height_map"].to(self.device)
            prop = batch["prop"].to(self.device)
            target = batch["image"].to(self.device)
            pred = self.depth_predictor(forward_height_map, prop)
            loss = (target - pred).pow(2).mean() * self.config.depth_predictor.loss_scale
            self.depth_predictor_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.depth_predictor.parameters(), 1.0)
            self.depth_predictor_opt.step()
            total_loss += float((loss / self.config.depth_predictor.loss_scale).detach().cpu())
            used_iters += 1
        if used_iters == 0:
            return None
        return total_loss / used_iters

    def _train_world_model(self) -> dict[str, float]:
        metrics_sum = {}
        used_steps = 0
        for _ in range(self.config.train_steps_per_iter):
            batch = self.replay.sample(self.config.batch_size, self.config.batch_length)
            batch = self._materialize_sampled_images(batch)
            batch = {
                k: v
                for k, v in batch.items()
                if k not in ("forward_height_map", "has_real_depth", "depth_index_slot")
            }
            _, _, metrics = self.world_model._train(batch)
            for key, value in metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value)
            used_steps += 1
        if used_steps == 0:
            return {}
        return {key: value / used_steps for key, value in metrics_sum.items()}

    def _materialize_sampled_images(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        depth_index_slot = batch.get("depth_index_slot")
        if depth_index_slot is None:
            has_real_depth = batch.get("has_real_depth")
            if has_real_depth is None:
                return batch
            missing_mask = has_real_depth[..., 0] <= 0.5
        else:
            missing_mask = depth_index_slot[..., 0] < 0
        if not torch.any(missing_mask):
            return batch

        image = batch["image"].to(self.device)
        forward_height_map = batch["forward_height_map"].to(self.device)
        prop = batch["prop"].to(self.device)
        flat_mask = missing_mask.reshape(-1).to(self.device)
        with torch.no_grad():
            pred = self.depth_predictor(
                forward_height_map.reshape(-1, forward_height_map.shape[-1])[flat_mask],
                prop.reshape(-1, prop.shape[-1])[flat_mask],
            )
        image = image.reshape(-1, *image.shape[2:])
        image[flat_mask] = pred
        batch["image"] = image.reshape(batch["prop"].shape[0], batch["prop"].shape[1], *image.shape[1:])
        return batch
