# -*- coding: utf-8 -*-
"""WMP WorldModel 前向与训练封装。

参考 ByteDance WMP 的 dreamer/models.py，保留 RSSM 架构并补齐第一版
在线训练所需的 KL、decoder、reward 损失。
"""

import torch
from torch import nn

from . import networks, tools


class WorldModel(nn.Module):
    def __init__(self, config, obs_shape, use_camera: bool = True):
        super().__init__()
        self._config = config
        self.device = config.device
        self.encoder = networks.MultiEncoder(obs_shape, **config.encoder, use_camera=use_camera)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter if config.dyn_discrete else config.dyn_stoch + config.dyn_deter
        self.feature_dim = feat_size
        self.deter_dim = config.dyn_deter
        self.heads = nn.ModuleDict()
        self.heads["decoder"] = networks.MultiDecoder(feat_size, obs_shape, **config.decoder, use_camera=use_camera)
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
        )
        self.model_opt = torch.optim.Adam(self.parameters(), lr=config.model_lr, eps=config.opt_eps)
        self._scales = {"image": 1.0, "reward": config.reward_head["loss_scale"]}

    def decode(self, features):
        return self.heads["decoder"](features)

    def preprocess(self, obs):
        out = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                out[key] = value.to(self.device).float()
            else:
                out[key] = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        return out

    def _train(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.dynamics.observe(embed, data["action"], data["is_first"])
        kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
            post,
            prior,
            self._config.kl_free,
            self._config.dyn_scale,
            self._config.rep_scale,
        )
        feat = self.dynamics.get_feat(post)
        preds = {}
        for name, head in self.heads.items():
            head_feat = feat if name in self._config.grad_heads else feat.detach()
            pred = head(head_feat)
            preds.update(pred if isinstance(pred, dict) else {name: pred})

        losses = {}
        for name, pred in preds.items():
            if name not in data:
                continue
            loss = -pred.log_prob(data[name])
            if loss.ndim == 3 and loss.shape[-1] == 1:
                loss = loss.squeeze(-1)
            losses[name] = loss

        model_loss = kl_loss
        for name, loss in losses.items():
            model_loss = model_loss + self._scales.get(name, 1.0) * loss
        loss = model_loss.mean()

        self.model_opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self._config.grad_clip)
        self.model_opt.step()

        metrics = {
            "model_loss": float(loss.detach().cpu()),
            "model_grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
            "kl": float(kl_value.mean().detach().cpu()),
            "dyn_loss": float(dyn_loss.mean().detach().cpu()),
            "rep_loss": float(rep_loss.mean().detach().cpu()),
        }
        for name, loss_value in losses.items():
            metrics[f"{name}_loss"] = float(loss_value.mean().detach().cpu())
        context = {
            "embed": embed.detach(),
            "feat": feat.detach(),
            "post": {k: v.detach() for k, v in post.items()},
        }
        return context["post"], context, metrics


class WMPReplayBuffer:
    """按时间顺序缓存 WMP 训练数据，并采样连续序列。"""

    def __init__(self, capacity: int, device: str):
        self.capacity = int(capacity)
        self.device = device
        self.storage: list[dict[str, torch.Tensor]] = []
        self.position = 0

    def __len__(self):
        return len(self.storage)

    def add(self, transition: dict[str, torch.Tensor]):
        item = {k: v.detach().to(self.device).float().clone() for k, v in transition.items()}
        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def can_sample(self, batch_size: int, batch_length: int) -> bool:
        return len(self.storage) >= batch_length and batch_size > 0

    def sample(self, batch_size: int, batch_length: int) -> dict[str, torch.Tensor]:
        if not self.can_sample(batch_size, batch_length):
            raise RuntimeError("WMP replay buffer does not have enough data.")
        max_start = len(self.storage) - batch_length
        starts = torch.randint(0, max_start + 1, (batch_size,), device=self.device)
        keys = self.storage[0].keys()
        batch = {}
        for key in keys:
            seqs = []
            for start in starts.tolist():
                seqs.append(torch.stack([self.storage[start + offset][key] for offset in range(batch_length)], dim=1))
            batch[key] = torch.cat(seqs, dim=0)
        return batch


class WMPEpisodeReplayBuffer:
    """按 episode 存储 WMP 序列并采样连续片段。

    原版 WMP 的 world model 以 depth update interval 为时间步。每条 transition
    已经包含聚合后的动作：

        a_wm[t] = concat(a[t-k+1], ..., a[t]), k = update_interval
    """

    def __init__(self, capacity_episodes: int, device: str):
        self.capacity_episodes = int(capacity_episodes)
        self.device = torch.device(device)
        self.episodes: list[dict[str, torch.Tensor]] = []
        self._current: dict[int, list[dict[str, torch.Tensor]]] = {}

    def __len__(self):
        return len(self.episodes)

    @property
    def num_steps(self) -> int:
        return sum(int(next(iter(ep.values())).shape[0]) for ep in self.episodes)

    def add_step(self, env_id: int, transition: dict[str, torch.Tensor]):
        item = {k: v.detach().to(self.device).float().clone() for k, v in transition.items()}
        self._current.setdefault(int(env_id), []).append(item)

    def finish_episode(self, env_id: int):
        env_id = int(env_id)
        steps = self._current.pop(env_id, [])
        if len(steps) < 2:
            return
        keys = steps[0].keys()
        episode = {key: torch.stack([step[key] for step in steps], dim=0).to(self.device) for key in keys}
        self.episodes.append(episode)
        if len(self.episodes) > self.capacity_episodes:
            self.episodes = self.episodes[-self.capacity_episodes :]

    def can_sample(self, batch_size: int, batch_length: int) -> bool:
        if batch_size <= 0:
            return False
        return any(next(iter(ep.values())).shape[0] >= batch_length for ep in self.episodes)

    def sample(self, batch_size: int, batch_length: int) -> dict[str, torch.Tensor]:
        valid = [ep for ep in self.episodes if next(iter(ep.values())).shape[0] >= batch_length]
        if not valid:
            raise RuntimeError("WMP episode replay buffer does not have enough sequence data.")
        lengths = torch.tensor([next(iter(ep.values())).shape[0] for ep in valid], device=self.device, dtype=torch.float32)
        probs = lengths / lengths.sum()
        indices = torch.multinomial(probs, batch_size, replacement=True).tolist()
        keys = valid[0].keys()
        batch = {key: [] for key in keys}
        for idx in indices:
            ep = valid[idx]
            ep_len = next(iter(ep.values())).shape[0]
            start = int(torch.randint(0, ep_len - batch_length + 1, (1,), device=self.device).item())
            for key in keys:
                batch[key].append(ep[key][start : start + batch_length])
        return {key: torch.stack(values, dim=0) for key, values in batch.items()}


class WMPFixedEpisodeReplayBuffer:
    """原版 WMP 风格的固定容量 replay。

    原项目只保存每个 env 最近完成的一条 WMP episode：

        dataset[env_id, t] <- current_episode[env_id, t]

    depth 图像也只为真实相机 env 保存；非相机 env 在采样后由
    DepthPredictor(forward_height_map, prop) 动态生成，避免把 64x64 图像
    为所有 env 历史步都落到主机内存里。
    """

    def __init__(
        self,
        num_envs: int,
        max_episode_steps: int,
        camera_env_ids: torch.Tensor,
        device: str,
        depth_index_without_crawl_tilt: torch.Tensor | None = None,
        depth_index_inverse: torch.Tensor | None = None,
        depth_predictor_excluded_env_ids: torch.Tensor | None = None,
    ):
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)
        self.device = torch.device(device)
        # 保留 env.depth_index 的顺序，避免真实 depth 的 camera slot 与 env id 映射错位。
        self.camera_env_ids = self._ordered_unique_valid(camera_env_ids)
        self.env_to_camera_slot = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        if self.camera_env_ids.numel() > 0:
            self.env_to_camera_slot[self.camera_env_ids] = torch.arange(
                self.camera_env_ids.numel(), device=self.device, dtype=torch.long
            )
        self.depth_index_inverse = self._make_depth_index_inverse(depth_index_inverse)
        if depth_index_without_crawl_tilt is None:
            depth_index_without_crawl_tilt = self.camera_env_ids
        self.depth_index_without_crawl_tilt = self._ordered_unique_valid(depth_index_without_crawl_tilt)
        if depth_predictor_excluded_env_ids is not None:
            excluded = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            excluded[self._ordered_unique_valid(depth_predictor_excluded_env_ids)] = True
            self.depth_index_without_crawl_tilt = self.depth_index_without_crawl_tilt[
                ~excluded[self.depth_index_without_crawl_tilt]
            ]
        if self.depth_index_without_crawl_tilt.numel() > 0:
            has_camera = self.env_to_camera_slot[self.depth_index_without_crawl_tilt] >= 0
            self.depth_index_without_crawl_tilt = self.depth_index_without_crawl_tilt[has_camera]

        self.current_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.dataset_size = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.has_episode = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._current: dict[str, torch.Tensor] | None = None
        self._dataset: dict[str, torch.Tensor] | None = None
        self._current_image: torch.Tensor | None = None
        self._dataset_image: torch.Tensor | None = None
        self._image_shape: tuple[int, ...] | None = None

    def __len__(self):
        return int(self.has_episode.sum().item())

    @property
    def num_steps(self) -> int:
        return int(self.dataset_size[self.has_episode].sum().item())

    def add_step(self, env_id: int, transition: dict[str, torch.Tensor]):
        batched = {key: value.unsqueeze(0) for key, value in transition.items()}
        self.add_steps(torch.tensor([env_id], device=self.device), batched)

    def add_steps(self, env_ids: torch.Tensor, transitions: dict[str, torch.Tensor]):
        """Append one transition for many environments without Python-side per-env copies."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if env_ids.numel() == 0:
            return
        if self._current is None:
            self._init_storage(transitions, batched=True)
        assert self._current is not None
        assert self._dataset is not None

        input_count = env_ids.numel()
        step_ids = self.current_index[env_ids]
        valid = step_ids < self.max_episode_steps
        env_ids = env_ids[valid]
        step_ids = step_ids[valid]
        if env_ids.numel() == 0:
            return

        for key, value in transitions.items():
            if key == "image":
                continue
            batch_value = self._select_transition_batch(value, env_ids, valid, input_count)
            self._current[key][env_ids, step_ids] = batch_value

        camera_slots = self.env_to_camera_slot[env_ids]
        has_real_depth = self._select_transition_batch(
            transitions["has_real_depth"], env_ids, valid, input_count
        ).reshape(env_ids.numel(), -1)[:, 0] > 0.5
        store_image = (camera_slots >= 0) & has_real_depth
        if self._current_image is not None:
            images = self._select_transition_subset(
                transitions["image"], env_ids, valid, store_image, input_count
            )
            self._current_image[camera_slots[store_image], step_ids[store_image]] = images
        self.current_index[env_ids] = step_ids + 1

    def finish_episode(self, env_id: int):
        self.finish_episodes(torch.tensor([env_id], device=self.device))

    def finish_episodes(self, env_ids: torch.Tensor):
        """Commit completed episodes in one device-side indexed copy."""
        if self._current is None:
            return
        assert self._dataset is not None

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if env_ids.numel() == 0:
            return
        lengths = self.current_index[env_ids].clone()
        self.current_index[env_ids] = 0
        valid = lengths >= 2
        env_ids = env_ids[valid]
        lengths = lengths[valid]
        if env_ids.numel() == 0:
            return
        for key, current_value in self._current.items():
            self._dataset[key][env_ids] = current_value[env_ids]
        camera_slots = self.env_to_camera_slot[env_ids]
        has_camera = camera_slots >= 0
        if self._current_image is not None and self._dataset_image is not None:
            slots = camera_slots[has_camera]
            self._dataset_image[slots] = self._current_image[slots]

        self.dataset_size[env_ids] = lengths
        self.has_episode[env_ids] = True

    def can_sample(self, batch_size: int, batch_length: int) -> bool:
        del batch_length
        if batch_size <= 0 or self._dataset is None:
            return False
        return bool(torch.any(self.dataset_size > 1).item())

    def can_sample_real_depth(self, batch_size: int) -> bool:
        if batch_size <= 0 or self._dataset is None or self._dataset_image is None:
            return False
        valid_envs, _ = self._valid_real_depth_envs_and_counts()
        return bool(valid_envs.numel() > 0)

    def sample(self, batch_size: int, batch_length: int) -> dict[str, torch.Tensor]:
        if not self.can_sample(batch_size, batch_length):
            raise RuntimeError("WMP fixed replay buffer does not have enough sequence data.")
        assert self._dataset is not None

        requested_batch_length = int(batch_length)
        valid_envs = (self.dataset_size > 1).nonzero(as_tuple=False).flatten()
        lengths = self.dataset_size[valid_envs].float()
        # 原版 WMP 采样概率: p_i = dataset_size_i / sum_j(dataset_size_j)。
        probs = lengths / lengths.sum().clamp_min(1.0)
        sampled_slots = torch.multinomial(probs, int(batch_size), replacement=True)
        env_ids = valid_envs[sampled_slots]
        batch_length = min(int(self.dataset_size[env_ids].min().item()), requested_batch_length)
        if batch_length <= 1:
            raise RuntimeError("WMP fixed replay sampled sequence length must be greater than 1.")

        sampled_lengths = self.dataset_size[env_ids]
        max_start = sampled_lengths - batch_length
        starts = (torch.rand(env_ids.numel(), device=self.device) * (max_start + 1).float()).long()
        step_ids = starts.unsqueeze(1) + torch.arange(batch_length, device=self.device).unsqueeze(0)
        env_grid = env_ids.unsqueeze(1)
        out = {key: value[env_grid, step_ids] for key, value in self._dataset.items()}

        image_shape = self._image_shape or ()
        out["image"] = torch.zeros(
            (env_ids.numel(), batch_length) + image_shape, device=self.device, dtype=torch.float32
        )
        camera_slots = self.env_to_camera_slot[env_ids]
        has_camera = camera_slots >= 0
        if self._dataset_image is not None:
            slots = camera_slots[has_camera].unsqueeze(1)
            out["image"][has_camera] = self._dataset_image[slots, step_ids[has_camera]]
        if "is_first" in out:
            out["is_first"].zero_()
            out["is_first"][:, 0] = 1.0
        return out

    def sample_real_depth(self, batch_size: int) -> dict[str, torch.Tensor]:
        """只从真实相机 env 的有效时间步采样 DepthPredictor 监督数据。"""
        if not self.can_sample_real_depth(batch_size):
            raise RuntimeError("WMP fixed replay buffer does not have enough real depth samples.")
        assert self._dataset is not None
        assert self._dataset_image is not None

        valid_envs, valid_counts = self._valid_real_depth_envs_and_counts()
        probs = valid_counts.float() / valid_counts.float().sum().clamp_min(1.0)
        sampled_slots = torch.multinomial(probs, int(batch_size), replacement=True)
        env_ids = valid_envs[sampled_slots]

        step_ids = self._sample_real_depth_steps(env_ids)
        camera_slots = self.env_to_camera_slot[env_ids]
        return {
            "forward_height_map": self._dataset["forward_height_map"][env_ids, step_ids],
            "prop": self._dataset["prop"][env_ids, step_ids],
            "image": self._dataset_image[camera_slots, step_ids],
        }

    def _init_storage(self, transition: dict[str, torch.Tensor], batched: bool = False):
        self._current = {}
        self._dataset = {}
        for key, value in transition.items():
            if key == "image":
                continue
            item_shape = tuple(value.shape[1:]) if batched else tuple(value.shape)
            shape = (self.num_envs, self.max_episode_steps) + item_shape
            self._current[key] = torch.zeros(shape, device=self.device, dtype=torch.float32)
            self._dataset[key] = torch.zeros(shape, device=self.device, dtype=torch.float32)

        image_shape = tuple(transition["image"].shape[1:]) if batched else tuple(transition["image"].shape)
        self._image_shape = image_shape
        camera_count = int(self.camera_env_ids.numel())
        if camera_count > 0:
            shape = (camera_count, self.max_episode_steps) + image_shape
            self._current_image = torch.zeros(shape, device=self.device, dtype=torch.float32)
            self._dataset_image = torch.zeros(shape, device=self.device, dtype=torch.float32)

    def _ordered_unique_valid(self, ids: torch.Tensor | None) -> torch.Tensor:
        if ids is None:
            return torch.empty(0, device=self.device, dtype=torch.long)
        ids = torch.as_tensor(ids, device="cpu", dtype=torch.long).flatten()
        keep = torch.zeros(self.num_envs, dtype=torch.bool)
        out = []
        for env_id in ids.tolist():
            env_id = int(env_id)
            if 0 <= env_id < self.num_envs and not bool(keep[env_id].item()):
                keep[env_id] = True
                out.append(env_id)
        return torch.tensor(out, device=self.device, dtype=torch.long)

    def _make_depth_index_inverse(self, depth_index_inverse: torch.Tensor | None) -> torch.Tensor:
        if depth_index_inverse is not None:
            inverse = torch.as_tensor(depth_index_inverse, device=self.device, dtype=torch.long).flatten()
            if inverse.numel() == self.num_envs:
                return inverse.clone()
        inverse = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        if self.camera_env_ids.numel() > 0:
            inverse[self.camera_env_ids] = torch.arange(
                self.camera_env_ids.numel(), device=self.device, dtype=torch.long
            )
        return inverse

    def _valid_real_depth_envs_and_counts(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._dataset is None or self.depth_index_without_crawl_tilt.numel() == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            return empty, empty
        has_real_depth = self._dataset.get("has_real_depth")
        env_ids = self.depth_index_without_crawl_tilt
        lengths = self.dataset_size[env_ids]
        if has_real_depth is None:
            counts = lengths
        else:
            steps = torch.arange(self.max_episode_steps, device=self.device).unsqueeze(0)
            within_episode = steps < lengths.unsqueeze(1)
            real_depth = has_real_depth[env_ids].reshape(env_ids.numel(), self.max_episode_steps, -1)[..., 0] > 0.5
            counts = (within_episode & real_depth).sum(dim=1)
        valid = counts > 0
        return env_ids[valid], counts[valid]

    def _sample_real_depth_steps(self, env_ids: torch.Tensor) -> torch.Tensor:
        assert self._dataset is not None
        lengths = self.dataset_size[env_ids]
        has_real_depth = self._dataset.get("has_real_depth")
        if has_real_depth is None:
            return (torch.rand(env_ids.numel(), device=self.device) * lengths.float()).long()
        valid = has_real_depth[env_ids].reshape(env_ids.numel(), self.max_episode_steps, -1)[..., 0] > 0.5
        steps = torch.arange(self.max_episode_steps, device=self.device).unsqueeze(0)
        valid &= steps < lengths.unsqueeze(1)
        scores = torch.rand(valid.shape, device=self.device).masked_fill_(~valid, -1.0)
        return scores.argmax(dim=1)

    def _select_transition_batch(
        self,
        value: torch.Tensor,
        env_ids: torch.Tensor,
        valid: torch.Tensor,
        input_count: int,
    ) -> torch.Tensor:
        value = value.detach()
        if value.shape[0] == self.num_envs:
            source_ids = env_ids.to(value.device)
            value = value[source_ids]
        elif value.shape[0] == input_count:
            value = value[valid.to(value.device)]
        else:
            raise ValueError(
                "Batched replay transition must be indexed by all envs or by the provided env_ids: "
                f"value_batch={value.shape[0]}, num_envs={self.num_envs}, env_ids={input_count}."
            )
        return value.to(device=self.device, dtype=torch.float32)

    def _select_transition_subset(
        self,
        value: torch.Tensor,
        env_ids: torch.Tensor,
        valid: torch.Tensor,
        subset: torch.Tensor,
        input_count: int,
    ) -> torch.Tensor:
        """Select a filtered subset before device transfer, notably for camera images."""
        value = value.detach()
        if value.shape[0] == self.num_envs:
            source_ids = env_ids[subset].to(value.device)
            value = value[source_ids]
        elif value.shape[0] == input_count:
            value = value[valid.to(value.device)]
            value = value[subset.to(value.device)]
        else:
            raise ValueError(
                "Batched replay transition must be indexed by all envs or by the provided env_ids: "
                f"value_batch={value.shape[0]}, num_envs={self.num_envs}, env_ids={input_count}."
            )
        return value.to(device=self.device, dtype=torch.float32)
