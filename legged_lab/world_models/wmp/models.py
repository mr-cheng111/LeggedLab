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
    ):
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)
        self.device = torch.device(device)
        self.camera_env_ids = camera_env_ids.detach().cpu().long().unique(sorted=True)
        self.camera_env_ids = self.camera_env_ids[
            (self.camera_env_ids >= 0) & (self.camera_env_ids < self.num_envs)
        ]
        self.env_to_camera_slot = torch.full((self.num_envs,), -1, dtype=torch.long)
        if self.camera_env_ids.numel() > 0:
            self.env_to_camera_slot[self.camera_env_ids] = torch.arange(self.camera_env_ids.numel(), dtype=torch.long)

        self.current_index = torch.zeros(self.num_envs, dtype=torch.long)
        self.dataset_size = torch.zeros(self.num_envs, dtype=torch.long)
        self.has_episode = torch.zeros(self.num_envs, dtype=torch.bool)
        self._current: dict[str, torch.Tensor] | None = None
        self._dataset: dict[str, torch.Tensor] | None = None
        self._current_image: torch.Tensor | None = None
        self._dataset_image: torch.Tensor | None = None

    def __len__(self):
        return int(self.has_episode.sum().item())

    @property
    def num_steps(self) -> int:
        return int(self.dataset_size[self.has_episode].sum().item())

    def add_step(self, env_id: int, transition: dict[str, torch.Tensor]):
        if self._current is None:
            self._init_storage(transition)
        assert self._current is not None
        assert self._dataset is not None

        env_id = int(env_id)
        step_id = int(self.current_index[env_id].item())
        if step_id >= self.max_episode_steps:
            return

        for key, value in transition.items():
            if key == "image":
                continue
            self._current[key][env_id, step_id].copy_(value.detach().to(self.device).float())

        camera_slot = int(self.env_to_camera_slot[env_id].item())
        has_real_depth = float(transition.get("has_real_depth", torch.zeros(1)).detach().flatten()[0].item()) > 0.5
        if camera_slot >= 0 and has_real_depth and self._current_image is not None:
            self._current_image[camera_slot, step_id].copy_(transition["image"].detach().to(self.device).float())
        self.current_index[env_id] += 1

    def finish_episode(self, env_id: int):
        if self._current is None:
            return
        assert self._dataset is not None

        env_id = int(env_id)
        length = int(self.current_index[env_id].item())
        self.current_index[env_id] = 0
        if length < 2:
            return

        for key, current_value in self._current.items():
            self._dataset[key][env_id, :length].copy_(current_value[env_id, :length])
        camera_slot = int(self.env_to_camera_slot[env_id].item())
        if camera_slot >= 0 and self._current_image is not None and self._dataset_image is not None:
            self._dataset_image[camera_slot, :length].copy_(self._current_image[camera_slot, :length])

        self.dataset_size[env_id] = length
        self.has_episode[env_id] = True

    def can_sample(self, batch_size: int, batch_length: int) -> bool:
        if batch_size <= 0 or self._dataset is None:
            return False
        return bool(torch.any(self.dataset_size >= int(batch_length)).item())

    def can_sample_real_depth(self, batch_size: int) -> bool:
        if batch_size <= 0 or self._dataset is None or self._dataset_image is None:
            return False
        if self.camera_env_ids.numel() == 0:
            return False
        valid_camera_envs = self.camera_env_ids[self.dataset_size[self.camera_env_ids] > 0]
        return bool(valid_camera_envs.numel() > 0)

    def sample(self, batch_size: int, batch_length: int) -> dict[str, torch.Tensor]:
        if not self.can_sample(batch_size, batch_length):
            raise RuntimeError("WMP fixed replay buffer does not have enough sequence data.")
        assert self._dataset is not None

        batch_length = int(batch_length)
        valid_envs = (self.dataset_size >= batch_length).nonzero(as_tuple=False).flatten()
        lengths = self.dataset_size[valid_envs].float()
        probs = lengths / lengths.sum().clamp_min(1.0)
        sampled_slots = torch.multinomial(probs, int(batch_size), replacement=True)
        env_ids = valid_envs[sampled_slots]

        batch = {key: [] for key in self._dataset.keys()}
        batch["image"] = []
        for env_id_tensor in env_ids:
            env_id = int(env_id_tensor.item())
            ep_len = int(self.dataset_size[env_id].item())
            start = int(torch.randint(0, ep_len - batch_length + 1, (1,)).item())
            for key, value in self._dataset.items():
                batch[key].append(value[env_id, start : start + batch_length])
            batch["image"].append(self._sample_image(env_id, start, batch_length))
        out = {key: torch.stack(values, dim=0) for key, values in batch.items()}
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

        valid_envs = self.camera_env_ids[self.dataset_size[self.camera_env_ids] > 0]
        lengths = self.dataset_size[valid_envs].float()
        probs = lengths / lengths.sum().clamp_min(1.0)
        sampled_slots = torch.multinomial(probs, int(batch_size), replacement=True)
        env_ids = valid_envs[sampled_slots]

        batch = {"forward_height_map": [], "prop": [], "image": []}
        for env_id_tensor in env_ids:
            env_id = int(env_id_tensor.item())
            ep_len = int(self.dataset_size[env_id].item())
            step_id = int(torch.randint(0, ep_len, (1,)).item())
            camera_slot = int(self.env_to_camera_slot[env_id].item())
            batch["forward_height_map"].append(self._dataset["forward_height_map"][env_id, step_id])
            batch["prop"].append(self._dataset["prop"][env_id, step_id])
            batch["image"].append(self._dataset_image[camera_slot, step_id])
        return {key: torch.stack(values, dim=0) for key, values in batch.items()}

    def _init_storage(self, transition: dict[str, torch.Tensor]):
        self._current = {}
        self._dataset = {}
        for key, value in transition.items():
            if key == "image":
                continue
            shape = (self.num_envs, self.max_episode_steps) + tuple(value.shape)
            self._current[key] = torch.zeros(shape, device=self.device, dtype=torch.float32)
            self._dataset[key] = torch.zeros(shape, device=self.device, dtype=torch.float32)

        image_shape = tuple(transition["image"].shape)
        camera_count = int(self.camera_env_ids.numel())
        if camera_count > 0:
            shape = (camera_count, self.max_episode_steps) + image_shape
            self._current_image = torch.zeros(shape, device=self.device, dtype=torch.float32)
            self._dataset_image = torch.zeros(shape, device=self.device, dtype=torch.float32)

    def _sample_image(self, env_id: int, start: int, batch_length: int) -> torch.Tensor:
        if self._dataset_image is None:
            raise RuntimeError("WMP fixed replay image storage has not been initialized.")
        camera_slot = int(self.env_to_camera_slot[env_id].item())
        if camera_slot >= 0:
            return self._dataset_image[camera_slot, start : start + batch_length]
        return torch.zeros_like(self._dataset_image[0, start : start + batch_length])
