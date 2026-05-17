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
