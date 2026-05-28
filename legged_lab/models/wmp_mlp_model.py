# -*- coding: utf-8 -*-
"""rsl_rl 5 兼容的 WMP MLP 模型。

该类不修改 rsl_rl 源码，而是复用其 `MLPModel` 扩展点，在本地对 WMP
RSSM feature 先做低维编码，再与常规 proprio/critic 观测拼接。这样保留
原 WMP 的分层结构：

    z = concat(obs_without_wmp, Enc_wmp(f_wmp))

其中 `f_wmp` 默认是 RSSM deter feature，维度 512；`Enc_wmp` 默认输出 32 维。
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization, HiddenState, MLP
from rsl_rl.utils import unpad_trajectories


class WMPMLPModel(MLPModel):
    """带 WMP feature encoder 的 rsl_rl 5 MLPModel 兼容类。"""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        wmp_key: str = "wmp",
        wmp_feature_dim: int = 512,
        wmp_latent_dim: int = 32,
        wmp_encoder_hidden_dims: tuple[int, ...] | list[int] = (64, 64),
        use_history_encoder: bool = False,
        history_steps: int = 5,
        history_dim_per_step: int | None = None,
        history_encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        history_latent_dim: int = 35,
        command_dim: int = 3,
        command_start: int = 6,
        height_scan_dim: int = 0,
        history_excludes_trailing_height: bool = True,
        critic_uses_history_encoder: bool = False,
    ) -> None:
        self.wmp_key = wmp_key
        self.wmp_feature_dim = int(wmp_feature_dim)
        self.wmp_latent_dim = int(wmp_latent_dim)
        self.wmp_encoder_hidden_dims = list(wmp_encoder_hidden_dims)
        self.use_history_encoder = bool(use_history_encoder)
        self.history_steps = int(history_steps)
        self.history_dim_per_step = history_dim_per_step
        self.history_encoder_hidden_dims = list(history_encoder_hidden_dims)
        self.history_latent_dim = int(history_latent_dim)
        self.command_dim = int(command_dim)
        self.command_start = int(command_start)
        self.height_scan_dim = int(height_scan_dim)
        self.history_excludes_trailing_height = bool(history_excludes_trailing_height)
        self.critic_uses_history_encoder = bool(critic_uses_history_encoder)
        self.obs_set = obs_set
        self.history_encoder = None
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        self.wmp_encoder = MLP(self.wmp_feature_dim, self.wmp_latent_dim, self.wmp_encoder_hidden_dims, activation)
        if self._uses_history_path():
            history_input_dim = self._get_history_input_dim()
            self.history_encoder = MLP(
                history_input_dim,
                self.history_latent_dim,
                self.history_encoder_hidden_dims,
                activation,
            )
        if self.obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self._get_latent_dim())
        # 替换父类按原始 obs_dim 构造的 MLP，使 head 输入变为非 WMP 观测 + WMP latent。
        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = MLP(self._get_latent_dim(), mlp_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        non_wmp = []
        wmp_feature = None
        for obs_group in self.obs_groups:
            value = obs[obs_group]
            if obs_group == self.wmp_key:
                wmp_feature = value
            else:
                non_wmp.append(value)
        if wmp_feature is None:
            raise KeyError(f"WMPMLPModel requires observation group '{self.wmp_key}'.")
        if wmp_feature.shape[-1] != self.wmp_feature_dim:
            raise ValueError(
                f"Expected WMP feature dim {self.wmp_feature_dim}, got {wmp_feature.shape[-1]}."
            )
        base_obs = torch.cat(non_wmp, dim=-1)
        if self.history_encoder is not None:
            history_obs, _ = self._split_history_and_extra(base_obs)
            history = self._extract_history(history_obs)
            command = self._extract_command(history_obs)
            latent_parts = [self.history_encoder(history), command]
        else:
            latent_parts = [base_obs]
        latent_parts.append(self.wmp_encoder(wmp_feature))
        latent = torch.cat(latent_parts, dim=-1)
        return self.obs_normalizer(latent)

    def predict_linear_velocity(self, obs: TensorDict) -> torch.Tensor:
        if self.history_encoder is None:
            raise RuntimeError("WMPMLPModel velocity prediction requires use_history_encoder=True.")
        base_obs = self._cat_non_wmp_obs(obs)
        history_obs, _ = self._split_history_and_extra(base_obs)
        latent = self.history_encoder(self._extract_history(history_obs))
        return latent[..., -3:]

    def update_normalization(self, obs: TensorDict) -> None:
        if not self.obs_normalization:
            return
        with torch.no_grad():
            latent = self.get_latent(obs)
            self.obs_normalizer.update(latent)  # type: ignore

    def _get_latent_dim(self) -> int:
        if self._uses_history_path():
            return self.history_latent_dim + self.command_dim + self.wmp_latent_dim
        return self.obs_dim - self.wmp_feature_dim + self.wmp_latent_dim

    def _uses_history_path(self) -> bool:
        return self.use_history_encoder and (self.obs_set == "actor" or self.critic_uses_history_encoder)

    def _cat_non_wmp_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups if group != self.wmp_key], dim=-1)

    def _get_history_input_dim(self) -> int:
        if self.history_dim_per_step is None:
            non_wmp_dim = self.obs_dim - self.wmp_feature_dim
            extra_dim = self._get_extra_obs_dim(non_wmp_dim=non_wmp_dim)
            if (non_wmp_dim - extra_dim) % self.history_steps != 0:
                raise ValueError(
                    "history_dim_per_step must be provided when non-WMP obs dim is not divisible "
                    f"by history_steps: {non_wmp_dim} - {extra_dim} vs {self.history_steps}."
                )
            self.history_dim_per_step = (non_wmp_dim - extra_dim) // self.history_steps
        # 原版 WMP history 去掉 command:
        #   h_t = concat(obs_t[0:command_start], obs_t[command_start + command_dim:])
        # 当前 LeggedLab 的 height scan 位于完整 history 之后，因此默认作为 extra_obs 进入 actor head。
        return self.history_steps * (self.history_dim_per_step - self.command_dim)

    def _get_extra_obs_dim(self, non_wmp_dim: int | None = None) -> int:
        if not self._uses_history_path() or not self.history_excludes_trailing_height:
            return 0
        if self.height_scan_dim <= 0:
            return 0
        non_wmp_dim = self.obs_dim - self.wmp_feature_dim if non_wmp_dim is None else int(non_wmp_dim)
        history_dim = self.history_steps * int(self.history_dim_per_step or 0)
        if history_dim > 0 and non_wmp_dim >= history_dim + self.height_scan_dim:
            return self.height_scan_dim
        return 0

    def _split_history_and_extra(self, base_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        extra_dim = self._get_extra_obs_dim(non_wmp_dim=base_obs.shape[-1])
        if extra_dim <= 0:
            return base_obs, None
        return base_obs[..., :-extra_dim], base_obs[..., -extra_dim:]

    def _extract_history(self, base_obs: torch.Tensor) -> torch.Tensor:
        if self.history_dim_per_step is None:
            raise RuntimeError("history_dim_per_step was not initialized.")
        obs = base_obs.reshape(*base_obs.shape[:-1], self.history_steps, self.history_dim_per_step)
        left = obs[..., : self.command_start]
        tail_start = self.command_start + self.command_dim
        right = obs[..., tail_start:]
        # 公式对应原版 runner: history = flatten(concat(obs_without_command_t))。
        return torch.cat((left, right), dim=-1).flatten(-2)

    def _extract_command(self, base_obs: torch.Tensor) -> torch.Tensor:
        obs = base_obs.reshape(*base_obs.shape[:-1], self.history_steps, self.history_dim_per_step)
        current = obs[..., -1, :]
        return current[..., self.command_start : self.command_start + self.command_dim]
