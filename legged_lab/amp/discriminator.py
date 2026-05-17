# -*- coding: utf-8 -*-
"""WMP AMP 判别器。

Derived from ByteDance WMP rsl_rl/algorithms/amp_discriminator.py
(BSD-3-Clause lineage from NVIDIA/ETH Zurich with ByteDance modifications).
"""

import torch
from torch import autograd, nn


class AMPDiscriminator(nn.Module):
    def __init__(self, input_dim: int, amp_reward_coef: float, hidden_layer_sizes: list[int], device: str, task_reward_lerp: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.amp_reward_coef = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp
        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        self.trunk = nn.Sequential(*layers).to(device)
        self.amp_linear = nn.Linear(curr_dim, 1).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.amp_linear(self.trunk(x))

    def compute_grad_pen(self, expert_state: torch.Tensor, expert_next_state: torch.Tensor, lambda_: float = 10.0):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True
        disc = self.forward(expert_data)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=torch.ones_like(disc),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return lambda_ * grad.norm(2, dim=1).pow(2).mean()

    def predict_amp_reward(self, state: torch.Tensor, next_state: torch.Tensor, task_reward: torch.Tensor, normalizer=None):
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, state.device)
                next_state = normalizer.normalize_torch(next_state, next_state.device)
            d = self.forward(torch.cat([state, next_state], dim=-1))
            # WMP 原式: r_amp = coef * clamp(1 - 1/4 * (D(s,s') - 1)^2, min=0)
            reward = self.amp_reward_coef * torch.clamp(1.0 - 0.25 * torch.square(d - 1.0), min=0.0)
            if self.task_reward_lerp > 0:
                # WMP 源码保留了 lerp 参数名，但实际返回 disc_r + task_r。
                reward = reward + task_reward.unsqueeze(-1)
            self.train()
        return reward.squeeze(-1), d.squeeze(-1)
