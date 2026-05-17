# -*- coding: utf-8 -*-
"""AMP 观测归一化器。"""

import torch


class Normalizer:
    def __init__(self, dim: int, eps: float = 1.0e-5, device: str = "cpu"):
        self.dim = dim
        self.eps = eps
        self.device = device
        self.count = torch.tensor(eps, device=device)
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)

    def update(self, x: torch.Tensor):
        x = x.detach().to(self.device)
        if x.numel() == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(x.shape[0], device=self.device, dtype=torch.float32)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total
        self.mean = new_mean
        self.var = torch.clamp(m2 / total, min=self.eps)
        self.count = total

    def normalize_torch(self, x: torch.Tensor, device: str | None = None) -> torch.Tensor:
        target_device = device or x.device
        mean = self.mean.to(target_device)
        var = self.var.to(target_device)
        return (x - mean) / torch.sqrt(var + self.eps)

    def state_dict(self):
        return {"count": self.count, "mean": self.mean, "var": self.var}

    def load_state_dict(self, state):
        self.count = state["count"].to(self.device)
        self.mean = state["mean"].to(self.device)
        self.var = state["var"].to(self.device)
