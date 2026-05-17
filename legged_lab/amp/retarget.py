# -*- coding: utf-8 -*-
"""AMP motion retarget adapter interface.

第一版算法层只消费 canonical AMP obs:
joint_pos(12) + base_lin_vel_b(3) + base_ang_vel_b(3) + joint_vel(12)。
retarget 的具体实现通过 config 注入，默认 NoOp 表示数据已经匹配目标机器人。
"""

from __future__ import annotations

import torch


class NoOpRetargetAdapter:
    """不做 retarget，仅校验 canonical AMP obs 维度。"""

    def __init__(self, canonical_obs_dim: int = 30, **_: object):
        self.canonical_obs_dim = canonical_obs_dim

    def __call__(self, amp_obs: torch.Tensor) -> torch.Tensor:
        if amp_obs.shape[-1] != self.canonical_obs_dim:
            raise ValueError(
                f"Expected canonical AMP obs dim={self.canonical_obs_dim}, got {amp_obs.shape[-1]}."
            )
        return amp_obs
