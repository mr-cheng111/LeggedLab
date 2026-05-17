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


class A1CanonicalRetargetAdapter:
    """Reorder canonical A1 AMP observations to match the simulator joint order.

    WMP motion files store A1 joints in the common Unitree SDK order:
    FR, FL, RR, RL with hip, thigh, calf for each leg. IsaacLab may expose the
    articulation joints in a different order, so the expert motion must be
    reordered before it is compared with policy AMP states.
    """

    SOURCE_JOINT_ORDER = (
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    )

    def __init__(
        self,
        canonical_obs_dim: int = 30,
        target_joint_order: list[str] | tuple[str, ...] | None = None,
        source_joint_order: list[str] | tuple[str, ...] | None = None,
        **_: object,
    ):
        self.canonical_obs_dim = canonical_obs_dim
        self.source_joint_order = tuple(source_joint_order or self.SOURCE_JOINT_ORDER)
        self.target_joint_order = tuple(target_joint_order or self.SOURCE_JOINT_ORDER)
        if len(self.source_joint_order) != 12 or len(self.target_joint_order) != 12:
            raise ValueError("A1 AMP retarget expects exactly 12 source and target joints.")
        missing = sorted(set(self.target_joint_order) - set(self.source_joint_order))
        if missing:
            raise ValueError(f"A1 AMP source joint order is missing target joints: {missing}")
        self.permutation = [self.source_joint_order.index(name) for name in self.target_joint_order]

    def __call__(self, amp_obs: torch.Tensor) -> torch.Tensor:
        if amp_obs.shape[-1] != self.canonical_obs_dim:
            raise ValueError(f"Expected canonical AMP obs dim={self.canonical_obs_dim}, got {amp_obs.shape[-1]}.")
        joint_pos = amp_obs[..., :12][..., self.permutation]
        root_lin_vel = amp_obs[..., 12:15]
        root_ang_vel = amp_obs[..., 15:18]
        joint_vel = amp_obs[..., 18:30][..., self.permutation]
        return torch.cat([joint_pos, root_lin_vel, root_ang_vel, joint_vel], dim=-1)
