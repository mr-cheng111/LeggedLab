# -*- coding: utf-8 -*-
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.unitree import B2_CFG
from legged_lab.envs.base.base_env_config import (  # noqa:F401
    BaseAgentCfg,
    BaseEnvCfg,
    BaseSceneCfg,
    DomainRandCfg,
    HeightScannerCfg,
    PhysxCfg,
    RewardCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG


@configclass
class B2RewardCfg(RewardCfg):
    """B2 奖励配置（默认用于平地训练）。

    总奖励形式:
        r_t = sum_i(w_i * r_i)
    其中 w_i 为各项权重，正值表示鼓励，负值表示惩罚。

    速度跟踪项采用指数奖励（常见于 IsaacLab / LeggedGym）：
        r_lin = exp(-||v_cmd_xy - v_xy_yaw||^2 / std^2)
        r_yaw = exp(-(w_cmd_z - w_z)^2 / std^2)

    其中 v_xy_yaw = Rz(yaw)^T * v_xy_world，
    即先把世界坐标系速度旋转到机器人 yaw 对齐坐标系，再进行误差计算。
    """

    # 1) 主任务：跟踪命令速度（前后/左右 + 偏航角速度）
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.5, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=0.75, params={"std": 0.5})

    # 2) 稳定性约束：抑制非期望速度与姿态
    # lin_vel_z_l2 = (v_z)^2，抑制机身上下弹跳
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # ang_vel_xy_l2 = (w_roll)^2 + (w_pitch)^2，抑制滚转/俯仰角速度振荡
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)

    # 3) 动作平滑与能耗约束
    # energy 近似关节功率/扭矩消耗项，鼓励更节能步态
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    # dof_acc_l2 = sum((q_ddot)^2)，抑制关节高频加速度
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # action_rate_l2 = sum((a_t - a_{t-1})^2)，抑制动作抖动
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    # 4) 接触与步态合理性约束
    # 非足端身体部位接触地面惩罚
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*foot.*).*"), "threshold": 1.0},
    )
    # 四足均不接触地面时惩罚，避免长时间腾空
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"), "threshold": 1.0},
    )

    # 预留项：机身姿态误差（当前权重为 0，不生效）
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}, weight=0.0
    )
    # 在平地上鼓励机身保持水平（roll/pitch 误差平方惩罚）
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    # 环境提前终止时一次性大惩罚
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # 足端接触期间水平滑移惩罚，减少打滑
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )

    # 足端受力奖励（当前权重 0，默认关闭）
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )

    # 足端绊碰惩罚
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*foot.*"])},
    )

    # 关节越界惩罚：关节接近/超过机械限位时惩罚
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)


@configclass
class B2FlatEnvCfg(BaseEnvCfg):
    """B2 平地环境配置。"""

    # 使用 B2 专用奖励组合
    reward = B2RewardCfg()

    def __post_init__(self):
        super().__post_init__()

        # 机器人与场景
        self.scene.height_scanner.prim_body_name = "base"
        self.scene.robot = B2_CFG
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = GRAVEL_TERRAINS_CFG

        # 接触终止与足端识别规则
        self.robot.terminate_contacts_body_names = [".*base.*"]
        self.robot.feet_body_names = [".*foot.*"]
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*base.*"]

        # 显式写出 B2 训练时的速度指令默认范围（单位：m/s, rad/s）
        # 采样形式：
        #   v_x ~ U(vx_min, vx_max)
        #   v_y ~ U(vy_min, vy_max)
        #   w_z ~ U(wz_min, wz_max)
        self.commands.resampling_time_range = (10.0, 10.0)
        self.commands.ranges.lin_vel_x = (-0.6, 1.0)
        self.commands.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.ranges.ang_vel_z = (-1.57, 1.57)
        # heading 范围 [-pi, pi]，用于偏航角闭环控制
        self.commands.ranges.heading = (-math.pi, math.pi)


@configclass
class B2FlatAgentCfg(BaseAgentCfg):
    """B2 平地任务训练器配置。"""

    experiment_name: str = "b2_flat"
    wandb_project: str = "b2_flat"


@configclass
class B2RoughEnvCfg(B2FlatEnvCfg):
    """B2 复杂地形环境配置（在平地配置基础上覆盖）。"""

    def __post_init__(self):
        super().__post_init__()

        # 粗糙地形：开启高度扫描并切换地形生成器
        self.scene.height_scanner.enable_height_scan = True
        self.scene.terrain_generator = ROUGH_TERRAINS_CFG

        # 粗糙地形下观测历史缩短，降低时序堆叠难度
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1

        # 保持当前速度跟踪与稳定性权重
        self.reward.track_lin_vel_xy_exp.weight = 1.5
        self.reward.track_ang_vel_z_exp.weight = 0.75
        self.reward.lin_vel_z_l2.weight = -2.0


@configclass
class B2RoughAgentCfg(BaseAgentCfg):
    """B2 复杂地形训练器配置。"""

    experiment_name: str = "b2_rough"
    wandb_project: str = "b2_rough"

    def __post_init__(self):
        super().__post_init__()

        # 粗糙地形采用循环策略网络（LSTM），增强时序建模能力
        self.policy.class_name = "ActorCriticRecurrent"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.policy.rnn_hidden_size = 256
        self.policy.rnn_num_layers = 1
        self.policy.rnn_type = "lstm"
