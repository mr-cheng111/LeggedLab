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
from legged_lab.assets.unitree import B2_CFG, B2_RGBD_CFG
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
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG, WMP_MIXED_TERRAINS_CFG


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
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=0.5, params={"std": 0.5})

    # 2) 稳定性约束：抑制非期望速度与姿态
    # lin_vel_z_l2 = (v_z)^2，抑制机身上下弹跳
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    # ang_vel_xy_l2 = (w_roll)^2 + (w_pitch)^2，抑制滚转/俯仰角速度振荡
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)

    # 3) 动作平滑与能耗约束
    # energy 近似关节功率/扭矩消耗项，鼓励更节能步态
    energy = RewTerm(func=mdp.energy, weight=-1e-5)
    # dof_acc_l2 = sum((q_ddot)^2)，抑制关节高频加速度
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # action_rate_l2 = sum((a_t - a_{t-1})^2)，抑制动作抖动
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.03)

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
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"), "threshold": 1.0},
    )
    # WMP/A1: r = sum((air_time - 0.5) * first_contact)，鼓励合理摆腿周期。
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_quadruped,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"), "threshold": 0.5},
    )

    # 预留项：机身姿态误差（当前权重为 0，不生效）
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}, weight=0.0
    )
    # 在平地上鼓励机身保持水平（roll/pitch 误差平方惩罚）
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    # 环境提前终止时一次性大惩罚
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # 足端接触期间水平滑移惩罚，减少打滑
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.03,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )

    # 足端受力奖励（当前权重 0）
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
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*foot.*"])},
    )

    # 关节越界惩罚：关节接近/超过机械限位时惩罚
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    dof_error = RewTerm(func=mdp.dof_error_l2, weight=-0.04)
    feet_edge = RewTerm(
        func=mdp.feet_edge,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )
    cheat = RewTerm(func=mdp.cheat_yaw, weight=-1.0, params={"heading_limit": 1.0})
    stuck = RewTerm(func=mdp.stuck, weight=-1.0, params={"velocity_threshold": 0.1, "command_threshold": 0.1})


@configclass
class B2WMPAMPRewardCfg(B2RewardCfg):
    """对齐 WMP A1 AMP 原版 rewards；原版没有的项暂时关闭。"""

    # WMP 原版 base cfg: ang_vel_xy = 0, orientation = 0；A1 AMP 未覆盖这两项。
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}, weight=0.0
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)

    # WMP A1 AMP 原版项与权重。
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.5, params={"std": 0.15**0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=0.5, params={"std": 0.15**0.5})
    energy = RewTerm(func=mdp.energy, weight=-0.0001)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_quadruped,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"), "threshold": 0.5},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*foot.*).*"), "threshold": 0.1},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*foot.*"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    feet_edge = RewTerm(
        func=mdp.feet_edge,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )
    dof_error = RewTerm(func=mdp.dof_error_l2, weight=-0.04)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    cheat = RewTerm(func=mdp.cheat_yaw, weight=-1.0, params={"heading_limit": 1.0})
    stuck = RewTerm(func=mdp.stuck, weight=-1.0, params={"velocity_threshold": 0.1, "command_threshold": 0.1})

    # WMP A1 AMP 原版没有这些项，暂时关闭。
    fly = RewTerm(
        func=mdp.fly,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"), "threshold": 1.0},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=0.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class B2StandRewardCfg(RewardCfg):
    """B2 强抗扰站立阶段奖励。

    目标是先学会最简单稳定站立：保持初始高度、机身水平、四足接触、足端低速。
    姿态项参考 G1 的 body/flat orientation 奖励，并额外提供指数型正奖励。
    """

    stand_height = RewTerm(func=mdp.base_height_exp, weight=2.0, params={"target_height": 0.58, "std": 0.12})
    feet_contact = RewTerm(
        func=mdp.feet_contact_count,
        weight=2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"), "threshold": 1.0},
    )
    feet_still = RewTerm(
        func=mdp.feet_still_exp,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
            "std": 0.5,
        },
    )
    push_recovery = RewTerm(
        func=mdp.push_recovery_time_exp,
        weight=4.0,
        params={
            "max_time": 2.0,
            "height_target": 0.58,
            "height_tol": 0.08,
            "lin_vel_tol": 0.10,
            "ang_vel_tol": 0.15,
            "orientation_tol": 0.14,
            "contact_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"),
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-6.0)
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}, weight=-3.0
    )
    lin_vel_xy_l2 = RewTerm(func=mdp.lin_vel_xy_l2, weight=-2.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-2.0, params={"target_height": 0.58})
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-4.0)
    ang_vel_z_l2 = RewTerm(func=mdp.ang_vel_z_l2, weight=-1.0)
    dof_error = RewTerm(func=mdp.dof_error_l2, weight=-0.45)
    energy = RewTerm(func=mdp.energy, weight=-1e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5e-6)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.12)
    fly = RewTerm(
        func=mdp.fly,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"), "threshold": 1.0},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-8.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*foot.*).*"), "threshold": 1.0},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-30.0)


@configclass
class B2SlowWalkRewardCfg(RewardCfg):
    """B2 慢速行走奖励。

    参考 G1 的行走奖励结构：速度跟踪为主，配合机身姿态、足端接触、滑移、动作平滑约束。
    慢走阶段不加入 WMP/AMP，也不加入推扰，先让站立策略过渡到低速步态。
    """

    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0, params={"std": 0.35})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.5, params={"std": 0.35})
    stand_height = RewTerm(func=mdp.base_height_exp, weight=0.8, params={"target_height": 0.58, "std": 0.15})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*base.*")}, weight=-1.0
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    energy = RewTerm(func=mdp.energy, weight=-1e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    dof_error = RewTerm(func=mdp.dof_error_l2, weight=-0.03)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_quadruped,
        weight=0.3,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"), "threshold": 0.35},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"), "threshold": 1.0},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.08,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*foot.*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*foot.*).*"), "threshold": 1.0},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)


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


@configclass
class B2RGBDFlatEnvCfg(B2FlatEnvCfg):
    """B2 RGBD 平地环境配置。

    当前仅切换机器人 USD 资产到 b2_rgbd 版本；PPO 观测仍沿用本体观测。
    后续世界模型训练可在该任务上接入 RGBD 相机数据。
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = B2_RGBD_CFG
        self.scene.height_scanner.prim_body_name = "base_link"
        self.scene.gemini2_camera.enable = True


@configclass
class B2RGBDFlatAgentCfg(B2FlatAgentCfg):
    """B2 RGBD 平地任务训练器配置。"""

    experiment_name: str = "b2_rgbd_flat"
    wandb_project: str = "b2_rgbd_flat"


@configclass
class B2RGBDStandEnvCfg(B2RGBDFlatEnvCfg):
    """B2 RGBD 站立预训练任务。"""

    reward = B2StandRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physx.gpu_max_rigid_patch_count = 2 * 2**15
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        self.scene.gemini2_camera.enable = False
        self.scene.gemini2_camera.enable_rgb = False
        self.scene.gemini2_camera.enable_depth = False
        self.scene.gemini2_camera.allow_missing_depth_fallback = True
        self.normalization.obs_scales.lin_vel = 1.0
        self.normalization.obs_scales.ang_vel = 0.25
        self.normalization.obs_scales.projected_gravity = 1.0
        self.normalization.obs_scales.commands = 1.0
        self.normalization.obs_scales.joint_pos = 1.0
        self.normalization.obs_scales.joint_vel = 0.05
        self.normalization.obs_scales.actions = 1.0
        self.normalization.clip_observations = 10.0
        self.commands.resampling_time_range = (10.0, 10.0)
        self.commands.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.ranges.heading = (0.0, 0.0)
        self.commands.heading_command = False
        self.commands.debug_vis = False
        self.commands.rel_standing_envs = 1.0
        self.commands.rel_heading_envs = 0.0
        self.domain_rand.events.physics_material = None
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*"]
        self.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (0.85, 1.15)
        self.domain_rand.events.add_base_mass.params["operation"] = "scale"
        self.domain_rand.events.add_base_mass.params["recompute_inertia"] = True
        self.domain_rand.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.domain_rand.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.domain_rand.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.domain_rand.events.push_robot.func = mdp.push_by_setting_velocity_with_recovery_marker
        self.domain_rand.events.push_robot.interval_range_s = (3.0, 6.0)
        self.domain_rand.events.push_robot.params["velocity_range"] = {
            "x": (-1.5, 1.5),
            "y": (-1.5, 1.5),
            "z": (-0.3, 0.3),
        }


@configclass
class B2RGBDStandAgentCfg(B2RGBDFlatAgentCfg):
    """B2 RGBD 站立预训练配置。"""

    experiment_name: str = "b2_rgbd_stand"
    wandb_project: str = "b2_rgbd_curriculum"

    def __post_init__(self):
        super().__post_init__()
        self.empirical_normalization = True
        self.policy.class_name = "ActorCritic"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.entropy_coef = 0.005
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4
        self.num_steps_per_env = 24
        self.save_interval = 500


@configclass
class B2RGBDSlowWalkEnvCfg(B2RGBDFlatEnvCfg):
    """B2 RGBD 慢速行走任务。"""

    reward = B2SlowWalkRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.sim.physx.gpu_max_rigid_patch_count = 2 * 2**15
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        self.scene.gemini2_camera.enable = False
        self.scene.gemini2_camera.enable_rgb = False
        self.scene.gemini2_camera.enable_depth = False
        self.scene.gemini2_camera.allow_missing_depth_fallback = True
        self.normalization.obs_scales.lin_vel = 1.0
        self.normalization.obs_scales.ang_vel = 0.25
        self.normalization.obs_scales.projected_gravity = 1.0
        self.normalization.obs_scales.commands = 1.0
        self.normalization.obs_scales.joint_pos = 1.0
        self.normalization.obs_scales.joint_vel = 0.05
        self.normalization.obs_scales.actions = 1.0
        self.normalization.clip_observations = 10.0
        self.commands.resampling_time_range = (6.0, 8.0)
        self.commands.ranges.lin_vel_x = (-0.2, 1.5)
        self.commands.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.ranges.heading = (0.0, 0.0)
        self.commands.heading_command = False
        self.commands.debug_vis = False
        self.commands.rel_standing_envs = 0.1
        self.commands.rel_heading_envs = 0.0
        self.robot.terminate_on_flight = True
        self.robot.terminate_on_flight_threshold = 1.0
        self.domain_rand.events.physics_material = None
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*"]
        self.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (0.85, 1.15)
        self.domain_rand.events.add_base_mass.params["operation"] = "scale"
        self.domain_rand.events.add_base_mass.params["recompute_inertia"] = True
        self.domain_rand.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.domain_rand.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.domain_rand.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.domain_rand.events.push_robot = None


@configclass
class B2RGBDSlowWalkAgentCfg(B2RGBDFlatAgentCfg):
    """B2 RGBD 慢速行走训练配置。"""

    experiment_name: str = "b2_rgbd_slow_walk"
    wandb_project: str = "b2_rgbd_curriculum"

    def __post_init__(self):
        super().__post_init__()
        self.empirical_normalization = True
        self.policy.class_name = "ActorCritic"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.entropy_coef = 0.01
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4
        self.num_steps_per_env = 24
        self.save_interval = 500


@configclass
class B2RGBDRoughEnvCfg(B2RoughEnvCfg):
    """B2 RGBD 复杂地形环境配置。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = B2_RGBD_CFG
        self.scene.height_scanner.prim_body_name = "base_link"
        self.scene.gemini2_camera.enable = True


@configclass
class B2RGBDRoughAgentCfg(B2RoughAgentCfg):
    """B2 RGBD 复杂地形任务训练器配置。"""

    experiment_name: str = "b2_rgbd_rough"
    wandb_project: str = "b2_rgbd_rough"


@configclass
class B2RGBDWMPAMPFlatEnvCfg(B2RGBDFlatEnvCfg):
    """B2 RGBD WMP+AMP 平地任务。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.gemini2_camera.enable = True
        self.scene.gemini2_camera.enable_rgb = False
        self.scene.gemini2_camera.enable_depth = True
        self.scene.gemini2_camera.allow_missing_depth_fallback = True


@configclass
class B2RGBDWMPAMPTerrainEnvCfg(B2RGBDWMPAMPFlatEnvCfg):
    """B2 RGBD WMP+AMP 原版混合地形任务。"""

    reward = B2WMPAMPRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = WMP_MIXED_TERRAINS_CFG
        self.scene.max_init_terrain_level = 0
        self.scene.height_scanner.enable_height_scan = True


@configclass
class B2RGBDWMPAMPFlatAgentCfg(B2RGBDFlatAgentCfg):
    """B2 RGBD WMP+AMP 平地训练配置。"""

    experiment_name: str = "b2_rgbd_wmp_amp_flat"
    wandb_project: str = "b2_rgbd_wmp_amp"
    runner_class_name: str = "legged_lab.runners.wmp_amp_runner:WMPAMPRunner"
    wmp: dict = {
        "feature_type": "deter",
        "train_start_steps": 10000,
        "train_steps_per_iter": 1,
        "batch_size": 16,
        "batch_length": 64,
        "replay_capacity": 50000,
    }
    amp: dict = {
        "motion_files": [],
        "num_preload_transitions": 2000000,
        "reward_coef": 0.01,
        "task_reward_lerp": 0.3,
        "discriminator_hidden_dims": [1024, 512],
        "replay_buffer_size": 1000000,
        "grad_penalty_coef": 1.0,
    }

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "MLPModel"
        self.algorithm.class_name = "legged_lab.algorithms.wmp_amp_ppo:WMPAMPPPO"
        self.policy.actor_hidden_dims = [256, 128, 64]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.entropy_coef = 0.01
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4
        self.num_steps_per_env = 24
        self.save_interval = 1000
        self.obs_groups = {"actor": ["policy", "wmp"], "critic": ["critic", "wmp"]}


@configclass
class B2RGBDWMPAMPTerrainAgentCfg(B2RGBDWMPAMPFlatAgentCfg):
    """B2 RGBD WMP+AMP 原版混合地形训练配置。"""

    experiment_name: str = "b2_rgbd_wmp_amp_terrain"
    wandb_project: str = "b2_rgbd_wmp_amp"
