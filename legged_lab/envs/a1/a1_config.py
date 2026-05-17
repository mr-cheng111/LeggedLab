# -*- coding: utf-8 -*-
"""Unitree A1 AMP-PPO 平地测试任务。

第一版目标是验证 AMP-PPO 框架本身：环境仍然提供常规速度跟踪奖励，
AMP 奖励由 runner 根据 canonical 30 维 AMP obs 单独混合。
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.unitree import A1_CFG
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg, RewardCfg


@configclass
class A1AMPRewardCfg(RewardCfg):
    """A1 平地 AMP 测试奖励。

    AMP 最终奖励在 runner 中计算:
        r = w_task * r_task + w_amp * r_amp
    这里的每个 RewTerm 只属于 r_task，用于让 PPO 保持基本可训练。
    """

    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.5, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=0.5, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=mdp.energy, weight=-1.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-2.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_quadruped,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"), "threshold": 0.5},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*foot.*).*"), "threshold": 1.0},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*foot.*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)


@configclass
class A1AMPFlatEnvCfg(BaseEnvCfg):
    reward = A1AMPRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = A1_CFG
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        self.scene.height_scanner.enable_height_scan = False
        self.scene.gemini2_camera.enable = False
        self.scene.env_spacing = 2.5
        self.robot.feet_body_names = [".*foot.*"]
        self.robot.terminate_contacts_body_names = [".*trunk.*"]
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.robot.action_scale = 0.25
        self.commands.heading_command = False
        self.commands.rel_standing_envs = 0.0
        self.commands.rel_heading_envs = 0.0
        self.commands.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.ranges.ang_vel_z = (-1.0, 1.0)
        self.noise.add_noise = False
        self.domain_rand.events.physics_material = None
        self.domain_rand.events.add_base_mass = None
        self.domain_rand.events.push_robot = None
        self.domain_rand.events.reset_base.params["pose_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)}
        self.domain_rand.events.reset_base.params["velocity_range"] = {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "z": (-0.1, 0.1),
            "roll": (-0.2, 0.2),
            "pitch": (-0.2, 0.2),
            "yaw": (-0.2, 0.2),
        }
        self.domain_rand.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (0.8, 1.2), "velocity_range": (0.0, 0.0)},
        )


@configclass
class A1AMPFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "a1_amp_flat"
    wandb_project: str = "a1_amp_flat"
    runner_class_name: str = "legged_lab.runners.amp_ppo_runner:AMPPPORunner"
    empirical_normalization: bool = True
    amp: dict = {
        "motion_files": [],
        "canonical_obs_dim": 30,
        "retarget_adapter": {
            "class_path": "legged_lab.amp.retarget:NoOpRetargetAdapter",
            "profile": "a1_canonical_v1",
        },
        "reward_coef": 0.01,
        "task_reward_weight": 1.0,
        "amp_reward_weight": 1.0,
        "discriminator_hidden_dims": [256, 128],
        "replay_buffer_size": 10000,
        "num_preload_transitions": 4096,
        "grad_penalty_coef": 0.0,
    }

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "MLPModel"
        self.policy.actor_hidden_dims = [256, 128, 64]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.class_name = "legged_lab.algorithms.amp_ppo:AMPPPO"
        self.algorithm.entropy_coef = 0.01
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4
        self.num_steps_per_env = 24
        self.save_interval = 1000
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
