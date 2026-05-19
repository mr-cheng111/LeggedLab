# -*- coding: utf-8 -*-
"""Minimal RB160W flat velocity-tracking task."""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.xuanji import RB160W_CFG
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg, RewardCfg


@configclass
class RB160WRewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=3.0, params={"std": 0.45})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"std": 0.45})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)
    energy = RewTerm(func=mdp.energy, weight=-2.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-7)
    action_rate_l2 = None
    leg_action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_joint,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
    )
    wheel_action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_joint,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_WHEEL_joint"])},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height_l2 = RewTerm(func=mdp.base_height_l2, weight=-2.0, params={"target_height": 0.72})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-5.0)
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*WHEEL.*"), "threshold": 1.0},
    )
    wheel_contact_count = RewTerm(
        func=mdp.feet_contact_count,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*WHEEL.*"), "threshold": 1.0},
    )
    all_wheels_contact = RewTerm(
        func=mdp.all_feet_contact,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*WHEEL.*"), "threshold": 1.0},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*WHEEL.*).*"), "threshold": 1.0},
    )
    wheel_slide = None
    leg_deviation_l2 = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
    )


@configclass
class RB160WFlatEnvCfg(BaseEnvCfg):
    reward = RB160WRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = RB160W_CFG
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        self.scene.height_scanner.enable_height_scan = False
        self.scene.height_scanner.prim_body_name = "base_link"
        self.scene.gemini2_camera.enable = False
        self.scene.env_spacing = 3.0

        self.robot.terminate_contacts_body_names = [".*base_link.*"]
        self.robot.feet_body_names = [".*WHEEL.*"]
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.robot.action_scale = 0.12
        self.robot.wheel_velocity_scale = 12.0

        self.normalization.obs_scales.lin_vel = 1.0
        self.normalization.obs_scales.ang_vel = 0.25
        self.normalization.obs_scales.projected_gravity = 1.0
        self.normalization.obs_scales.commands = 1.0
        self.normalization.obs_scales.joint_pos = 1.0
        self.normalization.obs_scales.joint_vel = 0.05
        self.normalization.obs_scales.actions = 1.0
        self.normalization.clip_observations = 10.0

        self.commands.resampling_time_range = (8.0, 10.0)
        self.commands.heading_command = False
        self.commands.debug_vis = False
        self.commands.rel_standing_envs = 0.1
        self.commands.rel_heading_envs = 0.0
        self.commands.ranges.lin_vel_x = (-0.6, 1.2)
        self.commands.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.ranges.heading = (-math.pi, math.pi)

        self.noise.add_noise = True
        self.domain_rand.events.physics_material.params["static_friction_range"] = (0.8, 1.6)
        self.domain_rand.events.physics_material.params["dynamic_friction_range"] = (0.6, 1.3)
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = [".*base.*"]
        self.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (0.95, 1.05)
        self.domain_rand.events.add_base_mass.params["operation"] = "scale"
        self.domain_rand.events.add_base_mass.params["recompute_inertia"] = True
        self.domain_rand.events.push_robot = None
        self.domain_rand.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-3.14, 3.14)}
        self.domain_rand.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.domain_rand.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)


@configclass
class RB160WFlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "rb160w_flat"
    wandb_project: str = "rb160w_flat"

    def __post_init__(self):
        super().__post_init__()
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.empirical_normalization = True
        self.algorithm.entropy_coef = 0.01
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4
        self.num_steps_per_env = 24
        self.save_interval = 500
