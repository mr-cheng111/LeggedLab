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
from legged_lab.terrains import WMP_MIXED_TERRAINS_CFG


@configclass
class A1AMPRewardCfg(RewardCfg):
    """A1 平地 AMP 测试奖励。

    AMP 最终奖励在 runner 中计算:
        r = w_task * r_task + w_amp * r_amp
    这里的每个 RewTerm 只属于 r_task，用于让 PPO 保持基本可训练。
    """

    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.5, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
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
        self.scene.rgbd_camera.enable = False
        self.scene.env_spacing = 2.5
        self.robot.feet_body_names = [".*foot.*"]
        self.robot.terminate_contacts_body_names = [".*trunk.*"]
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.robot.action_scale = 0.25
        self.commands.heading_command = False
        self.commands.rel_standing_envs = 0.0
        self.commands.rel_heading_envs = 0.0
        self.commands.ranges.lin_vel_x = (-0.6, 1.0)
        self.commands.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.ranges.ang_vel_z = (-1.57, 1.57)
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
            "class_path": "legged_lab.amp.retarget:A1CanonicalRetargetAdapter",
            "profile": "a1_canonical_v1",
            "target_joint_order": "env",
        },
        "reward_coef": 0.2,
        "task_reward_weight": 1.0,
        "amp_reward_weight": 0.2,
        "discriminator_hidden_dims": [256, 128],
        "replay_buffer_size": 10000,
        "num_preload_transitions": 4096,
        "preload_normalizer": True,
        "grad_penalty_coef": 0.1,
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


@configclass
class A1WMPAMPRewardCfg(A1AMPRewardCfg):
    """A1 WMP-AMP 地形奖励，对齐 bytedance/WMP A1AMPCfg。"""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"std": 0.15**0.5, "lin_vel_clip": 0.1},
    )
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=0.5, params={"std": 0.15**0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    energy = RewTerm(func=mdp.energy, weight=-1.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=0.0)
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
        func=mdp.wmp_feet_stumble,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*foot.*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)

    feet_edge = RewTerm(
        func=mdp.feet_edge,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*foot.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot.*"),
        },
    )
    dof_error = RewTerm(func=mdp.dof_error_l2, weight=-0.04)
    cheat = RewTerm(func=mdp.cheat_yaw, weight=-1.0, params={"heading_limit": 1.0})
    stuck = RewTerm(func=mdp.stuck, weight=-1.0, params={"velocity_threshold": 0.1, "command_threshold": 0.1})


@configclass
class A1WMPAMPTerrainEnvCfg(A1AMPFlatEnvCfg):
    """A1 完整 WMP-AMP 地形测试任务。"""

    reward = A1WMPAMPRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        wmp_update_interval = 5
        self.scene.robot = A1_CFG.replace(
            init_state=A1_CFG.init_state.replace(pos=(0.0, 0.0, 0.35)),
            actuators={
                "base_legs": A1_CFG.actuators["base_legs"].replace(
                    stiffness=40.0,
                    damping=1.0,
                )
            },
        )
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = WMP_MIXED_TERRAINS_CFG
        self.scene.max_init_terrain_level = 0
        self.scene.terrain_curriculum.enabled = True
        self.scene.terrain_curriculum.schedule = (0.0, 15000.0, 0.0, 9.0)
        self.scene.height_scanner.enable_height_scan = True
        self.scene.height_scanner.prim_body_name = "trunk"
        self.scene.rgbd_camera.enable = True
        self.scene.rgbd_camera.enable_rgb = False
        self.scene.rgbd_camera.enable_depth = True
        self.scene.rgbd_camera.partial_camera = True
        self.scene.rgbd_camera.partial_camera_num_envs = 1024
        self.scene.rgbd_camera.partial_camera_seed = 42
        self.scene.rgbd_camera.partial_camera_force_tilt_crawl = True
        self.scene.rgbd_camera.model_name = "wmp_front_depth"
        self.scene.rgbd_camera.camera_model = "pinhole"
        self.scene.rgbd_camera.spawn_prim_path = "trunk/rgbd_camera"
        self.scene.rgbd_camera.spawn_offset_pos = (0.3, 0.0, 0.03)
        self.scene.rgbd_camera.spawn_offset_rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.rgbd_camera.width = 64
        self.scene.rgbd_camera.height = 64
        self.scene.rgbd_camera.depth_near = 0.0
        self.scene.rgbd_camera.depth_far = 2.0
        self.scene.rgbd_camera.horizontal_fov_deg = 58.0
        self.scene.rgbd_camera.randomize_rotation = True
        self.scene.rgbd_camera.random_pitch_deg = (5.0, 15.0)
        self.scene.rgbd_camera.update_period = self.sim.dt * self.sim.decimation * wmp_update_interval
        self.scene.rgbd_camera.allow_missing_depth_fallback = False
        self.sim.render_interval = self.sim.decimation * wmp_update_interval
        self.robot.actor_obs_history_length = 5
        self.robot.critic_obs_history_length = 1
        self.robot.wmp_privileged_contact_body_names = [".*thigh.*", ".*calf.*"]
        self.robot.wmp_time_out_strictly_greater = True
        self.robot.terminate_on_wmp_velocity_violation = True
        self.robot.wmp_velocity_violation_threshold = 1.5
        self.robot.wmp_velocity_violation_min_terrain_level = 4
        self.robot.terminate_on_wmp_fall = True
        self.robot.wmp_fall_z_velocity_threshold = -3.0
        self.robot.wmp_fall_projected_gravity_z_threshold = 0.0
        self.normalization.obs_scales.ang_vel = 0.25
        self.normalization.obs_scales.joint_pos = 1.0
        self.normalization.obs_scales.joint_vel = 0.05
        self.normalization.obs_scales.height_scan = 5.0
        self.normalization.obs_scales.contact_force = 0.005
        self.normalization.obs_scales.pd_gains = 5.0
        self.normalization.obs_scales.com_pos = 20.0
        self.normalization.clip_actions = 6.0
        self.reward_settings.only_positive_rewards = True
        self.reward_settings.reward_curriculum = True
        self.reward_settings.reward_curriculum_term = ("feet_edge",)
        self.reward_settings.reward_curriculum_schedule = ((4000.0, 10000.0, 0.1, 1.0),)
        self.commands.heading_command = True
        self.commands.rel_heading_envs = 1.0
        self.commands.rel_standing_envs = 0.0
        self.commands.ranges.lin_vel_x = (0.0, 0.8)
        self.commands.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.ranges.heading = (0.0, 0.0)
        self.domain_rand.events.physics_material = EventTerm(
            func=mdp.wmp_recording_randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.5, 2.0),
                "dynamic_friction_range": (0.5, 2.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )
        self.domain_rand.events.add_base_mass = EventTerm(
            func=mdp.wmp_recording_randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*trunk.*"),
                "mass_distribution_params": (0.0, 3.0),
                "operation": "add",
                "recompute_inertia": True,
            },
        )
        self.domain_rand.events.add_link_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="(?!.*trunk.*).*"),
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )
        self.domain_rand.events.randomize_base_com = EventTerm(
            func=mdp.wmp_recording_randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*trunk.*"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )
        self.domain_rand.events.randomize_actuator_gains = EventTerm(
            func=mdp.wmp_recording_randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.8, 1.2),
                "operation": "scale",
            },
        )
        self.domain_rand.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(15.0, 15.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
        )
        self.domain_rand.action_delay.enable = True
        self.domain_rand.action_delay.params = {
            "max_delay": max(0, int((0.005 - 1.0e-8) / self.sim.dt) + 1),
            "min_delay": 0,
            "mode": "sim_step",
        }
        self.domain_rand.motor_strength.enable = True
        self.domain_rand.motor_strength.range = (0.8, 1.2)


@configclass
class A1WMPAMPTerrainAgentCfg(A1AMPFlatAgentCfg):
    """A1 完整 WMP-AMP 地形测试训练配置。"""

    experiment_name: str = "a1_wmp_amp_terrain"
    wandb_project: str = "a1_wmp_amp"
    runner_class_name: str = "legged_lab.runners.wmp_amp_runner:WMPAMPRunner"
    wmp: dict = {
        "feature_type": "deter",
        "update_interval": 5,
        "train_start_steps": 10000,
        "train_steps_per_iter": 10,
        "batch_size": 16,
        "batch_length": 64,
        "model_lr": 1.0e-4,
        "replay_capacity_episodes": 50000,
        "replay_device": "cpu",
        "use_depth_predictor": True,
        "camera_sample_all_envs": False,
        "camera_num_envs": 1024,
        "camera_env_seed": 42,
        "camera_force_tilt_crawl": True,
        "use_history_encoder": True,
        "history_steps": 5,
        "history_dim_per_step": 45,
        "history_encoder_hidden_dims": [256, 128],
        "history_latent_dim": 35,
        "height_scan_dim": 187,
        "command_start": 6,
        "command_dim": 3,
        "depth_predictor": {
            "training_interval": 10,
            "training_iters": 1000,
            "batch_size": 1024,
        },
    }
    amp: dict = {
        "motion_files": [],
        "canonical_obs_dim": 30,
        "retarget_adapter": {
            "class_path": "legged_lab.amp.retarget:A1CanonicalRetargetAdapter",
            "profile": "a1_canonical_v1",
            "target_joint_order": "env",
        },
        "num_preload_transitions": 2000000,
        "reward_coef": 0.01,
        "task_reward_lerp": 0.3,
        "discriminator_hidden_dims": [1024, 512],
        "replay_buffer_size": 1000000,
        "preload_normalizer": True,
        "grad_penalty_coef": 1.0,
        "min_normalized_std": [0.05, 0.02, 0.05] * 4,
    }

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "legged_lab.models:WMPMLPModel"
        self.policy.actor_hidden_dims = [256, 128, 64]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.class_name = "legged_lab.algorithms.wmp_amp_ppo:WMPAMPPPO"
        self.algorithm.learning_rate = 1.0e-3
        self.algorithm.schedule = "adaptive"
        self.algorithm.desired_kl = 0.01
        self.algorithm.entropy_coef = 0.01
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4
        self.algorithm.normalize_advantage_per_mini_batch = False
        self.algorithm.vel_predict_coef = 1.0
        self.algorithm.vel_target_start = 50
        self.algorithm.vel_target_dim = 3
        self.num_steps_per_env = 24
        self.save_interval = 1000
        self.obs_groups = {"actor": ["policy", "wmp"], "critic": ["critic", "wmp"]}
