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
from dataclasses import MISSING

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp


@configclass
class RewardCfg:
    pass


@configclass
class RewardSettingsCfg:
    """环境层奖励后处理配置。

    IsaacLab 的 RewardManager 只接受 RewardTermCfg 字段，因此原版 WMP 的
    `only_positive_rewards` 与 reward curriculum 放在独立配置里处理。
    """

    only_positive_rewards: bool = False
    reward_curriculum: bool = False
    reward_curriculum_term: tuple[str, ...] = ()
    reward_curriculum_schedule: tuple[tuple[float, float, float, float], ...] = ()


@configclass
class HeightScannerCfg:
    enable_height_scan: bool = False
    prim_body_name: str = MISSING
    resolution: float = 0.1
    size: tuple = (1.6, 1.0)
    forward_resolution: float = 0.1
    forward_size: tuple = (2.0, 2.4)
    debug_vis: bool = False
    drift_range: tuple = (0.0, 0.0)


@configclass
class Gemini2CameraCfg:
    """RGBD 相机配置。

    RGBD 相机统一在 InteractiveScene clone 完成后生成到机器人 body 下。
    WMP 可通过 partial_camera 只为部分 env 生成真实相机。
    """

    enable: bool = False
    enable_rgb: bool = True
    enable_depth: bool = True
    model_name: str = "wmp_front_depth"
    camera_model: str = "pinhole"
    width: int = 64
    height: int = 64
    update_period: float = 0.0
    depth_near: float = 0.0
    depth_far: float = 2.0
    render_depth_near: float = 0.01
    render_depth_far: float | None = None
    depth_clipping_behavior: str = "none"
    allow_missing_depth_fallback: bool = False
    partial_camera: bool = False
    partial_camera_num_envs: int | None = 1024
    partial_camera_seed: int = 42
    partial_camera_force_tilt_crawl: bool = True
    spawn_prim_path: str = "base/wmp_depth_camera"
    spawn_offset_pos: tuple[float, float, float] = (0.27, 0.0, 0.03)
    spawn_offset_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    spawn_offset_convention: str = "world"
    horizontal_aperture: float = 20.955
    vertical_aperture: float | None = None
    horizontal_fov_deg: float | None = 58.0
    randomize_rotation: bool = True
    randomize_rotation_seed: int | None = None
    random_roll_deg: tuple[float, float] = (0.0, 0.0)
    random_pitch_deg: tuple[float, float] = (-5.0, 5.0)
    random_yaw_deg: tuple[float, float] = (0.0, 0.0)
    focal_length: float = 18.9002
    focus_distance: float = 400.0


@configclass
class BaseSceneCfg:
    max_episode_length_s: float = 20.0
    num_envs: int = 4096
    env_spacing: float = 2.5
    robot: ArticulationCfg = MISSING
    terrain_type: str = MISSING
    terrain_generator: TerrainGeneratorCfg = None
    max_init_terrain_level: int = 5
    height_scanner: HeightScannerCfg = HeightScannerCfg()
    gemini2_camera: Gemini2CameraCfg = Gemini2CameraCfg()


@configclass
class RobotCfg:
    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    action_scale: float = 0.25
    wheel_velocity_scale: float = 8.0
    terminate_contacts_body_names: list = []
    feet_body_names: list = []
    terminate_on_flight: bool = False
    terminate_on_flight_threshold: float = 1.0


@configclass
class ObsScalesCfg:
    lin_vel: float = 1.0
    ang_vel: float = 1.0
    projected_gravity: float = 1.0
    commands: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 1.0
    actions: float = 1.0
    height_scan: float = 1.0


@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    height_scan_offset: float = 0.5


@configclass
class CommandRangesCfg:
    lin_vel_x: tuple = (-0.6, 1.0)
    lin_vel_y: tuple = (-0.5, 0.5)
    ang_vel_z: tuple = (-1.0, 1.0)
    heading: tuple = (-math.pi, math.pi)


@configclass
class CommandsCfg:
    resampling_time_range: tuple = (10.0, 10.0)
    rel_standing_envs: float = 0.2
    rel_heading_envs: float = 1.0
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    debug_vis: bool = True
    ranges: CommandRangesCfg = CommandRangesCfg()


@configclass
class NoiseScalesCfg:
    ang_vel: float = 0.2
    projected_gravity: float = 0.05
    joint_pos: float = 0.01
    joint_vel: float = 1.5
    height_scan: float = 0.1


@configclass
class NoiseCfg:
    add_noise: bool = True
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class ActionDelayCfg:
    enable: bool = False
    params: dict = {"max_delay": 5, "min_delay": 0}


@configclass
class MotorStrengthCfg:
    enable: bool = False
    range: tuple[float, float] = (1.0, 1.0)


@configclass
class DomainRandCfg:
    events: EventCfg = EventCfg()
    action_delay: ActionDelayCfg = ActionDelayCfg()
    motor_strength: MotorStrengthCfg = MotorStrengthCfg()


@configclass
class PhysxCfg:
    gpu_max_rigid_patch_count: int = 10 * 2**15


@configclass
class SimCfg:
    dt: float = 0.005
    decimation: int = 4
    render_interval: int | None = None
    physx: PhysxCfg = PhysxCfg()
