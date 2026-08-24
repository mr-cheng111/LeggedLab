"""DEEP Robotics M20 wheel-legged velocity-tracking tasks."""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.assets.deeprobotics import M20_CFG
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg, RewardCfg
from legged_lab.terrains import M20_ALL_TERRAINS_CFG


M20_LEG_JOINT_NAMES = [
    "fl_hipx_joint",
    "fl_hipy_joint",
    "fl_knee_joint",
    "fr_hipx_joint",
    "fr_hipy_joint",
    "fr_knee_joint",
    "hl_hipx_joint",
    "hl_hipy_joint",
    "hl_knee_joint",
    "hr_hipx_joint",
    "hr_hipy_joint",
    "hr_knee_joint",
]
M20_WHEEL_JOINT_NAMES = ["fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint"]
M20_LEG_JOINTS = [".*_hipx_joint", ".*_hipy_joint", ".*_knee_joint"]
M20_HIPX_JOINTS = [".*_hipx_joint"]
M20_WHEEL_JOINTS = [".*_wheel_joint"]
M20_WHEEL_BODIES = ".*_wheel"
M20_AMP_ENABLED = False


@configclass
class M20RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=5.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=3.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    leg_energy = RewTerm(
        func=mdp.energy,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=M20_LEG_JOINTS)},
    )
    leg_dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=M20_LEG_JOINTS)},
    )
    wheel_dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=M20_WHEEL_JOINTS)},
    )
    leg_action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_joint,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=M20_LEG_JOINTS)},
    )
    wheel_action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_joint,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=M20_WHEEL_JOINTS)},
    )
    hipx_action_l2 = RewTerm(
        func=mdp.action_l2_joint,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=M20_HIPX_JOINTS)},
    )
    stand_still = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-1.0,
        params={
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=M20_LEG_JOINTS),
        },
    )
    leg_deviation_l2 = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=M20_LEG_JOINTS)},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=M20_LEG_JOINTS)},
    )
    fly = RewTerm(
        func=mdp.fly,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=M20_WHEEL_BODIES), "threshold": 1.0},
    )
    wheel_contact_count = RewTerm(
        func=mdp.feet_contact_count,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=M20_WHEEL_BODIES), "threshold": 1.0},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["base_link", ".*_hipx", ".*_hipy", ".*_knee"]
            ),
            "threshold": 1.0,
        },
    )
    wheel_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=M20_WHEEL_BODIES)},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-5.0)


@configclass
class M20FlatEnvCfg(BaseEnvCfg):
    reward = M20RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = M20_CFG
        self.scene.terrain_type = "plane"
        self.scene.terrain_generator = None
        self.scene.height_scanner.enable_height_scan = False
        self.scene.height_scanner.prim_body_name = "base_link"
        self.scene.rgbd_camera.enable = False
        self.scene.env_spacing = 3.0

        self.robot.terminate_contacts_body_names = ["base_link"]
        self.robot.feet_body_names = [M20_WHEEL_BODIES]
        self.robot.actor_obs_history_length = 1
        self.robot.critic_obs_history_length = 1
        self.robot.action_scale = 0.25
        self.robot.leg_position_scale = {".*_hipx_joint": 0.125}
        self.robot.wheel_velocity_scale = 5.0
        self.robot.wheel_joint_names_expr = ".*_wheel_joint"
        self.robot.policy_joint_names = M20_LEG_JOINT_NAMES + M20_WHEEL_JOINT_NAMES

        self.normalization.obs_scales.lin_vel = 2.0
        self.normalization.obs_scales.ang_vel = 0.25
        self.normalization.obs_scales.projected_gravity = 1.0
        self.normalization.obs_scales.commands = 1.0
        self.normalization.obs_scales.joint_pos = 1.0
        self.normalization.obs_scales.joint_vel = 0.05
        self.normalization.obs_scales.actions = 1.0
        self.normalization.clip_observations = 100.0
        self.normalization.clip_actions = 100.0

        self.commands.resampling_time_range = (8.0, 10.0)
        self.commands.heading_command = False
        self.commands.debug_vis = False
        self.commands.rel_standing_envs = 0.2
        self.commands.rel_heading_envs = 0.0
        self.commands.ranges.lin_vel_x = (-2.0, 2.0)
        self.commands.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.ranges.ang_vel_z = (-2.0, 2.0)
        self.commands.ranges.heading = (-math.pi, math.pi)

        self.domain_rand.events.physics_material.params["static_friction_range"] = (0.35, 1.5)
        self.domain_rand.events.physics_material.params["dynamic_friction_range"] = (0.35, 1.5)
        self.domain_rand.events.physics_material.params["restitution_range"] = (0.0, 0.2)
        self.domain_rand.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.domain_rand.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 3.0)
        self.domain_rand.events.add_base_mass.params["recompute_inertia"] = True
        self.domain_rand.events.reset_base.params["pose_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-math.pi, math.pi),
        }
        self.domain_rand.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)


@configclass
class M20FlatAgentCfg(BaseAgentCfg):
    experiment_name: str = "m20_flat"
    wandb_project: str = "m20_flat"

    def __post_init__(self):
        super().__post_init__()
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.entropy_coef = 0.003
        self.num_steps_per_env = 24
        self.max_iterations = 5000
        self.save_interval = 500


@configclass
class M20RoughEnvCfg(M20FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = M20_ALL_TERRAINS_CFG
        self.scene.max_init_terrain_level = 0
        self.scene.terrain_curriculum.enabled = True
        self.scene.terrain_curriculum.schedule = (0.0, 15000.0, 0.0, 9.0)
        self.scene.height_scanner.enable_height_scan = True


@configclass
class M20RoughAgentCfg(M20FlatAgentCfg):
    experiment_name: str = "m20_rough"
    wandb_project: str = "m20_rough"

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 20000


@configclass
class M20DepthRoughEnvCfg(M20RoughEnvCfg):
    """M20 rough-terrain task supervised by a sparse set of depth cameras."""

    def __post_init__(self):
        super().__post_init__()
        wmp_update_interval = 5
        self.scene.rgbd_camera.enable = True
        self.scene.rgbd_camera.enable_rgb = False
        self.scene.rgbd_camera.enable_depth = True
        self.scene.rgbd_camera.partial_camera = True
        self.scene.rgbd_camera.partial_camera_num_envs = 256
        self.scene.rgbd_camera.partial_camera_seed = 42
        self.scene.rgbd_camera.partial_camera_force_tilt_crawl = False
        self.scene.rgbd_camera.model_name = "m20_front_depth"
        self.scene.rgbd_camera.camera_model = "pinhole"
        self.scene.rgbd_camera.spawn_prim_path = "base_link/depth_camera"
        self.scene.rgbd_camera.spawn_offset_pos = (0.38, 0.0, 0.10)
        # Quaternion order is wxyz. The previous (0, 0, 0, 1) was a 180-degree
        # yaw that pointed the camera backward; identity faces robot-forward.
        self.scene.rgbd_camera.spawn_offset_rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.rgbd_camera.width = 64
        self.scene.rgbd_camera.height = 64
        self.scene.rgbd_camera.depth_near = 0.05
        self.scene.rgbd_camera.depth_far = 3.0
        self.scene.rgbd_camera.horizontal_fov_deg = 70.0
        self.scene.rgbd_camera.randomize_rotation = True
        self.scene.rgbd_camera.random_pitch_deg = (5.0, 15.0)
        self.scene.rgbd_camera.update_period = self.sim.dt * self.sim.decimation * wmp_update_interval
        self.scene.rgbd_camera.allow_missing_depth_fallback = False
        self.sim.render_interval = self.sim.decimation * wmp_update_interval
        self.robot.actor_obs_history_length = 5
        self.robot.critic_obs_history_length = 1


@configclass
class M20DepthRoughAgentCfg(M20RoughAgentCfg):
    """M20 depth-supervised WMP-PPO training configuration."""

    experiment_name: str = "m20_depth_rough"
    wandb_project: str = "m20_depth_rough"
    runner_class_name: str = "legged_lab.runners.wmp_amp_runner:WMPRunner"
    wmp: dict = {
        "feature_type": "deter",
        "update_interval": 5,
        "train_start_steps": 10000,
        "train_steps_per_iter": 10,
        "batch_size": 16,
        "batch_length": 64,
        "model_lr": 1.0e-4,
        "replay_capacity_episodes": 50000,
        "replay_device": "cuda:0",
        "use_depth_predictor": True,
        "camera_sample_all_envs": False,
        "camera_num_envs": 256,
        "camera_env_seed": 42,
        "camera_force_tilt_crawl": False,
        "use_history_encoder": True,
        "history_steps": 5,
        "history_dim_per_step": 57,
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

    def __post_init__(self):
        super().__post_init__()
        self.policy.class_name = "legged_lab.models:WMPMLPModel"
        self.policy.actor_hidden_dims = [256, 128, 64]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.entropy_coef = 0.005
        self.num_steps_per_env = 24
        self.max_iterations = 20000
        self.save_interval = 500
        self.obs_groups = {"actor": ["policy", "wmp"], "critic": ["critic", "wmp"]}


@configclass
class M20DepthRoughAMPRewardCfg(M20RewardCfg):
    """Stage-2 rewards for adding leg assistance on WMP terrain."""

    # Keep rolling as the default behavior. Leg motion is made affordable, but
    # wheel air time is not globally rewarded because it caused needless gaiting.
    wheel_contact_count = RewTerm(
        func=mdp.feet_contact_count,
        weight=0.10,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=M20_WHEEL_BODIES), "threshold": 1.0},
    )
    wheel_air_time = RewTerm(
        func=mdp.feet_air_time_quadruped,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=M20_WHEEL_BODIES), "threshold": 0.2},
    )
    wheel_action_l2 = RewTerm(
        func=mdp.action_l2_joint,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=M20_WHEEL_JOINTS)},
    )
    # Relax the stage-1 posture constraint enough for obstacle negotiation.
    leg_deviation_l2 = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=M20_LEG_JOINTS)},
    )


@configclass
class M20DepthRoughAMPEnvCfg(M20DepthRoughEnvCfg):
    """M20 depth task retaining optional canonical 12-leg-joint AMP observations."""

    reward = M20DepthRoughAMPRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.robot.enable_amp_observations = M20_AMP_ENABLED


@configclass
class M20DepthRoughAMPAgentCfg(M20DepthRoughAgentCfg):
    """M20 WMP training with AMP retained behind a temporary feature flag."""

    experiment_name: str = "m20_depth_rough_amp"
    wandb_project: str = "m20_depth_rough_amp"
    runner_class_name: str = "legged_lab.runners.wmp_amp_runner:WMPAMPRunner"
    amp: dict = {
        "enabled": M20_AMP_ENABLED,
        "motion_files": [
            "datasets/retargeted/m20/hop1_left.txt",
            "datasets/retargeted/m20/hop1_right.txt",
            "datasets/retargeted/m20/hop2_left.txt",
            "datasets/retargeted/m20/hop2_right.txt",
            "datasets/retargeted/m20/trot1_left.txt",
            "datasets/retargeted/m20/trot1_right.txt",
            "datasets/retargeted/m20/trot2_left.txt",
            "datasets/retargeted/m20/trot2_right.txt",
        ],
        "canonical_obs_dim": 30,
        "retarget_adapter": {
            "class_path": "legged_lab.amp.retarget:NoOpRetargetAdapter",
        },
        "num_preload_transitions": 200000,
        "reward_coef": 0.01,
        "task_reward_lerp": 0.3,
        "discriminator_hidden_dims": [1024, 512],
        "replay_buffer_size": 200000,
        "preload_normalizer": True,
        "grad_penalty_coef": 1.0,
    }

    def __post_init__(self):
        super().__post_init__()
        if not self.amp.get("enabled", True):
            self.experiment_name = "m20_depth_rough_no_amp"
            self.wandb_project = "m20_depth_rough_no_amp"
            self.runner_class_name = "legged_lab.runners.wmp_amp_runner:WMPRunner"
            self.algorithm.class_name = "PPO"
            return

        self.algorithm.class_name = "legged_lab.algorithms.wmp_amp_ppo:WMPAMPPPO"
        self.algorithm.learning_rate = 1.0e-3
        self.algorithm.schedule = "adaptive"
        self.algorithm.desired_kl = 0.01
        self.algorithm.entropy_coef = 0.01
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4
        self.algorithm.normalize_advantage_per_mini_batch = False
