# -*- coding: utf-8 -*-
"""Xuanji / Astrall RB160W robot assets."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR


RB160W_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=(
            f"{ISAAC_ASSET_DIR}/xuanji/rb160w/usd/rb160w.usd"
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.72),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_WHEEL_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=11.5,
            stiffness={
                ".*_hip_joint": 70.0,
                ".*_thigh_joint": 100.0,
                ".*_calf_joint": 100.0,
            },
            damping={
                ".*_hip_joint": 3.0,
                ".*_thigh_joint": 4.0,
                ".*_calf_joint": 4.0,
            },
            armature=0.01,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_WHEEL_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=11.5,
            stiffness=0.0,
            damping=1.0,
            armature=0.005,
        ),
    },
)
