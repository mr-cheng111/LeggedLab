# -*- coding: utf-8 -*-
"""B2 从 IsaacLab 迁移到 mjlab 的最小 sim2sim 验证脚本。

功能目标：
1. 读取 MuJoCo 的 B2 MJCF 模型；
2. 复用 IsaacLab 导出的 TorchScript 策略（policy.pt）做推理；
3. 对齐核心观测/命令（角速度、重力投影、速度指令、关节状态、上一步动作）；
4. 支持观测噪声、观测延迟、动作延迟。

注意：
- 这是“可运行脚手架”，用于快速验证，不是 1:1 的训练闭环复现。
- 建议先用 `--obs_delay_steps 0 --act_delay_steps 0 --disable_noise` 跑通，再逐步加噪声与延迟。
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Callable

import torch
from tensordict import TensorDict


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="B2 mjlab sim2sim play script.")
    parser.add_argument("--mjcf", type=str, required=True, help="B2 的 MJCF/XML 路径。")
    parser.add_argument(
        "--policy_jit",
        type=str,
        required=True,
        help="IsaacLab 导出的 TorchScript 策略路径（通常是 exported/policy.pt）。",
    )
    parser.add_argument("--num_envs", type=int, default=64, help="并行环境数量。")
    parser.add_argument("--device", type=str, default=None, help="运行设备，例如 cuda:0 / cpu。")
    parser.add_argument(
        "--viewer",
        type=str,
        default="auto",
        choices=("auto", "native", "viser"),
        help="可视化后端。",
    )
    parser.add_argument(
        "--disable_noise",
        action="store_true",
        help="关闭观测噪声（默认开启，模拟训练分布）。",
    )
    parser.add_argument(
        "--obs_delay_steps",
        type=int,
        default=0,
        help="观测延迟（控制步），会作用于主要本体观测项。",
    )
    parser.add_argument(
        "--act_delay_steps",
        type=int,
        default=0,
        help="动作延迟（物理步），通过 DelayedActuatorCfg 固定延迟实现。",
    )
    parser.add_argument(
        "--resample_time",
        type=float,
        default=10.0,
        help="速度指令重采样时间（秒），默认对齐 B2 训练配置。",
    )
    parser.add_argument("--lin_vel_x_min", type=float, default=-0.6, help="指令线速度 x 最小值。")
    parser.add_argument("--lin_vel_x_max", type=float, default=1.0, help="指令线速度 x 最大值。")
    parser.add_argument("--lin_vel_y_min", type=float, default=-0.5, help="指令线速度 y 最小值。")
    parser.add_argument("--lin_vel_y_max", type=float, default=0.5, help="指令线速度 y 最大值。")
    parser.add_argument("--ang_vel_z_min", type=float, default=-1.57, help="指令角速度 z 最小值。")
    parser.add_argument("--ang_vel_z_max", type=float, default=1.57, help="指令角速度 z 最大值。")
    parser.add_argument(
        "--no_heading_command",
        action="store_false",
        dest="heading_command",
        default=True,
        help="关闭 heading 命令（默认开启）。",
    )
    return parser


def _import_mjlab():
    try:
        import mujoco  # noqa: F401
        from mjlab.actuator import BuiltinPositionActuatorCfg
        from mjlab.actuator.delayed_actuator import DelayedActuatorCfg
        from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.envs import mdp as envs_mdp
        from mjlab.managers.action_manager import ActionTermCfg
        from mjlab.managers.command_manager import CommandTermCfg
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
        from mjlab.managers.reward_manager import RewardTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        from mjlab.managers.termination_manager import TerminationTermCfg
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.scene import SceneCfg
        from mjlab.sensor import ContactMatch, ContactSensorCfg
        from mjlab.sim import MujocoCfg, SimulationCfg
        from mjlab.tasks.velocity import mdp as vel_mdp
        from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
        from mjlab.terrains import TerrainEntityCfg
        from mjlab.utils.noise import UniformNoiseCfg as Unoise
        from mjlab.utils.os import update_assets
        from mjlab.viewer import NativeMujocoViewer, ViewerConfig, ViserPlayViewer
    except ImportError as exc:
        raise RuntimeError(
            "未检测到 mjlab/mujoco 依赖，请先安装 mjlab 再运行本脚本。"
        ) from exc

    return {
        "BuiltinPositionActuatorCfg": BuiltinPositionActuatorCfg,
        "DelayedActuatorCfg": DelayedActuatorCfg,
        "EntityArticulationInfoCfg": EntityArticulationInfoCfg,
        "EntityCfg": EntityCfg,
        "ManagerBasedRlEnv": ManagerBasedRlEnv,
        "envs_mdp": envs_mdp,
        "ActionTermCfg": ActionTermCfg,
        "CommandTermCfg": CommandTermCfg,
        "EventTermCfg": EventTermCfg,
        "ObservationGroupCfg": ObservationGroupCfg,
        "ObservationTermCfg": ObservationTermCfg,
        "RewardTermCfg": RewardTermCfg,
        "SceneEntityCfg": SceneEntityCfg,
        "TerminationTermCfg": TerminationTermCfg,
        "RslRlVecEnvWrapper": RslRlVecEnvWrapper,
        "SceneCfg": SceneCfg,
        "ContactMatch": ContactMatch,
        "ContactSensorCfg": ContactSensorCfg,
        "MujocoCfg": MujocoCfg,
        "SimulationCfg": SimulationCfg,
        "vel_mdp": vel_mdp,
        "UniformVelocityCommandCfg": UniformVelocityCommandCfg,
        "TerrainEntityCfg": TerrainEntityCfg,
        "Unoise": Unoise,
        "ViewerConfig": ViewerConfig,
        "NativeMujocoViewer": NativeMujocoViewer,
        "ViserPlayViewer": ViserPlayViewer,
        "update_assets": update_assets,
    }


def _make_b2_entity_cfg(args: argparse.Namespace, m: dict):
    import mujoco

    EntityCfg = m["EntityCfg"]
    EntityArticulationInfoCfg = m["EntityArticulationInfoCfg"]
    BuiltinPositionActuatorCfg = m["BuiltinPositionActuatorCfg"]
    DelayedActuatorCfg = m["DelayedActuatorCfg"]
    update_assets = m["update_assets"]

    mjcf_path = Path(args.mjcf).expanduser().resolve()
    if not mjcf_path.exists():
        raise FileNotFoundError(f"MJCF 文件不存在: {mjcf_path}")

    def _spec_fn() -> mujoco.MjSpec:
        spec = mujoco.MjSpec.from_file(str(mjcf_path))
        # 尝试自动打包外部资源（mesh/texture），避免相对路径丢失导致加载失败。
        meshdir = spec.meshdir
        if meshdir:
            mesh_root = (mjcf_path.parent / meshdir).resolve()
            if mesh_root.exists():
                assets: dict[str, bytes] = {}
                update_assets(assets, mesh_root, meshdir)
                if assets:
                    spec.assets = assets
        return spec

    hip_thigh_cfg = BuiltinPositionActuatorCfg(
        target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
        stiffness=160.0,
        damping=5.0,
        effort_limit=200.0,
    )
    calf_cfg = BuiltinPositionActuatorCfg(
        target_names_expr=(".*_calf_joint",),
        stiffness=160.0,
        damping=5.0,
        effort_limit=320.0,
    )

    actuator_cfgs = []
    for base_cfg in (hip_thigh_cfg, calf_cfg):
        if args.act_delay_steps > 0:
            actuator_cfgs.append(
                DelayedActuatorCfg(
                    base_cfg=base_cfg,
                    delay_target="position",
                    delay_min_lag=args.act_delay_steps,
                    delay_max_lag=args.act_delay_steps,
                )
            )
        else:
            actuator_cfgs.append(base_cfg)

    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.58),
            joint_pos={
                ".*R_hip_joint": -0.1,
                ".*L_hip_joint": 0.1,
                "F[L,R]_thigh_joint": 0.8,
                "R[L,R]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
            joint_vel={".*": 0.0},
        ),
        spec_fn=_spec_fn,
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(actuator_cfgs),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


def _make_obs_term(
    ObservationTermCfg,
    func: Callable,
    *,
    params: dict | None = None,
    noise=None,
    obs_delay_steps: int = 0,
):
    return ObservationTermCfg(
        func=func,
        params=params or {},
        noise=noise,
        delay_min_lag=max(obs_delay_steps, 0),
        delay_max_lag=max(obs_delay_steps, 0),
    )


def _build_env_cfg(args: argparse.Namespace, m: dict):
    ManagerBasedRlEnv = m["ManagerBasedRlEnv"]
    del ManagerBasedRlEnv  # 仅用于类型语义，避免未使用告警。
    envs_mdp = m["envs_mdp"]
    vel_mdp = m["vel_mdp"]
    ObservationTermCfg = m["ObservationTermCfg"]
    ObservationGroupCfg = m["ObservationGroupCfg"]
    JointPositionActionCfg = __import__(
        "mjlab.envs.mdp.actions", fromlist=["JointPositionActionCfg"]
    ).JointPositionActionCfg
    SceneCfg = m["SceneCfg"]
    ContactMatch = m["ContactMatch"]
    ContactSensorCfg = m["ContactSensorCfg"]
    UniformVelocityCommandCfg = m["UniformVelocityCommandCfg"]
    TerrainEntityCfg = m["TerrainEntityCfg"]
    ViewerConfig = m["ViewerConfig"]
    SimulationCfg = m["SimulationCfg"]
    MujocoCfg = m["MujocoCfg"]
    RewardTermCfg = m["RewardTermCfg"]
    TerminationTermCfg = m["TerminationTermCfg"]
    Unoise = m["Unoise"]
    SceneEntityCfg = m["SceneEntityCfg"]

    robot_cfg = _make_b2_entity_cfg(args, m)

    obs_noise_enable = not args.disable_noise
    obs_delay_steps = max(args.obs_delay_steps, 0)

    actor_terms = {
        "base_ang_vel": _make_obs_term(
            ObservationTermCfg,
            envs_mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            obs_delay_steps=obs_delay_steps,
        ),
        "projected_gravity": _make_obs_term(
            ObservationTermCfg,
            envs_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            obs_delay_steps=obs_delay_steps,
        ),
        "command": _make_obs_term(
            ObservationTermCfg,
            envs_mdp.generated_commands,
            params={"command_name": "twist"},
            obs_delay_steps=0,
        ),
        "joint_pos": _make_obs_term(
            ObservationTermCfg,
            envs_mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            obs_delay_steps=obs_delay_steps,
        ),
        "joint_vel": _make_obs_term(
            ObservationTermCfg,
            envs_mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            obs_delay_steps=obs_delay_steps,
        ),
        "actions": _make_obs_term(
            ObservationTermCfg,
            envs_mdp.last_action,
            obs_delay_steps=0,
        ),
    }
    critic_terms = {
        **actor_terms,
        "base_lin_vel": _make_obs_term(
            ObservationTermCfg,
            envs_mdp.base_lin_vel,
            obs_delay_steps=0,
        ),
        "feet_contact": _make_obs_term(
            ObservationTermCfg,
            vel_mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
            obs_delay_steps=0,
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=obs_noise_enable,
            history_length=10,
            flatten_history_dim=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=10,
            flatten_history_dim=True,
        ),
    }

    # 对齐 IsaacLab B2：关节目标位置动作 + 全局 action_scale=0.25。
    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.25,
            use_default_offset=True,
        )
    }

    commands = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(args.resample_time, args.resample_time),
            rel_standing_envs=0.2,
            rel_heading_envs=1.0,
            rel_world_envs=0.0,
            rel_forward_envs=0.0,
            heading_command=args.heading_command,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(args.lin_vel_x_min, args.lin_vel_x_max),
                lin_vel_y=(args.lin_vel_y_min, args.lin_vel_y_max),
                ang_vel_z=(args.ang_vel_z_min, args.ang_vel_z_max),
                heading=(-math.pi, math.pi) if args.heading_command else None,
            ),
        )
    }

    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=vel_mdp.track_linear_velocity,
            weight=1.5,
            params={"command_name": "twist", "std": 0.5},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=vel_mdp.track_angular_velocity,
            weight=0.75,
            params={"command_name": "twist", "std": 0.5},
        ),
        "action_rate_l2": RewardTermCfg(func=vel_mdp.action_rate_l2, weight=-0.1),
        "dof_pos_limits": RewardTermCfg(func=vel_mdp.joint_pos_limits, weight=-10.0),
        # 注意：为避免和不同 MJCF 的足端/站点命名冲突，这里不启用高度扫描类奖励项。
    }

    terminations = {
        "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=vel_mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="body", pattern=".*foot.*", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    return __import__("mjlab.envs", fromlist=["ManagerBasedRlEnvCfg"]).ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot_cfg},
            sensors=(feet_ground_cfg,),
            num_envs=args.num_envs,
            extent=2.5,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events={},
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        metrics={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.AUTO,
            entity_name="robot",
            distance=2.4,
            elevation=-10.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=300,
            njmax=2000,
            contact_sensor_maxmatch=512,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=40.0,
    )


class TorchScriptPolicyAdapter:
    """把 TorchScript 策略适配到 mjlab viewer 的 policy 调用协议。"""

    def __init__(self, policy_module: torch.jit.ScriptModule, obs_key: str = "actor"):
        self.policy = policy_module
        self.obs_key = obs_key

    def __call__(self, obs):
        if isinstance(obs, TensorDict):
            if self.obs_key in obs.keys():
                obs_tensor = obs[self.obs_key]
            elif "policy" in obs.keys():
                # 兼容部分环境仍使用 "policy" 作为 actor 观测键名的场景。
                obs_tensor = obs["policy"]
            else:
                raise KeyError(f"观测中未找到 '{self.obs_key}' 或 'policy' 键。")
        else:
            obs_tensor = obs
        return self.policy(obs_tensor)


def main():
    args = _build_argparser().parse_args()
    m = _import_mjlab()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_path = Path(args.policy_jit).expanduser().resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"策略文件不存在: {policy_path}")

    env_cfg = _build_env_cfg(args, m)
    env = m["ManagerBasedRlEnv"](cfg=env_cfg, device=device)
    env = m["RslRlVecEnvWrapper"](env, clip_actions=None)

    policy_module = torch.jit.load(str(policy_path), map_location=device)
    policy_module.eval()
    policy = TorchScriptPolicyAdapter(policy_module, obs_key="actor")

    if args.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = args.viewer

    print("[INFO] Launch mjlab sim2sim")
    print(f"[INFO] device={device}, viewer={resolved_viewer}, num_envs={args.num_envs}")
    print(
        f"[INFO] noise={'off' if args.disable_noise else 'on'}, "
        f"obs_delay_steps={args.obs_delay_steps}, act_delay_steps={args.act_delay_steps}"
    )
    print(f"[INFO] mjcf={Path(args.mjcf).expanduser().resolve()}")
    print(f"[INFO] policy={policy_path}")

    if resolved_viewer == "native":
        m["NativeMujocoViewer"](env, policy).run()
    elif resolved_viewer == "viser":
        m["ViserPlayViewer"](env, policy).run()
    else:
        raise RuntimeError(f"不支持的 viewer: {resolved_viewer}")

    env.close()


if __name__ == "__main__":
    main()
