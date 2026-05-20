# -*- coding: utf-8 -*-
"""检查 RB160W 模型的初始关节角、关节限位和单关节运动方向。"""

import argparse
import math
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

parser = argparse.ArgumentParser(description="Inspect RB160W model joint defaults and motion directions.")
parser.add_argument("--task", type=str, default="rb160w_flat", help="Task name.")
parser.add_argument(
    "--mode",
    choices=("print", "hold", "set", "sweep"),
    default="print",
    help="Print metadata, hold initial pose, assign a fixed joint angle, or sweep joints.",
)
parser.add_argument(
    "--joint",
    type=str,
    default=".*_hip_joint",
    help="Regex used by IsaacLab find_joints(). Examples: FR_hip_joint, .*_hip_joint, .*_thigh_joint.",
)
parser.add_argument("--amplitude", type=float, default=0.15, help="Sweep amplitude in radians.")
parser.add_argument("--angle", type=float, default=0.15, help="Fixed joint angle offset in radians for --mode set.")
parser.add_argument("--cycles", type=float, default=2.0, help="Number of sine cycles per joint.")
parser.add_argument("--steps", type=int, default=240, help="Simulation steps per swept joint.")
parser.add_argument("--settle_steps", type=int, default=60, help="Zero-action warmup steps before inspection.")
parser.add_argument("--free_root", action="store_true", help="Do not pin the root pose during inspection.")
parser.add_argument("--urdf", type=str, default=None, help="Optional URDF path for axis reporting.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs import *  # noqa:F401,F403


def _log(message: str):
    print(f"[INFO] {message}", flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_urdf_path() -> Path:
    return _repo_root() / "legged_lab/assets/xuanji/rb160w/urdf/RB160W_isaaclab_actuated.urdf"


def _read_urdf_axes(urdf_path: Path) -> dict[str, str]:
    if not urdf_path.exists():
        return {}
    root = ET.parse(urdf_path).getroot()
    axes = {}
    for joint in root.findall("joint"):
        axis = joint.find("axis")
        if axis is not None:
            axes[joint.attrib.get("name", "")] = axis.attrib.get("xyz", "")
    return axes


def _set_deterministic_reset(env_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.scene.seed = 42
    env_cfg.noise.add_noise = False
    env_cfg.commands.debug_vis = False
    env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.domain_rand.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    env_cfg.domain_rand.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.domain_rand.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    if args_cli.device is not None:
        env_cfg.device = args_cli.device


def _print_joint_table(env, urdf_axes: dict[str, str]):
    names = env.robot.joint_names
    default_pos = env.robot.data.default_joint_pos[0].detach().cpu()
    current_pos = env.robot.data.joint_pos[0].detach().cpu()
    limits = env.robot.data.joint_pos_limits[0].detach().cpu()
    soft_limits = env.robot.data.soft_joint_pos_limits[0].detach().cpu()

    print("\n[INFO] RB160W joint table")
    print("idx  joint_name          group     urdf_axis   default(rad)  default(deg)  current(rad)  limit(rad)           soft_limit(rad)")
    for idx, name in enumerate(names):
        if "_hip_joint" in name:
            group = "shoulder/hip"
        elif "_thigh_joint" in name:
            group = "hip_pitch"
        elif "_calf_joint" in name:
            group = "knee"
        elif "_WHEEL_joint" in name:
            group = "wheel"
        else:
            group = "other"
        print(
            f"{idx:>3}  {name:<18} {group:<12} {urdf_axes.get(name, '-'):>9} "
            f"{default_pos[idx].item():>12.6f} {math.degrees(default_pos[idx].item()):>12.3f} "
            f"{current_pos[idx].item():>12.6f} "
            f"[{limits[idx, 0].item():>8.4f}, {limits[idx, 1].item():>8.4f}] "
            f"[{soft_limits[idx, 0].item():>8.4f}, {soft_limits[idx, 1].item():>8.4f}]"
        )


def _pin_root(env, root_state: torch.Tensor | None):
    if root_state is None:
        return
    env.robot.write_root_state_to_sim(root_state)


def _advance_model_step(env, root_state: torch.Tensor | None):
    _pin_root(env, root_state)
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    _pin_root(env, root_state)
    if not env.headless:
        env.sim.render()


def _zero_model_steps(env, steps: int, root_state: torch.Tensor | None):
    default_pos = env.robot.data.default_joint_pos.clone()
    zero_vel = torch.zeros_like(default_pos)
    for _ in range(max(steps, 0)):
        env.robot.write_joint_position_to_sim(default_pos)
        env.robot.write_joint_velocity_to_sim(zero_vel)
        env.robot.set_joint_position_target(default_pos)
        _advance_model_step(env, root_state)


def _hold_initial_pose(env, root_state: torch.Tensor | None):
    _log("holding initial joint angles; close the IsaacSim window to stop")
    if args_cli.headless:
        _zero_model_steps(env, args_cli.steps, root_state)
        return
    while simulation_app.is_running():
        _zero_model_steps(env, 1, root_state)


def _assign_joint_targets(env, joint_ids: list[int], target: torch.Tensor, root_state: torch.Tensor | None):
    env.robot.write_joint_position_to_sim(target[:, joint_ids], joint_ids=joint_ids)
    env.robot.write_joint_velocity_to_sim(torch.zeros_like(target[:, joint_ids]), joint_ids=joint_ids)
    env.robot.set_joint_position_target(target[:, joint_ids], joint_ids=joint_ids)
    _advance_model_step(env, root_state)


def _set_joints(env, joint_pattern: str, root_state: torch.Tensor | None):
    joint_ids, joint_names = env.robot.find_joints(joint_pattern)
    if not joint_ids:
        raise RuntimeError(f"No joints matched pattern: {joint_pattern}")

    base_pos = env.robot.data.default_joint_pos.clone()
    target = base_pos.clone()
    for joint_id in joint_ids:
        target[:, joint_id] = base_pos[:, joint_id] + args_cli.angle

    _log(f"setting joints and holding: {list(zip(joint_ids, joint_names))}, offset={args_cli.angle} rad")
    for _ in range(args_cli.steps):
        _assign_joint_targets(env, joint_ids, target, root_state)


def _sweep_joints(env, joint_pattern: str, root_state: torch.Tensor | None):
    joint_ids, joint_names = env.robot.find_joints(joint_pattern)
    if not joint_ids:
        raise RuntimeError(f"No joints matched pattern: {joint_pattern}")

    _log(f"sweeping joints: {list(zip(joint_ids, joint_names))}")
    base_pos = env.robot.data.default_joint_pos.clone()
    wheel_ids = set(getattr(env, "wheel_joint_ids", []))
    dt = env.physics_dt

    for joint_id, joint_name in zip(joint_ids, joint_names):
        _log(f"sweep joint={joint_name}, amplitude={args_cli.amplitude} rad")
        for step_idx in range(args_cli.steps):
            phase = 2.0 * math.pi * args_cli.cycles * step_idx / max(args_cli.steps - 1, 1)
            # 单关节正弦验证: q_target = q0 + A * sin(phase)
            # 其中 q0 是模型默认角，A 是 --amplitude，用于观察正方向是否符合机械定义。
            target = base_pos.clone()
            value = args_cli.amplitude * math.sin(phase)
            target[:, joint_id] = base_pos[:, joint_id] + value
            if joint_id in wheel_ids:
                env.robot.set_joint_velocity_target(
                    torch.full((env.num_envs, 1), value / max(dt, 1.0e-6), device=env.device),
                    joint_ids=[joint_id],
                )
                _advance_model_step(env, root_state)
            else:
                _assign_joint_targets(env, [joint_id], target, root_state)


def main():
    env_cfg, _ = task_registry.get_cfgs(args_cli.task)
    _set_deterministic_reset(env_cfg)
    urdf_path = Path(args_cli.urdf).expanduser() if args_cli.urdf else _default_urdf_path()
    urdf_axes = _read_urdf_axes(urdf_path)

    _log(f"constructing task={args_cli.task}")
    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)
    exit_code = 1
    try:
        env.sim.set_camera_view(eye=[2.0, -2.0, 1.2], target=[0.0, 0.0, 0.45])
        root_state = None if args_cli.free_root else env.robot.data.root_state_w.clone()
        if root_state is not None:
            root_state[:, 7:] = 0.0
            _pin_root(env, root_state)
        _zero_model_steps(env, args_cli.settle_steps, root_state)
        _log(f"URDF axis source: {urdf_path}")
        _print_joint_table(env, urdf_axes)
        if root_state is not None:
            _log("root is pinned to the initial pose during inspection")
        if args_cli.mode == "hold":
            _hold_initial_pose(env, root_state)
        if args_cli.mode == "set":
            _set_joints(env, args_cli.joint, root_state)
        if args_cli.mode == "sweep":
            _sweep_joints(env, args_cli.joint, root_state)
        exit_code = 0
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if hasattr(env, "close"):
            env.close()
        if args_cli.headless:
            _log(f"terminating headless IsaacSim process with exit_code={exit_code}")
            os._exit(exit_code)


if __name__ == "__main__":
    main()
    simulation_app.close()
