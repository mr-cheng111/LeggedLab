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

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import push_by_setting_velocity
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv


def push_by_setting_velocity_with_recovery_marker(
    env: BaseEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """施加速度推扰，并记录恢复计时起点。

    该函数包装 IsaacLab 原生 `push_by_setting_velocity`。推扰发生时记录当前 episode step：
        t_push[i] = episode_length_buf[i]
    后续 `push_recovery_time_exp` 使用 Δt = (episode_step - t_push) * step_dt 计算恢复耗时奖励。
    """
    push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
    if not hasattr(env, "last_push_step_buf"):
        env.last_push_step_buf = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        env.push_recovered_buf = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    env.last_push_step_buf[env_ids] = env.episode_length_buf[env_ids]
    env.push_recovered_buf[env_ids] = False


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_x_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """只跟踪机体系 yaw frame 下的正向速度。

    公式:
        r = exp(-((v_x_cmd - v_x)^2) / std^2)
    其中 v_x 先通过 yaw-only 四元数投影到机体朝向坐标系；该项不奖励/惩罚 y 方向速度，
    y 方向由 `base_lin_vel_yz_l2` 单独惩罚，避免策略把横移当作目标跟踪。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.square(env.command_generator.command[:, 0] - vel_yaw[:, 0])
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def lin_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机身水平速度平方惩罚。

    公式:
        r = v_x^2 + v_y^2
    站立任务中命令速度为 0，该项用于抑制被推后 base 在水平面持续规律摆动。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)


def base_lin_vel_yz_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """非目标 base 线速度平方惩罚。

    公式:
        r = v_y^2 + v_z^2
    x 方向是本任务的目标前进速度，不在这里惩罚；该项用于抑制横向漂移和上下窜动。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, 1:3]), dim=1)


def base_height_l2(
    env: BaseEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """机身高度误差平方。

    公式:
        r = (z_base - z_target)^2
    对 plane 站立任务，z_target 来自 B2 默认初始高度 0.58m，用来抑制跳起悬空或趴低。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def base_height_exp(
    env: BaseEnv, target_height: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """机身高度正奖励。

    公式:
        r = exp(-((z_base - z_target)^2) / std^2)
    其中 z_target 使用 B2 初始重心高度 0.58m；指数核来自常见速度跟踪奖励形式，
    能在目标高度附近提供接近 1 的正奖励，偏离目标时平滑下降。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-height_error / std**2)


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def ang_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """机身偏航角速度平方惩罚。

    公式:
        r = ω_z^2
    站立任务中 yaw 命令为 0，该项用于抑制机身绕竖直轴周期性摆动。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def action_rate_l2_joint(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, asset_cfg.joint_ids]
            - env.action_buffer._circular_buffer.buffer[:, -2, asset_cfg.joint_ids]
        ),
        dim=1,
    )


def action_l2_joint(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.sum(torch.square(env.action_buffer._circular_buffer.buffer[:, -1, asset_cfg.joint_ids]), dim=1)


def stand_still_joint_deviation_l1(
    env: BaseEnv, command_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """零速度命令下的关节姿态偏差。

    公式:
        r = 1(||cmd_xy|| < threshold) * sum_i |q_i - q_i_default|
    参考 B2W 的 stand_still 项，轮足机器人通常只对腿部姿态使用该项，轮子关节应从
    asset_cfg 中排除，避免速度控制轮在零命令时被位置偏差惩罚。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids])
    standing = torch.norm(env.command_generator.command[:, :2], dim=1) < command_threshold
    return torch.sum(joint_error, dim=1) * standing.float()


def undesired_contacts(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def feet_contact_count(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """四足接触正奖励。

    公式:
        r = (1 / N) * sum_i 1(||f_i|| > threshold)
    对四足 N=4，四只脚都接触时奖励为 1，全部离地时为 0。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.mean(is_contact.float(), dim=-1)


def all_feet_contact(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.all(is_contact, dim=-1).float()


def feet_still_exp(
    env: BaseEnv,
    std: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """接触足端低速正奖励。

    公式:
        r = exp(-mean_i(c_i * ||v_foot_i||^2) / std^2)
    其中 c_i 为足端接触指示。只奖励接触脚保持低速度，避免悬空摆腿通过低速度钻空子。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    foot_speed_sq = torch.sum(torch.square(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]), dim=-1)
    contact_count = torch.clamp(torch.sum(is_contact.float(), dim=-1), min=1.0)
    mean_contact_speed_sq = torch.sum(foot_speed_sq * is_contact.float(), dim=-1) / contact_count
    return torch.exp(-mean_contact_speed_sq / std**2) * (torch.sum(is_contact.float(), dim=-1) > 0).float()


def push_recovery_time_exp(
    env: BaseEnv,
    max_time: float,
    height_target: float,
    height_tol: float,
    lin_vel_tol: float,
    ang_vel_tol: float,
    orientation_tol: float,
    contact_threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """推扰后的快速恢复一次性奖励。

    稳定判据:
        |z - z_target| < height_tol
        ||v_base|| < lin_vel_tol
        ||ω_base|| < ang_vel_tol
        g_x^2 + g_y^2 < orientation_tol^2
        四足均接触地面

    奖励公式:
        r = 1(stable and not recovered) * clamp(1 - Δt / max_time, 0, 1)
    其中 Δt = (t_now - t_push) * dt。越快恢复，奖励越接近 1；超过 max_time 奖励为 0。
    """
    if not hasattr(env, "last_push_step_buf"):
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > contact_threshold
    all_feet_contact = torch.all(is_contact, dim=-1)

    height_ok = torch.abs(asset.data.root_pos_w[:, 2] - height_target) < height_tol
    lin_vel_ok = torch.norm(asset.data.root_lin_vel_b, dim=-1) < lin_vel_tol
    ang_vel_ok = torch.norm(asset.data.root_ang_vel_b, dim=-1) < ang_vel_tol
    orientation_ok = flat_orientation_l2(env, asset_cfg) < orientation_tol**2
    active = (env.last_push_step_buf >= 0) & (~env.push_recovered_buf)
    stable = height_ok & lin_vel_ok & ang_vel_ok & orientation_ok & all_feet_contact

    elapsed = (env.episode_length_buf - env.last_push_step_buf).float() * env.step_dt
    reward = torch.clamp(1.0 - elapsed / max_time, min=0.0, max=1.0)
    reward = torch.where(active & stable, reward, torch.zeros_like(reward))
    env.push_recovered_buf |= active & stable
    return reward


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_air_time_quadruped(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """WMP/A1 风格四足摆腿奖励。

    公式来自 WMP/legged_gym 的 `_reward_feet_air_time`:
        r = sum_i((t_air_i - threshold) * first_contact_i)
    其中 first_contact_i 表示第 i 只脚从离地状态首次重新接触地面。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum((air_time - threshold) * first_contact.float(), dim=1)
    reward *= torch.norm(env.command_generator.command[:, :2], dim=1) > 0.1
    return reward


def feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_deviation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def dof_error_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1)


def cheat_yaw(env: BaseEnv, heading_limit: float = 1.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    forward = math_utils.quat_apply(asset.data.root_quat_w, asset.data.FORWARD_VEC_B)
    heading = torch.atan2(forward[:, 1], forward[:, 0])
    return (torch.abs(heading) > heading_limit).float()


def stuck(env: BaseEnv, velocity_threshold: float = 0.1, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return ((torch.abs(asset.data.root_lin_vel_b[:, 0]) < velocity_threshold) & (torch.abs(env.command_generator.command[:, 0]) > command_threshold)).float()


def feet_edge(env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """WMP 原版 `feet_edge` 奖励。

    公式对齐 WMP/legged_gym:
        idx_xy = round((p_foot_xy + border_size) / horizontal_scale)
        feet_at_edge = contact_filt & x_edge_mask[idx_x, idx_y]
        r = 1(terrain_level > 3) * sum(feet_at_edge)

    其中 `x_edge_mask` 来自高度场沿 x 方向的高度突变检测：
        edge[i, j] = abs(h[i + 1, j] - h[i, j]) > threshold
    """
    edge_mask = getattr(env, "x_edge_mask", None)
    if edge_mask is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_filt = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0

    edge_offset = torch.tensor(getattr(env, "wmp_edge_query_offset", (0.0, 0.0)), device=env.device)
    horizontal_scale = getattr(env, "wmp_terrain_horizontal_scale", 1.0)
    feet_pos_xy = ((asset.data.body_pos_w[:, asset_cfg.body_ids, :2] + edge_offset) / horizontal_scale).round().long()
    feet_pos_xy[..., 0] = torch.clamp(feet_pos_xy[..., 0], 0, edge_mask.shape[0] - 1)
    feet_pos_xy[..., 1] = torch.clamp(feet_pos_xy[..., 1], 0, edge_mask.shape[1] - 1)
    feet_at_edge = edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]

    terrain_levels = getattr(env, "terrain_levels", None)
    terrain_types = getattr(env, "terrain_types", None)
    if terrain_levels is None or terrain_types is None:
        level_gate = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        type_gate = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        level_gate = terrain_levels > 3
        type_gate = (terrain_types >= getattr(env, "gap_start_col", 0)) & (
            terrain_types < getattr(env, "climb_end_col", terrain_types.max().item() + 1)
        )
    return (level_gate & type_gate).float() * torch.sum(contact_filt & feet_at_edge, dim=-1)
