# -*- coding: utf-8 -*-
"""MuJoCo XML + mink retarget solver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import (
    ANGULAR_VEL,
    FRAME_WIDTH,
    JOINT_POS,
    JOINT_VEL,
    LINEAR_VEL,
    ROOT_POS,
    ROOT_QUAT,
    TOE_POS_LOCAL,
    WMPMotion,
)
from .mapping import RetargetMapping
from .math_utils import angular_velocity_from_quat_wxyz, finite_difference, normalize_quat_wxyz, xyzw_to_wxyz


@dataclass
class RetargetResult:
    motion: WMPMotion
    source_feet_world: np.ndarray
    target_feet_world: np.ndarray
    foot_error: np.ndarray
    target_qpos: np.ndarray
    target_feet_goal_world: np.ndarray


def _import_runtime() -> tuple[Any, Any]:
    try:
        import mujoco  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少依赖 mujoco，请先运行: pip install mujoco mink pyyaml") from exc
    try:
        import mink  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少依赖 mink，请先运行: pip install mujoco mink pyyaml") from exc
    return mujoco, mink


class _MjcfModel:
    def __init__(self, xml_path: str | Path, model_mapping, mujoco_module):
        self.mujoco = mujoco_module
        self.model = self.mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = self.mujoco.MjData(self.model)
        self.mapping = model_mapping
        self.root_qpos_addr = 0
        self.joint_qpos_addrs = [self._joint_qpos_addr(name) for name in model_mapping.joints]
        self.site_ids = {key: self._site_id(site_name) for key, site_name in model_mapping.frames.items()}

    def _joint_qpos_addr(self, joint_name: str) -> int:
        joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"XML missing joint: {joint_name}")
        return int(self.model.jnt_qposadr[joint_id])

    def _site_id(self, site_name: str) -> int:
        site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"XML missing site: {site_name}")
        return int(site_id)

    def set_pose(self, root_pos: np.ndarray, root_quat_wxyz: np.ndarray, joints: np.ndarray) -> None:
        self.data.qpos[:] = 0.0
        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = normalize_quat_wxyz(root_quat_wxyz)
        for addr, value in zip(self.joint_qpos_addrs, joints):
            self.data.qpos[addr] = value
        self.mujoco.mj_forward(self.model, self.data)

    def site_pos(self, frame_key: str) -> np.ndarray:
        return np.asarray(self.data.site_xpos[self.site_ids[frame_key]], dtype=np.float64).copy()

    def sites(self) -> dict[str, np.ndarray]:
        return {key: self.site_pos(key) for key in self.site_ids}


def _neutral_qpos(model: _MjcfModel, joint_values: dict[str, float] | None = None) -> np.ndarray:
    qpos = np.zeros(model.model.nq, dtype=np.float64)
    qpos[3] = 1.0
    for name, addr in zip(model.mapping.joints, model.joint_qpos_addrs):
        qpos[addr] = 0.0 if joint_values is None else float(joint_values.get(name, 0.0))
    return qpos


def _joint_project_limits(model: _MjcfModel, mapping: RetargetMapping) -> list[tuple[int, float, float]]:
    limits = mapping.options.joint_project_limits or {}
    projected = []
    for name, addr in zip(model.mapping.joints, model.joint_qpos_addrs):
        if name not in limits:
            continue
        lower, upper = limits[name]
        projected.append((addr, float(lower), float(upper)))
    return projected


def _project_configuration(configuration, project_limits: list[tuple[int, float, float]]) -> None:
    if not project_limits:
        return
    qpos = np.asarray(configuration.q, dtype=np.float64).copy()
    changed = False
    for addr, lower, upper in project_limits:
        value = float(np.clip(qpos[addr], lower, upper))
        if value != qpos[addr]:
            qpos[addr] = value
            changed = True
    if changed:
        configuration.update(qpos)


def _set_model_qpos(model: _MjcfModel, qpos: np.ndarray) -> None:
    model.data.qpos[:] = qpos
    model.mujoco.mj_forward(model.model, model.data)


def _morphology_ratios(source_model: _MjcfModel, target_model: _MjcfModel, mapping: RetargetMapping) -> dict[str, float]:
    source_neutral = _neutral_qpos(source_model)
    target_neutral = _neutral_qpos(target_model, mapping.options.neutral_joint_pos)
    _set_model_qpos(source_model, source_neutral)
    _set_model_qpos(target_model, target_neutral)
    ratios: dict[str, float] = {}
    for leg in ("FR", "FL", "RR", "RL"):
        source_hip = source_model.site_pos(f"{leg}_hip")
        target_hip = target_model.site_pos(f"{leg}_hip")
        for suffix in ("thigh", "calf", "foot"):
            key = f"{leg}_{suffix}"
            source_len = float(np.linalg.norm(source_model.site_pos(key) - source_hip))
            target_len = float(np.linalg.norm(target_model.site_pos(key) - target_hip))
            ratios[key] = 1.0 if source_len < 1.0e-8 else target_len / source_len
    return ratios


def _build_frame_goals(
    target_root: np.ndarray,
    source_sites: dict[str, np.ndarray],
    target_neutral_sites: dict[str, np.ndarray],
    ratios: dict[str, float],
    mapping: RetargetMapping,
) -> dict[str, np.ndarray]:
    if mapping.options.frame_target_mode == "base_relative":
        return {
            key: target_root + (pos - source_sites["base"]) * mapping.options.frame_position_scale
            for key, pos in source_sites.items()
            if key != "base"
        }
    if mapping.options.frame_target_mode != "morphology_scaled":
        raise ValueError(f"Unsupported frame_target_mode={mapping.options.frame_target_mode!r}")

    goals: dict[str, np.ndarray] = {}
    for leg in ("FR", "FL", "RR", "RL"):
        source_hip = source_sites[f"{leg}_hip"]
        target_hip = target_root + target_neutral_sites[f"{leg}_hip"]
        goals[f"{leg}_hip"] = target_hip
        for suffix in ("thigh", "calf", "foot"):
            key = f"{leg}_{suffix}"
            rel = source_sites[key] - source_hip
            goals[key] = target_hip + rel * ratios.get(key, 1.0) * mapping.options.frame_position_scale
    return goals


def _make_mink_tasks(mink, target_model: _MjcfModel, mapping: RetargetMapping):
    opts = mapping.options
    tasks = []
    root_task = mink.FrameTask(
        frame_name=mapping.target.base_site,
        frame_type="site",
        position_cost=opts.root_cost,
        orientation_cost=opts.root_cost,
        lm_damping=opts.damping,
    )
    tasks.append(root_task)
    frame_tasks = {}
    for key, site_name in mapping.target.frames.items():
        if key == "base":
            continue
        cost = opts.foot_pos_cost if key.endswith("_foot") else opts.frame_pos_cost
        task = mink.FrameTask(
            frame_name=site_name,
            frame_type="site",
            position_cost=cost,
            orientation_cost=0.0,
            lm_damping=opts.damping,
        )
        frame_tasks[key] = task
        tasks.append(task)
    posture_task = mink.PostureTask(model=target_model.model, cost=opts.posture_cost)
    tasks.append(posture_task)
    return root_task, frame_tasks, posture_task, tasks


def _set_frame_target(mink, task, position: np.ndarray, quat_wxyz: np.ndarray | None = None) -> None:
    if hasattr(task, "set_target"):
        if quat_wxyz is None:
            quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(quat_wxyz), position))
    elif hasattr(task, "set_target_pos"):
        task.set_target_pos(position)
    else:
        raise RuntimeError("Unsupported mink FrameTask API: no set_target method found.")


def _solve_one_frame(
    mink,
    target_model: _MjcfModel,
    configuration,
    tasks,
    root_task,
    frame_tasks,
    posture_task,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    frame_goals: dict[str, np.ndarray],
    posture_qpos: np.ndarray,
    project_limits: list[tuple[int, float, float]],
    mapping: RetargetMapping,
) -> None:
    _set_frame_target(mink, root_task, root_pos, root_quat_wxyz)
    for key, task in frame_tasks.items():
        if key in frame_goals:
            _set_frame_target(mink, task, frame_goals[key])
    if hasattr(posture_task, "set_target"):
        posture_task.set_target(posture_qpos)
    elif hasattr(posture_task, "set_target_from_configuration"):
        posture_task.set_target_from_configuration(configuration)

    for _ in range(mapping.options.max_ik_iters):
        velocity = mink.solve_ik(configuration, tasks, dt=1.0, solver=mapping.options.qp_solver)
        configuration.integrate_inplace(velocity, 1.0)
        _project_configuration(configuration, project_limits)

    target_model.data.qpos[:] = np.asarray(configuration.q, dtype=np.float64)
    target_model.mujoco.mj_forward(target_model.model, target_model.data)


def retarget_motion(
    motion: WMPMotion,
    source_xml: str | Path,
    target_xml: str | Path,
    mapping: RetargetMapping,
    max_frames: int | None = None,
) -> RetargetResult:
    mujoco, mink = _import_runtime()
    source_model = _MjcfModel(source_xml, mapping.source, mujoco)
    target_model = _MjcfModel(target_xml, mapping.target, mujoco)

    frame_count = motion.frames.shape[0] if max_frames is None else min(int(max_frames), motion.frames.shape[0])
    out_frames = np.zeros((frame_count, FRAME_WIDTH), dtype=np.float64)
    target_qpos = np.zeros((frame_count, target_model.model.nq), dtype=np.float64)
    source_feet = np.zeros((frame_count, 4, 3), dtype=np.float64)
    target_feet = np.zeros((frame_count, 4, 3), dtype=np.float64)
    target_feet_goal = np.zeros((frame_count, 4, 3), dtype=np.float64)

    configuration = mink.Configuration(target_model.model)
    root_task, frame_tasks, posture_task, tasks = _make_mink_tasks(mink, target_model, mapping)
    foot_keys = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
    neutral_qpos = _neutral_qpos(target_model, mapping.options.neutral_joint_pos)
    project_limits = _joint_project_limits(target_model, mapping)
    _set_model_qpos(target_model, neutral_qpos)
    target_neutral_sites = target_model.sites()
    morphology_ratios = _morphology_ratios(source_model, target_model, mapping)

    for frame_id in range(frame_count):
        src_frame = motion.frames[frame_id]
        source_root = src_frame[ROOT_POS].copy()
        source_root[:2] *= mapping.options.root_xy_scale
        source_quat = xyzw_to_wxyz(src_frame[ROOT_QUAT])
        source_joint = src_frame[JOINT_POS]
        source_model.set_pose(source_root, source_quat, source_joint)
        sites = source_model.sites()

        target_root = source_root.copy()
        target_root[2] += mapping.options.root_height_offset
        frame_goals = _build_frame_goals(
            target_root,
            sites,
            target_neutral_sites,
            morphology_ratios,
            mapping,
        )

        if mapping.options.posture_mode == "source_joint":
            posture_qpos = np.asarray(configuration.q, dtype=np.float64).copy()
        elif mapping.options.posture_mode == "neutral":
            posture_qpos = neutral_qpos.copy()
        else:
            raise ValueError(f"Unsupported posture_mode={mapping.options.posture_mode!r}")
        posture_qpos[0:3] = target_root
        posture_qpos[3:7] = source_quat
        if mapping.options.posture_mode == "source_joint":
            for addr, value in zip(target_model.joint_qpos_addrs, source_joint):
                posture_qpos[addr] = value

        _solve_one_frame(
            mink,
            target_model,
            configuration,
            tasks,
            root_task,
            frame_tasks,
            posture_task,
            target_root,
            source_quat,
            frame_goals,
            posture_qpos,
            project_limits,
            mapping,
        )
        target_qpos[frame_id] = np.asarray(configuration.q, dtype=np.float64)
        out_frames[frame_id, ROOT_POS] = target_qpos[frame_id, 0:3]
        out_frames[frame_id, ROOT_QUAT] = src_frame[ROOT_QUAT]
        for joint_idx, addr in enumerate(target_model.joint_qpos_addrs):
            out_frames[frame_id, JOINT_POS.start + joint_idx] = target_qpos[frame_id, addr]

        base_pos = np.asarray(target_model.data.site_xpos[target_model.site_ids["base"]], dtype=np.float64)
        toe_local = []
        for foot_idx, foot_key in enumerate(foot_keys):
            source_feet[frame_id, foot_idx] = sites[foot_key]
            target_feet[frame_id, foot_idx] = target_model.site_pos(foot_key)
            target_feet_goal[frame_id, foot_idx] = frame_goals[foot_key]
            toe_local.extend((target_feet[frame_id, foot_idx] - base_pos).tolist())
        out_frames[frame_id, TOE_POS_LOCAL] = np.asarray(toe_local, dtype=np.float64)

    dt = motion.frame_duration
    out_frames[:, LINEAR_VEL] = finite_difference(out_frames[:, ROOT_POS], dt)
    out_frames[:, ANGULAR_VEL] = angular_velocity_from_quat_wxyz(xyzw_to_wxyz(out_frames[:, ROOT_QUAT]), dt)
    out_frames[:, JOINT_VEL] = finite_difference(out_frames[:, JOINT_POS], dt)

    foot_error = np.linalg.norm(target_feet - target_feet_goal, axis=-1)
    return RetargetResult(
        motion=WMPMotion(
            frames=out_frames,
            frame_duration=motion.frame_duration,
            motion_weight=motion.motion_weight,
            loop_mode=motion.loop_mode,
        ),
        source_feet_world=source_feet,
        target_feet_world=target_feet,
        foot_error=foot_error,
        target_qpos=target_qpos,
        target_feet_goal_world=target_feet_goal,
    )
