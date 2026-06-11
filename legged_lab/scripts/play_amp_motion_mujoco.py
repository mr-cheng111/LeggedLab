#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Use MuJoCo to kinematically replay a WMP AMP motion on an XML robot."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legged_lab.amp.mink_retarget.io import JOINT_POS, ROOT_POS, ROOT_QUAT, load_wmp_motion
from legged_lab.amp.mink_retarget.mapping import load_mapping
from legged_lab.amp.mink_retarget.math_utils import normalize_quat_wxyz, xyzw_to_wxyz


DEFAULT_XML = "legged_lab/assets/unitree/b2/mjcf/b2_retarget.xml"
DEFAULT_MAPPING = "legged_lab/assets/unitree/b2/mjcf/a1_to_b2_retarget.yaml"
DEFAULT_SCENE = "legged_lab/tools/mujoco_motion_scene.xml"
A1_JOINT_ORDER = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)


def _import_mujoco():
    try:
        import mujoco  # type: ignore
        import mujoco.viewer  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少依赖 mujoco，请先运行: pip install mujoco") from exc
    return mujoco


def _joint_qpos_addrs(mujoco, model, joint_names: tuple[str, ...]) -> list[int]:
    addrs = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"XML missing joint: {joint_name}")
        addrs.append(int(model.jnt_qposadr[joint_id]))
    return addrs


def _scene_xml_path(robot_xml: str | Path, scene_template: str | Path | None) -> tuple[str, tempfile.TemporaryDirectory | None]:
    if not scene_template:
        return str(robot_xml), None
    robot_path = Path(robot_xml).resolve()
    template_path = Path(scene_template)
    text = template_path.read_text(encoding="utf-8")
    temp_dir = tempfile.TemporaryDirectory(prefix="leggedlab_mujoco_scene_")
    scene_path = Path(temp_dir.name) / "scene.xml"
    scene_path.write_text(text.replace("{robot_xml}", str(robot_path)), encoding="utf-8")
    return str(scene_path), temp_dir


def _set_kinematic_pose(
    data,
    joint_addrs: list[int],
    frame: np.ndarray,
    fixed_root_pos: np.ndarray | None,
    fixed_root_quat: np.ndarray | None,
    root_xy_offset: np.ndarray | None,
    lock_xy: bool,
) -> None:
    if fixed_root_pos is None:
        data.qpos[0:3] = frame[ROOT_POS]
        if root_xy_offset is not None:
            data.qpos[0:2] -= root_xy_offset
        if lock_xy:
            data.qpos[0:2] = 0.0
    else:
        data.qpos[0:3] = fixed_root_pos

    if fixed_root_quat is None:
        data.qpos[3:7] = normalize_quat_wxyz(xyzw_to_wxyz(frame[ROOT_QUAT]))
    else:
        data.qpos[3:7] = fixed_root_quat

    for addr, value in zip(joint_addrs, frame[JOINT_POS]):
        data.qpos[addr] = value
    data.qvel[:] = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=DEFAULT_XML, help="MuJoCo XML/MJCF robot model.")
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="MuJoCo scene template with {robot_xml}, or empty string to disable.")
    parser.add_argument("--motion", default="datasets/retargeted/b2/hop1.txt", help="WMP AMP JSON motion file.")
    parser.add_argument("--mapping", default=DEFAULT_MAPPING, help="Mapping YAML that provides target joint order.")
    parser.add_argument("--joint_order", choices=("mapping_target", "mapping_source", "a1"), default="mapping_target")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int)
    parser.add_argument("--fixed_camera", action="store_true", help="Use the scene's fixed camera instead of MuJoCo free camera.")
    parser.add_argument("--loop", action="store_true", help="Loop playback until the viewer is closed.")
    parser.add_argument("--origin_xy", action="store_true", help="Subtract the first frame root x/y so the motion starts at the origin.")
    parser.add_argument("--lock_xy", action="store_true", help="Keep root x/y at the origin while preserving root height and orientation.")
    parser.add_argument("--fix_root", action="store_true", help="Keep floating root fixed in the air and only replay joints.")
    parser.add_argument("--root_height", type=float, default=None, help="Fixed root z when --fix_root is set.")
    parser.add_argument("--root_x", type=float, default=0.0)
    parser.add_argument("--root_y", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed <= 0.0:
        raise ValueError("--speed must be positive.")

    mujoco = _import_mujoco()
    motion = load_wmp_motion(args.motion)
    mapping = load_mapping(args.mapping) if args.mapping else None

    scene_path, scene_temp_dir = _scene_xml_path(args.xml, args.scene.strip() or None)
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0.0
    model.opt.wind[:] = 0.0
    model.opt.density = 0.0
    model.opt.viscosity = 0.0

    if args.joint_order == "a1":
        joint_names = A1_JOINT_ORDER
    elif args.joint_order == "mapping_source":
        if mapping is None:
            raise ValueError("--joint_order=mapping_source requires --mapping.")
        joint_names = mapping.source.joints
    else:
        if mapping is None:
            raise ValueError("--joint_order=mapping_target requires --mapping.")
        joint_names = mapping.target.joints
    joint_addrs = _joint_qpos_addrs(mujoco, model, joint_names)
    start = max(0, int(args.start_frame))
    end = motion.frames.shape[0] if args.end_frame is None else min(int(args.end_frame), motion.frames.shape[0])
    if start >= end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}.")

    fixed_root_pos = None
    fixed_root_quat = None
    root_xy_offset = None
    if args.fix_root:
        first_root = motion.frames[start, ROOT_POS].copy()
        fixed_root_pos = np.array(
            [
                args.root_x,
                args.root_y,
                first_root[2] if args.root_height is None else args.root_height,
            ],
            dtype=np.float64,
        )
        fixed_root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elif args.origin_xy:
        root_xy_offset = motion.frames[start, ROOT_POS][0:2].copy()

    frames = motion.frames[start:end]
    frame_dt = motion.frame_duration / args.speed
    print(
        f"[INFO] Playing {args.motion} on {args.xml}: frames={len(frames)}, "
        f"dt={motion.frame_duration}, speed={args.speed}, fix_root={args.fix_root}, "
        f"origin_xy={args.origin_xy}, lock_xy={args.lock_xy}, scene={args.scene or 'none'}"
    )

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "motion_view")
            if args.fixed_camera and camera_id >= 0:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = camera_id
            elif args.scene:
                viewer.cam.distance = 3.2
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -18
                viewer.cam.lookat[:] = (0.0, 0.0, 0.35)
            while viewer.is_running():
                for frame in frames:
                    if not viewer.is_running():
                        break
                    tick = time.perf_counter()
                    _set_kinematic_pose(
                        data,
                        joint_addrs,
                        frame,
                        fixed_root_pos,
                        fixed_root_quat,
                        root_xy_offset,
                        args.lock_xy,
                    )
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    sleep_time = frame_dt - (time.perf_counter() - tick)
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
                if not args.loop:
                    break
    finally:
        if scene_temp_dir is not None:
            scene_temp_dir.cleanup()


if __name__ == "__main__":
    main()
