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

import argparse
import os
import subprocess
import sys
import tempfile

import torch
from isaaclab.app import AppLauncher
from rsl_rl.runners import OnPolicyRunner

from legged_lab.utils import task_registry

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N play steps; 0 runs until the app closes.")
parser.add_argument(
    "--runner",
    type=str,
    default="default",
    choices=["default", "wmp", "wmp_amp"],
    help="Runner/checkpoint type.",
)
parser.add_argument("--play_flat", action="store_true", help="Play on a flat plane while keeping WMP sensor obs shapes.")
parser.add_argument(
    "--play_render_interval",
    type=int,
    default=None,
    help="Override sim render interval during play. Lower values make GUI smoother but heavier.",
)
parser.add_argument("--show_depth_image", action="store_true", help="Show the original WMP-style 64x64 depth image window.")
parser.add_argument(
    "--depth_image_mode",
    type=str,
    default="auto",
    choices=["auto", "window", "save"],
    help="How to show WMP depth images. window uses an isolated OpenCV process; auto falls back to PNG saving.",
)
parser.add_argument("--depth_image_dir", type=str, default=None, help="Directory for saved WMP depth image PNGs.")
parser.add_argument("--depth_image_save_interval", type=int, default=10, help="Save one depth image every N play steps.")
parser.add_argument("--show_depth_points", action="store_true", help="Visualize RGBD depth hits as red debug points.")
parser.add_argument("--show_height_scan_points", action="store_true", help="Visualize height scanner ray hits.")
parser.add_argument("--enable_play_push", action="store_true", help="Keep interval push disturbances enabled during play.")
parser.add_argument(
    "--hide_command", action="store_true", help="Hide command/current velocity debug visualization during play."
)
joystick_group = parser.add_mutually_exclusive_group()
joystick_group.add_argument(
    "--virtual_joystick",
    dest="virtual_joystick",
    action="store_true",
    default=True,
    help="Open the local desktop velocity controller (default).",
)
joystick_group.add_argument(
    "--no_virtual_joystick",
    dest="virtual_joystick",
    action="store_false",
    help="Disable the local desktop velocity controller.",
)
parser.add_argument("--joystick_port", type=int, default=8765, help="Local port for the virtual joystick page.")
parser.add_argument(
    "--joystick_max_vx", type=float, default=None, help="Maximum absolute joystick forward speed in m/s."
)
parser.add_argument(
    "--joystick_max_vy", type=float, default=None, help="Maximum absolute joystick lateral speed in m/s."
)
parser.add_argument(
    "--joystick_max_wz", type=float, default=None, help="Maximum absolute joystick yaw rate in rad/s."
)
parser.add_argument(
    "--joystick_timeout",
    type=float,
    default=2.0,
    help="Stop command after this many seconds without a joystick heartbeat.",
)
parser.add_argument("--no_open_joystick", action="store_true", help="Do not open the joystick desktop window.")
parser.add_argument("--depth_point_stride", type=int, default=16, help="Pixel stride for depth hit point visualization.")
parser.add_argument("--depth_point_max", type=int, default=300, help="Maximum depth hit points to draw.")
parser.add_argument("--depth_point_size", type=float, default=5.0, help="Debug draw size of each red depth hit point.")
parser.add_argument("--depth_point_forward_min", type=float, default=0.2, help="Minimum forward distance for depth hit points.")
parser.add_argument("--depth_point_forward_max", type=float, default=3.0, help="Maximum forward distance for depth hit points.")
parser.add_argument("--depth_point_min_z", type=float, default=None, help="Minimum world z for visualized depth hit points.")
parser.add_argument("--depth_point_max_z", type=float, default=None, help="Maximum world z for visualized depth hit points.")
parser.add_argument("--depth_point_debug", action="store_true", help="Print depth point visualization statistics.")
parser.add_argument("--depth_point_lift", type=float, default=0.05, help="Lift visualized depth points above surfaces.")
parser.add_argument("--depth_point_draw_rays", action="store_true", help="Draw yellow rays from camera origins to depth points.")
parser.add_argument(
    "--depth_point_camera_index",
    type=int,
    default=0,
    help="Depth camera index to visualize. Use -1 to visualize all camera envs.",
)
parser.add_argument(
    "--camera_offset_pos",
    type=float,
    nargs=3,
    default=None,
    help="Override spawned WMP camera xyz offset during play.",
)
parser.add_argument(
    "--camera_offset_rot",
    type=float,
    nargs=4,
    default=None,
    help="Override spawned WMP camera xyzw offset during play.",
)
parser.add_argument(
    "--camera_random_pitch_deg",
    type=float,
    nargs=2,
    default=None,
    help="Override WMP camera random pitch range in degrees during play.",
)
parser.add_argument("--camera_fov_deg", type=float, default=None, help="Override WMP camera horizontal FOV during play.")
parser.add_argument(
    "--camera_disable_random_rotation",
    action="store_true",
    help="Disable WMP camera random rotation during play.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

depth_viewer_process = None
if args_cli.show_depth_image and args_cli.depth_image_mode == "window":
    if args_cli.depth_image_dir is None:
        args_cli.depth_image_dir = tempfile.mkdtemp(prefix="leggedlab_depth_preview_")
    viewer_script = os.path.join(os.path.dirname(__file__), "depth_image_viewer.py")
    depth_viewer_process = subprocess.Popen(
        [sys.executable, viewer_script, args_cli.depth_image_dir, "--parent-pid", str(os.getpid())]
    )
    print(f"[INFO] External depth viewer started: {args_cli.depth_image_dir}", flush=True)

if args_cli.runner in ("wmp", "wmp_amp") or args_cli.show_depth_points or args_cli.show_depth_image:
    args_cli.enable_cameras = True
    if args_cli.show_depth_points:
        debug_draw_enable_arg = "--enable isaacsim.util.debug_draw"
        args_cli.kit_args = (
            f"{args_cli.kit_args} {debug_draw_enable_arg}".strip()
            if getattr(args_cli, "kit_args", None)
            else debug_draw_enable_arg
        )

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.utils.math import transform_points, unproject_depth

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from legged_lab.utils.rsl_rl_compat import adapt_legacy_cfg_for_rsl_rl_v5, is_rsl_rl_v5_plus
from legged_lab.world_models.wmp.preprocess import depth_to_wmp_image


def _acquire_debug_draw_interface():
    try:
        from isaacsim.util.debug_draw import _debug_draw
    except ModuleNotFoundError:
        try:
            from omni.isaac.debug_draw import _debug_draw
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Debug draw is unavailable in this IsaacSim environment. "
                "Run without --show_depth_points, or install/enable the debug draw extension."
            ) from exc
    return _debug_draw.acquire_debug_draw_interface()


def _depth_to_hit_points(
    depth_camera,
    near: float,
    far: float,
    stride: int,
    max_points: int,
    forward_min: float,
    forward_max: float,
    min_z: float | None,
    max_z: float | None,
    lift: float,
    camera_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    depth = depth_camera.data.output["distance_to_image_plane"]
    device = depth.device
    stride = max(stride, 1)
    camera_ids = _select_depth_camera_indices(depth.shape[0], camera_index, device)
    if camera_ids.numel() == 0:
        empty_points = torch.empty(0, 3, device=device)
        empty_origins = torch.empty(0, 3, device=device)
        return empty_points, empty_origins

    selected_depth = depth[camera_ids]
    selected_intrinsics = depth_camera.data.intrinsic_matrices[camera_ids]
    selected_pos = depth_camera.data.pos_w[camera_ids]
    selected_quat = depth_camera.data.quat_w_ros[camera_ids]

    points_cam = unproject_depth(selected_depth, selected_intrinsics, is_ortho=True)
    points_world = transform_points(points_cam, selected_pos, selected_quat)
    points_world = points_world[:, ::stride]
    point_origins = selected_pos[:, None, :].expand(-1, points_world.shape[1], -1)

    # unproject_depth 内部按 (u, v) 顺序展开 depth，因此这里用同样顺序对齐 mask。
    sampled_depth = selected_depth[..., 0].transpose(1, 2).reshape(selected_depth.shape[0], -1)[:, ::stride]
    valid = (
        torch.isfinite(sampled_depth)
        & (sampled_depth > near)
        & (sampled_depth < far)
        & (sampled_depth > forward_min)
        & (sampled_depth < forward_max)
    )
    points_world = points_world[valid]
    point_origins = point_origins[valid]
    if min_z is not None:
        keep = points_world[:, 2] >= float(min_z)
        points_world = points_world[keep]
        point_origins = point_origins[keep]
    if max_z is not None:
        keep = points_world[:, 2] <= float(max_z)
        points_world = points_world[keep]
        point_origins = point_origins[keep]
    if points_world.shape[0] > max_points:
        step = torch.ceil(torch.tensor(points_world.shape[0] / max_points, device=device)).long().item()
        points_world = points_world[::step][:max_points]
        point_origins = point_origins[::step][:max_points]
    if lift != 0.0 and points_world.numel() > 0:
        points_world = points_world.clone()
        points_world[:, 2] += float(lift)
    return points_world, point_origins


def _select_depth_camera_indices(camera_count: int, camera_index: int, device: torch.device) -> torch.Tensor:
    if camera_count <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if int(camera_index) < 0:
        return torch.arange(camera_count, dtype=torch.long, device=device)
    camera_index = max(0, min(int(camera_index), camera_count - 1))
    return torch.tensor([camera_index], dtype=torch.long, device=device)


def _depth_debug_stats(depth_camera, points: torch.Tensor, origins: torch.Tensor, camera_index: int) -> str:
    depth_all = depth_camera.data.output["distance_to_image_plane"][..., 0]
    camera_ids = _select_depth_camera_indices(depth_all.shape[0], camera_index, depth_all.device)
    depth = depth_all[camera_ids] if camera_ids.numel() > 0 else depth_all[:0]
    finite = torch.isfinite(depth)
    finite_depth = depth[finite]
    if finite_depth.numel() == 0:
        return f"finite_depth=0/{depth.numel()}, drawn_points={points.shape[0]}"
    near_hits = int((finite_depth <= 0.05).sum().item())
    near_ratio = near_hits / max(1, int(finite_depth.numel()))
    point_stats = ""
    if points.numel() > 0:
        point_min = points.amin(dim=0)
        point_max = points.amax(dim=0)
        cam_pos = origins[0] if origins.numel() > 0 else depth_camera.data.pos_w[camera_ids[0]]
        point_dist = torch.linalg.norm(points - origins, dim=-1) if origins.shape == points.shape else None
        dist_stats = ""
        if point_dist is not None and point_dist.numel() > 0:
            dist_stats = f", point_dist=({point_dist.min().item():.2f},{point_dist.max().item():.2f})"
        point_stats = (
            f", point_min=({point_min[0].item():.2f},{point_min[1].item():.2f},{point_min[2].item():.2f}), "
            f"point_max=({point_max[0].item():.2f},{point_max[1].item():.2f},{point_max[2].item():.2f}), "
            f"cam0=({cam_pos[0].item():.2f},{cam_pos[1].item():.2f},{cam_pos[2].item():.2f})"
            f"{dist_stats}"
        )
    camera_ids = getattr(depth_camera, "camera_env_ids", None)
    camera_info = ""
    if camera_ids is not None:
        camera_info = f", camera_envs={int(camera_ids.numel())}"
    return (
        f"finite_depth={finite_depth.numel()}/{depth.numel()}, "
        f"depth_min={finite_depth.min().item():.3f}, depth_max={finite_depth.max().item():.3f}, "
        f"near<=0.05m={near_hits}({near_ratio:.2%}), "
        f"drawn_points={points.shape[0]}"
        f"{camera_info}{point_stats}"
    )


def _draw_depth_hit_points(draw_interface, points: torch.Tensor, point_size: float):
    if draw_interface is None:
        return
    draw_interface.clear_points()
    if points.numel() == 0:
        return
    points_list = points.detach().cpu().tolist()
    colors = [(1.0, 0.0, 0.0, 1.0)] * len(points_list)
    sizes = [point_size] * len(points_list)
    draw_interface.draw_points(points_list, colors, sizes)


def _draw_depth_rays(draw_interface, origins: torch.Tensor, points: torch.Tensor, ray_count: int = 128):
    if draw_interface is None:
        return
    if hasattr(draw_interface, "clear_lines"):
        draw_interface.clear_lines()
    if points.numel() == 0 or origins.shape != points.shape:
        return
    stride = max(1, points.shape[0] // max(1, ray_count))
    ray_ends = points[::stride][:ray_count]
    ray_starts = origins[::stride][:ray_count]
    colors = [(1.0, 0.9, 0.0, 1.0)] * ray_ends.shape[0]
    sizes = [1.5] * ray_ends.shape[0]
    draw_interface.draw_lines(ray_starts.detach().cpu().tolist(), ray_ends.detach().cpu().tolist(), colors, sizes)


def _wmp_depth_image_numpy(depth_camera, near: float, far: float, camera_index: int = 0):
    depth = depth_camera.data.output["distance_to_image_plane"]
    if depth.shape[0] == 0:
        return None
    camera_index = max(0, min(int(camera_index), depth.shape[0] - 1))
    image = depth_to_wmp_image(depth[camera_index : camera_index + 1], near=near, far=far)[0, ..., 0]
    # 原版显示的是 depth_buffer + 0.5；这里同样把 WMP [-0.5, 0.5] 映射到 [0, 1]。
    return torch.clamp(image + 0.5, 0.0, 1.0).detach().cpu().numpy()


def _wmp_depth_raw_stats(depth_camera, near: float, far: float, camera_index: int = 0) -> str:
    depth = depth_camera.data.output["distance_to_image_plane"]
    if depth.shape[0] == 0:
        return "depth_batch=0"
    camera_index = max(0, min(int(camera_index), depth.shape[0] - 1))
    depth = depth[camera_index, ..., 0]
    finite = torch.isfinite(depth)
    finite_depth = depth[finite]
    valid = finite & (depth > near) & (depth < far)
    far_like = (~finite) | (depth >= far)
    if finite_depth.numel() == 0:
        return f"finite=0/{depth.numel()}, valid={int(valid.sum())}, far_like={int(far_like.sum())}"
    return (
        f"finite={finite_depth.numel()}/{depth.numel()}, "
        f"valid={int(valid.sum())}, far_like={int(far_like.sum())}, "
        f"min={finite_depth.min().item():.3f}, max={finite_depth.max().item():.3f}, "
        f"mean={finite_depth.mean().item():.3f}"
    )


def _show_wmp_depth_image_window(image) -> bool:
    import cv2

    try:
        cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth Image", 320, 320)
        cv2.imshow("Depth Image", image)
        cv2.waitKey(1)
        return True
    except cv2.error as exc:
        print(f"[WARN] OpenCV depth window unavailable, falling back to PNG saving: {exc}", flush=True)
        return False


def _save_wmp_depth_image(image, output_dir: str, step: int, latest_only: bool = False) -> str:
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "depth_latest.png" if latest_only else f"depth_{step:06d}.png")
    image_u8 = (image * 255.0).clip(0, 255).astype("uint8")
    if latest_only:
        temporary_path = os.path.join(output_dir, ".depth_latest.tmp.png")
        cv2.imwrite(temporary_path, image_u8)
        os.replace(temporary_path, path)
    else:
        cv2.imwrite(path, image_u8)
    return path


def _handle_wmp_depth_image(
    depth_camera,
    near: float,
    far: float,
    mode: str,
    output_dir: str,
    step: int,
    save_interval: int,
    window_available: bool,
) -> bool:
    image = _wmp_depth_image_numpy(depth_camera, near=near, far=far)
    if image is None:
        return window_available
    if mode in ("auto", "window") and window_available:
        shown = _show_wmp_depth_image_window(image)
        if shown or mode == "window":
            return shown
    save_interval = max(1, int(save_interval))
    if step % save_interval == 0:
        path = _save_wmp_depth_image(image, output_dir, step, latest_only=mode == "window")
        if step == 0 or step % (save_interval * 20) == 0:
            stats = _wmp_depth_raw_stats(depth_camera, near=near, far=far)
            print(f"[INFO] Saved WMP depth image: {path} | {stats}", flush=True)
    return False


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    command_ranges = env_cfg.commands.ranges
    joystick_limits = (
        args_cli.joystick_max_vx
        if args_cli.joystick_max_vx is not None
        else max(abs(float(command_ranges.lin_vel_x[0])), abs(float(command_ranges.lin_vel_x[1]))),
        args_cli.joystick_max_vy
        if args_cli.joystick_max_vy is not None
        else max(abs(float(command_ranges.lin_vel_y[0])), abs(float(command_ranges.lin_vel_y[1]))),
        args_cli.joystick_max_wz
        if args_cli.joystick_max_wz is not None
        else max(abs(float(command_ranges.ang_vel_z[0])), abs(float(command_ranges.ang_vel_z[1]))),
    )

    env_cfg.noise.add_noise = False
    if not args_cli.enable_play_push:
        env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    if "slow_walk" in env_class_name:
        env_cfg.commands.ranges.lin_vel_x = (0.0, 1.0)
        env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
        env_cfg.commands.ranges.heading = (0.0, 0.0)
    elif "rb160w" in env_class_name:
        env_cfg.commands.ranges.lin_vel_x = (0.2, 0.8)
        env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
        env_cfg.commands.ranges.heading = (0.0, 0.0)
    elif "stand" not in env_class_name:
        env_cfg.commands.ranges.lin_vel_x = (0.0, 0.8)
        env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
        env_cfg.commands.ranges.heading = (0.0, 0.0)
    if args_cli.virtual_joystick:
        env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
        env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
        env_cfg.commands.ranges.heading = (0.0, 0.0)
        env_cfg.commands.heading_command = False
        env_cfg.commands.rel_standing_envs = 0.0
        env_cfg.commands.rel_heading_envs = 0.0
    env_cfg.commands.debug_vis = not args_cli.hide_command
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)
    if args_cli.play_render_interval is not None:
        env_cfg.sim.render_interval = max(1, int(args_cli.play_render_interval))

    if args_cli.play_flat:
        # 播放平地时只替换地形，不关闭 WMP height scanner。
        # 这样 actor/critic 的 height_scan 维度仍与训练 checkpoint 保持一致。
        env_cfg.scene.terrain_generator = None
        env_cfg.scene.terrain_type = "plane"
        env_cfg.scene.max_init_terrain_level = 0

    if args_cli.show_height_scan_points:
        env_cfg.scene.height_scanner.debug_vis = True

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.show_depth_points or args_cli.show_depth_image:
        env_cfg.scene.rgbd_camera.enable = True
        env_cfg.scene.rgbd_camera.enable_depth = True
        if args_cli.camera_offset_pos is not None:
            env_cfg.scene.rgbd_camera.spawn_offset_pos = tuple(args_cli.camera_offset_pos)
        if args_cli.camera_offset_rot is not None:
            env_cfg.scene.rgbd_camera.spawn_offset_rot = tuple(args_cli.camera_offset_rot)
        if args_cli.camera_random_pitch_deg is not None:
            env_cfg.scene.rgbd_camera.random_pitch_deg = tuple(args_cli.camera_random_pitch_deg)
        if args_cli.camera_fov_deg is not None:
            env_cfg.scene.rgbd_camera.horizontal_fov_deg = float(args_cli.camera_fov_deg)
        if args_cli.camera_disable_random_rotation:
            env_cfg.scene.rgbd_camera.randomize_rotation = False
        if env_cfg.scene.rgbd_camera.partial_camera:
            env_cfg.scene.rgbd_camera.partial_camera_num_envs = env_cfg.scene.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)
    if args_cli.virtual_joystick:
        # Interactive playback resets only on explicit requests or safety terminations.
        env.time_out_enabled = False
    if not args_cli.headless:
        # A depth-only Isaac RTX camera may globally disable color rendering when the Kit visualizer
        # is not recognized by its legacy has_gui check, leaving the interactive viewport black.
        env.sim.set_setting("/rtx/sdg/force/disableColorRender", False)
        robot_pos = env.robot.data.root_pos_w[0].detach().cpu()
        target = [float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])]
        eye = [target[0] + 2.8, target[1] - 2.8, target[2] + 1.4]
        env.sim.set_camera_view(eye=eye, target=target)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    depth_image_dir = args_cli.depth_image_dir or os.path.join(log_dir, "depth_images")

    cfg_dict = agent_cfg.to_dict()
    if not cfg_dict.get("obs_groups"):
        cfg_dict["obs_groups"] = {
            "policy": ["policy"],
            "critic": ["critic"],
        }
    if is_rsl_rl_v5_plus():
        print("[INFO] Detected rsl_rl v5+, applying legacy cfg compatibility mapping.")
        cfg_dict = adapt_legacy_cfg_for_rsl_rl_v5(cfg_dict)

    if args_cli.runner in ("wmp", "wmp_amp"):
        from legged_lab.runners import WMPAMPRunner, WMPRunner

        runner_cls = WMPRunner if args_cli.runner == "wmp" else WMPAMPRunner
        runner = runner_cls(env, cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    if is_rsl_rl_v5_plus():
        # rsl_rl 5.x: load() 使用 load_cfg，不再支持 load_optimizer 参数。
        runner.load(
            resume_path,
            load_cfg={
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": True,
                "rnd": False,
            },
            map_location=agent_cfg.device,
        )
    else:
        # 旧版本兼容路径
        try:
            runner.load(resume_path, load_optimizer=False)
        except TypeError:
            runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    obs_normalizer = getattr(runner, "obs_normalizer", None)
    export_policy = getattr(getattr(runner, "alg", None), "policy", None)
    if export_policy is None and hasattr(getattr(runner, "alg", None), "get_policy"):
        export_policy = runner.alg.get_policy()
    if export_policy is None:
        export_policy = policy

    # 兼容不同 rsl_rl 版本: 部分版本没有 obs_normalizer，导出失败不应影响仿真评估主流程
    try:
        export_policy_as_jit(export_policy, obs_normalizer, path=export_model_dir, filename="policy.pt")
    except TypeError:
        export_policy_as_jit(export_policy, path=export_model_dir, filename="policy.pt")
    except Exception as exc:
        print(f"[WARN] Failed to export JIT policy: {exc}")

    try:
        export_policy_as_onnx(export_policy, normalizer=obs_normalizer, path=export_model_dir, filename="policy.onnx")
    except TypeError:
        export_policy_as_onnx(export_policy, path=export_model_dir, filename="policy.onnx")
    except Exception as exc:
        print(f"[WARN] Failed to export ONNX policy: {exc}")

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    joystick = None
    joystick_window = None
    if args_cli.virtual_joystick:
        from legged_lab.utils.virtual_joystick import VirtualJoystickServer, open_joystick_window

        joystick = VirtualJoystickServer(
            port=args_cli.joystick_port,
            max_vx=joystick_limits[0],
            max_vy=joystick_limits[1],
            max_wz=joystick_limits[2],
            timeout=args_cli.joystick_timeout,
        )
        joystick.start()
        env.set_command_override(torch.tensor(joystick.command(), device=env.device))
        print(f"[INFO] Virtual joystick: {joystick.url}", flush=True)
        if not args_cli.no_open_joystick:
            joystick_window = open_joystick_window(joystick.url)
            print(f"[INFO] Opened virtual joystick in {joystick_window}.", flush=True)

    obs = env.get_observations()
    depth_camera = env.scene.sensors.get("rgbd_camera") if args_cli.show_depth_points or args_cli.show_depth_image else None
    depth_draw = _acquire_debug_draw_interface() if args_cli.show_depth_points else None
    if depth_camera is not None:
        # TiledCamera 默认返回初始化时的相机位姿；红点需要跟随机器人上的 RGBD，
        # 因此播放调试时开启最新位姿更新。
        depth_camera.cfg.update_latest_camera_pose = True
        camera_ids = getattr(depth_camera, "camera_env_ids", None)
        if camera_ids is not None:
            print(f"[INFO] Depth point debug camera envs: {int(camera_ids.numel())}/{env.num_envs}", flush=True)

    depth_debug_counter = 0
    depth_image_step = 0
    # The explicit window mode is displayed by a separate process so Kit and OpenCV do not load Qt in one process.
    depth_window_available = args_cli.depth_image_mode == "auto"
    play_step = 0
    wmp_play_episode_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    try:
        while simulation_app.is_running() and (args_cli.max_steps <= 0 or play_step < args_cli.max_steps):

            with torch.inference_mode():
                if joystick is not None:
                    env.set_command_override(torch.tensor(joystick.command(), device=env.device))
                    if joystick.consume_reset():
                        env.request_reset()
                        print("[INFO] Virtual joystick requested robot reset.", flush=True)
                if args_cli.runner in ("wmp", "wmp_amp"):
                    wm_obs, wm_feature = runner.wmp_controller.observe_before_policy()
                    wm_feature = wm_feature.to(env.device)
                    obs["wmp"] = wm_feature
                actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)
                if args_cli.runner in ("wmp", "wmp_amp"):
                    wmp_dones = dones
                    if joystick is not None:
                        wmp_play_episode_steps += 1
                        logical_time_outs = wmp_play_episode_steps >= int(env.max_episode_length)
                        wmp_dones = dones | logical_time_outs
                        wmp_play_episode_steps[wmp_dones] = 0
                        if torch.any(logical_time_outs):
                            command0 = env.command_generator.command[0].detach().cpu().tolist()
                            print(
                                "[INFO] Reset WMP episode state without resetting the robot; "
                                f"command0=({command0[0]:.3f}, {command0[1]:.3f}, {command0[2]:.3f}).",
                                flush=True,
                            )
                    runner.wmp_controller.after_env_step(actions, rewards, wmp_dones, wm_obs)
                if depth_camera is not None and args_cli.show_depth_image:
                    depth_window_available = _handle_wmp_depth_image(
                        depth_camera,
                        near=env_cfg.scene.rgbd_camera.depth_near,
                        far=env_cfg.scene.rgbd_camera.depth_far,
                        mode=args_cli.depth_image_mode,
                        output_dir=depth_image_dir,
                        step=depth_image_step,
                        save_interval=args_cli.depth_image_save_interval,
                        window_available=depth_window_available,
                    )
                    depth_image_step += 1
                if depth_camera is not None and depth_draw is not None:
                    depth_points, depth_origins = _depth_to_hit_points(
                        depth_camera,
                        near=env_cfg.scene.rgbd_camera.depth_near,
                        far=env_cfg.scene.rgbd_camera.depth_far,
                        stride=args_cli.depth_point_stride,
                        max_points=args_cli.depth_point_max,
                        forward_min=args_cli.depth_point_forward_min,
                        forward_max=args_cli.depth_point_forward_max,
                        min_z=args_cli.depth_point_min_z,
                        max_z=args_cli.depth_point_max_z,
                        lift=args_cli.depth_point_lift,
                        camera_index=args_cli.depth_point_camera_index,
                    )
                    _draw_depth_hit_points(depth_draw, depth_points, args_cli.depth_point_size)
                    if args_cli.depth_point_draw_rays:
                        _draw_depth_rays(depth_draw, depth_origins, depth_points)
                    if args_cli.depth_point_debug and depth_debug_counter % 60 == 0:
                        print(
                            f"[INFO] Depth points: "
                            f"{_depth_debug_stats(depth_camera, depth_points, depth_origins, args_cli.depth_point_camera_index)}",
                            flush=True,
                        )
                    if not args_cli.headless:
                        env.sim.render()
                    depth_debug_counter += 1
                play_step += 1
    finally:
        if joystick_window is not None:
            joystick_window.close()
        if joystick is not None:
            joystick.stop()
        env.close()


if __name__ == "__main__":
    play()
    simulation_app.close()
