# -*- coding: utf-8 -*-
"""预览 b2_rgbd 地形，并检查 RGBD 渲染结果。"""

import argparse
import os
import sys
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

parser = argparse.ArgumentParser(description="Inspect terrain and RGBD rendering for b2_rgbd tasks.")
parser.add_argument("--task", type=str, default="b2_rgbd_rough", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps", type=int, default=30, help="Number of zero-action steps.")
parser.add_argument("--save_dir", type=str, default="logs/rgbd_preview", help="Directory for saved images.")
parser.add_argument("--save_images", action="store_true", help="Save RGB and depth preview images.")
parser.add_argument("--terrain", choices=("task", "plane"), default="task", help="Terrain to render.")
parser.add_argument(
    "--wmp_terrain",
    choices=("slope", "stair", "gap", "climb", "tilt", "crawl"),
    default=None,
    help="Override task terrain with a single WMP-style terrain preset.",
)
parser.add_argument(
    "--rgb_camera_path",
    type=str,
    default=None,
    help="RGB camera prim path under Robot.",
)
parser.add_argument(
    "--depth_camera_path",
    type=str,
    default=None,
    help="Depth camera prim path under Robot.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.camera.utils import save_images_to_file

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.terrains import WMP_TERRAIN_CFGS


def _normalize_depth(depth: torch.Tensor, near: float, far: float) -> torch.Tensor:
    finite = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    # 深度图可视化: d_vis = clamp((d - near) / (far - near), 0, 1)
    return torch.clamp((finite - near) / (far - near), 0.0, 1.0)


def main():
    env_cfg, _ = task_registry.get_cfgs(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.seed = 42
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    if args_cli.terrain == "plane":
        env_cfg.scene.terrain_type = "plane"
        env_cfg.scene.terrain_generator = None
    if args_cli.wmp_terrain is not None:
        env_cfg.scene.terrain_type = "generator"
        env_cfg.scene.terrain_generator = WMP_TERRAIN_CFGS[args_cli.wmp_terrain]
        env_cfg.scene.max_init_terrain_level = 0
    if args_cli.device is not None:
        env_cfg.device = args_cli.device

    env_cfg.scene.gemini2_camera.enable = True
    if args_cli.rgb_camera_path is not None:
        env_cfg.scene.gemini2_camera.rgb_camera_path = args_cli.rgb_camera_path
    if args_cli.depth_camera_path is not None:
        env_cfg.scene.gemini2_camera.depth_camera_path = args_cli.depth_camera_path
    rgb_camera_prim_path = "{ENV_REGEX_NS}/Robot/" + env_cfg.scene.gemini2_camera.rgb_camera_path.strip("/")
    depth_camera_prim_path = "{ENV_REGEX_NS}/Robot/" + env_cfg.scene.gemini2_camera.depth_camera_path.strip("/")

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)
    env.sim.set_camera_view(eye=[2.8, -2.8, 1.6], target=[0.0, 0.0, 0.45])
    rgb_camera = env.scene.sensors["gemini2_rgb_camera"]
    depth_camera = env.scene.sensors["gemini2_depth_camera"]
    print(f"[INFO] reading RGB camera prim: {rgb_camera_prim_path}")
    print(f"[INFO] reading depth camera prim: {depth_camera_prim_path}")

    rgb = None
    depth = None
    for _ in range(args_cli.steps):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        env.step(actions)
        rgb = rgb_camera.data.output["rgb"]
        depth = depth_camera.data.output["distance_to_image_plane"]

    assert rgb is not None and depth is not None
    finite_depth = depth[torch.isfinite(depth)]
    print(f"[INFO] rgb shape={tuple(rgb.shape)} dtype={rgb.dtype}")
    print(
        "[INFO] depth "
        f"shape={tuple(depth.shape)} dtype={depth.dtype} "
        f"min={finite_depth.min().item():.3f} "
        f"max={finite_depth.max().item():.3f} "
        f"mean={finite_depth.mean().item():.3f}"
    )

    if args_cli.save_images:
        save_dir = Path(args_cli.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = save_dir / "rgb.png"
        depth_path = save_dir / "depth.png"
        save_images_to_file(rgb.float() / 255.0, str(rgb_path))
        save_images_to_file(
            _normalize_depth(
                depth,
                env_cfg.scene.gemini2_camera.depth_near,
                env_cfg.scene.gemini2_camera.depth_far,
            ),
            str(depth_path),
        )
        print(f"[INFO] saved rgb: {os.path.abspath(rgb_path)}")
        print(f"[INFO] saved depth: {os.path.abspath(depth_path)}")

    while simulation_app.is_running() and not args_cli.headless:
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        env.step(actions)

    if hasattr(env, "close"):
        env.close()
    simulation_app.close()
    if args_cli.headless:
        sys.exit(0)


if __name__ == "__main__":
    main()
