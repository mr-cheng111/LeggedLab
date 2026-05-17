# -*- coding: utf-8 -*-
"""检查 WMP x_edge_mask 是否挂到 IsaacLab env。"""

import argparse
import sys

from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

parser = argparse.ArgumentParser(description="Inspect WMP x_edge_mask in LeggedLab env.")
parser.add_argument("--task", type=str, default="b2_rgbd_wmp_amp_flat", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--wmp_terrain", choices=("slope", "stair", "gap", "climb", "tilt", "crawl"), default="gap")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.terrains import WMP_TERRAIN_CFGS


def main():
    env_cfg, _ = task_registry.get_cfgs(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.seed = 42
    env_cfg.scene.terrain_type = "generator"
    env_cfg.scene.terrain_generator = WMP_TERRAIN_CFGS[args_cli.wmp_terrain]
    env_cfg.scene.max_init_terrain_level = 0
    env_cfg.scene.gemini2_camera.enable = False
    env_cfg.scene.gemini2_camera.enable_rgb = False
    env_cfg.scene.gemini2_camera.enable_depth = False
    env_cfg.scene.gemini2_camera.allow_missing_depth_fallback = True
    env_cfg.noise.add_noise = False

    if args_cli.device is not None:
        env_cfg.device = args_cli.device

    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)
    mask = getattr(env, "x_edge_mask", None)
    if mask is None:
        raise RuntimeError("env.x_edge_mask is missing.")
    print(f"[INFO] terrain={args_cli.wmp_terrain}")
    print(f"[INFO] x_edge_mask shape={tuple(mask.shape)} true_count={int(mask.sum().item())}")
    print(f"[INFO] edge_query_offset={env.wmp_edge_query_offset}")
    print(f"[INFO] horizontal_scale={env.wmp_terrain_horizontal_scale}")

    if hasattr(env, "close"):
        env.close()
    simulation_app.close()
    if args_cli.headless:
        sys.exit(0)


if __name__ == "__main__":
    main()
