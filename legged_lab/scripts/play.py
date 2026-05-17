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
parser.add_argument("--runner", type=str, default="default", choices=["default", "wmp_amp"], help="Runner/checkpoint type.")
parser.add_argument("--show_depth_points", action="store_true", help="Visualize Gemini2 depth hits as red debug points.")
parser.add_argument("--enable_play_push", action="store_true", help="Keep interval push disturbances enabled during play.")
parser.add_argument(
    "--hide_command", action="store_true", help="Hide command/current velocity debug visualization during play."
)
parser.add_argument("--depth_point_stride", type=int, default=16, help="Pixel stride for depth hit point visualization.")
parser.add_argument("--depth_point_max", type=int, default=300, help="Maximum depth hit points to draw.")
parser.add_argument("--depth_point_size", type=float, default=5.0, help="Debug draw size of each red depth hit point.")
parser.add_argument("--depth_point_forward_min", type=float, default=0.2, help="Minimum forward distance for depth hit points.")
parser.add_argument("--depth_point_forward_max", type=float, default=3.0, help="Maximum forward distance for depth hit points.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab.sensors import patterns
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.utils.math import quat_apply

try:
    from isaacsim.util.debug_draw import _debug_draw
except ModuleNotFoundError:
    from omni.isaac.debug_draw import _debug_draw

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from legged_lab.utils.rsl_rl_compat import adapt_legacy_cfg_for_rsl_rl_v5, is_rsl_rl_v5_plus
from legged_lab.world_models.wmp import depth_to_wmp_image


def _depth_to_hit_points(
    depth_camera,
    near: float,
    far: float,
    stride: int,
    max_points: int,
    forward_min: float,
    forward_max: float,
) -> torch.Tensor:
    depth = depth_camera.data.output["distance_to_image_plane"][..., 0]
    device = depth.device
    num_cameras, height, width = depth.shape
    stride = max(stride, 1)

    pattern_cfg = patterns.PinholeCameraPatternCfg(width=width, height=height)
    _, ray_dirs = patterns.pinhole_camera_pattern(pattern_cfg, depth_camera.data.intrinsic_matrices, device)
    ray_dirs = ray_dirs[:, ::stride, :]
    sampled_depth = depth.reshape(num_cameras, -1)[:, ::stride]
    valid = (
        torch.isfinite(sampled_depth)
        & (sampled_depth > near)
        & (sampled_depth < far)
        & (sampled_depth > forward_min)
        & (sampled_depth < forward_max)
    )
    # IsaacLab 的 pinhole pattern 已用 K^-1 [u, v, 1]^T 生成单位射线方向。
    # distance_to_image_plane 是光轴平面深度 d，pattern 的相机前向轴为 +X，
    # 因此射线尺度 s = d / ray_dir_x，点位 p_cam = s * ray_dir。
    points_cam = ray_dirs * (sampled_depth / torch.clamp(ray_dirs[..., 0], min=1.0e-6)).unsqueeze(-1)
    points_world = depth_camera.data.pos_w[:, None, :] + quat_apply(
        depth_camera.data.quat_w_world[:, None, :].expand(-1, points_cam.shape[1], -1).reshape(-1, 4),
        points_cam.reshape(-1, 3),
    ).reshape(num_cameras, -1, 3)
    points_world = points_world[valid]
    if points_world.shape[0] > max_points:
        step = torch.ceil(torch.tensor(points_world.shape[0] / max_points, device=device)).long().item()
        points_world = points_world[::step][:max_points]
    return points_world


def _draw_depth_hit_points(draw_interface, points: torch.Tensor, point_size: float):
    draw_interface.clear_points()
    if points.numel() == 0:
        return
    points_list = points.detach().cpu().tolist()
    colors = [(1.0, 0.0, 0.0, 1.0)] * len(points_list)
    sizes = [point_size] * len(points_list)
    draw_interface.draw_points(points_list, colors, sizes)


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    if not args_cli.enable_play_push:
        env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    if "slow_walk" in env_class_name:
        env_cfg.commands.ranges.lin_vel_x = (-0.2, 1.0)
        env_cfg.commands.ranges.lin_vel_y = (1.0, 1.0)
        env_cfg.commands.ranges.ang_vel_z = (-0.5, 0.5)
        env_cfg.commands.ranges.heading = (0.0, 0.0)
    elif "stand" not in env_class_name:
        env_cfg.commands.ranges.lin_vel_x = (-0.6, 0.6)
        env_cfg.commands.ranges.lin_vel_y = (0.0, 1.0)
        env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.commands.debug_vis = not args_cli.hide_command
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    # env_cfg.scene.terrain_generator = None
    # env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.show_depth_points:
        env_cfg.scene.gemini2_camera.enable = True

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    cfg_dict = agent_cfg.to_dict()
    if not cfg_dict.get("obs_groups"):
        cfg_dict["obs_groups"] = {
            "policy": ["policy"],
            "critic": ["critic"],
        }
    if is_rsl_rl_v5_plus():
        print("[INFO] Detected rsl_rl v5+, applying legacy cfg compatibility mapping.")
        cfg_dict = adapt_legacy_cfg_for_rsl_rl_v5(cfg_dict)

    if args_cli.runner == "wmp_amp":
        from legged_lab.runners import WMPAMPRunner

        runner = WMPAMPRunner(env, cfg_dict, log_dir=log_dir, device=agent_cfg.device)
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

    obs = env.get_observations()
    wm_latent = None
    wm_action = torch.zeros(env.num_envs, env.num_actions, device=getattr(runner, "wm_device", env.device))
    wm_is_first = torch.ones(env.num_envs, device=getattr(runner, "wm_device", env.device))
    depth_camera = env.scene.sensors.get("gemini2_depth_camera") if args_cli.show_depth_points else None
    depth_draw = _debug_draw.acquire_debug_draw_interface() if args_cli.show_depth_points else None
    if depth_camera is not None:
        # TiledCamera 默认返回初始化时的相机位姿；红点需要跟随机器人上的 Gemini2，
        # 因此播放调试时开启最新位姿更新。
        depth_camera.cfg.update_latest_camera_pose = True

    while simulation_app.is_running():

        with torch.inference_mode():
            if args_cli.runner == "wmp_amp":
                depth = env.get_depth_observations()
                wm_obs = {
                    "prop": env.get_wmp_proprioception().to(runner.wm_device),
                    "image": depth_to_wmp_image(
                        depth,
                        near=env_cfg.scene.gemini2_camera.depth_near,
                        far=env_cfg.scene.gemini2_camera.depth_far,
                    ).to(runner.wm_device),
                    "is_first": wm_is_first,
                }
                wm_embed = runner.world_model.encoder(wm_obs)
                wm_latent, _ = runner.world_model.dynamics.obs_step(wm_latent, wm_action, wm_embed, wm_is_first)
                wm_feature = runner._wm_feature(wm_latent).to(env.device)
                obs["wmp"] = wm_feature
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            if args_cli.runner == "wmp_amp":
                wm_action = actions.to(runner.wm_device)
                wm_is_first = torch.zeros(env.num_envs, device=runner.wm_device)
            if depth_camera is not None and depth_draw is not None:
                depth_points = _depth_to_hit_points(
                    depth_camera,
                    near=env_cfg.scene.gemini2_camera.depth_near,
                    far=env_cfg.scene.gemini2_camera.depth_far,
                    stride=args_cli.depth_point_stride,
                    max_points=args_cli.depth_point_max,
                    forward_min=args_cli.depth_point_forward_min,
                    forward_max=args_cli.depth_point_forward_max,
                )
                _draw_depth_hit_points(depth_draw, depth_points, args_cli.depth_point_size)


if __name__ == "__main__":
    play()
    simulation_app.close()
