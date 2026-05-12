# -*- coding: utf-8 -*-
"""检查 WMP RSSM world model 在 b2_rgbd 上的前向链路。"""

import argparse
import os
import traceback

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry

parser = argparse.ArgumentParser(description="Inspect WMP RSSM world model with b2_rgbd Gemini2 depth.")
parser.add_argument("--task", type=str, default="b2_rgbd_rough", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps", type=int, default=5, help="Number of zero-action warmup steps.")
parser.add_argument("--wm_device", type=str, default=None, help="World model device. Defaults to env device.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.envs import *  # noqa:F401,F403
from legged_lab.world_models.wmp import WorldModel, depth_to_nchw, depth_to_wmp_image, make_default_wmp_config


def _log(message: str):
    print(f"[INFO] {message}", flush=True)


def main():
    _log(f"loading task config: {args_cli.task}")
    env_cfg, _ = task_registry.get_cfgs(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.seed = 42
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.ranges.heading = (0.0, 0.0)
    env_cfg.scene.gemini2_camera.enable = True
    if args_cli.device is not None:
        env_cfg.device = args_cli.device

    _log("constructing IsaacLab environment")
    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)
    exit_code = 1
    try:
        env.sim.set_camera_view(eye=[2.8, -2.8, 1.6], target=[0.0, 0.0, 0.45])
        depth_camera = env.scene.sensors["gemini2_depth_camera"]

        depth = None
        actor_obs = None
        for step_idx in range(args_cli.steps):
            _log(f"warmup step {step_idx + 1}/{args_cli.steps}")
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
            env.step(actions)
            depth = depth_camera.data.output["distance_to_image_plane"]
            actor_obs, _ = env.compute_observations()

        assert depth is not None and actor_obs is not None
        wm_device = args_cli.wm_device or env.device
        _log(f"preprocessing Gemini2 depth on device={wm_device}")
        prop = actor_obs.to(wm_device)
        depth_nchw = depth_to_nchw(
            depth.to(wm_device),
            near=env_cfg.scene.gemini2_camera.depth_near,
            far=env_cfg.scene.gemini2_camera.depth_far,
        )
        image = depth_to_wmp_image(
            depth.to(wm_device),
            near=env_cfg.scene.gemini2_camera.depth_near,
            far=env_cfg.scene.gemini2_camera.depth_far,
        )
        is_first = torch.ones(env.num_envs, device=wm_device)

        _log("constructing WMP WorldModel")
        cfg = make_default_wmp_config(device=wm_device, num_actions=env.num_actions)
        obs_shape = {"prop": (prop.shape[-1],), "image": (64, 64, 1)}
        world_model = WorldModel(cfg, obs_shape, use_camera=True).to(wm_device)
        world_model.eval()

        _log("running encoder, RSSM obs_step, and decoder")
        wm_obs = {"prop": prop, "image": image, "is_first": is_first}
        with torch.inference_mode():
            embed = world_model.encoder(wm_obs)
            latent = world_model.dynamics.initial(env.num_envs)
            action = torch.zeros(env.num_envs, env.num_actions, device=wm_device)
            post, _ = world_model.dynamics.obs_step(latent, action, embed, is_first, sample=False)
            deter_feat = world_model.dynamics.get_deter_feat(post)
            full_feat = world_model.dynamics.get_feat(post)
            decoded = world_model.decode(full_feat)["image"].mode()

        _log(f"depth raw shape={tuple(depth.shape)}")
        _log(f"depth preprocessed NCHW shape={tuple(depth_nchw.shape)}")
        _log(f"WMP image NHWC shape={tuple(image.shape)}")
        _log(f"prop shape={tuple(prop.shape)}")
        _log(f"encoder embed shape={tuple(embed.shape)}")
        _log(f"RSSM deter feature shape={tuple(deter_feat.shape)}")
        _log(f"RSSM full feature shape={tuple(full_feat.shape)}")
        _log(f"decoded image shape={tuple(decoded.shape)}")

        assert depth_nchw.shape == (env.num_envs, 1, 64, 64)
        assert image.shape == (env.num_envs, 64, 64, 1)
        assert deter_feat.shape == (env.num_envs, 512)
        assert full_feat.shape == (env.num_envs, 1536)
        assert decoded.shape == (env.num_envs, 64, 64, 1)
        _log("WMP WorldModel smoke test passed")
        exit_code = 0
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if args_cli.headless:
            _log(f"terminating headless IsaacSim process with exit_code={exit_code}")
            os._exit(exit_code)
        if hasattr(env, "close"):
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
