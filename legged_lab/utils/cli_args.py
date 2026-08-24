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

import argparse
import os
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env_config import BaseAgentConfig


# 默认 wandb 参数（可被命令行覆盖）
DEFAULT_WANDB_ENTITY = ""
DEFAULT_WANDB_MODE = "online"
DEFAULT_WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
    arg_group.add_argument("--num_steps_per_env", type=int, default=None, help="Rollout steps per environment.")
    arg_group.add_argument(
        "--num_mini_batches",
        type=int,
        default=None,
        help="Number of mini-batches per PPO update.",
    )
    # 兼容常见拼写错误参数
    arg_group.add_argument("--num_mini_batces", type=int, default=None, help=argparse.SUPPRESS)

    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", type=bool, default=None, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    arg_group.add_argument(
        "--warm_start_checkpoint",
        type=str,
        default=None,
        help="Load model weights from this checkpoint but start a fresh training run from iteration 0.",
    )
    arg_group.add_argument(
        "--warm_start_load_optimizer",
        action="store_true",
        default=False,
        help="Also load optimizer states during warm-start. Defaults to false.",
    )
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default="wandb", choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    # -- wandb arguments
    arg_group.add_argument("--wandb_entity", type=str, default=DEFAULT_WANDB_ENTITY, help="Wandb entity/username.")
    arg_group.add_argument(
        "--wandb_mode",
        type=str,
        default=DEFAULT_WANDB_MODE,
        choices={"online", "offline", "disabled"},
        help="Wandb mode.",
    )
    arg_group.add_argument("--wandb_api_key", type=str, default=DEFAULT_WANDB_API_KEY, help="Wandb API key (default from WANDB_API_KEY env).")

    arg_group.add_argument(
        "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
    )
    arg_group.add_argument(
        "--amp_num_preload_transitions",
        type=int,
        default=None,
        help="Override AMP expert preload transitions for smoke tests.",
    )
    arg_group.add_argument(
        "--amp_reward_coef",
        type=float,
        default=None,
        help="Override AMP discriminator reward coefficient.",
    )
    arg_group.add_argument(
        "--amp_task_reward_lerp",
        type=float,
        default=None,
        help="Override AMP/task reward interpolation alpha in [0, 1].",
    )
    arg_group.add_argument(
        "--wmp_camera_num_envs",
        type=int,
        default=None,
        help="Override number of real WMP depth camera environments.",
    )
    arg_group.add_argument(
        "--wmp_depth_training_iters",
        type=int,
        default=None,
        help="Override DepthPredictor training iterations per training interval.",
    )
    arg_group.add_argument(
        "--wmp_depth_batch_size",
        type=int,
        default=None,
        help="Override DepthPredictor training batch size.",
    )
    arg_group.add_argument(
        "--wmp_train_steps_per_iter",
        type=int,
        default=None,
        help="Override world model gradient steps per PPO iteration.",
    )
    arg_group.add_argument(
        "--wmp_train_interval",
        type=int,
        default=None,
        help="Train world model every N PPO iterations after train_start_steps.",
    )


def update_rsl_rl_cfg(agent_cfg: BaseAgentConfig, args_cli: argparse.Namespace):

    # override the default configuration with CLI arguments
    if args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args_cli.num_steps_per_env

    num_mini_batches = args_cli.num_mini_batches
    if num_mini_batches is None and args_cli.num_mini_batces is not None:
        num_mini_batches = args_cli.num_mini_batces
    if num_mini_batches is not None:
        agent_cfg.algorithm.num_mini_batches = num_mini_batches

    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger

    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    if getattr(args_cli, "amp_num_preload_transitions", None) is not None and hasattr(agent_cfg, "amp"):
        agent_cfg.amp["num_preload_transitions"] = args_cli.amp_num_preload_transitions
    if getattr(args_cli, "amp_reward_coef", None) is not None and hasattr(agent_cfg, "amp"):
        agent_cfg.amp["reward_coef"] = args_cli.amp_reward_coef
    if getattr(args_cli, "amp_task_reward_lerp", None) is not None and hasattr(agent_cfg, "amp"):
        agent_cfg.amp["task_reward_lerp"] = args_cli.amp_task_reward_lerp
    if hasattr(agent_cfg, "wmp"):
        if getattr(args_cli, "wmp_camera_num_envs", None) is not None:
            agent_cfg.wmp["camera_num_envs"] = args_cli.wmp_camera_num_envs
        if getattr(args_cli, "wmp_train_steps_per_iter", None) is not None:
            agent_cfg.wmp["train_steps_per_iter"] = args_cli.wmp_train_steps_per_iter
        if getattr(args_cli, "wmp_train_interval", None) is not None:
            agent_cfg.wmp["train_interval"] = args_cli.wmp_train_interval
        depth_cfg = agent_cfg.wmp.setdefault("depth_predictor", {})
        if getattr(args_cli, "wmp_depth_training_iters", None) is not None:
            depth_cfg["training_iters"] = args_cli.wmp_depth_training_iters
        if getattr(args_cli, "wmp_depth_batch_size", None) is not None:
            depth_cfg["batch_size"] = args_cli.wmp_depth_batch_size

    return agent_cfg
