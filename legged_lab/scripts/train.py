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

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import os
from datetime import datetime

import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def setup_wandb_env(logger_name: str):
    """根据 CLI 参数设置 wandb 环境变量，兼容 rsl_rl 的 WandbSummaryWriter。"""
    if logger_name != "wandb":
        return

    if args_cli.wandb_entity:
        os.environ["WANDB_USERNAME"] = args_cli.wandb_entity
        os.environ["WANDB_ENTITY"] = args_cli.wandb_entity

    if args_cli.wandb_mode:
        os.environ["WANDB_MODE"] = args_cli.wandb_mode

    if args_cli.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args_cli.wandb_api_key


def train():
    runner: OnPolicyRunner

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    env_class = task_registry.get_task_class(env_class_name)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed
    print(f"[INFO] Effective logger: {agent_cfg.logger}")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.scene.seed = seed
        agent_cfg.seed = seed

    # 在构造 runner 前配置 wandb 环境变量
    setup_wandb_env(agent_cfg.logger)

    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    cfg_dict = agent_cfg.to_dict()

    print("\n========== agent cfg keys ==========")
    print(cfg_dict.keys())
    print("\n========== obs_groups before ==========")
    print(cfg_dict.get("obs_groups", None))

    if not cfg_dict.get("obs_groups"):
        cfg_dict["obs_groups"] = {
            "policy": ["policy"],
            "critic": ["critic"],
        }

    print("\n========== obs_groups after ==========")
    print(cfg_dict["obs_groups"])

    runner = OnPolicyRunner(
        env,
        cfg_dict,
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    remaining_iterations = agent_cfg.max_iterations

    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

        # 兼容 rsl_rl 的语义: learn(num_learning_iterations) 在 resume 时表示“追加轮次”
        # 这里将其转换为“训练到 max_iterations 为止”
        completed_iterations = runner.current_learning_iteration + 1
        remaining_iterations = max(agent_cfg.max_iterations - completed_iterations, 0)
        print(
            f"[INFO] Resume iteration: {runner.current_learning_iteration}, "
            f"target max_iterations: {agent_cfg.max_iterations}, "
            f"remaining_iterations: {remaining_iterations}"
        )

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    if remaining_iterations <= 0:
        print("[INFO] Max iterations reached. Skip training.")
        return

    runner.learn(num_learning_iterations=remaining_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    train()
    simulation_app.close()
