# LeggedLab (Fork by mr-cheng111)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RL](https://img.shields.io/badge/RSL_RL-3.1.2-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Overview

这是一个基于 IsaacLab 的足式机器人强化学习训练仓库（个人维护分支）。

当前分支重点是：
- 保持原始 LeggedLab 的直接环境工作流
- 便于按个人需求快速修改训练脚本、日志和任务配置
- 支持使用 `wandb` 作为默认训练日志后端

## Fork Statement

本仓库基于原作者项目二次开发：
- Original repository: https://github.com/Hellod035/LeggedLab
- Original author: Wandong Sun

本仓库在保留原始许可证（BSD-3-Clause）的前提下进行修改与扩展。

## Installation

建议先安装 Isaac Lab（推荐 conda 方式）：
- Isaac Lab install guide: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

然后克隆本仓库：

```bash
git clone git@github.com:mr-cheng111/LeggedLab.git
# 或
# git clone https://github.com/mr-cheng111/LeggedLab.git

cd LeggedLab
pip install -e .
```

## Quick Start

默认训练（当前默认 logger 为 wandb）：

```bash
python legged_lab/scripts/train.py --task=g1_flat --headless --num_envs=64
```

显式指定 logger：

```bash
python legged_lab/scripts/train.py --task=g1_flat --headless --logger=wandb --num_envs=64
```

策略回放：

```bash
python legged_lab/scripts/play.py --task=g1_flat --load_run=<run_dir> --checkpoint=<model_xxx.pt>
```

## Available Tasks

当前注册任务包括：
- `h1_flat`, `h1_rough`
- `g1_flat`, `g1_rough`
- `gr2_flat`, `gr2_rough`

注册位置：`legged_lab/envs/__init__.py`

## Use Your Own Robot

如需接入自己的机器人：
1. 将机器人资产转换为 USD
2. 在 `legged_lab/assets` 中添加资产配置
3. 在 `legged_lab/envs` 中新增环境与 agent 配置
4. 在 `legged_lab/envs/__init__.py` 中注册新 task

参考文档：
- https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html

## Multi-GPU and Multi-Node Training

本仓库沿用 IsaacLab / rsl_rl 的并行训练方式：
- https://isaac-sim.github.io/IsaacLab/main/source/features/multi_gpu.html

## Troubleshooting

### Pylance 缺少索引

如果 VSCode 无法正确索引扩展，可在 `.vscode/settings.json` 增加：

```json
{
  "python.analysis.extraPaths": [
    "${workspaceFolder}/legged_lab",
    "<path-to-IsaacLab>/source/isaaclab_tasks",
    "<path-to-IsaacLab>/source/isaaclab_mimic",
    "<path-to-IsaacLab>/source/extensions",
    "<path-to-IsaacLab>/source/isaaclab_assets",
    "<path-to-IsaacLab>/source/isaaclab_rl",
    "<path-to-IsaacLab>/source/isaaclab"
  ]
}
```

## Acknowledgements

感谢以下开源项目：
- IsaacLab: https://github.com/isaac-sim/IsaacLab
- rsl_rl: https://github.com/leggedrobotics/rsl_rl
- legged_gym: https://github.com/leggedrobotics/legged_gym
- ProtoMotions: https://github.com/NVlabs/ProtoMotions

同时特别感谢原始 LeggedLab 作者及贡献者。

## Citation

如果你在学术工作中使用原始 LeggedLab，请优先引用原项目：

```bibtex
@software{LeggedLab,
  author = {Wandong, Sun},
  license = {BSD-3-Clause},
  title = {Legged Lab: Direct IsaacLab Workflow for Legged Robots},
  url = {https://github.com/Hellod035/LeggedLab},
  version = {1.0.0},
  year = {2025}
}
```

如果引用本 Fork，请在文中注明基于原始 LeggedLab 的二次开发版本。

## System Architecture

当前 LeggedLab 的核心链路是：`scripts -> task_registry -> BaseEnv(VecEnv) -> rsl_rl(OnPolicyRunner)`。

```text
                        CLI / AppLauncher
                               |
                +--------------+--------------+
                |                             |
         scripts/train.py                scripts/play.py
                |                             |
                +---------- task_registry -----+
                           (task -> cfg/class)
                               |
             +-----------------+-----------------+
             |                                   |
       EnvCfg / AgentCfg                    EnvClass(BaseEnv)
             |                                   |
             +-----------------+-----------------+
                               |
                         BaseEnv (VecEnv)
      (Scene/Sim + CommandGenerator + RewardManager + EventManager)
                               |
                   obs(TensorDict), reward, done, info
                               |
                      rsl_rl OnPolicyRunner
                 (PPO update / inference policy)
                               |
              logs, checkpoints, exported policy(jit/onnx)
```

训练时序（简化）：

```text
train.py -> get_cfgs -> BaseEnv()
       -> OnPolicyRunner(env,cfg)
       -> learn():
          for iter:
            act -> env.step -> collect rollout
            PPO update
            log/save
```
