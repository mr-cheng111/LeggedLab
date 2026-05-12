# LeggedLab (Fork by mr-cheng111)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-0.54.3-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RL](https://img.shields.io/badge/RSL_RL-5.0.1-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
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

当前维护环境：
- conda 环境：`isaaclab`
- Python：`3.11`
- IsaacSim：`5.1.0.0`
- IsaacLab：`0.54.3`
- rsl_rl / rsl-rl-lib：`5.0.1`

然后克隆本仓库：

```bash
git clone git@github.com:mr-cheng111/LeggedLab.git
# 或
# git clone https://github.com/mr-cheng111/LeggedLab.git

cd LeggedLab
pip install -e .
```

## Quick Start

进入训练环境：

```bash
conda activate isaaclab
cd /home/tower/Bags/LeggedLab
```

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

可视化启动器（可填写 train/play 参数并一键运行）：

```bash
python legged_lab/scripts/launcher_gui.py
```

说明：
- 启动器会优先通过 `train.py --help` / `play.py --help` 自动提取参数（包含 IsaacLab `AppLauncher` 参数）
- 如果当前环境未正确安装 IsaacLab，会回退到源码解析（此时不含 `AppLauncher` 动态参数）
- 可通过“额外参数”输入框补充任意原生命令行参数

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

## WMP RSSM World Model Smoke Test

当前分支已开始移植 WMP（World Model-based Perception for Visual Legged Locomotion）的 RSSM 世界模型到
`b2_rgbd`。本阶段只提供模型前向检查，不接入 PPO 联训，也不改变现有 `b2_rgbd_*` 训练观测。

当前链路：

```text
b2_rgbd Gemini2 depth
      -> clean/clip/normalize
      -> resize to 1x64x64 (NCHW)
      -> convert to WMP image format 64x64x1 (NHWC)
      -> WMP MultiEncoder
      -> RSSM obs_step
      -> deter feature(512) / full feature(1536)
      -> MultiDecoder image reconstruction smoke test
```

运行检查：

```bash
env CONDA_PREFIX=/home/tower/miniconda/envs/isaaclab \
/home/tower/Bags/IsaacLab/isaaclab.sh -p legged_lab/scripts/inspect_wmp_world_model.py \
  --task=b2_rgbd_rough \
  --num_envs=1 \
  --headless \
  --enable_cameras \
  --steps=5
```

预期输出包含：
- `depth preprocessed NCHW shape=(1, 1, 64, 64)`
- `WMP image NHWC shape=(1, 64, 64, 1)`
- `RSSM deter feature shape=(1, 512)`
- `RSSM full feature shape=(1, 1536)`
- `decoded image shape=(1, 64, 64, 1)`

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
