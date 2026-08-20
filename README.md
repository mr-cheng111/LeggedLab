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
- 增加 A1 WMP-AMP 训练链路、原版 WMP 对齐检查与辅助脚本
- 保留 B2 RGBD / WMP world model 前向检查脚本
- 增加 RB160W 轮腿机器人任务、混合腿位置/轮速度控制与模型验证脚本
- 增加 DEEP Robotics M20 官方 USD、平地/粗糙地形任务和轮足混合控制

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
python legged_lab/scripts/play.py --task=g1_flat --load_run=<run_dir> --checkpoint=<model_xxx.pt> --num_envs=1
```

说明：
- `play.py` 默认会把环境数设为 `50`，调试模型或相机任务时建议显式加 `--num_envs=1`。
- `--num_envs` 是完整参数名；`--num_env` 可能被 argparse 识别为缩写，但不建议依赖。

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
- `a1_amp_flat`, `a1_wmp_amp_terrain`
- `h1_flat`, `h1_rough`
- `g1_flat`, `g1_rough`
- `gr2_flat`, `gr2_rough`
- `b2_flat`, `b2_rough`
- `b2_rgbd_flat`, `b2_rgbd_stand`, `b2_rgbd_slow_walk`, `b2_rgbd_rough`
- `b2_rgbd_wmp_amp_flat`, `b2_rgbd_wmp_amp_terrain`
- `rb160w_flat`
- `m20_flat`, `m20_rough`, `m20_depth_rough`, `m20_depth_rough_amp`

注册位置：`legged_lab/envs/__init__.py`

## RB160W

`rb160w_flat` 使用 `RB160WEnv`，动作空间仍按机器人关节顺序输出：

- 腿部关节：位置目标，包含 `*_hip_joint`, `*_thigh_joint`, `*_calf_joint`
- 轮子关节：速度目标，包含 `*_WHEEL_joint`

训练示例：

```bash
python legged_lab/scripts/train.py --task=rb160w_flat --headless --num_envs=64 --logger=wandb
```

播放示例：

```bash
python legged_lab/scripts/play.py --task=rb160w_flat --num_envs=1 --hide_command
```

模型验证脚本：

```bash
# 打印 IsaacLab 实际读到的关节顺序、初始角、限位和 URDF 轴向
python legged_lab/scripts/inspect_rb160w_model.py --task=rb160w_flat --mode=print --headless

# 固定 root 和全部初始关节角，观察模型初始姿态
python legged_lab/scripts/inspect_rb160w_model.py --task=rb160w_flat --mode=hold

# 给单个关节赋固定偏移，并保持最后角度
python legged_lab/scripts/inspect_rb160w_model.py --task=rb160w_flat --mode=set --joint=FR_thigh_joint --angle=0.15

# 单关节正弦扫描，用于检查旋转方向
python legged_lab/scripts/inspect_rb160w_model.py --task=rb160w_flat --mode=sweep --joint=".*_thigh_joint"
```

注意：
- `inspect_rb160w_model.py` 默认会固定 root，防止验证时机器人自由下落；如需放开 root，可加 `--free_root`。
- 实际仿真加载的是 `legged_lab/assets/xuanji/rb160w/usd/rb160w.usd`。修改 URDF 只会影响源文件记录，除非重新导出 USD。
- `rb160w.usd` 当前接近 GitHub 单文件建议上限。若需要长期瘦身，推荐生成新的轻量版 USD 或使用 Git LFS，不建议直接在原 USD 上反复手工改。

## DEEP Robotics M20

文档中的机器人是云深处山猫 M20（Lynx M20），每条腿包含 `hipx`, `hipy`, `knee`, `wheel`
四个关节，共 16 个自由度。本仓库直接使用厂商
[`deep_robotics_model`](https://github.com/DeepRoboticsLab/deep_robotics_model) 中的官方 USD，
资产提交和许可证记录在 `legged_lab/assets/deeprobotics/SOURCE.md`。

动作空间对齐厂商训练和 ONNX 部署顺序（12 个腿关节在前，4 个轮关节在后）：

- 12 个腿关节使用位置目标；`hipx` 动作缩放为 `0.125 rad`，`hipy/knee` 为 `0.25 rad`
- 4 个轮关节使用速度目标，动作缩放为 `5.0 rad/s`

先用平地任务做资产和控制验证：

```bash
python legged_lab/scripts/train.py --task=m20_flat --headless --num_envs=64 --logger=tensorboard
```

平地策略稳定后，可启动本仓库通用粗糙地形基线：

```bash
python legged_lab/scripts/train.py --task=m20_rough --headless --num_envs=4096 --logger=tensorboard
```

`m20_rough` 复用本仓库的地形生成器和基础轮足奖励，并非厂商 `rl_training` 中全部 M20
专用转向步态奖励的逐项移植。需要直接复现厂商任务时，可使用官方
[`rl_training`](https://github.com/DeepRoboticsLab/rl_training) 的
`Rough-Deeprobotics-M20-v0`。

`m20_flat` 和 `m20_rough` 面向无视觉本体运动训练。`m20_depth_rough` 在 `base_link` 下挂载
64x64 前向深度相机，并使用不依赖 AMP 专家动作的 WMP-PPO：部分环境的真实深度监督
DepthPredictor，RSSM depth feature 进入 actor；理想高度扫描只用于深度预测监督和 critic。

深度任务训练示例：

```bash
python legged_lab/scripts/train.py \
  --task=m20_depth_rough \
  --num_envs=64 \
  --enable_cameras \
  --wmp_camera_num_envs=8 \
  --logger=tensorboard
```

使用 WMP checkpoint 回放时必须选择 `--runner=wmp`。例如回放仓库当前阶段验证模型：

```bash
python legged_lab/scripts/play.py \
  --task=m20_depth_rough \
  --runner=wmp \
  --num_envs=1 \
  --viz=kit \
  --load_run=2026-08-19_17-39-49_depth64_corrected \
  --checkpoint=model_19.pt
```

`model_19.pt` 只训练了 20 轮，用于验证 RTX depth -> DepthPredictor/RSSM -> PPO 的完整链路，
不是收敛或可部署模型。

### M20 AMP motion retarget

`m20_depth_rough_amp` 在深度 WMP 任务上增加 AMP。专家状态保持原版 30 维契约：
`joint_pos(12) + base_lin_vel_b(3) + base_ang_vel_b(3) + joint_vel(12)`。12 个关节严格按
`M20_LEG_JOINT_NAMES` 排列，4 个连续转动轮关节不进入 AMP 判别器。

使用已安装好 MuJoCo 和 Mink 的 `mujoco` 环境批量重定向原始 A1 motion：

```bash
conda run --no-capture-output -n mujoco \
  python legged_lab/scripts/retarget_amp_motion.py \
  --target_robot=m20 \
  --input_motion 'datasets/wmp_mocap_motions/*.txt' \
  --debug_npz
```

输出位于 `datasets/retargeted/m20/`。M20 前腿和 A1 的屈曲方向相反，映射会镜像前腿
局部 sagittal x 轨迹后再做 Mink IK；不要用逐关节复制替代该步骤。

小规模真相机训练 smoke：

```bash
conda run --no-capture-output -n isaaclab3 \
  python legged_lab/scripts/train.py \
  --task=m20_depth_rough_amp \
  --runner=wmp_amp \
  --num_envs=2 \
  --enable_cameras \
  --wmp_camera_num_envs=2 \
  --max_iterations=1 \
  --num_steps_per_env=2 \
  --num_mini_batches=1 \
  --amp_num_preload_transitions=256 \
  --viz=none \
  --logger=tensorboard
```

正式训练：

```bash
conda run --no-capture-output -n isaaclab3 \
  python legged_lab/scripts/train.py \
  --task=m20_depth_rough_amp \
  --runner=wmp_amp \
  --num_envs=4096 \
  --enable_cameras \
  --wmp_camera_num_envs=256 \
  --viz=none \
  --logger=tensorboard
```

播放和模型检查：

```bash
python legged_lab/scripts/play.py --task=m20_flat --num_envs=1 --hide_command
python legged_lab/scripts/inspect_rb160w_model.py --task=m20_flat --mode=print --headless
python legged_lab/scripts/inspect_rb160w_model.py --task=m20_flat --mode=sweep --joint=".*_hipx_joint"
```

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

## A1 WMP-AMP

当前主线 WMP 任务是 `a1_wmp_amp_terrain`，目标是在 IsaacLab + rsl_rl 5 中复现 ByteDance WMP 的 A1 地形训练链路。它不是对原仓库的逐文件搬运，而是把原版 WMP 的 actor/critic/RSSM/DepthPredictor/AMP 语义接入当前 LeggedLab 结构。

核心文件：
- 任务配置：`legged_lab/envs/a1/a1_config.py`
- runner：`legged_lab/runners/wmp_amp_runner.py`
- PPO/AMP 算法：`legged_lab/algorithms/wmp_amp_ppo.py`
- WMP 模型控制器：`legged_lab/world_models/wmp/controller.py`
- WMP policy/critic 模型：`legged_lab/models/wmp_mlp_model.py`
- partial depth camera：`legged_lab/sensors/wmp_partial_tiled_camera.py`

训练命令：

```bash
python legged_lab/scripts/train.py \
  --task=a1_wmp_amp_terrain \
  --num_envs=4096 \
  --headless \
  --enable_cameras \
  --runner=wmp_amp \
  --logger=wandb
```

小规模 smoke：

```bash
python legged_lab/scripts/train.py \
  --task=a1_wmp_amp_terrain \
  --num_envs=2 \
  --headless \
  --enable_cameras \
  --runner=wmp_amp \
  --max_iterations=1 \
  --num_steps_per_env=2 \
  --num_mini_batches=1 \
  --logger=tensorboard
```

不开相机的 Python 链路 dry smoke：

```bash
python legged_lab/scripts/train.py \
  --task=a1_wmp_amp_terrain \
  --num_envs=2 \
  --headless \
  --runner=wmp_amp \
  --max_iterations=1 \
  --num_steps_per_env=2 \
  --num_mini_batches=1 \
  --logger=tensorboard
```

说明：`--runner=wmp_amp` 且不传 `--enable_cameras` 时，`train.py` 会关闭 RGBD camera 并启用 zero-depth fallback；这只适合检查 Python 训练链路，不适合评估 WMP 视觉能力。

### 与原版 WMP 的当前对齐状态

- Actor 主干与原版对齐：`history_encoder(210 -> 35) + command(3) + wmp_encoder(512 -> 32) -> MLP(70 -> 12)`。
- Critic 输入维度已按原版恢复：`critic_obs(285) + wmp_latent(32) = 317`，第一层对应原版 `critic.0.weight: (512, 317)`。
- WMP/RSSM 默认使用 `deter=512` 作为 policy feature；`update_interval=5`，因此 WMP action 维度为 `12 * 5 = 60`。
- WMP proprioception 为 33 维：`ang_vel(3) + projected_gravity(3) + command(3) + joint_pos(12) + joint_vel(12)`。
- DepthPredictor 使用 forward height map 525 维和 prop 33 维，输出 `64x64x1` WMP depth image。
- forward height map 已按原版修正为前方 `x=[0,2], y=[-1.2,1.2]` 采样语义；IsaacLab centered grid 通过 `forward_offset=(1,0,0)` 得到 `x'=x+1`。
- AMP discriminator 输入为 60 维，即 `AMP obs(30) + next AMP obs(30)`；motion 默认来自 `datasets/wmp_mocap_motions/*.txt`。

原版 critic layout 在当前环境中构造为：

```text
contact_flag(8)
+ contact_force(12)
+ d_gain_scale(12)
+ p_gain_scale(12)
+ com_pos(3)
+ added_mass(1)
+ restitution(1)
+ friction(1)
+ base_lin_vel(3)
+ actor_obs(45)
= 98

98 + height_scan(187) = critic_obs(285)
critic_obs(285) + wmp_latent(32) = critic_input(317)
```

### 仍需注意的差异

当前代码已经修正了旧报告中提到的 `critic 271` 问题；如果看到 `WMP_NETWORK_DESIGN_DIFF_REPORT.txt` 中相关描述，应以当前源码和本 README 为准。
A1 WMP privileged scale 也已恢复到原版：`contact_force=0.005`、`pd_gains=5.0`、`com_pos=20.0`。
当前从零训练不要求兼容旧 `.pt`，actor obs、critic obs、WMP prop、AMP obs 和 action 使用 IsaacLab runtime joint order 保持内部自洽。

仍可能影响训练效果的差异：
- IsaacLab 与原版 IsaacGym/legged_gym 的仿真、接触、相机和资产加载不同。
- 原版 checkpoint 的 joint order 与当前 runtime joint order 不直接兼容，需要额外转换。
- WMP replay 由 `WMPFixedEpisodeReplayBuffer` 管理，不是原版 `wm_dataset/wm_buffer` 的逐行实现。
- partial camera 和 depth buffer 使用 IsaacLab sensor/render 调度，depth 帧时序需要重点检查。
- AMP motion 经过 `A1CanonicalRetargetAdapter` 映射到当前 env joint order，专家数据均值和当前机器人初态应在启动日志中核对。

### WMP 检查脚本

World model 前向检查：

```bash
python legged_lab/scripts/inspect_wmp_world_model.py \
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

地形与边缘 mask 检查：

```bash
python legged_lab/scripts/inspect_wmp_terrain_generator.py --headless
python legged_lab/scripts/inspect_wmp_edge_mask.py --task=a1_wmp_amp_terrain --headless --num_envs=4
```

原版 checkpoint 转换：

```bash
python legged_lab/scripts/convert_original_wmp_checkpoint.py \
  /path/to/original/model.pt \
  --reference /path/to/leggedlab/reference.pt \
  --output /path/to/converted.pt
```

转换脚本会迁移 actor、world model、DepthPredictor；critic、optimizer、AMP 状态保留 reference 模板，主要用于推理/播放检查。

## Troubleshooting

### GitHub 大文件 warning

GitHub 对单文件超过 50MB 会给 warning，超过 100MB 会拒绝 push。当前 RB160W USD 资产体积较大：

```text
legged_lab/assets/xuanji/rb160w/usd/rb160w.usd
```

建议处理方式：
- 短期：50MB warning 不会阻止 push，可先保留。
- 长期：使用 Git LFS 管理大资产，或生成 `rb160w_light.usd` 并在 `xuanji.py` 中切换到轻量资产。
- 不建议直接在原始 USD 上做不可追踪的手工瘦身；优先保留可复现的导出流程。

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
