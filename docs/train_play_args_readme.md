# Train / Play 参数说明

本文记录 `legged_lab/scripts/train.py` 与 `legged_lab/scripts/play.py` 当前暴露的命令行参数。参数默认值以脚本 `--help` 和代码中的 `argparse` 默认值为准；当默认值为 `None` 时，通常表示不覆盖任务配置文件里的默认值。

## 快速结论

- A1 WMP-AMP-PPO 训练使用 `train.py --runner=wmp_amp --task=a1_wmp_amp_terrain`。
- WMP 训练如需真实深度相机，必须加 `--enable_cameras`；否则 `train.py` 会进入 dry smoke 的 zero-depth fallback。
- 播放 WMP checkpoint 使用 `play.py --runner=wmp_amp --load_run ... --checkpoint ...`。
- 播放时只要打开 `--show_depth_image` 或 `--show_depth_points`，脚本会自动启用 `--enable_cameras`。
- `--wmp_camera_num_envs` 控制真实 depth 相机环境数量，不等于总环境数；例如 `--num_envs=4096 --wmp_camera_num_envs=1024` 表示 4096 个并行环境中 1024 个有真实相机。

## 训练脚本

入口：

```bash
/home/tower/miniconda/envs/isaaclab/bin/python -u legged_lab/scripts/train.py [参数]
```

### train.py 专属参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--task TASK` | `None` | 任务名称，例如 `a1_wmp_amp_terrain`。 |
| `--num_envs NUM_ENVS` | `None` | 覆盖并行环境数量；不传则使用任务配置。 |
| `--seed SEED` | `None` | 覆盖随机种子；传 `-1` 时随机生成。 |
| `--runner {default,wmp_amp,amp_ppo}` | `default` | 训练 Runner。`wmp_amp` 用 WMP+AMP+PPO，`amp_ppo` 用 AMP+PPO，`default` 用普通 RSL-RL runner。 |

### train.py 行为补充

- `--runner=wmp_amp` 且未传 `--enable_cameras` 时，脚本会关闭 RGBD camera depth/rgb，相当于只做 smoke dry run，不适合完整 WMP 训练。
- `--num_envs` 会覆盖 `env_cfg.scene.num_envs`。
- `--wmp_camera_num_envs` 会同时覆盖环境相机配置和 WMP 配置里的真实相机环境数量。
- resume 时，`max_iterations` 被解释为最终目标迭代数，而不是额外追加的迭代数。
- 日志目录格式为 `logs/{experiment_name}/{日期时间}_{run_name}`。

## 播放脚本

入口：

```bash
/home/tower/miniconda/envs/isaaclab/bin/python -u legged_lab/scripts/play.py [参数]
```

### play.py 基础参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--task TASK` | `None` | 任务名称，例如 `a1_wmp_amp_terrain`。 |
| `--num_envs NUM_ENVS` | `4` | 覆盖播放环境数量；如果不传，脚本默认使用 4 个环境。 |
| `--seed SEED` | `None` | 覆盖随机种子。 |
| `--runner {default,wmp_amp}` | `default` | checkpoint 类型。WMP checkpoint 需要用 `wmp_amp`。 |
| `--play_flat` | `False` | 播放时将地形替换为平地，同时保留 WMP sensor obs 形状。 |
| `--play_render_interval PLAY_RENDER_INTERVAL` | `4` | 覆盖播放渲染间隔。数值越小 GUI 越流畅但越重；`4` 约等于 50Hz 渲染。 |
| `--enable_play_push` | `False` | 播放时保留 interval push 扰动；默认会关闭 push。 |
| `--hide_command` | `False` | 隐藏命令速度/当前速度调试可视化。 |

### play.py 深度图参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--show_depth_image` | `False` | 显示或保存 WMP 风格 `64x64` depth 图。开启后自动启用相机。 |
| `--depth_image_mode {auto,window,save}` | `auto` | depth 图显示模式。`auto` 优先窗口，OpenCV GUI 不可用时自动保存 PNG；`save` 只保存。 |
| `--depth_image_dir DEPTH_IMAGE_DIR` | `None` | depth PNG 保存目录。不传时保存到当前 run 的 `depth_images` 目录。 |
| `--depth_image_save_interval DEPTH_IMAGE_SAVE_INTERVAL` | `10` | 每 N 个 play step 保存一张 depth 图。 |

### play.py 深度点云参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--show_depth_points` | `False` | 将 depth 图反投影成红色 debug points。开启后自动启用相机和 `isaacsim.util.debug_draw`。 |
| `--depth_point_stride DEPTH_POINT_STRIDE` | `16` | 采样像素步长。值越小点越密，渲染越重。 |
| `--depth_point_max DEPTH_POINT_MAX` | `300` | 最多绘制多少个 depth 点。 |
| `--depth_point_size DEPTH_POINT_SIZE` | `5.0` | 红色 depth 点的 debug draw 尺寸。 |
| `--depth_point_forward_min DEPTH_POINT_FORWARD_MIN` | `0.2` | 只显示大于该深度距离的点。 |
| `--depth_point_forward_max DEPTH_POINT_FORWARD_MAX` | `3.0` | 只显示小于该深度距离的点。 |
| `--depth_point_min_z DEPTH_POINT_MIN_Z` | `None` | 只显示世界坐标 z 大于该值的点。 |
| `--depth_point_max_z DEPTH_POINT_MAX_Z` | `None` | 只显示世界坐标 z 小于该值的点。 |
| `--depth_point_debug` | `False` | 定期打印 depth 点统计信息。 |
| `--depth_point_lift DEPTH_POINT_LIFT` | `0.05` | 绘制时将点沿世界 z 方向抬高，避免被地面遮住。 |
| `--depth_point_draw_rays` | `False` | 从相机到 depth 点绘制黄色射线。该模式较重，且 UI 中更容易卡顿。 |
| `--depth_point_camera_index DEPTH_POINT_CAMERA_INDEX` | `0` | 可视化第几个 depth camera。传 `-1` 表示显示所有相机环境的点。 |

### play.py height scanner 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--show_height_scan_points` | `False` | 显示 height scanner ray hit 点。 |

### play.py 相机覆盖参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--camera_offset_pos X Y Z` | `None` | 播放时覆盖 WMP depth camera 的位置偏移。常用调试值：`0.27 0.0 0.10`。 |
| `--camera_offset_rot W X Y Z` | `None` | 播放时覆盖 WMP depth camera 的四元数偏移，顺序为 `wxyz`。 |
| `--camera_random_pitch_deg MIN MAX` | `None` | 播放时覆盖相机 pitch 随机范围，单位为度。 |
| `--camera_fov_deg DEG` | `None` | 播放时覆盖 WMP camera 水平视场角。 |
| `--camera_disable_random_rotation` | `False` | 播放时关闭相机随机旋转。 |

### play.py 行为补充

- 播放默认关闭噪声：`env_cfg.noise.add_noise=False`。
- 播放默认关闭 push，除非传 `--enable_play_push`。
- 播放默认将 episode length 设为 `40s`。
- 播放默认 command 范围：
  - 非 `stand` / `slow_walk` / `rb160w` 任务：`lin_vel_x=(0.0,0.8)`，`lin_vel_y=(0.0,0.0)`，`ang_vel_z=(0.0,0.0)`，`heading=(0.0,0.0)`。
  - `slow_walk`：`lin_vel_x=(0.0,1.0)`，其余为 0。
  - `rb160w`：`lin_vel_x=(0.2,0.8)`，其余为 0。
- 如果打开 depth 图或 depth 点，且任务使用 partial camera，播放脚本会把 partial camera 数量设置为当前 `num_envs`，即播放环境都挂相机。
- `play.py` 会尝试导出 JIT/ONNX；WMP 分层模型可能导出失败，但不影响播放。

## 共享 RSL-RL / AMP / WMP 参数

这些参数由 `legged_lab/utils/cli_args.py` 注入，`train.py` 和 `play.py` 都能接收。播放时大多数训练参数不会参与控制，但 `--load_run`、`--checkpoint`、`--experiment_name` 会用于查找 checkpoint。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--max_iterations MAX_ITERATIONS` | `None` | PPO/RL 最大训练迭代数。 |
| `--num_steps_per_env NUM_STEPS_PER_ENV` | `None` | 每个环境每次 rollout 的步数。 |
| `--num_mini_batches NUM_MINI_BATCHES` | `None` | PPO update 的 mini-batch 数量。 |
| `--experiment_name EXPERIMENT_NAME` | `None` | 日志根目录名；实际路径为 `logs/{experiment_name}`。 |
| `--run_name RUN_NAME` | `None` | 追加到本次 run 日志目录名。 |
| `--resume RESUME` | `None` | 是否从 checkpoint 恢复训练。注意这是 bool 参数，常用 `--resume=True`。 |
| `--load_run LOAD_RUN` | `None` | 要加载的 run 文件夹名。 |
| `--checkpoint CHECKPOINT` | `None` | 要加载的 checkpoint 文件，例如 `model_10000.pt`。 |
| `--logger {wandb,tensorboard,neptune}` | `wandb` | 训练日志后端。 |
| `--log_project_name LOG_PROJECT_NAME` | `None` | wandb/neptune project 名。 |
| `--wandb_entity WANDB_ENTITY` | `""` | wandb entity/username。 |
| `--wandb_mode {online,offline,disabled}` | `online` | wandb 模式。 |
| `--wandb_api_key WANDB_API_KEY` | 环境变量 `WANDB_API_KEY` | wandb API key。 |
| `--distributed` | `False` | 多 GPU / 多节点训练。 |
| `--amp_num_preload_transitions AMP_NUM_PRELOAD_TRANSITIONS` | `None` | 覆盖 AMP expert preload transitions。 |
| `--amp_reward_coef AMP_REWARD_COEF` | `None` | 覆盖 AMP discriminator reward 系数。 |
| `--amp_task_reward_lerp AMP_TASK_REWARD_LERP` | `None` | 覆盖 AMP task reward 混合系数。 |
| `--wmp_camera_num_envs WMP_CAMERA_NUM_ENVS` | `None` | 覆盖真实 WMP depth camera 环境数量。 |
| `--wmp_depth_training_iters WMP_DEPTH_TRAINING_ITERS` | `None` | 覆盖 DepthPredictor 每次触发时训练迭代数。 |
| `--wmp_depth_batch_size WMP_DEPTH_BATCH_SIZE` | `None` | 覆盖 DepthPredictor batch size。 |
| `--wmp_train_steps_per_iter WMP_TRAIN_STEPS_PER_ITER` | `None` | 覆盖每个 PPO iteration 中 world model 梯度步数。 |
| `--wmp_train_interval WMP_TRAIN_INTERVAL` | `None` | train_start_steps 后，每 N 个 PPO iteration 训练一次 world model。 |

兼容参数：

| 参数 | 说明 |
| --- | --- |
| `--num_mini_batces` | `--num_mini_batches` 的拼写错误兼容参数，help 中隐藏。 |

## IsaacLab AppLauncher 参数

这些参数由 IsaacLab `AppLauncher` 注入，两个脚本都支持。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--headless` | `False` | 强制无 GUI 运行。训练通常建议打开。 |
| `--livestream {0,1,2}` | 由 IsaacLab 决定 | 强制启用 livestream。 |
| `--enable_cameras` | `False` | 启用 camera sensors 和相关扩展。WMP 真实 depth 训练需要打开。 |
| `--xr` | `False` | 启用 VR/AR XR 模式。 |
| `--device DEVICE` | 由 IsaacLab 决定 | 仿真设备，例如 `cuda`、`cuda:0`、`cpu`。 |
| `--verbose` | `False` | SimulationApp verbose 日志。 |
| `--info` | `False` | SimulationApp info 日志。 |
| `--experience EXPERIENCE` | 由 IsaacLab 决定 | 指定 IsaacSim experience 文件。 |
| `--rendering_mode {performance,balanced,quality}` | 由 IsaacLab 决定 | 渲染模式。训练建议 `performance`。 |
| `--kit_args KIT_ARGS` | `None` | 额外 Omniverse Kit 参数字符串。 |
| `--anim_recording_enabled` | `False` | 启用 USD animation 记录。 |
| `--anim_recording_start_time ANIM_RECORDING_START_TIME` | `None` | animation 记录开始时间。 |
| `--anim_recording_stop_time ANIM_RECORDING_STOP_TIME` | `None` | animation 记录停止时间。 |

## 常用命令

### A1 WMP-AMP-PPO 完整训练

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/tower/miniconda/envs/isaaclab/bin/python -u legged_lab/scripts/train.py \
  --task=a1_wmp_amp_terrain \
  --headless \
  --enable_cameras \
  --runner=wmp_amp \
  --num_envs=4096 \
  --wmp_camera_num_envs=1024 \
  --max_iterations=20000 \
  --num_steps_per_env=24 \
  --wmp_depth_training_iters=1000 \
  --wmp_depth_batch_size=1024 \
  --wmp_train_steps_per_iter=10 \
  --wmp_train_interval=1 \
  --amp_num_preload_transitions=2000000 \
  --logger=wandb \
  --log_project_name=a1_wmp_amp \
  --run_name=a1_wmp_amp_4096env_1024cam_v1
```

### A1 WMP 快速 dry train

```bash
/home/tower/miniconda/envs/isaaclab/bin/python -u legged_lab/scripts/train.py \
  --task=a1_wmp_amp_terrain \
  --headless \
  --enable_cameras \
  --runner=wmp_amp \
  --num_envs=2 \
  --wmp_camera_num_envs=2 \
  --max_iterations=1 \
  --num_steps_per_env=2 \
  --num_mini_batches=1 \
  --wmp_depth_training_iters=1 \
  --wmp_depth_batch_size=2 \
  --wmp_train_steps_per_iter=1 \
  --logger=tensorboard \
  --run_name=a1_wmp_smoke
```

### 播放 A1 WMP checkpoint

```bash
/home/tower/miniconda/envs/isaaclab/bin/python -u legged_lab/scripts/play.py \
  --task=a1_wmp_amp_terrain \
  --runner=wmp_amp \
  --load_run=2026-05-27_14-00-28_a1_wmp_amp_4096env_1024cam_fixed_camera_v1 \
  --checkpoint=model_10000.pt \
  --num_envs=4
```

### 播放平地验证步态

```bash
/home/tower/miniconda/envs/isaaclab/bin/python -u legged_lab/scripts/play.py \
  --task=a1_wmp_amp_terrain \
  --runner=wmp_amp \
  --load_run=2026-05-27_14-00-28_a1_wmp_amp_4096env_1024cam_fixed_camera_v1 \
  --checkpoint=model_10000.pt \
  --num_envs=4 \
  --play_flat
```

### 保存 64x64 WMP depth 图

```bash
/home/tower/miniconda/envs/isaaclab/bin/python -u legged_lab/scripts/play.py \
  --task=a1_wmp_amp_terrain \
  --runner=wmp_amp \
  --load_run=2026-05-27_14-00-28_a1_wmp_amp_4096env_1024cam_fixed_camera_v1 \
  --checkpoint=model_10000.pt \
  --num_envs=4 \
  --show_depth_image \
  --depth_image_mode=save \
  --depth_image_dir=/tmp/wmp_depth_images \
  --depth_image_save_interval=5 \
  --camera_offset_pos 0.27 0.0 0.10 \
  --camera_disable_random_rotation
```

### 显示所有狗的 depth 点云

```bash
/home/tower/miniconda/envs/isaaclab/bin/python -u legged_lab/scripts/play.py \
  --task=a1_wmp_amp_terrain \
  --runner=wmp_amp \
  --load_run=2026-05-27_14-00-28_a1_wmp_amp_4096env_1024cam_fixed_camera_v1 \
  --checkpoint=model_10000.pt \
  --num_envs=4 \
  --show_depth_points \
  --depth_point_camera_index=-1 \
  --depth_point_stride=8 \
  --depth_point_max=1200 \
  --depth_point_debug \
  --camera_offset_pos 0.27 0.0 0.10 \
  --camera_disable_random_rotation
```

### 播放时减轻 GUI 卡顿

```bash
/home/tower/miniconda/envs/isaaclab/bin/python -u legged_lab/scripts/play.py \
  --task=a1_wmp_amp_terrain \
  --runner=wmp_amp \
  --load_run=2026-05-27_14-00-28_a1_wmp_amp_4096env_1024cam_fixed_camera_v1 \
  --checkpoint=model_10000.pt \
  --num_envs=1 \
  --play_render_interval=8
```

## 建议组合

| 场景 | 推荐参数 |
| --- | --- |
| 正式 A1 WMP 训练 | `--headless --enable_cameras --runner=wmp_amp --num_envs=4096 --wmp_camera_num_envs=1024` |
| 快速排查 WMP 链路 | `--num_envs=2 --wmp_camera_num_envs=2 --max_iterations=1 --num_steps_per_env=2 --num_mini_batches=1` |
| 看 64x64 depth 输出 | `--show_depth_image --depth_image_mode=save --depth_image_save_interval=5` |
| 看地面反投影点 | `--show_depth_points --depth_point_camera_index=-1 --depth_point_stride=8` |
| 播放卡顿 | 减小 `--num_envs`，增大 `--play_render_interval`，关闭 `--show_depth_points` 或增大 `--depth_point_stride`。 |
