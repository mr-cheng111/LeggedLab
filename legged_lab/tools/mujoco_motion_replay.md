# MuJoCo Motion Replay Tool

这个工具用于在 MuJoCo Viewer 中运动学播放 WMP/AMP motion 文件，方便检查 `datasets/wmp_mocap_motions/*.txt` 或 retarget 后的 motion 是否合理。它只按 motion 逐帧设置 root pose 和关节角，不运行策略、不跑动力学控制。

## 文件位置

- 播放脚本：`legged_lab/scripts/play_amp_motion_mujoco.py`
- 默认场景：`legged_lab/tools/mujoco_motion_scene.xml`
- 原版 A1 motion：`datasets/wmp_mocap_motions/*.txt`
- B2 retarget motion：`datasets/retargeted/b2/*.txt`

## 功能

- 播放 WMP AMP JSON motion。
- 支持 A1 原版关节顺序、mapping source 顺序、mapping target 顺序。
- 支持默认可视化场景：天空背景、棋盘地面、灯光、初始观察相机。
- 默认使用 MuJoCo 自由相机，鼠标可以旋转、平移、缩放。
- 可选固定相机 `--fixed_camera`。
- 可循环播放 `--loop`。
- 可把 motion 第一帧 XY 平移到原点 `--origin_xy`。
- 可锁定 root XY `--lock_xy`，保留 root 高度和姿态，用于原地观察跳跃/步态。
- 可固定 root `--fix_root`，只看腿部关节动作。

## 常用命令

播放原版 A1 跳跃 `hop1`，锁住 XY，方便原地观察：

```bash
/home/tower/miniconda/envs/isaaclab/bin/python -u legged_lab/scripts/play_amp_motion_mujoco.py \
  --xml legged_lab/assets/unitree/a1/mjcf/a1_retarget.xml \
  --motion datasets/wmp_mocap_motions/hop1.txt \
  --joint_order a1 \
  --lock_xy \
  --loop
```

播放原版 A1 跳跃 `hop2`：

```bash
/home/tower/miniconda/envs/isaaclab/bin/python -u legged_lab/scripts/play_amp_motion_mujoco.py \
  --xml legged_lab/assets/unitree/a1/mjcf/a1_retarget.xml \
  --motion datasets/wmp_mocap_motions/hop2.txt \
  --joint_order a1 \
  --lock_xy \
  --loop
```

播放原版 A1 小跑 `trot1`，保留前进轨迹但从原点开始：

```bash
/home/tower/miniconda/envs/isaaclab/bin/python -u legged_lab/scripts/play_amp_motion_mujoco.py \
  --xml legged_lab/assets/unitree/a1/mjcf/a1_retarget.xml \
  --motion datasets/wmp_mocap_motions/trot1.txt \
  --joint_order a1 \
  --origin_xy \
  --loop
```

播放 B2 retarget 后的 motion：

```bash
/home/tower/miniconda/envs/isaaclab/bin/python -u legged_lab/scripts/play_amp_motion_mujoco.py \
  --xml legged_lab/assets/unitree/b2/mjcf/b2_retarget.xml \
  --motion datasets/retargeted/b2/hop1.txt \
  --mapping legged_lab/assets/unitree/b2/mjcf/a1_to_b2_retarget.yaml \
  --joint_order mapping_target \
  --lock_xy \
  --loop
```

关闭默认场景，只加载 robot XML：

```bash
/home/tower/miniconda/envs/isaaclab/bin/python -u legged_lab/scripts/play_amp_motion_mujoco.py \
  --xml legged_lab/assets/unitree/a1/mjcf/a1_retarget.xml \
  --scene "" \
  --motion datasets/wmp_mocap_motions/hop1.txt \
  --joint_order a1 \
  --lock_xy \
  --loop
```

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--xml` | `legged_lab/assets/unitree/b2/mjcf/b2_retarget.xml` | MuJoCo XML/MJCF 机器人模型路径。播放 A1 原版 motion 时应改为 `legged_lab/assets/unitree/a1/mjcf/a1_retarget.xml`。 |
| `--scene` | `legged_lab/tools/mujoco_motion_scene.xml` | MuJoCo 场景模板。模板里用 `{robot_xml}` 引入机器人 XML。传空字符串 `--scene ""` 可关闭默认场景。 |
| `--motion` | `datasets/retargeted/b2/hop1.txt` | WMP AMP JSON motion 文件路径。 |
| `--mapping` | `legged_lab/assets/unitree/b2/mjcf/a1_to_b2_retarget.yaml` | mapping YAML，提供 source/target 关节顺序。使用 `--joint_order a1` 时不依赖 mapping。 |
| `--joint_order` | `mapping_target` | 关节顺序来源。可选 `mapping_target`、`mapping_source`、`a1`。 |
| `--speed` | `1.0` | 播放速度倍率。大于 1 更快，小于 1 更慢。 |
| `--start_frame` | `0` | 起始帧。 |
| `--end_frame` | 空 | 结束帧。为空时播放到 motion 末尾。 |
| `--fixed_camera` | 关闭 | 使用场景里的固定相机 `motion_view`。默认关闭，默认允许自由移动观察相机。 |
| `--loop` | 关闭 | 循环播放，直到关闭 viewer。 |
| `--origin_xy` | 关闭 | 减去第一帧 root 的 XY，使 motion 从原点附近开始，但仍保留原始前进轨迹。 |
| `--lock_xy` | 关闭 | 将 root 的 XY 固定为原点，保留 root 高度和姿态。适合原地观察跳跃动作，避免跑出视野。 |
| `--fix_root` | 关闭 | 固定 floating root，只播放关节角。会固定 root 位置和 root 姿态。 |
| `--root_height` | 空 | `--fix_root` 打开时的固定 root 高度。为空时使用起始帧 root z。 |
| `--root_x` | `0.0` | `--fix_root` 打开时的固定 root x。 |
| `--root_y` | `0.0` | `--fix_root` 打开时的固定 root y。 |

## 关节顺序

`--joint_order a1` 使用内置 A1 顺序：

```text
FR_hip_joint
FR_thigh_joint
FR_calf_joint
FL_hip_joint
FL_thigh_joint
FL_calf_joint
RR_hip_joint
RR_thigh_joint
RR_calf_joint
RL_hip_joint
RL_thigh_joint
RL_calf_joint
```

`--joint_order mapping_source` 使用 mapping YAML 的 `source.joints`。

`--joint_order mapping_target` 使用 mapping YAML 的 `target.joints`。

## 场景模板

默认场景 `legged_lab/tools/mujoco_motion_scene.xml` 会给 robot XML 外层包一层可视化环境：

- `skybox`：蓝灰渐变背景。
- `ground`：棋盘地面。
- `key_light` / `fill_light`：主光和补光。
- `motion_view`：固定相机。默认不锁定，只作为 `--fixed_camera` 时使用。

模板通过 `{robot_xml}` 占位符引入机器人 XML。脚本运行时会生成临时 XML，不会修改原始机器人资产。

## root 轨迹选项区别

`--origin_xy`：

```text
root_xy = motion_root_xy - first_frame_root_xy
```

适合想看完整前进轨迹，但不想一开始就在 `(6, 6)` 远处的情况。

`--lock_xy`：

```text
root_xy = (0, 0)
root_z = motion_root_z
root_quat = motion_root_quat
```

适合看跳跃和腿部节奏，不让机器人跑出镜头。

`--fix_root`：

```text
root_xyz = (root_x, root_y, root_height or first_frame_root_z)
root_quat = identity
```

适合只检查关节角序列，但会丢掉原 motion 的身体姿态和上下跳动。

## 注意事项

- 原版 `hop1/hop2` 的 root XY 本身会前进很多米，不加 `--origin_xy` 或 `--lock_xy` 时容易跑出视野。
- `datasets/wmp_mocap_motions/*.txt` 是单行 JSON，`wc -l` 可能显示 `0`，这不代表文件为空。
- 工具播放的是 motion 数据，不代表当前训练策略已经学会同样动作。
