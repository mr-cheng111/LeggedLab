# XML 版 A1 到 B2 AMP Retarget 工具

本文档记录当前实现的设计边界和使用方式。工具目标是把现有 WMP AMP JSON 动作数据从 A1 几何模型重定向到 B2 几何模型，并导出仍兼容 `AMPMotionDataset` 的 61 维 WMP AMP JSON。

## 设计结论

- 运行时模型输入使用 MuJoCo XML/MJCF，不再依赖 URDF。
- XML 是 retarget 专用的轻量模型，只要求 base、腿部关节轴、关键 body/site 和视觉几何足够正确，不作为真实动力学资产。
- 当前版本固定输出 12 关节四足格式：`FR, FL, RR, RL`，每条腿 `hip, thigh, calf`。
- 当前 AMP discriminator 和 runner 不改动，输出文件可直接作为 B2 WMP AMP motion file 使用。

## 使用方式

```bash
python legged_lab/scripts/retarget_amp_motion.py \
  --input_motion datasets/wmp_mocap_motions/hop1.txt \
  --source_xml legged_lab/assets/unitree/a1/mjcf/a1_retarget.xml \
  --target_xml legged_lab/assets/unitree/b2/mjcf/b2_retarget.xml \
  --mapping legged_lab/assets/unitree/b2/mjcf/a1_to_b2_retarget.yaml \
  --output_motion datasets/retargeted/b2/hop1.txt \
  --debug_npz datasets/retargeted/b2/hop1_debug.npz
```

批量转换：

```bash
python legged_lab/scripts/retarget_amp_motion.py \
  --input_motion datasets/wmp_mocap_motions/*.txt \
  --output_dir datasets/retargeted/b2 \
  --debug_npz
```

依赖：

```bash
pip install mujoco mink pyyaml
```

默认 QP 后端使用 `daqp`，这是 `mink` 的 pip 依赖会自动安装的 solver。

GMR 前/后处理插件：

```bash
python legged_lab/scripts/retarget_amp_motion.py \
  --input_motion datasets/wmp_mocap_motions/hop1.txt \
  --output_motion datasets/retargeted/b2/hop1_gmr.txt \
  --gmr_pre \
  --gmr_post \
  --gmr_features joint_pos,toe_pos_local \
  --gmr_components 8
```

也可以单独平滑一个 motion：

```bash
python legged_lab/tools/gmr_motion/smooth_motion.py \
  --input_motion datasets/retargeted/b2/hop1.txt \
  --output_motion datasets/retargeted/b2/hop1_gmr.txt
```

## 数据流

1. 读取 WMP AMP JSON：`root_pos(3), root_quat(4), joint_pos(12), toe_pos_local(12), lin_vel(3), ang_vel(3), joint_vel(12)`。
2. 在 A1 XML 上做 FK，得到 root、腿部关键 body/site 和足端轨迹。
3. 在 B2 XML 上用 `mink` 做 IK，默认把 A1 每条腿相对 hip 的 frame/足端轨迹按 B2 腿长比例缩放后作为主目标，12 个目标关节只作为弱姿态正则。
4. 导出 B2 WMP AMP JSON，保留原始 `LoopMode`、`FrameDuration`、`MotionWeight` 字段。

## 速度计算

- 关节速度使用有限差分。内部帧使用中心差分：
  `qdot[t] = (q[t+1] - q[t-1]) / (2 * dt)`。
- 首尾帧分别使用前向和后向差分。
- base 线速度由 root position 差分得到。
- base 角速度由相邻 quaternion delta 的旋转向量计算得到，公式见实现注释。

## 验证清单

- `--validate_only`：检查 XML/YAML/输入 JSON 的静态一致性。
- 前 100 帧 smoke：
  ```bash
  python legged_lab/scripts/retarget_amp_motion.py \
    --input_motion datasets/wmp_mocap_motions/hop1.txt \
    --output_motion datasets/retargeted/b2/hop1_100.txt \
    --max_frames 100 \
    --debug_npz datasets/retargeted/b2/hop1_100_debug.npz
  ```
- 导出的 motion 应能被 `AMPMotionDataset` 读取，帧宽保持 61。

## MuJoCo 播放

可以用 MuJoCo 对导出的 B2 motion 做运动学播放：

```bash
python legged_lab/scripts/play_amp_motion_mujoco.py \
  --xml legged_lab/assets/unitree/b2/mjcf/b2_retarget.xml \
  --motion datasets/retargeted/b2/hop1.txt \
  --mapping legged_lab/assets/unitree/b2/mjcf/a1_to_b2_retarget.yaml \
  --loop
```

固定 root 悬空，只检查关节轨迹：

```bash
python legged_lab/scripts/play_amp_motion_mujoco.py \
  --xml legged_lab/assets/unitree/b2/mjcf/b2_retarget.xml \
  --motion datasets/retargeted/b2/hop1.txt \
  --fix_root \
  --root_height 0.75 \
  --loop
```

播放器每帧直接写入 floating root 的 6DoF 和 12 个关节位置，然后调用 `mj_forward`，不做动力学积分；同时将 gravity、wind、density、viscosity 置零。

## 后续扩展

- 如果目标机器人不是 12 关节四足，需要扩展 AMP obs 维度、discriminator 输入维度和 runner 配置。
- 如果 B2 XML 的轴向或比例需要调优，应优先改 retarget XML/YAML，而不是改 USD。
- 不建议强行逐关节复制 A1 角度到 B2。A1 与 B2 连杆比例不同，默认 `frame_target_mode: morphology_scaled` 会把 A1 每条腿相对 hip 的轨迹按目标腿长缩放后再求 IK，`posture_mode: neutral` 仅用 B2 默认站姿做弱正则。
