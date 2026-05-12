# B2 到 mjlab 的最小迁移说明

## 结论
- 你可以先用 `legged_lab/scripts/mjlab_b2_sim2sim.py` 跑通 B2 的 MuJoCo sim2sim 验证。
- 该脚本已包含：观测噪声、观测延迟、动作延迟、速度命令范围对齐。

## 前置条件
1. 已安装 `mjlab` 与 `mujoco`。
2. 已准备 B2 的 MJCF（`.xml/.mjcf`）。
3. 已从 IsaacLab 导出 TorchScript 策略（`policy.pt`）。

## 最小运行命令
```bash
python legged_lab/scripts/mjlab_b2_sim2sim.py \
  --mjcf /path/to/b2.xml \
  --policy_jit /path/to/policy.pt \
  --num_envs 64 \
  --viewer auto
```

## 先对齐再加扰动
1. 先验证“纯净”：
```bash
python legged_lab/scripts/mjlab_b2_sim2sim.py \
  --mjcf /path/to/b2.xml \
  --policy_jit /path/to/policy.pt \
  --disable_noise \
  --obs_delay_steps 0 \
  --act_delay_steps 0
```
2. 再逐步打开噪声与延迟：
```bash
python legged_lab/scripts/mjlab_b2_sim2sim.py \
  --mjcf /path/to/b2.xml \
  --policy_jit /path/to/policy.pt \
  --obs_delay_steps 1 \
  --act_delay_steps 2
```

## 当前脚手架对齐项
- 观测（actor）：`ang_vel`、`projected_gravity`、`command`、`joint_pos`、`joint_vel`、`actions`
- 历史堆叠：`history_length = 10`（actor/critic）
- 指令范围：
  - `lin_vel_x: [-0.6, 1.0]`
  - `lin_vel_y: [-0.5, 0.5]`
  - `ang_vel_z: [-1.57, 1.57]`
  - `heading: [-pi, pi]`
- 控制参数：`dt=0.005`，`decimation=4`

## 说明
- 这是“验证脚手架”，目的是尽快跑通 sim2sim；不是完整训练环境复刻。
- 若你的 B2 MJCF 里足端/body 命名和正则不一致，需要改脚本里的接触匹配规则。
