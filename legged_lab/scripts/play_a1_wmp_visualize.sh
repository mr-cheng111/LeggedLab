#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-2026-06-02_17-47-55_a1_wmp_amp_pitch_5_15_4096env_1024cam}"
CHECKPOINT="${2:-model_19999.pt}"
NUM_ENVS="${3:-4}"
MODE="${4:-fast}"

cd "$(dirname "$0")/../.."

COMMON_ARGS=(
  --task=a1_wmp_amp_terrain
  --runner=wmp_amp
  --load_run="${RUN_NAME}"
  --checkpoint="${CHECKPOINT}"
  --num_envs="${NUM_ENVS}"
  --enable_cameras
  --rendering_mode performance
  --show_depth_image
  --depth_image_mode=auto
  --depth_image_save_interval=30
  --show_camera_model
  --play_wmp_terrain slope
)

case "${MODE}" in
  fast)
    # 快速推理查看：保持相机和 depth 小窗，但关闭点云/RSSM 对比，避免每步重渲染拖慢动作。
    EXTRA_ARGS=(
      --play_render_interval=10
      --camera_random_pitch_deg 10 10
    )
    ;;
  compare)
    # 模型诊断：打开 RSSM real/prior/posterior 对比，但不画点云。
    EXTRA_ARGS=(
      --play_render_interval=10
      --camera_random_pitch_deg 10 10
      --show_rssm_depth_compare
    )
    ;;
  full)
    # 全量相机调试：会明显变慢，只建议短时间检查相机朝向和点云。
    EXTRA_ARGS=(
      --play_render_interval=5
      --camera_random_pitch_deg 10 10
      --show_rssm_depth_compare
      --show_depth_points
      --show_camera_axes
      --depth_point_camera_index=0
      --depth_point_stride=8
      --depth_point_max=1024
      --depth_point_size=2.0
      --depth_point_forward_min=0.01
      --depth_point_forward_max=3.0
      --depth_point_lift=0.03
      --depth_point_debug
    )
    ;;
  original)
    # 原版 bytedance/WMP 权重兼容检查：还原原版 A1 关节顺序、相机外参和较干净的播放环境。
    EXTRA_ARGS=(
      --play_render_interval=10
      --policy_joint_order=original_wmp_a1
      --disable_play_domain_rand
      --camera_offset_pos 0.27 0.0 0.03
      --camera_random_pitch_deg -5 5
    )
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use one of: fast, compare, full, original." >&2
    exit 2
    ;;
esac

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
conda run -n isaaclab python -u legged_lab/scripts/play.py \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
