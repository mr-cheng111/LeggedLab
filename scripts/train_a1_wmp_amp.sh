#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${ISAACLAB_DIR:-/home/tower/Bags/IsaacLab}"
CONDA_SH="${CONDA_SH:-/home/tower/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"

MODE="${1:-smoke}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

case "${MODE}" in
  smoke)
    NUM_ENVS="${NUM_ENVS:-2}"
    MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
    NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-2}"
    NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-1}"
    WMP_CAMERA_NUM_ENVS="${WMP_CAMERA_NUM_ENVS:-2}"
    LOGGER="${LOGGER:-tensorboard}"
    WANDB_MODE="${WANDB_MODE:-disabled}"
    EXTRA_MODE_ARGS=(
      --wmp_depth_training_iters "${WMP_DEPTH_TRAINING_ITERS:-1}"
      --wmp_depth_batch_size "${WMP_DEPTH_BATCH_SIZE:-2}"
      --wmp_train_steps_per_iter "${WMP_TRAIN_STEPS_PER_ITER:-1}"
      --amp_num_preload_transitions "${AMP_NUM_PRELOAD_TRANSITIONS:-4096}"
    )
    ;;
  train)
    NUM_ENVS="${NUM_ENVS:-4096}"
    MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
    NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-24}"
    NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-4}"
    WMP_CAMERA_NUM_ENVS="${WMP_CAMERA_NUM_ENVS:-1024}"
    LOGGER="${LOGGER:-wandb}"
    WANDB_MODE="${WANDB_MODE:-online}"
    EXTRA_MODE_ARGS=()
    ;;
  dry)
    NUM_ENVS="${NUM_ENVS:-2}"
    MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
    NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-2}"
    NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-1}"
    WMP_CAMERA_NUM_ENVS="${WMP_CAMERA_NUM_ENVS:-0}"
    LOGGER="${LOGGER:-tensorboard}"
    WANDB_MODE="${WANDB_MODE:-disabled}"
    EXTRA_MODE_ARGS=(
      --wmp_depth_training_iters "${WMP_DEPTH_TRAINING_ITERS:-1}"
      --wmp_depth_batch_size "${WMP_DEPTH_BATCH_SIZE:-2}"
      --wmp_train_steps_per_iter "${WMP_TRAIN_STEPS_PER_ITER:-1}"
      --amp_num_preload_transitions "${AMP_NUM_PRELOAD_TRANSITIONS:-4096}"
    )
    ;;
  *)
    echo "Usage: $0 [smoke|train|dry] [extra train.py args...]" >&2
    exit 2
    ;;
esac

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
fi

if [[ -f "${ISAACLAB_DIR}/_isaac_sim/setup_conda_env.sh" ]]; then
  # Isaac Sim 的运行时环境变量由这个脚本注入到当前 shell。
  # 公式无关；这是训练入口所需的环境准备。
  # shellcheck disable=SC1090
  set +u
  source "${ISAACLAB_DIR}/_isaac_sim/setup_conda_env.sh"
  set -u
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export WANDB_MODE

CMD=(
  python legged_lab/scripts/train.py
  --task a1_wmp_amp_terrain
  --runner wmp_amp
  --headless
  --num_envs "${NUM_ENVS}"
  --max_iterations "${MAX_ITERATIONS}"
  --num_steps_per_env "${NUM_STEPS_PER_ENV}"
  --num_mini_batches "${NUM_MINI_BATCHES}"
  --logger "${LOGGER}"
  --wandb_mode "${WANDB_MODE}"
  --wmp_camera_num_envs "${WMP_CAMERA_NUM_ENVS}"
  "${EXTRA_MODE_ARGS[@]}"
)

if [[ "${MODE}" != "dry" ]]; then
  CMD+=(--enable_cameras)
fi

CMD+=("$@")

echo "[INFO] mode=${MODE}"
echo "[INFO] root=${ROOT_DIR}"
echo "[INFO] command: ${CMD[*]}"
exec "${CMD[@]}"
