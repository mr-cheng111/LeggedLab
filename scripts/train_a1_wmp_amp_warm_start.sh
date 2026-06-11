#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CKPT="${ROOT_DIR}/logs/a1_wmp_amp_terrain/2026-06-09_10-48-49/model_8000.pt"

WARM_START_CHECKPOINT="${WARM_START_CHECKPOINT:-${1:-${DEFAULT_CKPT}}}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

if [[ ! -f "${WARM_START_CHECKPOINT}" ]]; then
  echo "[ERROR] Warm-start checkpoint not found: ${WARM_START_CHECKPOINT}" >&2
  exit 1
fi

export RUN_NAME="${RUN_NAME:-warm_start_from_model_8000_fixed_curriculum}"

EXTRA_ARGS=(--warm_start_checkpoint "${WARM_START_CHECKPOINT}" --run_name "${RUN_NAME}")

if [[ "${WARM_START_LOAD_OPTIMIZER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--warm_start_load_optimizer)
fi

exec "${ROOT_DIR}/scripts/train_a1_wmp_amp.sh" train "${EXTRA_ARGS[@]}" "$@"
