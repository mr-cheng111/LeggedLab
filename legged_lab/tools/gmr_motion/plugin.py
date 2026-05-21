# -*- coding: utf-8 -*-
"""Gaussian Mixture Regression smoothing for WMP AMP motions."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from gmr import GMM

from legged_lab.amp.mink_retarget.io import (
    ANGULAR_VEL,
    JOINT_POS,
    JOINT_VEL,
    LINEAR_VEL,
    ROOT_POS,
    ROOT_QUAT,
    TOE_POS_LOCAL,
    WMPMotion,
)
from legged_lab.amp.mink_retarget.math_utils import angular_velocity_from_quat_wxyz, finite_difference, xyzw_to_wxyz


FEATURE_SLICES = {
    "root_pos": ROOT_POS,
    "joint_pos": JOINT_POS,
    "toe_pos_local": TOE_POS_LOCAL,
}


@dataclass(frozen=True)
class GMRMotionConfig:
    n_components: int = 8
    features: tuple[str, ...] = ("joint_pos", "toe_pos_local")
    random_state: int = 42
    n_iter: int = 100
    covariance_regularization: float = 1.0e-4


def _phase(frame_count: int) -> np.ndarray:
    if frame_count <= 1:
        return np.zeros((frame_count, 1), dtype=np.float64)
    return np.linspace(0.0, 1.0, frame_count, dtype=np.float64).reshape(-1, 1)


def _feature_indices(features: tuple[str, ...]) -> list[int]:
    indices: list[int] = []
    for name in features:
        if name not in FEATURE_SLICES:
            raise ValueError(f"Unsupported GMR feature={name!r}. Expected one of {sorted(FEATURE_SLICES)}.")
        indices.extend(range(FEATURE_SLICES[name].start, FEATURE_SLICES[name].stop))
    return indices


def _fit_predict_gmr(values: np.ndarray, cfg: GMRMotionConfig) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.maximum(std, 1.0e-6)
    normalized_values = (values - mean) / std
    phase = _phase(values.shape[0])
    samples = np.concatenate([phase, normalized_values], axis=1)
    # GMR 在高维小样本上容易出现退化协方差；组件数按样本数保守降级。
    n_components = max(1, min(cfg.n_components, values.shape[0] // 10, values.shape[0]))
    last_error: Exception | None = None
    for init_params, use_oas in (("kmeans++", True), ("random", False)):
        try:
            gmm = GMM(n_components=n_components, random_state=cfg.random_state)
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                gmm.from_samples(
                    samples,
                    n_iter=cfg.n_iter,
                    R_diff=cfg.covariance_regularization,
                    init_params=init_params,
                    oracle_approximating_shrinkage=use_oas,
                )
            break
        except (FloatingPointError, RuntimeWarning, ValueError, np.linalg.LinAlgError) as exc:
            last_error = exc
    else:
        raise RuntimeError(f"GMR fitting failed after fallback attempts: {last_error}") from last_error
    prediction = gmm.predict(np.array([0]), phase)
    means = prediction[0] if isinstance(prediction, tuple) else prediction
    return np.asarray(means, dtype=np.float64) * std + mean


def _recompute_velocities(frames: np.ndarray, dt: float) -> None:
    frames[:, LINEAR_VEL] = finite_difference(frames[:, ROOT_POS], dt)
    # 数据集 quaternion 存储为 xyzw；角速度公式在 math_utils 中使用 wxyz。
    frames[:, ANGULAR_VEL] = angular_velocity_from_quat_wxyz(xyzw_to_wxyz(frames[:, ROOT_QUAT]), dt)
    frames[:, JOINT_VEL] = finite_difference(frames[:, JOINT_POS], dt)


def smooth_motion(motion: WMPMotion, cfg: GMRMotionConfig) -> WMPMotion:
    frames = np.asarray(motion.frames, dtype=np.float64).copy()
    if frames.shape[0] < 3:
        return WMPMotion(frames=frames, frame_duration=motion.frame_duration, motion_weight=motion.motion_weight, loop_mode=motion.loop_mode)

    indices = _feature_indices(cfg.features)
    smoothed = _fit_predict_gmr(frames[:, indices], cfg)
    frames[:, indices] = smoothed
    _recompute_velocities(frames, motion.frame_duration)
    return WMPMotion(
        frames=frames,
        frame_duration=motion.frame_duration,
        motion_weight=motion.motion_weight,
        loop_mode=motion.loop_mode,
    )


class GMRMotionPlugin:
    """Small plugin wrapper used by CLI tools."""

    def __init__(self, cfg: GMRMotionConfig):
        self.cfg = cfg

    def preprocess(self, motion: WMPMotion) -> WMPMotion:
        return smooth_motion(motion, self.cfg)

    def postprocess(self, motion: WMPMotion) -> WMPMotion:
        return smooth_motion(motion, self.cfg)
