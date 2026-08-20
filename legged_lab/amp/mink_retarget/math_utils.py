# -*- coding: utf-8 -*-
"""Quaternion and finite-difference helpers."""

from __future__ import annotations

import numpy as np


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat)
    return np.concatenate([quat[..., 3:4], quat[..., 0:3]], axis=-1)


def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat)
    return np.concatenate([quat[..., 1:4], quat[..., 0:1]], axis=-1)


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1.0e-12)


def quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    out = np.asarray(quat, dtype=np.float64).copy()
    out[..., 1:4] *= -1.0
    return out


def quat_mul_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(lhs, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(rhs, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_rotate_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate vectors by normalized wxyz quaternions."""
    quat = normalize_quat_wxyz(quat)
    vector = np.asarray(vector, dtype=np.float64)
    quat_vector = quat[..., 1:4]
    twice_cross = 2.0 * np.cross(quat_vector, vector)
    return vector + quat[..., 0:1] * twice_cross + np.cross(quat_vector, twice_cross)


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < 2:
        return np.zeros_like(values)
    out = np.zeros_like(values)
    out[0] = (values[1] - values[0]) / dt
    out[-1] = (values[-1] - values[-2]) / dt
    if values.shape[0] > 2:
        out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    return out


def angular_velocity_from_quat_wxyz(quats: np.ndarray, dt: float) -> np.ndarray:
    quats = normalize_quat_wxyz(quats)
    if quats.shape[0] < 2:
        return np.zeros((quats.shape[0], 3), dtype=np.float64)

    out = np.zeros((quats.shape[0], 3), dtype=np.float64)
    for i in range(quats.shape[0]):
        if i == 0:
            q_prev, q_next, denom = quats[0], quats[1], dt
        elif i == quats.shape[0] - 1:
            q_prev, q_next, denom = quats[-2], quats[-1], dt
        else:
            q_prev, q_next, denom = quats[i - 1], quats[i + 1], 2.0 * dt
        delta = normalize_quat_wxyz(quat_mul_wxyz(q_next, quat_conjugate_wxyz(q_prev)))
        if delta[0] < 0.0:
            delta = -delta
        sin_half = np.linalg.norm(delta[1:4])
        if sin_half < 1.0e-12:
            rotvec = np.zeros(3, dtype=np.float64)
        else:
            axis = delta[1:4] / sin_half
            angle = 2.0 * np.arctan2(sin_half, delta[0])
            rotvec = axis * angle
        # 相邻姿态满足 q_next = exp(0.5 * omega * dt) * q_prev，
        # 因此 SO(3) 对数映射得到的旋转向量 rotvec = omega * dt。
        out[i] = rotvec / denom
    return out
