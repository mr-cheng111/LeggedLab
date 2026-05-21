# -*- coding: utf-8 -*-
"""WMP AMP JSON IO helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FRAME_WIDTH = 61
ROOT_POS = slice(0, 3)
ROOT_QUAT = slice(3, 7)
JOINT_POS = slice(7, 19)
TOE_POS_LOCAL = slice(19, 31)
LINEAR_VEL = slice(31, 34)
ANGULAR_VEL = slice(34, 37)
JOINT_VEL = slice(37, 49)


@dataclass
class WMPMotion:
    frames: np.ndarray
    frame_duration: float
    motion_weight: float
    loop_mode: str = "Wrap"


def load_wmp_motion(path: str | Path) -> WMPMotion:
    motion_path = Path(path)
    with motion_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    frames = np.asarray(payload["Frames"], dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] < FRAME_WIDTH:
        raise ValueError(f"{motion_path} expected frames with at least {FRAME_WIDTH} columns, got {frames.shape}.")
    return WMPMotion(
        frames=frames[:, :FRAME_WIDTH].copy(),
        frame_duration=float(payload["FrameDuration"]),
        motion_weight=float(payload.get("MotionWeight", 1.0)),
        loop_mode=str(payload.get("LoopMode", "Wrap")),
    )


def save_wmp_motion(path: str | Path, motion: WMPMotion) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "LoopMode": motion.loop_mode,
        "FrameDuration": float(motion.frame_duration),
        "MotionWeight": float(motion.motion_weight),
        "Frames": np.asarray(motion.frames, dtype=np.float32).tolist(),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
