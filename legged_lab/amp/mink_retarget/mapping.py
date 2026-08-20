# -*- coding: utf-8 -*-
"""Retarget mapping schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelMapping:
    xml: str | None
    root_freejoint: str
    base_site: str
    joints: tuple[str, ...]
    frames: dict[str, str]


@dataclass(frozen=True)
class RetargetOptions:
    root_height_offset: float = 0.0
    root_xy_scale: float = 1.0
    frame_position_scale: float = 1.0
    frame_target_mode: str = "morphology_scaled"
    posture_mode: str = "neutral"
    max_ik_iters: int = 40
    solver: str = "mink"
    qp_solver: str = "daqp"
    damping: float = 1.0e-5
    posture_cost: float = 0.005
    frame_pos_cost: float = 1.0
    foot_pos_cost: float = 4.0
    root_cost: float = 2.0
    neutral_joint_pos: dict[str, float] | None = None
    joint_project_limits: dict[str, tuple[float, float] | list[float]] | None = None
    leg_axis_scale: dict[str, tuple[float, float, float] | list[float]] | None = None


@dataclass(frozen=True)
class RetargetMapping:
    source: ModelMapping
    target: ModelMapping
    options: RetargetOptions


def _read_model_mapping(raw: dict, section: str) -> ModelMapping:
    joints = tuple(raw.get("joints", ()))
    frames = dict(raw.get("frames", {}))
    if len(joints) != 12:
        raise ValueError(f"{section}.joints must contain exactly 12 joints, got {len(joints)}.")
    if not frames:
        raise ValueError(f"{section}.frames is empty.")
    return ModelMapping(
        xml=raw.get("xml"),
        root_freejoint=str(raw.get("root_freejoint", "root")),
        base_site=str(raw.get("base_site", "base_site")),
        joints=joints,
        frames=frames,
    )


def load_mapping(path: str | Path) -> RetargetMapping:
    mapping_path = Path(path)
    with mapping_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    retarget_raw = raw.get("retarget", {}) or {}
    return RetargetMapping(
        source=_read_model_mapping(raw.get("source", {}) or {}, "source"),
        target=_read_model_mapping(raw.get("target", {}) or {}, "target"),
        options=RetargetOptions(**{k: v for k, v in retarget_raw.items() if k in RetargetOptions.__annotations__}),
    )
