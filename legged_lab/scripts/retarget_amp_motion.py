#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retarget WMP AMP motions from A1 to another quadruped with mink."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legged_lab.amp.mink_retarget import load_mapping, load_wmp_motion, retarget_motion, save_wmp_motion
from legged_lab.amp.mink_retarget.io import JOINT_POS, JOINT_VEL, TOE_POS_LOCAL, WMPMotion


DEFAULT_SOURCE_XML = "legged_lab/assets/unitree/a1/mjcf/a1_retarget.xml"
TARGET_PRESETS = {
    "b2": {
        "target_xml": "legged_lab/assets/unitree/b2/mjcf/b2_retarget.xml",
        "mapping": "legged_lab/assets/unitree/b2/mjcf/a1_to_b2_retarget.yaml",
        "output_dir": "datasets/retargeted/b2",
    },
    "m20": {
        "target_xml": "legged_lab/assets/deeprobotics/m20/mjcf/m20_retarget.xml",
        "mapping": "legged_lab/assets/deeprobotics/m20/mjcf/a1_to_m20_retarget.yaml",
        "output_dir": "datasets/retargeted/m20",
    },
}


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(Path(match) for match in matches)
    if not paths:
        raise FileNotFoundError(f"No input motion matched: {patterns}")
    return paths


def _validate_xml_names(xml_path: str | Path, joint_names: tuple[str, ...], site_names: list[str]) -> None:
    root = ElementTree.parse(xml_path).getroot()
    joints = {node.attrib.get("name") for node in root.iter("joint")}
    sites = {node.attrib.get("name") for node in root.iter("site")}
    missing_joints = [name for name in joint_names if name not in joints]
    missing_sites = [name for name in site_names if name not in sites]
    if missing_joints or missing_sites:
        raise ValueError(f"{xml_path} missing joints={missing_joints}, sites={missing_sites}")


def _validate_static(args: argparse.Namespace, mapping) -> None:
    _validate_xml_names(args.source_xml, mapping.source.joints, list(mapping.source.frames.values()))
    _validate_xml_names(args.target_xml, mapping.target.joints, list(mapping.target.frames.values()))
    for input_path in _expand_inputs(args.input_motion):
        motion = load_wmp_motion(input_path)
        if motion.frames.shape[1] != 61:
            raise ValueError(f"{input_path} frame width must be 61, got {motion.frames.shape[1]}")
        print(f"[OK] {input_path}: {motion.frames.shape[0]} frames, dt={motion.frame_duration}")


def _output_path_for(input_path: Path, args: argparse.Namespace, multiple: bool, variant: str | None = None) -> Path:
    if args.output_motion and not multiple:
        output_path = Path(args.output_motion)
    else:
        output_path = Path(args.output_dir) / input_path.name
    if variant:
        output_path = output_path.with_name(f"{output_path.stem}_{variant}{output_path.suffix}")
    return output_path


def _best_phase_lag(reference: np.ndarray, target: np.ndarray) -> int:
    """Find the smallest signed lag that best aligns two periodic signals."""
    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    reference = (reference - reference.mean()) / max(reference.std(), 1.0e-8)
    target = (target - target.mean()) / max(target.std(), 1.0e-8)
    max_lag = min(reference.shape[0] // 4, 100)
    candidates = [
        (float(np.mean((np.roll(reference, lag) - target) ** 2)), lag)
        for lag in range(-max_lag, max_lag + 1)
    ]
    return min(candidates, key=lambda item: (item[0], abs(item[1])))[1]


def _shift_without_wrap(values: np.ndarray, lag: int) -> np.ndarray:
    """Shift a trajectory without moving its endpoint discontinuity into the middle."""
    shifted = np.empty_like(values)
    if lag > 0:
        shifted[:lag] = values[0]
        shifted[lag:] = values[:-lag]
    elif lag < 0:
        shifted[:lag] = values[-lag:]
        shifted[lag:] = values[-1]
    else:
        shifted[:] = values
    return shifted


def _bilateral_variant(
    motion: WMPMotion, source_joint_names: tuple[str, ...], retained_side: str
) -> WMPMotion:
    """Mirror one side's front/rear joints onto the other side."""
    if retained_side not in {"left", "right"}:
        raise ValueError(f"Unsupported retained_side={retained_side!r}.")

    leg_indices: dict[str, list[int]] = {}
    for leg in ("FR", "FL", "RR", "RL"):
        indices = [index for index, name in enumerate(source_joint_names) if name.startswith(f"{leg}_")]
        if len(indices) != 3:
            raise ValueError(f"Expected three source joints for {leg}, got {indices}.")
        leg_indices[leg] = indices

    frames = motion.frames.copy()
    joint_pos = frames[:, JOINT_POS].copy()
    toe_z = motion.frames[:, TOE_POS_LOCAL].reshape(-1, 4, 3)[:, :, 2]
    left_to_right_lags = {
        "front": _best_phase_lag(toe_z[:, 1], toe_z[:, 0]),
        "rear": _best_phase_lag(toe_z[:, 3], toe_z[:, 2]),
    }
    pairs = (
        (("FL", "FR", left_to_right_lags["front"]), ("RL", "RR", left_to_right_lags["rear"]))
        if retained_side == "left"
        else (
            ("FR", "FL", -left_to_right_lags["front"]),
            ("RR", "RL", -left_to_right_lags["rear"]),
        )
    )
    mirror_sign = np.array([-1.0, 1.0, 1.0], dtype=np.float64)
    for donor_leg, receiver_leg, lag in pairs:
        donor = joint_pos[:, leg_indices[donor_leg]].copy()
        joint_pos[:, leg_indices[receiver_leg]] = _shift_without_wrap(donor, lag) * mirror_sign
    frames[:, JOINT_POS] = joint_pos
    frames[:, JOINT_VEL] = 0.0
    return WMPMotion(
        frames=frames,
        frame_duration=motion.frame_duration,
        motion_weight=motion.motion_weight,
        loop_mode=motion.loop_mode,
    )


def _debug_path_for(output_path: Path, args: argparse.Namespace) -> Path | None:
    if args.debug_npz is None:
        return None
    debug_arg = str(args.debug_npz)
    if debug_arg.lower() in {"1", "true", "yes", "auto"}:
        return output_path.with_suffix(".debug.npz")
    path = Path(debug_arg)
    if path.suffix == ".npz":
        return path
    return path / output_path.with_suffix(".debug.npz").name


def _validate_result(result, mapping, output_path: Path) -> tuple[float, float, float]:
    frames = result.motion.frames
    if not np.isfinite(frames).all():
        raise ValueError(f"{output_path} contains NaN or Inf values.")
    if not np.isfinite(result.foot_error).all():
        raise ValueError(f"{output_path} contains non-finite foot tracking errors.")

    joint_pos = frames[:, JOINT_POS]
    limits = mapping.options.joint_project_limits or {}
    violations = []
    for joint_idx, joint_name in enumerate(mapping.target.joints):
        if joint_name not in limits:
            continue
        lower, upper = (float(value) for value in limits[joint_name])
        value_min = float(joint_pos[:, joint_idx].min())
        value_max = float(joint_pos[:, joint_idx].max())
        if value_min < lower - 1.0e-6 or value_max > upper + 1.0e-6:
            violations.append(f"{joint_name}=[{value_min:.4f},{value_max:.4f}] outside [{lower:.4f},{upper:.4f}]")
    if violations:
        raise ValueError(f"{output_path} violates target joint limits: {'; '.join(violations)}")

    mean_foot_error = float(np.mean(result.foot_error))
    max_foot_error = float(np.max(result.foot_error))
    max_joint_velocity = float(np.max(np.abs(frames[:, JOINT_VEL])))
    return mean_foot_error, max_foot_error, max_joint_velocity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_motion", nargs="+", default=["datasets/wmp_mocap_motions/*.txt"])
    parser.add_argument("--target_robot", choices=sorted(TARGET_PRESETS), default="b2")
    parser.add_argument("--source_xml", default=DEFAULT_SOURCE_XML)
    parser.add_argument("--target_xml")
    parser.add_argument("--mapping")
    parser.add_argument("--output_motion")
    parser.add_argument("--output_dir")
    parser.add_argument("--debug_npz", nargs="?", const="auto")
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--gmr_pre", action="store_true", help="Smooth source motion before IK retargeting.")
    parser.add_argument("--gmr_post", action="store_true", help="Smooth retargeted motion after IK.")
    parser.add_argument("--gmr_features", default="joint_pos,toe_pos_local")
    parser.add_argument("--gmr_components", type=int, default=8)
    parser.add_argument("--gmr_n_iter", type=int, default=100)
    parser.add_argument("--gmr_covariance_regularization", type=float, default=1.0e-4)
    parser.add_argument(
        "--bilateral_variants",
        action="store_true",
        help="Generate left- and right-retained symmetric variants for every input motion.",
    )
    args = parser.parse_args()
    preset = TARGET_PRESETS[args.target_robot]
    args.target_xml = args.target_xml or preset["target_xml"]
    args.mapping = args.mapping or preset["mapping"]
    args.output_dir = args.output_dir or preset["output_dir"]
    return args


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.mapping)
    _validate_static(args, mapping)
    if args.validate_only:
        print("[OK] static validation complete.")
        return

    input_paths = _expand_inputs(args.input_motion)
    multiple = len(input_paths) > 1
    if multiple and args.output_motion:
        raise ValueError("--output_motion can only be used with a single input file.")
    gmr_plugin = None
    if args.gmr_pre or args.gmr_post:
        from legged_lab.tools.gmr_motion import GMRMotionConfig, GMRMotionPlugin

        gmr_plugin = GMRMotionPlugin(
            GMRMotionConfig(
                n_components=args.gmr_components,
                features=tuple(x.strip() for x in args.gmr_features.split(",") if x.strip()),
                n_iter=args.gmr_n_iter,
                covariance_regularization=args.gmr_covariance_regularization,
            )
        )

    for input_path in input_paths:
        motion = load_wmp_motion(input_path)
        if args.gmr_pre:
            assert gmr_plugin is not None
            motion = gmr_plugin.preprocess(motion)
        variants = (
            [
                ("left", _bilateral_variant(motion, mapping.source.joints, "left")),
                ("right", _bilateral_variant(motion, mapping.source.joints, "right")),
            ]
            if args.bilateral_variants
            else [(None, motion)]
        )
        for variant_name, variant_motion in variants:
            output_path = _output_path_for(input_path, args, multiple, variant_name)
            result = retarget_motion(
                variant_motion,
                source_xml=args.source_xml,
                target_xml=args.target_xml,
                mapping=mapping,
                max_frames=args.max_frames,
            )
            if args.gmr_post:
                assert gmr_plugin is not None
                result.motion = gmr_plugin.postprocess(result.motion)
            mean_foot_error, max_foot_error, max_joint_velocity = _validate_result(result, mapping, output_path)
            save_wmp_motion(output_path, result.motion)
            print(
                "[OK] wrote "
                f"{output_path} frames={result.motion.frames.shape[0]} "
                f"foot_error_mean/max={mean_foot_error:.5f}/{max_foot_error:.5f} "
                f"max_joint_velocity={max_joint_velocity:.3f}"
            )

            debug_path = _debug_path_for(output_path, args)
            if debug_path is not None:
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    debug_path,
                    source_feet_world=result.source_feet_world,
                    target_feet_world=result.target_feet_world,
                    foot_error=result.foot_error,
                    target_qpos=result.target_qpos,
                )
                print(f"[OK] wrote debug {debug_path}")


if __name__ == "__main__":
    main()
