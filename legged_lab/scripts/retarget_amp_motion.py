#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retarget WMP AMP motions from A1 XML to B2 XML with mink."""

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
from legged_lab.tools.gmr_motion import GMRMotionConfig, GMRMotionPlugin


DEFAULT_SOURCE_XML = "legged_lab/assets/unitree/a1/mjcf/a1_retarget.xml"
DEFAULT_TARGET_XML = "legged_lab/assets/unitree/b2/mjcf/b2_retarget.xml"
DEFAULT_MAPPING = "legged_lab/assets/unitree/b2/mjcf/a1_to_b2_retarget.yaml"


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


def _output_path_for(input_path: Path, args: argparse.Namespace, multiple: bool) -> Path:
    if args.output_motion and not multiple:
        return Path(args.output_motion)
    output_dir = Path(args.output_dir)
    return output_dir / input_path.name


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_motion", nargs="+", default=["datasets/wmp_mocap_motions/*.txt"])
    parser.add_argument("--source_xml", default=DEFAULT_SOURCE_XML)
    parser.add_argument("--target_xml", default=DEFAULT_TARGET_XML)
    parser.add_argument("--mapping", default=DEFAULT_MAPPING)
    parser.add_argument("--output_motion")
    parser.add_argument("--output_dir", default="datasets/retargeted/b2")
    parser.add_argument("--debug_npz", nargs="?", const="auto")
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--gmr_pre", action="store_true", help="Smooth source motion before IK retargeting.")
    parser.add_argument("--gmr_post", action="store_true", help="Smooth retargeted motion after IK.")
    parser.add_argument("--gmr_features", default="joint_pos,toe_pos_local")
    parser.add_argument("--gmr_components", type=int, default=8)
    parser.add_argument("--gmr_n_iter", type=int, default=100)
    parser.add_argument("--gmr_covariance_regularization", type=float, default=1.0e-4)
    return parser.parse_args()


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
    gmr_plugin = GMRMotionPlugin(
        GMRMotionConfig(
            n_components=args.gmr_components,
            features=tuple(x.strip() for x in args.gmr_features.split(",") if x.strip()),
            n_iter=args.gmr_n_iter,
            covariance_regularization=args.gmr_covariance_regularization,
        )
    )

    for input_path in input_paths:
        output_path = _output_path_for(input_path, args, multiple)
        motion = load_wmp_motion(input_path)
        if args.gmr_pre:
            motion = gmr_plugin.preprocess(motion)
        result = retarget_motion(
            motion,
            source_xml=args.source_xml,
            target_xml=args.target_xml,
            mapping=mapping,
            max_frames=args.max_frames,
        )
        if args.gmr_post:
            result.motion = gmr_plugin.postprocess(result.motion)
        save_wmp_motion(output_path, result.motion)
        print(
            "[OK] wrote "
            f"{output_path} frames={result.motion.frames.shape[0]} "
            f"mean_foot_error={float(np.mean(result.foot_error)):.5f}"
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
