#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smooth WMP AMP motions with Gaussian Mixture Regression."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legged_lab.amp.mink_retarget import load_wmp_motion, save_wmp_motion
from legged_lab.tools.gmr_motion import GMRMotionConfig, smooth_motion


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(match) for match in sorted(glob.glob(pattern)))
    if not paths:
        raise FileNotFoundError(f"No input motion matched: {patterns}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_motion", nargs="+", required=True)
    parser.add_argument("--output_motion")
    parser.add_argument("--output_dir", default="datasets/gmr_smoothed")
    parser.add_argument("--features", default="joint_pos,toe_pos_local")
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--covariance_regularization", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = _expand_inputs(args.input_motion)
    if len(inputs) > 1 and args.output_motion:
        raise ValueError("--output_motion can only be used with a single input file.")
    cfg = GMRMotionConfig(
        n_components=args.components,
        features=tuple(x.strip() for x in args.features.split(",") if x.strip()),
        random_state=args.random_state,
        n_iter=args.n_iter,
        covariance_regularization=args.covariance_regularization,
    )
    for input_path in inputs:
        output_path = Path(args.output_motion) if args.output_motion else Path(args.output_dir) / input_path.name
        motion = load_wmp_motion(input_path)
        smoothed = smooth_motion(motion, cfg)
        save_wmp_motion(output_path, smoothed)
        print(f"[OK] wrote {output_path} frames={smoothed.frames.shape[0]} features={','.join(cfg.features)}")


if __name__ == "__main__":
    main()
