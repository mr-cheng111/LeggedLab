#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将 ByteDance 原版 WMP checkpoint 转成当前 LeggedLab WMP 播放格式。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _rename_world_model_key(key: str) -> str:
    replacements = (
        ("encoder._mlp.layers.Encoder_linear", "encoder._mlp.layers.linear"),
        ("encoder._mlp.layers.Encoder_norm", "encoder._mlp.layers.norm"),
        ("heads.decoder._mlp.layers.Decoder_linear", "heads.decoder._mlp.layers.linear"),
        ("heads.decoder._mlp.layers.Decoder_norm", "heads.decoder._mlp.layers.norm"),
        ("heads.reward.layers.Reward_linear", "heads.reward.layers.linear"),
        ("heads.reward.layers.Reward_norm", "heads.reward.layers.norm"),
        ("dynamics._cell.layers.GRU_linear.weight", "dynamics._cell.layers.0.weight"),
    )
    for old, new in replacements:
        key = key.replace(old, new)
    return key


def _convert_world_model(src: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {_rename_world_model_key(key): value for key, value in src.items()}


def _convert_actor(src: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    dst = {key: value.clone() for key, value in reference.items()}
    mapping = {
        "std": "distribution.std_param",
        "actor.0.weight": "mlp.0.weight",
        "actor.0.bias": "mlp.0.bias",
        "actor.2.weight": "mlp.2.weight",
        "actor.2.bias": "mlp.2.bias",
        "actor.4.weight": "mlp.4.weight",
        "actor.4.bias": "mlp.4.bias",
        "actor.6.weight": "mlp.6.weight",
        "actor.6.bias": "mlp.6.bias",
        "wm_feature_encoder.0.weight": "wmp_encoder.0.weight",
        "wm_feature_encoder.0.bias": "wmp_encoder.0.bias",
        "wm_feature_encoder.2.weight": "wmp_encoder.2.weight",
        "wm_feature_encoder.2.bias": "wmp_encoder.2.bias",
        "wm_feature_encoder.4.weight": "wmp_encoder.4.weight",
        "wm_feature_encoder.4.bias": "wmp_encoder.4.bias",
        "history_encoder.0.weight": "history_encoder.0.weight",
        "history_encoder.0.bias": "history_encoder.0.bias",
        "history_encoder.2.weight": "history_encoder.2.weight",
        "history_encoder.2.bias": "history_encoder.2.bias",
        "history_encoder.4.weight": "history_encoder.4.weight",
        "history_encoder.4.bias": "history_encoder.4.bias",
    }
    for src_key, dst_key in mapping.items():
        if src_key not in src or dst_key not in dst:
            raise KeyError(f"Missing actor key mapping {src_key} -> {dst_key}")
        if tuple(src[src_key].shape) != tuple(dst[dst_key].shape):
            raise ValueError(f"Actor shape mismatch {src_key} -> {dst_key}: {src[src_key].shape} vs {dst[dst_key].shape}")
        dst[dst_key] = src[src_key].clone()
    # 原版 ActorCriticWMP 没有 actor latent normalizer；当前 WMPMLPModel 有 normalizer。
    # 为保持原版推理语义，设为恒等变换: norm(x) = (x - 0) / 1。
    for key in ("obs_normalizer._mean",):
        if key in dst:
            dst[key] = torch.zeros_like(dst[key])
    for key in ("obs_normalizer._var", "obs_normalizer._std"):
        if key in dst:
            dst[key] = torch.ones_like(dst[key])
    return dst


def _copy_matching(src: dict[str, torch.Tensor], reference: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    dst = {key: value.clone() for key, value in reference.items()}
    for key, value in src.items():
        if key not in dst:
            raise KeyError(f"{prefix}: unexpected key {key}")
        if tuple(value.shape) != tuple(dst[key].shape):
            raise ValueError(f"{prefix}: shape mismatch {key}: {value.shape} vs {dst[key].shape}")
        dst[key] = value.clone()
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert original ByteDance WMP checkpoint to LeggedLab format.")
    parser.add_argument("source", type=Path, help="Original WMP checkpoint path.")
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="A LeggedLab WMP checkpoint used as structural template.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output converted checkpoint path.")
    args = parser.parse_args()

    source = torch.load(args.source.expanduser().resolve(), map_location="cpu", weights_only=False)
    reference = torch.load(args.reference.expanduser().resolve(), map_location="cpu", weights_only=False)

    required = ("model_state_dict", "world_model_dict", "depth_predictor")
    missing = [key for key in required if key not in source]
    if missing:
        raise KeyError(f"Source is not an original WMP checkpoint, missing: {missing}")

    converted = dict(reference)
    converted["actor_state_dict"] = _convert_actor(source["model_state_dict"], reference["actor_state_dict"])
    converted["world_model_state_dict"] = _copy_matching(
        _convert_world_model(source["world_model_dict"]),
        reference["world_model_state_dict"],
        "world_model",
    )
    converted["depth_predictor_state_dict"] = _copy_matching(
        source["depth_predictor"],
        reference["depth_predictor_state_dict"],
        "depth_predictor",
    )
    converted["iter"] = int(source.get("iter", reference.get("iter", 0)))
    converted["infos"] = source.get("infos")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output)
    print(f"[INFO] converted checkpoint saved: {output}")
    print("[INFO] converted modules: actor_state_dict, world_model_state_dict, depth_predictor_state_dict")
    print("[WARN] critic/optimizer/AMP states are kept from reference template; inference uses converted actor + WMP modules.")


if __name__ == "__main__":
    main()
