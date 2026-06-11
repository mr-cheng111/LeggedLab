#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 PyTorch checkpoint 的 key、shape 和 WMP 兼容性。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch


def _shape(value) -> str:
    if isinstance(value, torch.Tensor):
        return f"shape={tuple(value.shape)} dtype={value.dtype}"
    return type(value).__name__


def _summarize_mapping(name: str, mapping: Mapping, max_items: int) -> None:
    print(f"\n[{name}] dict keys={len(mapping)}")
    for index, (key, value) in enumerate(mapping.items()):
        if index >= max_items:
            print(f"  ... ({len(mapping) - max_items} more)")
            break
        print(f"  {key}: {_shape(value)}")


def _state_dict_like(value) -> bool:
    return isinstance(value, Mapping) and any(isinstance(v, torch.Tensor) for v in value.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a PyTorch .pt/.pth checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Checkpoint path.")
    parser.add_argument("--max_items", type=int, default=80, help="Max items printed for each dict.")
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional reference checkpoint. Prints shape matches/mismatches for state_dict-like entries.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    print(f"[INFO] checkpoint={checkpoint}")
    print(f"[INFO] root={type(data).__name__}")

    if isinstance(data, Mapping):
        _summarize_mapping("root", data, args.max_items)
        state_dicts = {key: value for key, value in data.items() if _state_dict_like(value)}
        for key, value in state_dicts.items():
            _summarize_mapping(key, value, args.max_items)
    else:
        print(_shape(data))
        state_dicts = {}

    if args.compare is None:
        return

    reference = torch.load(args.compare.expanduser().resolve(), map_location="cpu", weights_only=False)
    if not isinstance(reference, Mapping) or not isinstance(data, Mapping):
        raise TypeError("--compare requires both checkpoints to be mapping-like.")

    print(f"\n[COMPARE] reference={args.compare.expanduser().resolve()}")
    for key, value in data.items():
        if not _state_dict_like(value):
            continue
        ref = reference.get(key)
        if not _state_dict_like(ref):
            print(f"\n[{key}] missing in reference")
            continue
        value_keys = set(value.keys())
        ref_keys = set(ref.keys())
        common = sorted(value_keys & ref_keys)
        missing = sorted(ref_keys - value_keys)
        unexpected = sorted(value_keys - ref_keys)
        mismatched = [
            name
            for name in common
            if isinstance(value[name], torch.Tensor)
            and isinstance(ref[name], torch.Tensor)
            and tuple(value[name].shape) != tuple(ref[name].shape)
        ]
        print(
            f"\n[{key}] common={len(common)} missing={len(missing)} "
            f"unexpected={len(unexpected)} shape_mismatch={len(mismatched)}"
        )
        for name in mismatched[: args.max_items]:
            print(f"  mismatch {name}: src={tuple(value[name].shape)} ref={tuple(ref[name].shape)}")
        for name in missing[: min(args.max_items, 20)]:
            print(f"  missing {name}: ref={_shape(ref[name])}")
        for name in unexpected[: min(args.max_items, 20)]:
            print(f"  unexpected {name}: src={_shape(value[name])}")


if __name__ == "__main__":
    main()
