# -*- coding: utf-8 -*-
"""Gemini2 depth 到 WMP image 输入的预处理。"""

import torch
from torch.nn import functional as F


def depth_to_nchw(
    depth: torch.Tensor,
    near: float = 0.15,
    far: float = 10.0,
    image_size: tuple[int, int] = (64, 64),
) -> torch.Tensor:
    """将 Gemini2 depth 转换为 B,1,H,W。

    归一化公式:
        d_norm = clamp((d - near) / (far - near), 0, 1)
    其中 near/far 来自相机裁剪范围。
    """
    if depth.ndim != 4:
        raise ValueError(f"depth must have shape B,H,W,1 or B,1,H,W, got {tuple(depth.shape)}")
    depth = torch.nan_to_num(depth.float(), nan=far, posinf=far, neginf=near)
    if depth.shape[-1] == 1:
        depth = depth.permute(0, 3, 1, 2)
    elif depth.shape[1] != 1:
        raise ValueError(f"depth channel dimension must be 1, got {tuple(depth.shape)}")
    depth = torch.clamp((depth - near) / (far - near), 0.0, 1.0)
    return F.interpolate(depth, size=image_size, mode="bilinear", align_corners=False)


def depth_to_wmp_image(
    depth: torch.Tensor,
    near: float = 0.15,
    far: float = 10.0,
    image_size: tuple[int, int] = (64, 64),
) -> torch.Tensor:
    """将 Gemini2 depth 转换为 WMP 内部 NHWC 图像 B,64,64,1。"""
    return depth_to_nchw(depth, near=near, far=far, image_size=image_size).permute(0, 2, 3, 1).contiguous()

