# -*- coding: utf-8 -*-
"""RGBD depth 到 WMP image 输入的预处理。"""

import torch
from torch.nn import functional as F


def depth_to_nchw(
    depth: torch.Tensor,
    near: float = 0.0,
    far: float = 2.0,
    image_size: tuple[int, int] = (64, 64),
) -> torch.Tensor:
    """将 depth 转换为 WMP 原生 B,1,H,W 输入。

    归一化公式:
        d_wmp = clamp(d, near, far)
        image = (d_wmp - near) / (far - near) - 0.5
    原版 Isaac Gym 返回负深度，所以写作:
        image = ((-depth) - near) / (far - near) - 0.5
    IsaacLab 的 distance_to_image_plane 为正深度，因此这里先统一成正深度再套用同一公式。
    """
    if depth.ndim != 4:
        raise ValueError(f"depth must have shape B,H,W,1 or B,1,H,W, got {tuple(depth.shape)}")
    depth = torch.nan_to_num(depth.float(), nan=far, posinf=far, neginf=-far)
    if depth.shape[-1] == 1:
        depth = depth.permute(0, 3, 1, 2)
    elif depth.shape[1] != 1:
        raise ValueError(f"depth channel dimension must be 1, got {tuple(depth.shape)}")
    depth = torch.where(depth < 0.0, -depth, depth)
    depth = torch.clamp(depth, near, far)
    depth = (depth - near) / max(far - near, 1.0e-6) - 0.5
    return F.interpolate(depth, size=image_size, mode="bilinear", align_corners=False)


def depth_to_wmp_image(
    depth: torch.Tensor,
    near: float = 0.0,
    far: float = 2.0,
    image_size: tuple[int, int] = (64, 64),
) -> torch.Tensor:
    """将 RGBD depth 转换为 WMP 内部 NHWC 图像 B,64,64,1。"""
    return depth_to_nchw(depth, near=near, far=far, image_size=image_size).permute(0, 2, 3, 1).contiguous()
