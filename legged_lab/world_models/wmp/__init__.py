# -*- coding: utf-8 -*-
"""WMP 风格 RSSM 世界模型。

代码结构参考 ByteDance WMP 与 dreamerv3-torch，当前仅提供模型前向与
b2_rgbd smoke test 所需接口，不接入 PPO 联训。
"""

from .config import WMPWorldModelConfig, make_default_wmp_config
from .models import WorldModel
from .preprocess import depth_to_nchw, depth_to_wmp_image

__all__ = [
    "WMPWorldModelConfig",
    "WorldModel",
    "depth_to_nchw",
    "depth_to_wmp_image",
    "make_default_wmp_config",
]

