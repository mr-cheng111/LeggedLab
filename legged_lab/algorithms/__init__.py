# -*- coding: utf-8 -*-
"""LeggedLab 自定义训练算法。"""

from .amp_ppo import AMPPPO
from .wmp_amp_ppo import WMPAMPPPO

__all__ = ["AMPPPO", "WMPAMPPPO"]
