# -*- coding: utf-8 -*-
"""WMP DepthPredictor。

参考 bytedance/WMP 的 `rsl_rl/modules/depth_predictor.py`。输入前方高度图
和本体感觉，输出 WMP 归一化 depth image：

    image = Decoder(MLP([forward_height_map, prop]))

其中 forward_height_map 默认 21x25=525 维，prop 默认 33 维。
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from . import tools


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch: int, eps: float = 1.0e-3):
        super().__init__()
        self.norm = nn.LayerNorm(ch, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class DepthPredictor(nn.Module):
    def __init__(
        self,
        forward_heightmap_dim: int = 525,
        prop_dim: int = 33,
        depth_image_dims: tuple[int, int] = (64, 64),
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        depth: int = 32,
        act: str = "ELU",
        norm: bool = True,
        kernel_size: int = 4,
        minres: int = 4,
        outscale: float = 1.0,
        cnn_sigmoid: bool = False,
    ):
        super().__init__()
        h, w = depth_image_dims
        stages = int(math.log2(w) - math.log2(minres))
        h_list = []
        w_list = []
        for _ in range(stages):
            h, w = (h + 1) // 2, (w + 1) // 2
            h_list.append(h)
            w_list.append(w)
        self.h_list = list(reversed(h_list)) + [depth_image_dims[0]]
        self.w_list = list(reversed(w_list)) + [depth_image_dims[1]]
        self._cnn_sigmoid = cnn_sigmoid
        act_cls = getattr(nn, act)
        layer_num = len(self.h_list) - 1
        out_ch = self.h_list[0] * self.w_list[0] * depth * 2 ** (len(self.h_list) - 2)
        self._embed_size = out_ch

        encoder_layers: list[nn.Module] = [nn.Linear(forward_heightmap_dim + prop_dim, encoder_hidden_dims[0]), act_cls()]
        for idx in range(len(encoder_hidden_dims)):
            if idx == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[idx], self._embed_size))
            else:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[idx], encoder_hidden_dims[idx + 1]))
                encoder_layers.append(act_cls())
        self.encoder = nn.Sequential(*encoder_layers)

        in_dim = out_ch // (self.h_list[0] * self.w_list[0])
        out_dim = in_dim // 2
        layers: list[nn.Module] = []
        for idx in range(layer_num):
            use_act = True
            use_norm = norm
            bias = False
            if idx == layer_num - 1:
                out_dim = 1
                use_act = False
                use_norm = False
                bias = True
            if idx != 0:
                in_dim = 2 ** (layer_num - (idx - 1) - 2) * depth
            pad_h, outpad_h = (1, 0) if self.h_list[idx] * 2 == self.h_list[idx + 1] else (2, 1)
            pad_w, outpad_w = (1, 0) if self.w_list[idx] * 2 == self.w_list[idx + 1] else (2, 1)
            layers.append(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size,
                    2,
                    padding=(pad_h, pad_w),
                    output_padding=(outpad_h, outpad_w),
                    bias=bias,
                )
            )
            if use_norm:
                layers.append(ImgChLayerNorm(out_dim))
            if use_act:
                layers.append(act_cls())
            in_dim = out_dim
            out_dim //= 2
        for layer in layers[:-1]:
            layer.apply(tools.weight_init)
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def forward(self, forward_heightmap: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        leading_shape = forward_heightmap.shape[:-1]
        x = torch.cat((forward_heightmap, prop), dim=-1)
        x = self.encoder(x.reshape(-1, x.shape[-1]))
        x = x.reshape(-1, self.h_list[0], self.w_list[0], self._embed_size // (self.h_list[0] * self.w_list[0]))
        x = self.layers(x.permute(0, 3, 1, 2))
        mean = x.permute(0, 2, 3, 1)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        return mean.reshape(*leading_shape, self.h_list[-1], self.w_list[-1], 1)
