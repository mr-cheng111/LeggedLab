# -*- coding: utf-8 -*-
"""WMP/Dreamer 工具函数与轻量分布封装。

本文件包含从 ByteDance WMP 的 dreamer/tools.py 裁剪出的模型前向所需组件。

MIT License
Copyright (c) 2023 NM512
This file may include modifications inspired by Bytedance WMP.
"""

import numpy as np
import torch
from torch import distributions as torchd
from torch import nn
from torch.nn import functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        mode = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
        return mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("seeded sampling is not supported")
        sample = super().sample(sample_shape)
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        return sample + probs - probs.detach()


class ContDist:
    def __init__(self, dist=None, absmax=None):
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


class MSEDist:
    def __init__(self, mode, agg="sum"):
        self._mode = mode
        self._agg = agg

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        dims = list(range(len(distance.shape)))[2:]
        if self._agg == "mean":
            loss = distance.mean(dims)
        elif self._agg == "sum":
            loss = distance.sum(dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class SymlogDist:
    def __init__(self, mode, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dist = dist
        self._agg = agg
        self._tol = tol

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2.0
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        dims = list(range(len(distance.shape)))[2:]
        return -distance.sum(dims)


class DiscDist:
    def __init__(self, logits, low=-20.0, high=20.0, device="cuda"):
        self.logits = logits
        self.probs = torch.softmax(logits, -1)
        self.buckets = torch.linspace(low, high, steps=255, device=device)

    def mode(self):
        mode = self.probs * self.buckets
        return symexp(torch.sum(mode, dim=-1, keepdim=True))

    def log_prob(self, x):
        x = symlog(x)
        below = torch.sum((self.buckets <= x[..., None]).to(torch.int32), dim=-1) - 1
        above = len(self.buckets) - torch.sum((self.buckets > x[..., None]).to(torch.int32), dim=-1)
        below = torch.clip(below, 0, len(self.buckets) - 1)
        above = torch.clip(above, 0, len(self.buckets) - 1)
        equal = below == above
        dist_to_below = torch.where(equal, 1, torch.abs(self.buckets[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.buckets[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = F.one_hot(below, num_classes=len(self.buckets)) * weight_below[..., None]
        target += F.one_hot(above, num_classes=len(self.buckets)) * weight_above[..., None]
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdim=True)
        return (target.squeeze(-2) * log_pred).sum(-1)


def static_scan(fn, inputs, start):
    last = start
    outputs = None
    for index in range(inputs[0].shape[0]):
        inp = lambda x: (_input[x] for _input in inputs)
        last = fn(last, *inp(index))
        if outputs is None:
            if isinstance(last, dict):
                outputs = {key: value.clone().unsqueeze(0) for key, value in last.items()}
            else:
                outputs = []
                for item in last:
                    if isinstance(item, dict):
                        outputs.append({key: value.clone().unsqueeze(0) for key, value in item.items()})
                    else:
                        outputs.append(item.clone().unsqueeze(0))
        else:
            if isinstance(last, dict):
                for key in last.keys():
                    outputs[key] = torch.cat([outputs[key], last[key].unsqueeze(0)], dim=0)
            else:
                for idx, item in enumerate(last):
                    if isinstance(item, dict):
                        for key in item.keys():
                            outputs[idx][key] = torch.cat([outputs[idx][key], item[key].unsqueeze(0)], dim=0)
                    else:
                        outputs[idx] = torch.cat([outputs[idx], item.unsqueeze(0)], dim=0)
    if isinstance(last, dict):
        outputs = [outputs]
    return outputs


def weight_init(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        denoms = (in_num + out_num) / 2.0
        scale = 1.0 / denoms
        std = np.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    return f

