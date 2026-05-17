# -*- coding: utf-8 -*-
"""WMP RSSM 网络组件。

参考 ByteDance WMP 的 dreamer/networks.py，保留 RSSM、MultiEncoder、
MultiDecoder 与 b2_rgbd smoke test 所需分布接口。
"""

import math
import re

import numpy as np
import torch
from torch import distributions as torchd
from torch import nn
from torch.nn import functional as F

from . import tools


class RSSM(nn.Module):
    def __init__(
        self,
        stoch=30,
        deter=200,
        hidden=200,
        rec_depth=1,
        discrete=False,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
    ):
        super().__init__()
        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._min_std = min_std
        self._rec_depth = rec_depth
        self._discrete = discrete
        self._mean_act = mean_act
        self._std_act = std_act
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._num_actions = num_actions
        self._embed = embed
        self._device = device
        act_cls = getattr(torch.nn, act)

        inp_dim = (self._stoch * self._discrete if self._discrete else self._stoch) + num_actions
        self._img_in_layers = self._make_layers(inp_dim, self._hidden, act_cls, norm)
        self._img_in_layers.apply(tools.weight_init)
        self._cell = GRUCell(self._hidden, self._deter, norm=norm)
        self._cell.apply(tools.weight_init)

        self._img_out_layers = self._make_layers(self._deter, self._hidden, act_cls, norm)
        self._img_out_layers.apply(tools.weight_init)
        self._obs_out_layers = self._make_layers(self._deter + self._embed, self._hidden, act_cls, norm)
        self._obs_out_layers.apply(tools.weight_init)

        if self._discrete:
            self._imgs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
            self._obs_stat_layer = nn.Linear(self._hidden, self._stoch * self._discrete)
        else:
            self._imgs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
            self._obs_stat_layer = nn.Linear(self._hidden, 2 * self._stoch)
        self._imgs_stat_layer.apply(tools.uniform_weight_init(1.0))
        self._obs_stat_layer.apply(tools.uniform_weight_init(1.0))

        if self._initial == "learned":
            self.W = torch.nn.Parameter(torch.zeros((1, self._deter), device=torch.device(self._device)))

    @staticmethod
    def _make_layers(inp_dim, out_dim, act_cls, norm):
        layers = [nn.Linear(inp_dim, out_dim, bias=False)]
        if norm:
            layers.append(nn.LayerNorm(out_dim, eps=1e-03))
        layers.append(act_cls())
        return nn.Sequential(*layers)

    def initial(self, batch_size):
        deter = torch.zeros(batch_size, self._deter, device=self._device)
        if self._discrete:
            state = {
                "logit": torch.zeros([batch_size, self._stoch, self._discrete], device=self._device),
                "stoch": torch.zeros([batch_size, self._stoch, self._discrete], device=self._device),
                "deter": deter,
            }
        else:
            state = {
                "mean": torch.zeros([batch_size, self._stoch], device=self._device),
                "std": torch.zeros([batch_size, self._stoch], device=self._device),
                "stoch": torch.zeros([batch_size, self._stoch], device=self._device),
                "deter": deter,
            }
        if self._initial == "zeros":
            return state
        if self._initial == "learned":
            state["deter"] = torch.tanh(self.W).repeat(batch_size, 1)
            state["stoch"] = self.get_stoch(state["deter"])
            return state
        raise NotImplementedError(self._initial)

    def observe(self, embed, action, is_first, state=None):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        embed, action, is_first = swap(embed), swap(action), swap(is_first)
        post, prior = tools.static_scan(
            lambda prev_state, prev_act, embed, is_first: self.obs_step(prev_state[0], prev_act, embed, is_first),
            (action, embed, is_first),
            (state, state),
        )
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        return post, prior

    def imagine_with_action(self, action, state):
        swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
        action = swap(action)
        prior = tools.static_scan(self.img_step, [action], state)[0]
        return {k: swap(v) for k, v in prior.items()}

    def get_feat(self, state):
        stoch = state["stoch"]
        if self._discrete:
            stoch = stoch.reshape(list(stoch.shape[:-2]) + [self._stoch * self._discrete])
        return torch.cat([stoch, state["deter"]], -1)

    def get_deter_feat(self, state):
        return state["deter"]

    def get_dist(self, state):
        if self._discrete:
            dist = tools.OneHotDist(state["logit"], unimix_ratio=self._unimix_ratio)
            return torchd.independent.Independent(dist, 1)
        return tools.ContDist(torchd.independent.Independent(torchd.normal.Normal(state["mean"], state["std"]), 1))

    def kl_loss(self, post, prior, free, dyn_scale, rep_scale):
        """DreamerV3/WMP KL 损失。

        公式来源为 WMP dreamer/networks.py:
        L = dyn_scale * max(KL(sg(post) || prior), free)
          + rep_scale * max(KL(post || sg(prior)), free)
        其中 sg 表示 stop-gradient。
        """
        dist = lambda x: self.get_dist(x)
        sg = lambda x: {k: v.detach() for k, v in x.items()}
        rep_loss = value = torchd.kl.kl_divergence(dist(post), dist(sg(prior)))
        dyn_loss = torchd.kl.kl_divergence(dist(sg(post)), dist(prior))
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        loss = dyn_scale * dyn_loss + rep_scale * rep_loss
        return loss, value, dyn_loss, rep_loss

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        if prev_state is None or torch.sum(is_first) == len(is_first):
            prev_state = self.initial(len(is_first))
            prev_action = torch.zeros((len(is_first), self._num_actions), device=self._device)
        elif torch.sum(is_first) > 0:
            is_first = is_first[:, None]
            prev_action *= 1.0 - is_first
            init_state = self.initial(len(is_first))
            for key, val in prev_state.items():
                is_first_r = torch.reshape(is_first, is_first.shape + (1,) * (len(val.shape) - len(is_first.shape)))
                prev_state[key] = val * (1.0 - is_first_r) + init_state[key] * is_first_r

        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior["deter"], embed], -1)
        stats = self._suff_stats_layer("obs", self._obs_out_layers(x))
        stoch = self.get_dist(stats).sample() if sample else self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def img_step(self, prev_state, prev_action, sample=True):
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            prev_stoch = prev_stoch.reshape(list(prev_stoch.shape[:-2]) + [self._stoch * self._discrete])
        x = torch.cat([prev_stoch, prev_action], -1)
        x = self._img_in_layers(x)
        for _ in range(self._rec_depth):
            deter = prev_state["deter"]
            x, deter = self._cell(x, [deter])
            deter = deter[0]
        stats = self._suff_stats_layer("ims", self._img_out_layers(deter))
        stoch = self.get_dist(stats).sample() if sample else self.get_dist(stats).mode()
        return {"stoch": stoch, "deter": deter, **stats}

    def get_stoch(self, deter):
        stats = self._suff_stats_layer("ims", self._img_out_layers(deter))
        return self.get_dist(stats).mode()

    def _suff_stats_layer(self, name, x):
        x = self._imgs_stat_layer(x) if name == "ims" else self._obs_stat_layer(x)
        if self._discrete:
            return {"logit": x.reshape(list(x.shape[:-1]) + [self._stoch, self._discrete])}
        mean, std = torch.split(x, [self._stoch] * 2, -1)
        mean = {"none": lambda: mean, "tanh5": lambda: 5.0 * torch.tanh(mean / 5.0)}[self._mean_act]()
        std = {
            "softplus": lambda: F.softplus(std),
            "abs": lambda: torch.abs(std + 1),
            "sigmoid": lambda: torch.sigmoid(std),
            "sigmoid2": lambda: 2 * torch.sigmoid(std / 2),
        }[self._std_act]()
        return {"mean": mean, "std": std + self._min_std}


class MultiEncoder(nn.Module):
    def __init__(self, shapes, mlp_keys, cnn_keys, act, norm, cnn_depth, kernel_size, minres, mlp_layers, mlp_units, symlog_inputs, use_camera=False):
        super().__init__()
        self.use_camera = use_camera
        excluded = ("is_first", "is_last", "is_terminal", "reward", "height_map")
        shapes = {k: v for k, v in shapes.items() if k not in excluded and not k.startswith("log_")}
        self.cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)}
        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(mlp_keys, k)}
        self.outdim = 0
        if self.cnn_shapes:
            input_ch = sum(v[-1] for v in self.cnn_shapes.values())
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self._cnn = ConvEncoder(input_shape, cnn_depth, act, norm, kernel_size, minres)
            self.outdim += self._cnn.outdim
        if self.mlp_shapes:
            input_size = sum(sum(v) for v in self.mlp_shapes.values())
            self._mlp = MLP(input_size, None, mlp_layers, mlp_units, act, norm, symlog_inputs=symlog_inputs)
            self.outdim += mlp_units

    def forward(self, obs):
        outputs = []
        if self.cnn_shapes:
            if self.use_camera:
                inputs = torch.cat([obs[k] for k in self.cnn_shapes], -1)
                outputs.append(self._cnn(inputs))
            else:
                outputs.append(torch.zeros((obs["is_first"].shape + (self._cnn.outdim,)), device=obs["is_first"].device))
        if self.mlp_shapes:
            inputs = torch.cat([obs[k] for k in self.mlp_shapes], -1)
            outputs.append(self._mlp(inputs))
        return torch.cat(outputs, -1)


class MultiDecoder(nn.Module):
    def __init__(self, feat_size, shapes, mlp_keys, cnn_keys, act, norm, cnn_depth, kernel_size, minres, mlp_layers, mlp_units, cnn_sigmoid, image_dist, vector_dist, outscale, use_camera=False):
        super().__init__()
        self.use_camera = use_camera
        shapes = {k: v for k, v in shapes.items() if k not in ("is_first", "is_last", "is_terminal", "height_map")}
        self.cnn_shapes = {k: v for k, v in shapes.items() if len(v) == 3 and re.match(cnn_keys, k)}
        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(mlp_keys, k)}
        if self.cnn_shapes:
            some_shape = list(self.cnn_shapes.values())[0]
            shape = (sum(x[-1] for x in self.cnn_shapes.values()),) + some_shape[:-1]
            self._cnn = ConvDecoder(feat_size, shape, cnn_depth, act, norm, kernel_size, minres, outscale, cnn_sigmoid)
        if self.mlp_shapes:
            self._mlp = MLP(feat_size, self.mlp_shapes, mlp_layers, mlp_units, act, norm, vector_dist, outscale=outscale)
        self._image_dist = image_dist

    def forward(self, features):
        dists = {}
        if self.cnn_shapes and self.use_camera:
            outputs = self._cnn(features)
            split_sizes = [v[-1] for v in self.cnn_shapes.values()]
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update({key: self._make_image_dist(output) for key, output in zip(self.cnn_shapes.keys(), outputs)})
        if self.mlp_shapes:
            dists.update(self._mlp(features))
        return dists

    def _make_image_dist(self, mean):
        if self._image_dist == "mse":
            return tools.MSEDist(mean)
        if self._image_dist == "normal":
            return tools.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, 1), 3))
        raise NotImplementedError(self._image_dist)


class ConvEncoder(nn.Module):
    def __init__(self, input_shape, depth=32, act="SiLU", norm=True, kernel_size=4, minres=4):
        super().__init__()
        act_cls = getattr(torch.nn, act)
        h, w, input_ch = input_shape
        stages = int(np.log2(w) - np.log2(minres))
        in_dim = input_ch
        out_dim = depth
        layers = []
        for _ in range(stages):
            layers.append(Conv2dSamePad(in_dim, out_dim, kernel_size=kernel_size, stride=2, bias=False))
            if norm:
                layers.append(ImgChLayerNorm(out_dim))
            layers.append(act_cls())
            in_dim = out_dim
            out_dim *= 2
            h, w = (h + 1) // 2, (w + 1) // 2
        self.outdim = out_dim // 2 * h * w
        self.layers = nn.Sequential(*layers)
        self.layers.apply(tools.weight_init)

    def forward(self, obs):
        obs = obs - 0.5
        x = obs.reshape((-1,) + tuple(obs.shape[-3:]))
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        return x.reshape(list(obs.shape[:-3]) + [x.shape[-1]])


class ConvDecoder(nn.Module):
    def __init__(self, feat_size, shape=(3, 64, 64), depth=32, act="SiLU", norm=True, kernel_size=4, minres=4, outscale=1.0, cnn_sigmoid=False):
        input_ch, h, w = shape
        stages = int(np.log2(w) - np.log2(minres))
        self.h_list = []
        self.w_list = []
        for _ in range(stages):
            h, w = (h + 1) // 2, (w + 1) // 2
            self.h_list.append(h)
            self.w_list.append(w)
        self.h_list = self.h_list[::-1]
        self.w_list = self.w_list[::-1]
        self.h_list.append(shape[1])
        self.w_list.append(shape[2])
        super().__init__()
        act_cls = getattr(torch.nn, act)
        self._shape = shape
        self._cnn_sigmoid = cnn_sigmoid
        layer_num = len(self.h_list) - 1
        out_ch = self.h_list[0] * self.w_list[0] * depth * 2 ** (len(self.h_list) - 2)
        self._embed_size = out_ch
        self._linear_layer = nn.Linear(feat_size, out_ch)
        self._linear_layer.apply(tools.uniform_weight_init(outscale))
        in_dim = out_ch // (self.h_list[0] * self.w_list[0])
        out_dim = in_dim // 2
        layers = []
        for i in range(layer_num):
            bias = i == layer_num - 1
            layer_norm = norm and not bias
            layer_act = None if bias else act_cls()
            if i != 0:
                in_dim = 2 ** (layer_num - (i - 1) - 2) * depth
            pad_h, outpad_h = (1, 0) if self.h_list[i] * 2 == self.h_list[i + 1] else (2, 1)
            pad_w, outpad_w = (1, 0) if self.w_list[i] * 2 == self.w_list[i + 1] else (2, 1)
            layers.append(nn.ConvTranspose2d(in_dim, self._shape[0] if bias else out_dim, kernel_size, 2, padding=(pad_h, pad_w), output_padding=(outpad_h, outpad_w), bias=bias))
            if layer_norm:
                layers.append(ImgChLayerNorm(out_dim))
            if layer_act is not None:
                layers.append(layer_act)
            in_dim = out_dim
            out_dim //= 2
        [m.apply(tools.weight_init) for m in layers[:-1]]
        layers[-1].apply(tools.uniform_weight_init(outscale))
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        squeeze_time = features.ndim == 2
        if squeeze_time:
            features = features.unsqueeze(1)
        x = self._linear_layer(features)
        x = x.reshape([-1, self.h_list[0], self.w_list[0], self._embed_size // (self.h_list[0] * self.w_list[0])])
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)
        mean = x.reshape(features.shape[:-1] + self._shape)
        mean = mean.permute(0, 1, 3, 4, 2)
        if squeeze_time:
            mean = mean.squeeze(1)
        if self._cnn_sigmoid:
            mean = F.sigmoid(mean)
        return mean


class MLP(nn.Module):
    def __init__(self, inp_dim, shape, layers, units, act="SiLU", norm=True, dist="normal", std=1.0, min_std=0.1, max_std=1.0, outscale=1.0, symlog_inputs=False):
        super().__init__()
        self._shape = (shape,) if isinstance(shape, int) else shape
        if self._shape is not None and len(self._shape) == 0:
            self._shape = (1,)
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,))
        self._min_std = min_std
        self._max_std = max_std
        self._symlog_inputs = symlog_inputs
        act_cls = getattr(torch.nn, act)

        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(f"linear{i}", nn.Linear(inp_dim, units, bias=False))
            if norm:
                self.layers.add_module(f"norm{i}", nn.LayerNorm(units, eps=1e-03))
            self.layers.add_module(f"act{i}", act_cls())
            inp_dim = units
        self.layers.apply(tools.weight_init)

        if isinstance(self._shape, dict):
            self.mean_layer = nn.ModuleDict({name: nn.Linear(inp_dim, int(np.prod(shape))) for name, shape in self._shape.items()})
            self.mean_layer.apply(tools.uniform_weight_init(outscale))
        elif self._shape is not None:
            self.mean_layer = nn.Linear(inp_dim, int(np.prod(self._shape)))
            self.mean_layer.apply(tools.uniform_weight_init(outscale))

    def forward(self, features):
        x = tools.symlog(features) if self._symlog_inputs else features
        out = self.layers(x)
        if self._shape is None:
            return out
        if isinstance(self._shape, dict):
            return {name: self.dist(self._dist, self.mean_layer[name](out), shape) for name, shape in self._shape.items()}
        return self.dist(self._dist, self.mean_layer(out), self._shape)

    def dist(self, dist, mean, shape):
        if dist == "symlog_mse":
            return tools.SymlogDist(mean.reshape(mean.shape[:-1] + tuple(shape)))
        if dist == "symlog_disc":
            return tools.DiscDist(logits=mean)
        if dist == "normal":
            std = torch.ones_like(mean)
            return tools.ContDist(torchd.independent.Independent(torchd.normal.Normal(mean, std), 1))
        raise NotImplementedError(dist)


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=True, act=torch.tanh, update_bias=-1):
        super().__init__()
        self._size = size
        self._act = act
        self._update_bias = update_bias
        self.layers = nn.Sequential(nn.Linear(inp_size + size, 3 * size, bias=False))
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * size, eps=1e-03))

    def forward(self, inputs, state):
        state = state[0]
        reset, cand, update = torch.split(self.layers(torch.cat([inputs, state], -1)), [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]


class Conv2dSamePad(torch.nn.Conv2d):
    @staticmethod
    def calc_same_pad(i, k, s, d):
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = self.calc_same_pad(iw, self.kernel_size[1], self.stride[1], self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ImgChLayerNorm(nn.Module):
    def __init__(self, ch, eps=1e-03):
        super().__init__()
        self.norm = torch.nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)
