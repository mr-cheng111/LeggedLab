# -*- coding: utf-8 -*-
"""WMP WorldModel 前向封装。

参考 ByteDance WMP 的 dreamer/models.py，当前实现 encode/RSSM/decoder
smoke test 所需路径，不包含训练优化器和 replay dataset。
"""

from torch import nn

from . import networks


class WorldModel(nn.Module):
    def __init__(self, config, obs_shape, use_camera: bool = True):
        super().__init__()
        self._config = config
        self.device = config.device
        self.encoder = networks.MultiEncoder(obs_shape, **config.encoder, use_camera=use_camera)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter if config.dyn_discrete else config.dyn_stoch + config.dyn_deter
        self.feature_dim = feat_size
        self.deter_dim = config.dyn_deter
        self.heads = nn.ModuleDict()
        self.heads["decoder"] = networks.MultiDecoder(feat_size, obs_shape, **config.decoder, use_camera=use_camera)
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
        )

    def decode(self, features):
        return self.heads["decoder"](features)

