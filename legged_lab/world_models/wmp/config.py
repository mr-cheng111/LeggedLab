# -*- coding: utf-8 -*-
"""WMP RSSM 默认配置。

默认值对齐 ByteDance WMP 的 dreamer/configs.yaml：
dyn_deter=512, dyn_stoch=32, dyn_discrete=32, dyn_hidden=512。
"""

from dataclasses import dataclass, field


@dataclass
class WMPWorldModelConfig:
    device: str = "cuda:0"
    precision: int = 32

    num_actions: int = 12
    dyn_hidden: int = 512
    dyn_deter: int = 512
    dyn_stoch: int = 32
    dyn_discrete: int = 32
    dyn_rec_depth: int = 1
    dyn_mean_act: str = "none"
    dyn_std_act: str = "sigmoid2"
    dyn_min_std: float = 0.1
    unimix_ratio: float = 0.01
    initial: str = "learned"
    act: str = "SiLU"
    norm: bool = True
    units: int = 512

    encoder: dict = field(
        default_factory=lambda: {
            "mlp_keys": ".*",
            "cnn_keys": "image",
            "act": "SiLU",
            "norm": True,
            "cnn_depth": 32,
            "kernel_size": 4,
            "minres": 4,
            "mlp_layers": 5,
            "mlp_units": 1024,
            "symlog_inputs": True,
        }
    )
    decoder: dict = field(
        default_factory=lambda: {
            "mlp_keys": ".*",
            "cnn_keys": "image",
            "act": "SiLU",
            "norm": True,
            "cnn_depth": 32,
            "kernel_size": 4,
            "minres": 4,
            "mlp_layers": 5,
            "mlp_units": 1024,
            "cnn_sigmoid": False,
            "image_dist": "mse",
            "vector_dist": "symlog_mse",
            "outscale": 1.0,
        }
    )
    reward_head: dict = field(
        default_factory=lambda: {"layers": 2, "dist": "symlog_disc", "loss_scale": 0.0, "outscale": 0.0}
    )
    grad_heads: tuple[str, ...] = ("decoder", "reward")
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    kl_free: float = 1.0
    weight_decay: float = 0.0
    train_steps_per_iter: int = 10
    train_start_steps: int = 10000
    batch_size: int = 16
    batch_length: int = 64
    model_lr: float = 1.0e-4
    opt_eps: float = 1.0e-8
    grad_clip: float = 1000.0
    opt: str = "adam"
    feature_type: str = "deter"


def make_default_wmp_config(device: str = "cuda:0", num_actions: int = 12) -> WMPWorldModelConfig:
    cfg = WMPWorldModelConfig()
    cfg.device = device
    cfg.num_actions = num_actions
    return cfg
