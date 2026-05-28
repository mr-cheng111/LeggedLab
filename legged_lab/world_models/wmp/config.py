# -*- coding: utf-8 -*-
"""WMP RSSM 默认配置。

默认值对齐 ByteDance WMP 的 dreamer/configs.yaml：
dyn_deter=512, dyn_stoch=32, dyn_discrete=32, dyn_hidden=512。
"""

from dataclasses import dataclass, field


@dataclass
class DepthPredictorConfig:
    forward_heightmap_dim: int = 525
    prop_dim: int = 33
    depth_image_dims: tuple[int, int] = (64, 64)
    encoder_hidden_dims: tuple[int, ...] = (256, 128)
    depth: int = 32
    act: str = "ELU"
    norm: bool = True
    kernel_size: int = 4
    minres: int = 4
    outscale: float = 1.0
    cnn_sigmoid: bool = False
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    training_interval: int = 10
    training_iters: int = 1000
    batch_size: int = 1024
    loss_scale: float = 100.0


@dataclass
class WMPWorldModelConfig:
    device: str = "cuda:0"
    precision: int = 32

    num_actions: int = 12
    env_num_actions: int | None = None
    action_dim: int | None = None
    update_interval: int = 5
    prop_dim: int = 33
    forward_heightmap_dim: int = 525
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
    train_interval: int = 1
    train_start_steps: int = 10000
    batch_size: int = 16
    batch_length: int = 64
    model_lr: float = 1.0e-4
    opt_eps: float = 1.0e-8
    grad_clip: float = 1000.0
    opt: str = "adam"
    feature_type: str = "deter"
    replay_capacity_episodes: int = 50000
    replay_device: str = "cpu"
    use_depth_predictor: bool = True
    camera_sample_all_envs: bool = False
    camera_num_envs: int | None = 1024
    camera_env_seed: int = 42
    camera_force_tilt_crawl: bool = True
    camera_env_ids: tuple[int, ...] | None = None
    depth_predictor: DepthPredictorConfig = field(default_factory=DepthPredictorConfig)


def make_default_wmp_config(device: str = "cuda:0", num_actions: int = 12) -> WMPWorldModelConfig:
    cfg = WMPWorldModelConfig()
    cfg.device = device
    cfg.num_actions = num_actions
    cfg.env_num_actions = num_actions
    cfg.action_dim = num_actions
    return cfg
