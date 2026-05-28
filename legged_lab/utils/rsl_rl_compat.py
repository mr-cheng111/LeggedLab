# -*- coding: utf-8 -*-
"""rsl_rl 配置兼容层。

目的：
1. 兼容旧版配置（policy/algorithm）与新版 rsl_rl（actor/critic/algorithm）。
2. 尽量不改变现有任务配置写法，避免一次性大改。
"""

from __future__ import annotations

import copy
from importlib.metadata import PackageNotFoundError, version


def _parse_major(ver: str) -> int:
    token = ver.split(".", 1)[0].strip()
    return int(token) if token.isdigit() else 0


def is_rsl_rl_v5_plus() -> bool:
    """判断当前环境中的 rsl-rl-lib 是否为 5.x 及以上。"""
    for pkg_name in ("rsl-rl-lib", "rsl_rl"):
        try:
            return _parse_major(version(pkg_name)) >= 5
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return False


def adapt_legacy_cfg_for_rsl_rl_v5(cfg_dict: dict) -> dict:
    """把旧格式配置转换为 rsl_rl v5 可识别格式。

    旧格式核心字段：
    - policy: {class_name, actor_hidden_dims, critic_hidden_dims, ...}
    - obs_groups: {"policy": [...], "critic": [...]}

    新格式核心字段：
    - actor: {class_name, hidden_dims, ...}
    - critic: {class_name, hidden_dims, ...}
    - obs_groups: {"actor": [...], "critic": [...]}
    """
    if not is_rsl_rl_v5_plus():
        return cfg_dict

    out = copy.deepcopy(cfg_dict)

    def _is_complete_model_cfg(model_cfg: object, *, needs_distribution: bool) -> bool:
        if not isinstance(model_cfg, dict):
            return False
        if not model_cfg.get("class_name"):
            return False
        if not model_cfg.get("hidden_dims"):
            return False
        if not model_cfg.get("activation"):
            return False
        if needs_distribution and not isinstance(model_cfg.get("distribution_cfg"), dict):
            return False
        return True

    # 已是完整新格式则直接返回；不完整时继续从 legacy policy 重建。
    if _is_complete_model_cfg(out.get("actor"), needs_distribution=True) and _is_complete_model_cfg(
        out.get("critic"), needs_distribution=False
    ):
        obs_groups = out.get("obs_groups") or {}
        if "actor" not in obs_groups and "policy" in obs_groups:
            obs_groups["actor"] = obs_groups.pop("policy")
        if "critic" not in obs_groups:
            obs_groups["critic"] = ["critic"]
        out["obs_groups"] = obs_groups
        return out

    policy = out.get("policy")
    if not isinstance(policy, dict):
        return out

    policy_class = str(policy.get("class_name", "ActorCritic"))
    is_recurrent = "Recurrent" in policy_class
    requested_class = str(policy.get("class_name", "ActorCritic"))
    model_class_name = requested_class if ":" in requested_class or "." in requested_class else ("RNNModel" if is_recurrent else "MLPModel")
    obs_norm = bool(out.get("empirical_normalization", False))

    actor_cfg = {
        "class_name": model_class_name,
        "hidden_dims": policy.get("actor_hidden_dims", [512, 256, 128]),
        "activation": policy.get("activation", "elu"),
        "obs_normalization": obs_norm,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": policy.get("init_noise_std", 1.0),
            "std_type": policy.get("noise_std_type", "scalar"),
        },
    }
    critic_cfg = {
        "class_name": model_class_name,
        "hidden_dims": policy.get("critic_hidden_dims", [512, 256, 128]),
        "activation": policy.get("activation", "elu"),
        "obs_normalization": obs_norm,
    }

    if "WMPMLPModel" in model_class_name:
        wmp_cfg = out.get("wmp") if isinstance(out.get("wmp"), dict) else {}
        feature_type = wmp_cfg.get("feature_type", "deter")
        wmp_feature_dim = 1536 if feature_type == "full" else 512
        wmp_model_cfg = {
            "wmp_key": wmp_cfg.get("obs_key", "wmp"),
            "wmp_feature_dim": wmp_cfg.get("feature_dim", wmp_feature_dim),
            "wmp_latent_dim": wmp_cfg.get("wm_latent_dim", 32),
            "wmp_encoder_hidden_dims": wmp_cfg.get("wm_encoder_hidden_dims", [64, 64]),
            "use_history_encoder": wmp_cfg.get("use_history_encoder", False),
            "history_steps": wmp_cfg.get("history_steps", 5),
            "history_dim_per_step": wmp_cfg.get("history_dim_per_step"),
            "history_encoder_hidden_dims": wmp_cfg.get("history_encoder_hidden_dims", [256, 128]),
            "history_latent_dim": wmp_cfg.get("history_latent_dim", 35),
            "command_dim": wmp_cfg.get("command_dim", 3),
            "command_start": wmp_cfg.get("command_start", 6),
            "height_scan_dim": wmp_cfg.get("height_scan_dim", 0),
            "history_excludes_trailing_height": wmp_cfg.get("history_excludes_trailing_height", True),
            "critic_uses_history_encoder": wmp_cfg.get("critic_uses_history_encoder", False),
        }
        actor_cfg.update(wmp_model_cfg)
        critic_cfg.update(wmp_model_cfg)

    if is_recurrent:
        # rsl_rl v5 的 RNNModel 参数名是 rnn_hidden_dim。
        rnn_hidden_dim = policy.get("rnn_hidden_dim", policy.get("rnn_hidden_size", 256))
        rnn_type = policy.get("rnn_type", "lstm")
        rnn_num_layers = policy.get("rnn_num_layers", 1)
        actor_cfg.update(
            {
                "rnn_type": rnn_type,
                "rnn_hidden_dim": rnn_hidden_dim,
                "rnn_num_layers": rnn_num_layers,
            }
        )
        critic_cfg.update(
            {
                "rnn_type": rnn_type,
                "rnn_hidden_dim": rnn_hidden_dim,
                "rnn_num_layers": rnn_num_layers,
            }
        )

    out["actor"] = actor_cfg
    out["critic"] = critic_cfg

    obs_groups = out.get("obs_groups") or {}
    if "actor" not in obs_groups:
        if "policy" in obs_groups:
            obs_groups["actor"] = obs_groups.pop("policy")
        else:
            # 默认把 actor 连接到环境的 policy 观测键。
            obs_groups["actor"] = ["policy"]
    if "critic" not in obs_groups:
        obs_groups["critic"] = ["critic"]
    out["obs_groups"] = obs_groups

    return out
