# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp import events as isaac_events
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _env_ids_cpu(env: ManagerBasedEnv, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.scene.num_envs, device="cpu")
    if isinstance(env_ids, torch.Tensor):
        return env_ids.detach().cpu()
    return torch.tensor(env_ids, dtype=torch.long, device="cpu")


def _env_ids_device(env: ManagerBasedEnv, env_ids_cpu: torch.Tensor) -> torch.Tensor:
    return env_ids_cpu.to(device=env.device, dtype=torch.long)


def _sample_distribution(
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    shape: tuple[int, ...],
    device: str | torch.device,
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    if distribution == "uniform":
        return math_utils.sample_uniform(*distribution_parameters, shape, device=device)
    if distribution == "log_uniform":
        return math_utils.sample_log_uniform(*distribution_parameters, shape, device=device)
    if distribution == "gaussian":
        return math_utils.sample_gaussian(*distribution_parameters, shape, device=device)
    raise NotImplementedError(f"Unknown randomization distribution: '{distribution}'.")


def _resolve_body_ids(asset: RigidObject | Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    if asset_cfg.body_ids == slice(None):
        return torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    return torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")


class wmp_recording_randomize_rigid_body_material(isaac_events.randomize_rigid_body_material):
    """Randomize material properties and record WMP critic privileged values.

    原版 WMP 在 env 创建时为每个 env 采样一个 friction/restitution 标量，并把同一个值
    写到该 env 的 rigid shapes；critic 也读取同一个标量。这里保持同源采样：
    `wmp_priv_friction = sampled_static_friction`,
    `wmp_priv_restitution = sampled_restitution`。
    """

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        env_ids_cpu = _env_ids_cpu(env, env_ids)
        env_ids_device = _env_ids_device(env, env_ids_cpu)
        total_num_shapes = self.asset.root_physx_view.max_shapes

        static_friction = math_utils.sample_uniform(
            static_friction_range[0], static_friction_range[1], (len(env_ids_cpu), 1, 1), device="cpu"
        )
        dynamic_friction = math_utils.sample_uniform(
            dynamic_friction_range[0], dynamic_friction_range[1], (len(env_ids_cpu), 1, 1), device="cpu"
        )
        if make_consistent:
            dynamic_friction = torch.minimum(static_friction, dynamic_friction)
        restitution = math_utils.sample_uniform(
            restitution_range[0], restitution_range[1], (len(env_ids_cpu), 1, 1), device="cpu"
        )
        material_samples = torch.cat([static_friction, dynamic_friction, restitution], dim=-1).repeat(
            1, total_num_shapes, 1
        )

        materials = self.asset.root_physx_view.get_material_properties()
        if self.num_shapes_per_body is not None:
            for body_id in self.asset_cfg.body_ids:
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                materials[env_ids_cpu, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            materials[env_ids_cpu] = material_samples
        self.asset.root_physx_view.set_material_properties(materials, env_ids_cpu)

        if hasattr(env, "wmp_priv_friction"):
            env.wmp_priv_friction[env_ids_device] = static_friction[:, 0].to(env.device)
        if hasattr(env, "wmp_priv_restitution"):
            env.wmp_priv_restitution[env_ids_device] = restitution[:, 0].to(env.device)


class wmp_recording_randomize_rigid_body_mass(ManagerTermBase):
    """Randomize body mass and record the additive base-mass term used by WMP critic."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        operation = cfg.params["operation"]
        if operation == "scale":
            if "mass_distribution_params" in cfg.params:
                isaac_events._validate_scale_range(
                    cfg.params["mass_distribution_params"], "mass_distribution_params", allow_zero=False
                )
        elif operation not in ("abs", "add"):
            raise ValueError(f"WMP mass randomization does not support operation: '{operation}'.")
        if cfg.params.get("min_mass") is not None and cfg.params.get("min_mass") < 1.0e-6:
            raise ValueError("WMP mass randomization requires 'min_mass' >= 1e-6.")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        mass_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        recompute_inertia: bool = True,
        min_mass: float = 1.0e-6,
    ):
        env_ids_cpu = _env_ids_cpu(env, env_ids)
        env_ids_device = _env_ids_device(env, env_ids_cpu)
        body_ids = _resolve_body_ids(self.asset, self.asset_cfg)

        masses = self.asset.root_physx_view.get_masses()
        default_mass = self.asset.data.default_mass[env_ids_cpu[:, None], body_ids].clone()
        masses[env_ids_cpu[:, None], body_ids] = default_mass

        samples = _sample_distribution(mass_distribution_params, (len(env_ids_cpu), 1), masses.device, distribution)
        if operation == "add":
            randomized_mass = default_mass + samples
            recorded_added_mass = samples
        elif operation == "scale":
            randomized_mass = default_mass * samples
            recorded_added_mass = randomized_mass[:, :1] - default_mass[:, :1]
        elif operation == "abs":
            randomized_mass = samples.expand_as(default_mass)
            recorded_added_mass = randomized_mass[:, :1] - default_mass[:, :1]
        else:
            raise NotImplementedError(f"Unknown mass randomization operation: '{operation}'.")

        masses[env_ids_cpu[:, None], body_ids] = torch.clamp(randomized_mass, min=min_mass)
        self.asset.root_physx_view.set_masses(masses, env_ids_cpu)

        if recompute_inertia:
            ratios = masses[env_ids_cpu[:, None], body_ids] / self.asset.data.default_mass[env_ids_cpu[:, None], body_ids]
            inertias = self.asset.root_physx_view.get_inertias()
            if isinstance(self.asset, Articulation):
                inertias[env_ids_cpu[:, None], body_ids] = (
                    self.asset.data.default_inertia[env_ids_cpu[:, None], body_ids] * ratios[..., None]
                )
            else:
                inertias[env_ids_cpu] = self.asset.data.default_inertia[env_ids_cpu] * ratios
            self.asset.root_physx_view.set_inertias(inertias, env_ids_cpu)

        if hasattr(env, "wmp_priv_added_mass"):
            env.wmp_priv_added_mass[env_ids_device] = recorded_added_mass.to(env.device)


def wmp_recording_randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize CoM offset and record the sampled offset used by WMP critic.

    记录量为 `new_com - default_com`，与原版 critic 中 `randomized_com_pos * obs_scales.com_pos`
    的语义一致。
    """

    asset: Articulation = env.scene[asset_cfg.name]
    env_ids_cpu = _env_ids_cpu(env, env_ids)
    env_ids_device = _env_ids_device(env, env_ids_cpu)
    body_ids = _resolve_body_ids(asset, asset_cfg)

    ranges = torch.tensor([com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]], device="cpu")
    offsets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids_cpu), 3), device="cpu")

    coms = asset.root_physx_view.get_coms().clone()
    if hasattr(env, "wmp_priv_default_coms"):
        default_coms = env.wmp_priv_default_coms.detach().cpu()
        coms[env_ids_cpu[:, None], body_ids, :3] = default_coms[env_ids_cpu[:, None], body_ids, :3]
    coms[env_ids_cpu[:, None], body_ids, :3] += offsets.unsqueeze(1)
    asset.root_physx_view.set_coms(coms, env_ids_cpu)

    if hasattr(env, "wmp_priv_com_pos"):
        env.wmp_priv_com_pos[env_ids_device] = offsets.to(env.device)


class wmp_recording_randomize_actuator_gains(ManagerTermBase):
    """Randomize actuator gains and record `(randomized_gain / default_gain - 1)`.

    原版 WMP critic 使用:
        p_gain_scale = randomized_p_gains / p_gains - 1
        d_gain_scale = randomized_d_gains / d_gains - 1
    BaseEnv 后续会再乘 `obs_scales.pd_gains`。
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        operation = cfg.params["operation"]
        if operation == "scale":
            if "stiffness_distribution_params" in cfg.params:
                isaac_events._validate_scale_range(
                    cfg.params["stiffness_distribution_params"], "stiffness_distribution_params", allow_zero=False
                )
            if "damping_distribution_params" in cfg.params:
                isaac_events._validate_scale_range(
                    cfg.params["damping_distribution_params"], "damping_distribution_params"
                )
        elif operation not in ("abs", "add"):
            raise ValueError(f"WMP actuator gain randomization does not support operation: '{operation}'.")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        if env_ids is None or isinstance(env_ids, slice):
            env_ids_asset = torch.arange(env.scene.num_envs, device=self.asset.device)
        else:
            env_ids_asset = env_ids.to(self.asset.device)
        env_ids_device = env_ids_asset.to(env.device)

        for actuator in self.asset.actuators.values():
            if isinstance(self.asset_cfg.joint_ids, slice):
                actuator_indices = slice(None)
                if isinstance(actuator.joint_indices, slice):
                    global_indices = slice(None)
                elif isinstance(actuator.joint_indices, torch.Tensor):
                    global_indices = actuator.joint_indices.to(self.asset.device)
                else:
                    raise TypeError("Actuator joint indices must be a slice or a torch.Tensor.")
            elif isinstance(actuator.joint_indices, slice):
                global_indices = actuator_indices = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
            else:
                actuator_joint_indices = actuator.joint_indices
                asset_joint_ids = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
                actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
                if len(actuator_indices) == 0:
                    continue
                global_indices = actuator_joint_indices[actuator_indices]

            if stiffness_distribution_params is not None:
                stiffness = actuator.stiffness[env_ids_asset].clone()
                default_stiffness = self.asset.data.default_joint_stiffness[env_ids_asset][:, global_indices].clone()
                stiffness[:, actuator_indices] = self._randomize_gain(
                    default_stiffness, stiffness_distribution_params, operation, distribution
                )
                actuator.stiffness[env_ids_asset] = stiffness
                if hasattr(env, "wmp_priv_p_gain_scale"):
                    self._record_gain_scale(
                        env.wmp_priv_p_gain_scale,
                        env_ids_device,
                        global_indices,
                        default_stiffness,
                        stiffness[:, actuator_indices],
                    )
                if isinstance(actuator, ImplicitActuator):
                    self.asset.write_joint_stiffness_to_sim(
                        stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids_asset
                    )

            if damping_distribution_params is not None:
                damping = actuator.damping[env_ids_asset].clone()
                default_damping = self.asset.data.default_joint_damping[env_ids_asset][:, global_indices].clone()
                damping[:, actuator_indices] = self._randomize_gain(
                    default_damping, damping_distribution_params, operation, distribution
                )
                actuator.damping[env_ids_asset] = damping
                if hasattr(env, "wmp_priv_d_gain_scale"):
                    self._record_gain_scale(
                        env.wmp_priv_d_gain_scale,
                        env_ids_device,
                        global_indices,
                        default_damping,
                        damping[:, actuator_indices],
                    )
                if isinstance(actuator, ImplicitActuator):
                    self.asset.write_joint_damping_to_sim(
                        damping, joint_ids=actuator.joint_indices, env_ids=env_ids_asset
                    )

    def _randomize_gain(
        self,
        default_gain: torch.Tensor,
        distribution_parameters: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"],
    ) -> torch.Tensor:
        samples = _sample_distribution(distribution_parameters, default_gain.shape, default_gain.device, distribution)
        if operation == "add":
            return default_gain + samples
        if operation == "scale":
            return default_gain * samples
        if operation == "abs":
            return samples
        raise NotImplementedError(f"Unknown actuator gain randomization operation: '{operation}'.")

    def _record_gain_scale(
        self,
        target: torch.Tensor,
        env_ids: torch.Tensor,
        global_indices: torch.Tensor | slice,
        default_gain: torch.Tensor,
        randomized_gain: torch.Tensor,
    ):
        gain_scale = randomized_gain / torch.clamp(default_gain, min=1.0e-8) - 1.0
        if isinstance(global_indices, slice):
            target[env_ids] = gain_scale.to(target.device)
        else:
            target[env_ids[:, None], global_indices.to(target.device)] = gain_scale.to(target.device)


__all__ = [
    "wmp_recording_randomize_rigid_body_material",
    "wmp_recording_randomize_rigid_body_mass",
    "wmp_recording_randomize_rigid_body_com",
    "wmp_recording_randomize_actuator_gains",
]
