# -*- coding: utf-8 -*-
"""WMP 专用部分环境深度相机。

原版 ByteDance WMP 在 Isaac Gym 中只给 `depth_index` 对应的环境创建相机：

    camera_envs = random_non_tilt_crawl + all_tilt_crawl

这里保持同样思想，但不修改 IsaacLab 源码。传感器仍复用 IsaacLab 的
TiledCamera，只把 `prim_path` 限制到选中的 env 子集，并在 reset 时把全局
env_id 映射到相机局部 id。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.math import convert_camera_frame_orientation_convention


def select_wmp_camera_env_ids(
    num_envs: int,
    camera_num_envs: int | None = 1024,
    seed: int = 42,
    terrain_generator=None,
    terrain_types: torch.Tensor | None = None,
    terrain_cols_by_kind: dict[str, tuple[int, ...]] | None = None,
    force_tilt_crawl: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """选择 WMP 真实相机环境。

    对齐原版公式：
        depth_index = sample(non_tilt_crawl) + all_tilt_crawl
    """
    _, depth_index, _ = select_wmp_depth_indices(
        num_envs=num_envs,
        camera_num_envs=camera_num_envs,
        seed=seed,
        terrain_generator=terrain_generator,
        terrain_types=terrain_types,
        terrain_cols_by_kind=terrain_cols_by_kind,
        force_tilt_crawl=force_tilt_crawl,
        device=device,
    )
    return depth_index


def select_wmp_depth_indices(
    num_envs: int,
    camera_num_envs: int | None = 1024,
    seed: int = 42,
    terrain_generator=None,
    terrain_types: torch.Tensor | None = None,
    terrain_cols_by_kind: dict[str, tuple[int, ...]] | None = None,
    force_tilt_crawl: bool = True,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成原版 WMP 三元 depth index。"""
    device = torch.device(device)
    inverse = torch.full((max(int(num_envs), 0),), -1, device=device, dtype=torch.long)
    if num_envs <= 0:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty, inverse
    if camera_num_envs is None:
        camera_num_envs = 1024
    camera_num_envs = min(int(camera_num_envs), int(num_envs))
    camera_num_envs = max(camera_num_envs, 0)

    forced = _tilt_crawl_env_ids(
        num_envs,
        terrain_generator,
        force_tilt_crawl,
        device,
        terrain_types=terrain_types,
        terrain_cols_by_kind=terrain_cols_by_kind,
    )
    forced = _ordered_unique_valid(forced, num_envs, device)
    if forced.numel() > camera_num_envs:
        print(
            "[WARN] WMP depth_index expands camera_num_envs to include all tilt/crawl envs: "
            f"requested={camera_num_envs}, tilt_crawl={int(forced.numel())}"
        )
        camera_num_envs = int(forced.numel())
    remaining = camera_num_envs - int(forced.numel())

    pool = _depth_index_without_tilt_crawl_pool(
        num_envs,
        device,
        terrain_types=terrain_types,
        terrain_cols_by_kind=terrain_cols_by_kind,
        forced=forced,
    )
    if remaining <= 0:
        sampled = torch.empty(0, device=device, dtype=torch.long)
    elif pool.numel() <= remaining:
        sampled = pool
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        perm = torch.randperm(pool.numel(), generator=generator, device="cpu")[:remaining].to(device)
        sampled = pool[perm]
    depth_index_without_crawl_tilt = torch.sort(_ordered_unique_valid(sampled, num_envs, device)).values
    forced = torch.sort(forced).values
    depth_index = torch.cat((depth_index_without_crawl_tilt, forced), dim=0)
    inverse[depth_index] = torch.arange(depth_index.numel(), device=device, dtype=torch.long)
    return depth_index_without_crawl_tilt, depth_index, inverse


def _ordered_unique_valid(ids: torch.Tensor, num_envs: int, device: torch.device) -> torch.Tensor:
    ids = torch.as_tensor(ids, device=device, dtype=torch.long).flatten()
    if ids.numel() == 0:
        return ids
    keep = torch.zeros(num_envs, device=device, dtype=torch.bool)
    out = []
    for env_id in ids.tolist():
        if 0 <= int(env_id) < num_envs and not bool(keep[int(env_id)].item()):
            keep[int(env_id)] = True
            out.append(int(env_id))
    return torch.tensor(out, device=device, dtype=torch.long)


def _depth_index_without_tilt_crawl_pool(
    num_envs: int,
    device: torch.device,
    terrain_types: torch.Tensor | None,
    terrain_cols_by_kind: dict[str, tuple[int, ...]] | None,
    forced: torch.Tensor,
) -> torch.Tensor:
    cols_by_kind = terrain_cols_by_kind or {}
    forced_cols = list(cols_by_kind.get("tilt", ())) + list(cols_by_kind.get("crawl", ()))
    if forced_cols and terrain_types is not None:
        terrain_types = torch.as_tensor(terrain_types, device=device)
        # 原版从 tilt_start_idx 之前采样，因此 rough_flat 等 crawl 后方地形不进入随机相机池。
        return (terrain_types < min(forced_cols)).nonzero(as_tuple=False).flatten()
    pool_mask = torch.ones(num_envs, device=device, dtype=torch.bool)
    if forced.numel() > 0:
        pool_mask[forced] = False
    return pool_mask.nonzero(as_tuple=False).flatten()


def _tilt_crawl_env_ids(
    num_envs: int,
    terrain_generator,
    force_tilt_crawl: bool,
    device: torch.device,
    terrain_types: torch.Tensor | None = None,
    terrain_cols_by_kind: dict[str, tuple[int, ...]] | None = None,
) -> torch.Tensor:
    if not force_tilt_crawl:
        return torch.empty(0, device=device, dtype=torch.long)

    cols_by_kind = terrain_cols_by_kind or {}
    forced_cols = list(cols_by_kind.get("tilt", ())) + list(cols_by_kind.get("crawl", ()))
    if forced_cols and terrain_types is not None:
        terrain_types = torch.as_tensor(terrain_types, device=device)
        col_tensor = torch.tensor(forced_cols, device=device, dtype=terrain_types.dtype)
        return torch.isin(terrain_types, col_tensor).nonzero(as_tuple=False).flatten()

    if terrain_generator is None or len(terrain_generator.sub_terrains) <= 1:
        return torch.empty(0, device=device, dtype=torch.long)
    keys = list(terrain_generator.sub_terrains.keys())
    forced_kind_ids = [idx for idx, key in enumerate(keys) if key.removeprefix("wmp_") in ("tilt", "crawl")]
    if not forced_kind_ids:
        return torch.empty(0, device=device, dtype=torch.long)

    proportions = torch.tensor([float(terrain_generator.sub_terrains[key].proportion) for key in keys], device=device)
    proportions = proportions / proportions.sum().clamp_min(1.0e-6)
    cumulative = torch.cumsum(proportions, dim=0)
    num_cols = int(terrain_generator.num_cols)
    forced_cols = []
    for col in range(num_cols):
        kind = int(torch.searchsorted(cumulative, torch.tensor(col / num_cols + 0.001, device=device)).item())
        if kind in forced_kind_ids:
            forced_cols.append(col)
    if not forced_cols:
        return torch.empty(0, device=device, dtype=torch.long)

    terrain_types = torch.div(torch.arange(num_envs, device=device), (num_envs / num_cols), rounding_mode="floor").long()
    return torch.isin(terrain_types, torch.tensor(forced_cols, device=device, dtype=torch.long)).nonzero(as_tuple=False).flatten()


@configclass
class WMPPartialTiledCameraCfg(TiledCameraCfg):
    """只绑定 WMP 选中环境的 tiled camera 配置。"""

    full_num_envs: int = 0
    camera_env_ids: tuple[int, ...] = ()
    camera_model_name: str = "RGBD_camera"


class WMPPartialTiledCamera(TiledCamera):
    """WMP 部分相机传感器。

    IsaacLab 的 InteractiveScene 会用全局 env ids reset 传感器，而本传感器只
    有 camera_env_ids 对应的局部实例。因此 reset 时需要映射：
        local_id = inverse_map[global_env_id]
    """

    cfg: WMPPartialTiledCameraCfg

    def __init__(self, cfg: WMPPartialTiledCameraCfg):
        camera_ids = torch.as_tensor(cfg.camera_env_ids, dtype=torch.long)
        if camera_ids.numel() == 0:
            raise ValueError("WMPPartialTiledCamera requires at least one camera env id.")
        ordered_num_envs = int(cfg.full_num_envs or int(camera_ids.max().item()) + 1)
        self.requested_camera_env_ids = _ordered_unique_valid(camera_ids, ordered_num_envs, torch.device("cpu"))
        self.camera_env_ids = self.requested_camera_env_ids.clone()
        full_num_envs = int(cfg.full_num_envs or (int(self.camera_env_ids.max().item()) + 1))
        self.full_num_envs = full_num_envs
        self._set_camera_env_ids(self.camera_env_ids)
        self._wmp_full_prim_path = cfg.prim_path
        self._wmp_subset_prim_path = self._camera_subset_prim_path(cfg.prim_path, self.camera_env_ids)
        spawn_cfg = cfg.spawn
        if spawn_cfg is not None:
            self._spawn_partial_cameras_from_cfg(cfg, spawn_cfg)
        cfg.spawn = None
        super().__init__(cfg)
        self.cfg.spawn = spawn_cfg
        self.cfg.prim_path = self._wmp_subset_prim_path

    def _initialize_impl(self):
        super()._initialize_impl()
        actual_env_ids = []
        for prim in self._view.prims:
            match = re.search(r"/env_(\d+)/", prim.GetPath().pathString)
            if match is not None:
                actual_env_ids.append(int(match.group(1)))
        if len(actual_env_ids) == self._view.count:
            actual = torch.tensor(actual_env_ids, dtype=torch.long)
            if actual.numel() != self.camera_env_ids.numel() or not torch.equal(actual, self.camera_env_ids):
                requested_set = set(self.camera_env_ids.tolist())
                actual_set = set(actual.tolist())
                if actual.numel() == self.camera_env_ids.numel() and requested_set == actual_set:
                    print(
                        "[WARN] WMPPartialTiledCamera image slot order differs from requested depth_index; "
                        "using actual IsaacLab camera view order for depth_index mapping. "
                        f"requested_head={self.camera_env_ids[:16].tolist()}, actual_head={actual[:16].tolist()}"
                    )
                    self._set_camera_env_ids(actual)
                else:
                    missing = sorted(requested_set - actual_set)[:16]
                    extra = sorted(actual_set - requested_set)[:16]
                    raise RuntimeError(
                        "WMPPartialTiledCamera camera env set mismatch. "
                        f"requested_count={self.camera_env_ids.numel()}, actual_count={actual.numel()}, "
                        f"requested_head={self.camera_env_ids[:16].tolist()}, actual_head={actual[:16].tolist()}, "
                        f"missing_head={missing}, extra_head={extra}"
                    )

    def _set_camera_env_ids(self, camera_env_ids: torch.Tensor):
        self.camera_env_ids = torch.as_tensor(camera_env_ids, dtype=torch.long, device="cpu").flatten().clone()
        self.camera_env_mask = torch.zeros(self.full_num_envs, dtype=torch.bool)
        self.camera_env_mask[self.camera_env_ids] = True
        self.camera_env_id_inverse = torch.full((self.full_num_envs,), -1, dtype=torch.long)
        self.camera_env_id_inverse[self.camera_env_ids] = torch.arange(self.camera_env_ids.numel(), dtype=torch.long)

    def _spawn_partial_cameras_from_cfg(self, cfg: WMPPartialTiledCameraCfg, spawn_cfg):
        stage = get_current_stage()
        offset_rots = getattr(cfg, "wmp_camera_offset_rots", None)
        if offset_rots is None:
            offset_rots = torch.tensor(cfg.offset.rot, dtype=torch.float32, device="cpu").repeat(
                self.camera_env_ids.numel(), 1
            )
        else:
            offset_rots = torch.as_tensor(offset_rots, dtype=torch.float32, device="cpu")
        rot = offset_rots
        rot_offset = convert_camera_frame_orientation_convention(
            rot, origin=cfg.offset.convention, target="opengl"
        ).cpu().numpy()
        if spawn_cfg.vertical_aperture is None:
            spawn_cfg.vertical_aperture = spawn_cfg.horizontal_aperture * cfg.height / cfg.width
        for local_id, env_id in enumerate(self.camera_env_ids.cpu().tolist()):
            prim_path = self._wmp_full_prim_path.replace("env_.*", f"env_{int(env_id)}", 1)
            if stage.GetPrimAtPath(prim_path).IsValid():
                continue
            spawn_cfg.func(
                prim_path,
                spawn_cfg,
                translation=cfg.offset.pos,
                orientation=rot_offset[local_id],
            )

    @staticmethod
    def _camera_subset_prim_path(prim_path: str, camera_env_ids: torch.Tensor) -> str:
        env_pattern = "|".join(f"env_{int(env_id)}" for env_id in camera_env_ids.cpu().tolist())
        return prim_path.replace("env_.*", f"({env_pattern})", 1)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
        local_ids = self._global_to_local_env_ids(env_ids)
        super().reset(local_ids)

    def _global_to_local_env_ids(self, env_ids):
        if env_ids is None:
            return None
        if isinstance(env_ids, slice):
            return env_ids
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu")
        valid = (env_ids >= 0) & (env_ids < self.full_num_envs)
        if not torch.any(valid):
            return torch.empty(0, dtype=torch.long, device=self.device)
        local = self.camera_env_id_inverse[env_ids[valid]]
        local = local[local >= 0]
        return local.to(self.device)
