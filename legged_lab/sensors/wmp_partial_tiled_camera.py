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
    force_tilt_crawl: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """选择 WMP 真实相机环境。

    对齐原版公式：
        depth_index = sample(non_tilt_crawl) U range(tilt_start_idx, crawl_end_idx)
    在 IsaacLab 中 terrain type 由 env index 按列分配，因此先由 terrain
    proportions 还原每列的地形类型，再强制加入 tilt/crawl 列对应的 env。
    """
    device = torch.device(device)
    if num_envs <= 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if camera_num_envs is None:
        camera_num_envs = 1024
    camera_num_envs = min(int(camera_num_envs), int(num_envs))
    if camera_num_envs <= 0:
        return torch.empty(0, device=device, dtype=torch.long)

    forced = _tilt_crawl_env_ids(num_envs, terrain_generator, force_tilt_crawl, device)
    remaining = camera_num_envs - int(forced.numel())
    if remaining <= 0:
        return forced[:camera_num_envs].unique(sorted=True)

    pool_mask = torch.ones(num_envs, device=device, dtype=torch.bool)
    if forced.numel() > 0:
        pool_mask[forced] = False
    pool = pool_mask.nonzero(as_tuple=False).flatten()
    if pool.numel() <= remaining:
        sampled = pool
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        perm = torch.randperm(pool.numel(), generator=generator, device="cpu")[:remaining].to(device)
        sampled = pool[perm]
    return torch.cat((forced, sampled), dim=0).unique(sorted=True)


def _tilt_crawl_env_ids(num_envs: int, terrain_generator, force_tilt_crawl: bool, device: torch.device) -> torch.Tensor:
    if not force_tilt_crawl or terrain_generator is None or len(terrain_generator.sub_terrains) <= 1:
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
    camera_model_name: str = "gemini2"


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
        self.camera_env_ids = camera_ids.unique(sorted=True)
        full_num_envs = int(cfg.full_num_envs or (int(self.camera_env_ids.max().item()) + 1))
        self.full_num_envs = full_num_envs
        self.camera_env_mask = torch.zeros(full_num_envs, dtype=torch.bool)
        self.camera_env_mask[self.camera_env_ids] = True
        self.camera_env_id_inverse = torch.full((full_num_envs,), -1, dtype=torch.long)
        self.camera_env_id_inverse[self.camera_env_ids] = torch.arange(self.camera_env_ids.numel(), dtype=torch.long)
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
            self.camera_env_ids = torch.tensor(actual_env_ids, dtype=torch.long)
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
