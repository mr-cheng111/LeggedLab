# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
#
# This file contains code adapted from ByteDance WMP
# (`legged_gym/utils/terrain.py`) and IsaacLab terrain utilities, with
# modifications for LeggedLab/IsaacLab APIs.

"""WMP height-field terrain generator with x-edge mask support.

WMP 原版的 `feet_edge` 奖励依赖 `x_edge_mask`：该 mask 来自高度场沿 x
方向的突变检测。公式与原版一致：

    edge[i, j] = abs(h[i + 1, j] - h[i, j]) > slope_threshold

其中 `slope_threshold` 先按 IsaacGym/IsaacLab 约定缩放：

    threshold_grid = slope_threshold * horizontal_scale / vertical_scale

这样比较发生在离散高度场单位上，而不是米制高度上。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
from scipy.ndimage import binary_dilation
from scipy import interpolate

import isaaclab.sim as sim_utils
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from isaaclab.terrains.terrain_importer import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
    from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg


class WMPHeightFieldTerrainGenerator:
    """WMP 原版风格 height-field terrain generator。

    这个类只实现 IsaacLab `TerrainImporter` 需要的最小接口：
    `terrain_mesh`, `terrain_origins`, `flat_patches`, `x_edge_mask`。
    """

    x_edge_mask: torch.Tensor
    height_field_raw: np.ndarray
    wmp_border_cells: int

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.flat_patches = {}
        self.terrain_meshes = []
        self.added_meshes = []
        self.wmp_border_cells = int(round(cfg.border_width / cfg.horizontal_scale))
        self._sub_rows = int(round(cfg.size[0] / cfg.horizontal_scale))
        self._sub_cols = int(round(cfg.size[1] / cfg.horizontal_scale))
        self._tot_rows = cfg.num_rows * self._sub_rows + 2 * self.wmp_border_cells
        self._tot_cols = cfg.num_cols * self._sub_cols + 2 * self.wmp_border_cells
        self.height_field_raw = np.zeros((self._tot_rows, self._tot_cols), dtype=np.int16)
        self.terrain_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)

        self._generate_wmp_terrains()
        vertices, triangles, move_x = _convert_heightfield_to_trimesh_with_x_edges(
            self.height_field_raw,
            cfg.horizontal_scale,
            cfg.vertical_scale,
            cfg.slope_threshold,
        )
        heightfield_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        meshes = [heightfield_mesh, *self.added_meshes]

        transform = np.eye(4)
        transform[:2, -1] = -cfg.border_width - np.asarray(cfg.size) * np.asarray((cfg.num_rows, cfg.num_cols)) * 0.5
        for mesh in meshes:
            mesh.apply_transform(transform)
        self.terrain_mesh = trimesh.util.concatenate(meshes)
        self.terrain_origins += transform[:3, -1]

        # WMP 原版从 `convert_heightfield_to_trimesh()` 的 move_x != 0 得到 mask。
        mask = binary_dilation(move_x, structure=np.ones((3, 1), dtype=bool))
        self.x_edge_mask = torch.tensor(mask, dtype=torch.bool, device=self.device)

    def _terrain_kind(self) -> str:
        name = next(iter(self.cfg.sub_terrains.keys()))
        return name.removeprefix("wmp_")

    def _terrain_kind_for_col(self, col: int) -> str:
        if len(self.cfg.sub_terrains) == 1:
            return self._terrain_kind()
        proportions = np.asarray([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()], dtype=np.float64)
        proportions /= np.sum(proportions)
        sub_index = int(np.min(np.where(col / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0]))
        return list(self.cfg.sub_terrains.keys())[sub_index].removeprefix("wmp_")

    def _generate_wmp_terrains(self):
        for row in range(self.cfg.num_rows):
            for col in range(self.cfg.num_cols):
                kind = self._terrain_kind_for_col(col)
                if self.cfg.curriculum:
                    lower, upper = self.cfg.difficulty_range
                    difficulty = lower + (upper - lower) * ((row + np.random.random()) / self.cfg.num_rows)
                else:
                    difficulty = float(np.random.uniform(*self.cfg.difficulty_range))
                terrain = np.zeros((self._sub_rows, self._sub_cols), dtype=np.int16)
                self._make_wmp_subterrain(terrain, kind, difficulty, row, col)
                self._add_heightfield_to_map(terrain, row, col)

    def _make_wmp_subterrain(self, terrain: np.ndarray, kind: str, difficulty: float, row: int, col: int):
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        gap_size = 1.0 * difficulty
        pit_depth = 0.6 * difficulty
        tilt_width = 0.32 - 0.04 * difficulty
        stair_step_width = 0.30 + np.random.random() * 0.04

        if kind == "wave":
            _wave_terrain(terrain, self.cfg.horizontal_scale, self.cfg.vertical_scale, amplitude=0.1 + 0.2 * difficulty)
            _random_uniform_terrain(terrain, self.cfg.horizontal_scale, self.cfg.vertical_scale)
        elif kind == "slope":
            if np.random.random() < 0.5:
                slope *= -1
            _pyramid_sloped_terrain(terrain, self.cfg.horizontal_scale, self.cfg.vertical_scale, slope, platform_size=3.0)
            _random_uniform_terrain(terrain, self.cfg.horizontal_scale, self.cfg.vertical_scale)
        elif kind in ("stair", "stair_up", "stair_down"):
            if kind in ("stair", "stair_up"):
                step_height *= -1
            _pyramid_stairs_terrain(
                terrain,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                stair_step_width,
                step_height,
                platform_size=3.0,
            )
        elif kind == "gap":
            _gap_terrain(terrain, self.cfg.horizontal_scale, gap_size=gap_size)
            _random_uniform_terrain(terrain, self.cfg.horizontal_scale, self.cfg.vertical_scale)
        elif kind == "climb":
            _climb_terrain(terrain, self.cfg.horizontal_scale, self.cfg.vertical_scale, depth=pit_depth)
            _random_uniform_terrain(terrain, self.cfg.horizontal_scale, self.cfg.vertical_scale)
        elif kind == "discrete":
            _random_uniform_terrain(terrain, self.cfg.horizontal_scale, self.cfg.vertical_scale)
        elif kind == "tilt":
            self._add_wmp_tilt_mesh(row, col, tilt_width)
        elif kind == "crawl":
            self._add_wmp_crawl_mesh(row, col, difficulty)
        elif kind == "rough_flat":
            _random_uniform_terrain(terrain, self.cfg.horizontal_scale, self.cfg.vertical_scale)
        else:
            raise ValueError(f"Unsupported WMP terrain kind: {kind}")

    def _add_heightfield_to_map(self, terrain: np.ndarray, row: int, col: int):
        start_x = self.wmp_border_cells + row * self._sub_rows
        start_y = self.wmp_border_cells + col * self._sub_cols
        self.height_field_raw[start_x : start_x + self._sub_rows, start_y : start_y + self._sub_cols] = terrain

        x1 = int((self.cfg.size[0] / 2.0 - 1.0) / self.cfg.horizontal_scale)
        x2 = int((self.cfg.size[0] / 2.0 + 1.0) / self.cfg.horizontal_scale)
        y1 = int((self.cfg.size[1] / 2.0 - 1.0) / self.cfg.horizontal_scale)
        y2 = int((self.cfg.size[1] / 2.0 + 1.0) / self.cfg.horizontal_scale)
        origin_z = np.max(terrain[x1:x2, y1:y2]) * self.cfg.vertical_scale
        self.terrain_origins[row, col] = [
            self.cfg.border_width + (row + 0.5) * self.cfg.size[0],
            self.cfg.border_width + (col + 0.5) * self.cfg.size[1],
            origin_z,
        ]

    def _env_center(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.cfg.border_width + (row + 0.5) * self.cfg.size[0],
            self.cfg.border_width + (col + 0.5) * self.cfg.size[1],
        )

    def _add_wmp_tilt_mesh(self, row: int, col: int, tilt_width: float):
        env_x, env_y = self._env_center(row, col)
        box_z = 1.0
        box_x = 0.4 + 0.4 * np.random.random()
        side_y = (self.cfg.size[1] - tilt_width) / 2.0
        for x_sign in (1.0, -1.0):
            for y_sign in (-1.0, 1.0):
                center = (
                    env_x + x_sign * (2.0 + box_x / 2.0),
                    env_y + y_sign * (tilt_width / 2.0 + side_y / 2.0),
                    box_z / 2.0,
                )
                self.added_meshes.append(_box_mesh((box_x, side_y, box_z), center))

    def _add_wmp_crawl_mesh(self, row: int, col: int, difficulty: float):
        crawl_height = 0.35 - 0.15 * difficulty
        env_x, env_y = self._env_center(row, col)
        box_x = 0.2 + 0.2 * np.random.random()
        box_z = 1.0
        # WMP 原版两个 bar 的 x 坐标相同；这里保留该行为以严格对齐源码。
        for _ in range(2):
            center = (env_x + 2.0 + box_x / 2.0, env_y, crawl_height + box_z / 2.0)
            self.added_meshes.append(_box_mesh((box_x, self.cfg.size[1], box_z), center))


def _convert_heightfield_to_trimesh_with_x_edges(
    height_field_raw: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    slope_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return IsaacLab heightfield mesh plus WMP's `move_x != 0` edge mask."""
    hf = height_field_raw
    num_rows, num_cols = hf.shape
    move_x = np.zeros((num_rows, num_cols), dtype=np.float32)
    if slope_threshold is not None:
        threshold = slope_threshold * horizontal_scale / vertical_scale
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > threshold
    vertices, triangles = convert_height_field_to_mesh(hf, horizontal_scale, vertical_scale, slope_threshold)
    return vertices, triangles, move_x != 0


def _box_mesh(size: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    return trimesh.creation.box(size, transform=trimesh.transformations.translation_matrix(center))


def _random_uniform_terrain(
    terrain: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    min_height: float = -0.05,
    max_height: float = 0.05,
    step: float = 0.005,
    downsampled_scale: float = 0.2,
):
    """WMP/IsaacGym 风格随机粗糙高度叠加。

    高度先在低分辨率网格采样，再用双线性插值回 terrain 分辨率：
        h(x, y) = interp2d(h_low)(x, y)
    最后除以 `vertical_scale` 转成 int16 高度场单位。
    """
    height_choices = np.arange(min_height, max_height + step, step)
    height_choices = np.rint(height_choices / vertical_scale).astype(np.int16)
    downsample = max(1, int(round(downsampled_scale / horizontal_scale)))
    rows = max(2, terrain.shape[0] // downsample)
    cols = max(2, terrain.shape[1] // downsample)
    low = np.random.choice(height_choices, size=(rows, cols))
    x_low = np.linspace(0, terrain.shape[0] - 1, rows)
    y_low = np.linspace(0, terrain.shape[1] - 1, cols)
    interpolator = interpolate.RegularGridInterpolator((x_low, y_low), low, bounds_error=False, fill_value=None)
    x = np.arange(terrain.shape[0])
    y = np.arange(terrain.shape[1])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    noise = interpolator(np.stack((xx.reshape(-1), yy.reshape(-1)), axis=-1)).reshape(terrain.shape)
    terrain += np.rint(noise).astype(np.int16)


def _wave_terrain(
    terrain: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    num_waves: int = 5,
    amplitude: float = 0.2,
):
    """IsaacGym `wave_terrain` 等价实现。"""
    x = np.arange(terrain.shape[0], dtype=np.float32) * horizontal_scale
    y = np.arange(terrain.shape[1], dtype=np.float32) * horizontal_scale
    xx, yy = np.meshgrid(x, y, indexing="ij")
    length = terrain.shape[0] * horizontal_scale
    width = terrain.shape[1] * horizontal_scale
    heights = amplitude * (np.sin(2 * np.pi * num_waves * xx / length) + np.sin(2 * np.pi * num_waves * yy / width))
    terrain += np.rint(heights / vertical_scale).astype(np.int16)


def _pyramid_sloped_terrain(
    terrain: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    slope: float,
    platform_size: float = 1.0,
):
    """IsaacGym `pyramid_sloped_terrain` 等价实现。"""
    x = (np.arange(terrain.shape[0]) - terrain.shape[0] / 2) * horizontal_scale
    y = (np.arange(terrain.shape[1]) - terrain.shape[1] / 2) * horizontal_scale
    xx, yy = np.meshgrid(x, y, indexing="ij")
    max_dist = np.maximum(np.abs(xx), np.abs(yy))
    platform_half = platform_size / 2.0
    heights = slope * np.maximum(max_dist - platform_half, 0.0)
    heights -= heights.min()
    if slope < 0:
        heights = heights.max() - heights
    terrain += np.rint(heights / vertical_scale).astype(np.int16)


def _pyramid_stairs_terrain(
    terrain: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    step_width: float,
    step_height: float,
    platform_size: float = 1.0,
):
    """IsaacGym `pyramid_stairs_terrain` 等价实现。"""
    step_width_cells = max(1, int(round(step_width / horizontal_scale)))
    step_height_cells = int(round(step_height / vertical_scale))
    platform_cells = int(round(platform_size / horizontal_scale))
    center_x = terrain.shape[0] // 2
    center_y = terrain.shape[1] // 2
    max_steps = int((min(terrain.shape) - platform_cells) // (2 * step_width_cells))
    for step in range(max_steps):
        x1 = step * step_width_cells
        x2 = terrain.shape[0] - step * step_width_cells
        y1 = step * step_width_cells
        y2 = terrain.shape[1] - step * step_width_cells
        terrain[x1:x2, y1:y2] = step_height_cells * (step + 1)
    px1 = center_x - platform_cells // 2
    px2 = center_x + platform_cells // 2
    py1 = center_y - platform_cells // 2
    py2 = center_y + platform_cells // 2
    terrain[px1:px2, py1:py2] = step_height_cells * max_steps


def _gap_terrain(terrain: np.ndarray, horizontal_scale: float, gap_size: float, platform_size: float = 4.0):
    """WMP 原版 `gap_terrain()`。"""
    del platform_size
    gap_cells = int(gap_size / horizontal_scale)
    center_x = terrain.shape[0] // 2
    center_y = terrain.shape[1] // 2
    x1 = int(center_x - 1 / horizontal_scale)
    x2 = int(center_x + 2 / horizontal_scale)
    x3 = x1 - gap_cells
    x4 = x2 + gap_cells
    width = 1.0 + 1.0 * np.random.random()
    half_width = width / 2.0
    y1 = int(center_y - half_width / horizontal_scale)
    y2 = int(center_y + half_width / horizontal_scale)
    x5 = gap_cells

    terrain[:, :] = -1000
    terrain[x5:x3, y1:y2] = 0
    terrain[x1:x2, y1:y2] = 0
    terrain[x4:, y1:y2] = 0


def _climb_terrain(terrain: np.ndarray, horizontal_scale: float, vertical_scale: float, depth: float):
    """WMP 原版 `climb_terrain()`。"""
    depth_cells = int(depth / vertical_scale)
    x1 = int(1 / horizontal_scale)
    length = 1.0 + 0.2 * np.random.random()
    x2 = int((1 + length) / horizontal_scale)
    x3 = int(6 / horizontal_scale)
    length = 1.0 + 0.2 * np.random.random()
    x4 = int((6 + length) / horizontal_scale)
    terrain[x1:x2, :] = depth_cells
    terrain[x3:x4, :] = depth_cells


class WMPTerrainImporter(TerrainImporter):
    """TerrainImporter that preserves WMP generator metadata on `scene.terrain`."""

    x_edge_mask: torch.Tensor | None
    wmp_edge_query_offset: tuple[float, float]
    wmp_horizontal_scale: float
    gap_start_col: int
    climb_end_col: int

    def __init__(self, cfg: TerrainImporterCfg):
        cfg.validate()
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore
        self.terrain_prim_paths = []
        self.terrain_origins = None
        self.env_origins = None
        self._terrain_flat_patches = {}
        self.x_edge_mask = None
        self.wmp_edge_query_offset = (0.0, 0.0)
        self.wmp_horizontal_scale = 1.0
        self.gap_start_col = 0
        self.climb_end_col = 0

        if self.cfg.terrain_type == "generator":
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            generator = cfg.terrain_generator.class_type(cfg=cfg.terrain_generator, device=self.device)
            self.import_mesh("terrain", generator.terrain_mesh)
            if self.cfg.use_terrain_origins:
                self.configure_env_origins(generator.terrain_origins)
            else:
                self.configure_env_origins()
            self._terrain_flat_patches = generator.flat_patches
            if hasattr(generator, "x_edge_mask"):
                self.x_edge_mask = generator.x_edge_mask
                self.wmp_edge_query_offset = (cfg.terrain_generator.border_width, cfg.terrain_generator.border_width)
                self.wmp_horizontal_scale = cfg.terrain_generator.horizontal_scale
                if len(cfg.terrain_generator.sub_terrains) > 1:
                    keys = list(cfg.terrain_generator.sub_terrains.keys())
                    proportions = np.asarray(
                        [cfg.terrain_generator.sub_terrains[key].proportion for key in keys], dtype=np.float64
                    )
                    proportions /= np.sum(proportions)
                    col_kinds = []
                    for col in range(cfg.terrain_generator.num_cols):
                        idx = int(np.min(np.where(col / cfg.terrain_generator.num_cols + 0.001 < np.cumsum(proportions))[0]))
                        col_kinds.append(keys[idx].removeprefix("wmp_"))
                    gap_cols = [i for i, kind in enumerate(col_kinds) if kind == "gap"]
                    climb_cols = [i for i, kind in enumerate(col_kinds) if kind == "climb"]
                    if gap_cols:
                        self.gap_start_col = min(gap_cols)
                    if climb_cols:
                        self.climb_end_col = max(climb_cols) + 1
        elif self.cfg.terrain_type == "usd":
            if self.cfg.usd_path is None:
                raise ValueError("Input terrain type is 'usd' but no value provided for 'usd_path'.")
            self.import_usd("terrain", self.cfg.usd_path)
            self.configure_env_origins()
        elif self.cfg.terrain_type == "plane":
            self.import_ground_plane("terrain")
            self.configure_env_origins()
        else:
            raise ValueError(f"Terrain type '{self.cfg.terrain_type}' not available.")

        self.set_debug_vis(self.cfg.debug_vis)
