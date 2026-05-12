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


"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


def make_single_terrain_cfg(name: str, sub_terrain, difficulty: float = 0.7) -> TerrainGeneratorCfg:
    """构造单一场景地形。

    difficulty 会传给 IsaacLab 地形生成器，地形参数通常按线性插值计算：
        value = min + difficulty * (max - min)
    这里固定 difficulty，便于预览时每个 tile 呈现一致难度。
    """
    return TerrainGeneratorCfg(
        curriculum=False,
        difficulty_range=(difficulty, difficulty),
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={name: sub_terrain},
    )

GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        )
    },
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs_28": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_32": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.32,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_34": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.15, amplitude_range=(0.0, 0.2), num_waves=5.0),
        "high_platform": terrain_gen.MeshPitTerrainCfg(
            proportion=0.15, pit_depth_range=(0.0, 0.3), platform_width=2.0, double_pit=True
        ),
        # "star": terrain_gen.MeshStarTerrainCfg(
        #     proportion=0.15, num_bars=6, bar_width_range=(0.05, 0.05), bar_height_range=(0.0, 0.25), platform_width=1.0
        # ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.15, gap_width_range=(0.1, 0.4), platform_width=2.0
        # )
    },
)

WMP_SLOPE_TERRAINS_CFG = make_single_terrain_cfg(
    "wmp_slope",
    terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=1.0,
        slope_range=(0.08, 0.45),
        platform_width=2.0,
        border_width=0.25,
    ),
)

WMP_STAIR_TERRAINS_CFG = make_single_terrain_cfg(
    "wmp_stair",
    terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        proportion=1.0,
        step_height_range=(0.05, 0.23),
        step_width=0.30,
        platform_width=3.0,
        border_width=1.0,
        holes=False,
    ),
)

WMP_GAP_TERRAINS_CFG = make_single_terrain_cfg(
    "wmp_gap",
    terrain_gen.MeshGapTerrainCfg(
        proportion=1.0,
        gap_width_range=(0.15, 0.60),
        platform_width=2.0,
    ),
)

WMP_CLIMB_TERRAINS_CFG = make_single_terrain_cfg(
    "wmp_climb",
    terrain_gen.MeshBoxTerrainCfg(
        proportion=1.0,
        box_height_range=(0.08, 0.35),
        platform_width=2.0,
        double_box=True,
    ),
)

WMP_TILT_TERRAINS_CFG = make_single_terrain_cfg(
    "wmp_tilt",
    terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=1.0,
        slope_range=(0.12, 0.55),
        platform_width=1.5,
        border_width=0.25,
    ),
)

WMP_CRAWL_TERRAINS_CFG = make_single_terrain_cfg(
    "wmp_crawl",
    terrain_gen.MeshFloatingRingTerrainCfg(
        proportion=1.0,
        ring_width_range=(0.25, 0.75),
        ring_height_range=(0.45, 0.85),
        ring_thickness=0.12,
        platform_width=1.8,
    ),
    difficulty=0.4,
)

WMP_TERRAIN_CFGS = {
    "slope": WMP_SLOPE_TERRAINS_CFG,
    "stair": WMP_STAIR_TERRAINS_CFG,
    "gap": WMP_GAP_TERRAINS_CFG,
    "climb": WMP_CLIMB_TERRAINS_CFG,
    "tilt": WMP_TILT_TERRAINS_CFG,
    "crawl": WMP_CRAWL_TERRAINS_CFG,
}
