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


from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.a1.a1_config import A1AMPFlatAgentCfg, A1AMPFlatEnvCfg
from legged_lab.envs.g1.g1_config import (
    G1FlatAgentCfg,
    G1FlatEnvCfg,
    G1RoughAgentCfg,
    G1RoughEnvCfg,
)
from legged_lab.envs.gr2.gr2_config import (
    GR2FlatAgentCfg,
    GR2FlatEnvCfg,
    GR2RoughAgentCfg,
    GR2RoughEnvCfg,
)
from legged_lab.envs.h1.h1_config import (
    H1FlatAgentCfg,
    H1FlatEnvCfg,
    H1RoughAgentCfg,
    H1RoughEnvCfg,
)
from legged_lab.envs.b2.b2_config import (
    B2FlatAgentCfg,
    B2FlatEnvCfg,
    B2RGBDFlatAgentCfg,
    B2RGBDFlatEnvCfg,
    B2RGBDRoughAgentCfg,
    B2RGBDRoughEnvCfg,
    B2RGBDSlowWalkAgentCfg,
    B2RGBDSlowWalkEnvCfg,
    B2RGBDStandAgentCfg,
    B2RGBDStandEnvCfg,
    B2RGBDWMPAMPFlatAgentCfg,
    B2RGBDWMPAMPFlatEnvCfg,
    B2RGBDWMPAMPTerrainAgentCfg,
    B2RGBDWMPAMPTerrainEnvCfg,
    B2RoughAgentCfg,
    B2RoughEnvCfg,
)
from legged_lab.utils.task_registry import task_registry

task_registry.register("a1_amp_flat", BaseEnv, A1AMPFlatEnvCfg(), A1AMPFlatAgentCfg())
task_registry.register("h1_flat", BaseEnv, H1FlatEnvCfg(), H1FlatAgentCfg())
task_registry.register("h1_rough", BaseEnv, H1RoughEnvCfg(), H1RoughAgentCfg())
task_registry.register("g1_flat", BaseEnv, G1FlatEnvCfg(), G1FlatAgentCfg())
task_registry.register("g1_rough", BaseEnv, G1RoughEnvCfg(), G1RoughAgentCfg())
task_registry.register("gr2_flat", BaseEnv, GR2FlatEnvCfg(), GR2FlatAgentCfg())
task_registry.register("gr2_rough", BaseEnv, GR2RoughEnvCfg(), GR2RoughAgentCfg())

task_registry.register("b2_flat", BaseEnv, B2FlatEnvCfg(), B2FlatAgentCfg())
task_registry.register("b2_rough", BaseEnv, B2RoughEnvCfg(), B2RoughAgentCfg())
task_registry.register("b2_rgbd_flat", BaseEnv, B2RGBDFlatEnvCfg(), B2RGBDFlatAgentCfg())
task_registry.register("b2_rgbd_stand", BaseEnv, B2RGBDStandEnvCfg(), B2RGBDStandAgentCfg())
task_registry.register("b2_rgbd_slow_walk", BaseEnv, B2RGBDSlowWalkEnvCfg(), B2RGBDSlowWalkAgentCfg())
task_registry.register("b2_rgbd_rough", BaseEnv, B2RGBDRoughEnvCfg(), B2RGBDRoughAgentCfg())
task_registry.register("b2_rgbd_wmp_amp_flat", BaseEnv, B2RGBDWMPAMPFlatEnvCfg(), B2RGBDWMPAMPFlatAgentCfg())
task_registry.register(
    "b2_rgbd_wmp_amp_terrain",
    BaseEnv,
    B2RGBDWMPAMPTerrainEnvCfg(),
    B2RGBDWMPAMPTerrainAgentCfg(),
)
