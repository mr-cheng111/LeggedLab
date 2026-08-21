import unittest
from collections import Counter

import numpy as np

from legged_lab.terrains import M20_ALL_TERRAINS_CFG


CURRENT_TERRAINS = {
    "pyramid_stairs_28",
    "pyramid_stairs_30",
    "pyramid_stairs_32",
    "pyramid_stairs_34",
    "boxes",
    "random_rough",
    "wave",
    "high_platform",
}
SPECIAL_TERRAINS = {
    "wmp_wave",
    "wmp_slope",
    "wmp_stair_up",
    "wmp_stair_down",
    "wmp_discrete",
    "wmp_gap",
    "wmp_climb",
    "wmp_tilt",
    "wmp_crawl",
    "wmp_rough_flat",
}


class M20AllTerrainsTest(unittest.TestCase):
    def test_every_configured_terrain_gets_a_curriculum_column(self):
        cfg = M20_ALL_TERRAINS_CFG.copy()
        self.assertEqual(set(cfg.sub_terrains), CURRENT_TERRAINS | SPECIAL_TERRAINS)
        self.assertAlmostEqual(sum(sub_cfg.proportion for sub_cfg in cfg.sub_terrains.values()), 1.0)

        generator = cfg.class_type(cfg, device="cpu")
        counts = Counter(generator.col_names)
        self.assertEqual(set(counts), CURRENT_TERRAINS | SPECIAL_TERRAINS)
        self.assertEqual(counts["wmp_gap"], 2)
        self.assertEqual(counts["wmp_climb"], 2)
        self.assertTrue(all(count >= 1 for count in counts.values()))

        self.assertEqual(generator.terrain_origins.shape, (10, 20, 3))
        self.assertTrue(np.isfinite(generator.terrain_origins).all())
        self.assertGreater(len(generator.terrain_mesh.vertices), 0)
        self.assertGreater(len(generator.terrain_mesh.faces), 0)
        self.assertTrue(np.isfinite(generator.terrain_mesh.vertices).all())


if __name__ == "__main__":
    unittest.main()
