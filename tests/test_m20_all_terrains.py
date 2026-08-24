import unittest
from collections import Counter

import numpy as np

from legged_lab.terrains import M20_ALL_TERRAINS_CFG


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

EXPECTED_COLUMN_COUNTS = {
    "wmp_slope": 1,
    "wmp_stair_up": 3,
    "wmp_stair_down": 3,
    "wmp_gap": 5,
    "wmp_climb": 5,
    "wmp_tilt": 1,
    "wmp_crawl": 1,
    "wmp_rough_flat": 1,
}


class M20AllTerrainsTest(unittest.TestCase):
    def test_m20_uses_stage_two_wmp_weights(self):
        cfg = M20_ALL_TERRAINS_CFG.copy()
        self.assertEqual(set(cfg.sub_terrains), SPECIAL_TERRAINS)
        self.assertAlmostEqual(sum(sub_cfg.proportion for sub_cfg in cfg.sub_terrains.values()), 1.0)

        generator = cfg.class_type(cfg, device="cpu")
        counts = Counter(generator.col_names)
        self.assertEqual(dict(counts), EXPECTED_COLUMN_COUNTS)

        self.assertEqual(generator.terrain_origins.shape, (10, 20, 3))
        self.assertTrue(np.isfinite(generator.terrain_origins).all())
        self.assertGreater(len(generator.terrain_mesh.vertices), 0)
        self.assertGreater(len(generator.terrain_mesh.faces), 0)
        self.assertTrue(np.isfinite(generator.terrain_mesh.vertices).all())


if __name__ == "__main__":
    unittest.main()
