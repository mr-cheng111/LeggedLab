import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from legged_lab.amp.mink_retarget.mapping import load_mapping
from legged_lab.amp.mink_retarget.solver import _joint_project_limits


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "datasets" / "retargeted" / "m20"
MAPPING_PATH = REPO_ROOT / "legged_lab" / "assets" / "deeprobotics" / "m20" / "mjcf" / "a1_to_m20_retarget.yaml"
EXPECTED_MOTIONS = {
    f"{motion}_{side}.txt"
    for motion in ("hop1", "hop2", "trot1", "trot2")
    for side in ("left", "right")
}


class M20AMPDatasetTest(unittest.TestCase):
    def test_mapping_selects_inward_knee_branch(self):
        with MAPPING_PATH.open("r", encoding="utf-8") as stream:
            retarget = yaml.safe_load(stream)["retarget"]

        self.assertNotIn("leg_axis_scale", retarget)
        self.assertEqual(
            retarget["frame_axis_scale"],
            {
                "RR_thigh": [-1.0, 1.0, 1.0],
                "RR_calf": [-1.0, 1.0, 1.0],
                "RL_thigh": [-1.0, 1.0, 1.0],
                "RL_calf": [-1.0, 1.0, 1.0],
            },
        )
        expected_signs = {
            "fl_hipy_joint": 1,
            "fl_knee_joint": -1,
            "fr_hipy_joint": 1,
            "fr_knee_joint": -1,
            "hl_hipy_joint": -1,
            "hl_knee_joint": 1,
            "hr_hipy_joint": -1,
            "hr_knee_joint": 1,
        }
        self.assertEqual(retarget["joint_branch_signs"], expected_signs)
        for joint_name, expected_sign in expected_signs.items():
            self.assertGreater(retarget["neutral_joint_pos"][joint_name] * expected_sign, 0.0)

    def test_inward_knee_signs_are_applied_as_ik_projection_limits(self):
        mapping = load_mapping(MAPPING_PATH)
        joint_addrs = list(range(len(mapping.target.joints)))
        model = SimpleNamespace(mapping=mapping.target, joint_qpos_addrs=joint_addrs)
        projected = {
            name: bounds
            for name, (_, *bounds) in zip(mapping.target.joints, _joint_project_limits(model, mapping), strict=True)
        }

        for joint_name, expected_sign in mapping.options.joint_branch_signs.items():
            lower, upper = projected[joint_name]
            if expected_sign > 0:
                self.assertGreaterEqual(lower, 0.0)
            else:
                self.assertLessEqual(upper, 0.0)

    def test_all_eight_motions_stay_on_inward_knee_branch(self):
        paths = sorted(DATASET_DIR.glob("*.txt"))
        self.assertEqual({path.name for path in paths}, EXPECTED_MOTIONS)

        constrained_columns = {
            1: 1,
            2: -1,
            4: 1,
            5: -1,
            7: -1,
            8: 1,
            10: -1,
            11: 1,
        }
        for path in paths:
            with self.subTest(motion=path.name), path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
                frames = np.asarray(payload["Frames"], dtype=np.float64)
                self.assertEqual(frames.shape, (501, 61))
                self.assertTrue(np.isfinite(frames).all())
                joint_pos = frames[:, 7:19]
                for column, expected_sign in constrained_columns.items():
                    self.assertGreaterEqual(float(np.min(joint_pos[:, column] * expected_sign)), -1.0e-6)

    def test_trot_stance_feet_sweep_backward_relative_to_body(self):
        for path in sorted(DATASET_DIR.glob("trot*.txt")):
            with self.subTest(motion=path.name), path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
                frames = np.asarray(payload["Frames"], dtype=np.float64)
                dt = float(payload["FrameDuration"])
                forward_speed = float(np.median(frames[:, 31]))
                self.assertGreater(forward_speed, 0.1)

                feet = frames[:, 19:31].reshape(-1, 4, 3)
                for foot_index in range(4):
                    foot_height = feet[:, foot_index, 2]
                    stance = foot_height <= np.quantile(foot_height, 0.35)
                    relative_vx = np.gradient(feet[:, foot_index, 0], dt)
                    stance_vx = float(np.median(relative_vx[stance]))
                    self.assertLess(stance_vx, -0.2 * forward_speed)


if __name__ == "__main__":
    unittest.main()
