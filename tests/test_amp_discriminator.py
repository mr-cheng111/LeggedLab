import unittest

import torch

from legged_lab.amp import AMPDiscriminator


class AMPDiscriminatorTest(unittest.TestCase):
    def _make_discriminator(self, task_reward_lerp: float) -> AMPDiscriminator:
        discriminator = AMPDiscriminator(
            input_dim=4,
            amp_reward_coef=2.0,
            hidden_layer_sizes=[],
            device="cpu",
            task_reward_lerp=task_reward_lerp,
        )
        with torch.no_grad():
            discriminator.amp_linear.weight.zero_()
            discriminator.amp_linear.bias.fill_(1.0)
        return discriminator

    def test_task_reward_lerp_blends_amp_and_task_rewards(self):
        state = torch.zeros(2, 2)
        next_state = torch.zeros(2, 2)
        task_reward = torch.tensor([4.0, 8.0])

        discriminator = self._make_discriminator(task_reward_lerp=0.25)
        reward, prediction, amp_reward = discriminator.predict_amp_reward(
            state,
            next_state,
            task_reward,
            return_details=True,
        )

        torch.testing.assert_close(prediction, torch.ones(2))
        torch.testing.assert_close(amp_reward, torch.full((2,), 2.0))
        torch.testing.assert_close(reward, torch.tensor([2.5, 3.5]))

    def test_task_reward_lerp_endpoints(self):
        state = torch.zeros(1, 2)
        next_state = torch.zeros(1, 2)
        task_reward = torch.tensor([4.0])

        amp_only, _ = self._make_discriminator(0.0).predict_amp_reward(state, next_state, task_reward)
        task_only, _ = self._make_discriminator(1.0).predict_amp_reward(state, next_state, task_reward)

        torch.testing.assert_close(amp_only, torch.tensor([2.0]))
        torch.testing.assert_close(task_only, task_reward)

    def test_task_reward_lerp_rejects_values_outside_unit_interval(self):
        for value in (-0.1, 1.1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "task_reward_lerp"):
                self._make_discriminator(value)


if __name__ == "__main__":
    unittest.main()
