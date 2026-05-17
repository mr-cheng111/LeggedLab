# -*- coding: utf-8 -*-
"""松耦合 AMP-PPO for rsl_rl 5。

该模块只继承 rsl_rl.algorithms.PPO，不修改 rsl_rl site-packages。
AMP reward 在 runner 中显式计算；算法层负责存储 AMP transition，并在
PPO minibatch update 中联合优化 discriminator。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO

from legged_lab.amp import AMPDiscriminator, AMPMotionDataset, AMPReplayBuffer, Normalizer


class AMPPPO(PPO):
    def attach_amp(
        self,
        discriminator: AMPDiscriminator,
        amp_data: AMPMotionDataset,
        amp_normalizer: Normalizer,
        replay_buffer_size: int,
        grad_penalty_coef: float,
    ) -> None:
        self.discriminator = discriminator
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer
        self.amp_storage = AMPReplayBuffer(amp_data.observation_dim, replay_buffer_size, self.device)
        self.grad_penalty_coef = grad_penalty_coef
        self.optimizer.add_param_group({"params": self.discriminator.trunk.parameters(), "weight_decay": 10.0e-4})
        self.optimizer.add_param_group({"params": self.discriminator.amp_linear.parameters(), "weight_decay": 10.0e-2})
        self._last_amp_obs = None
        self._last_combined_reward = None
        self._last_amp_reward = None
        self._last_task_reward = None

    def act(self, obs, amp_obs=None):
        self._last_amp_obs = amp_obs.detach().to(self.device) if amp_obs is not None else None
        return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras, next_amp_obs=None, task_rewards=None, amp_rewards=None):
        if self._last_amp_obs is not None and next_amp_obs is not None:
            next_amp_obs = next_amp_obs.detach().to(self.device)
            self.amp_storage.insert(self._last_amp_obs, next_amp_obs)
            self._last_combined_reward = rewards.detach().to(self.device)
            if amp_rewards is not None:
                self._last_amp_reward = amp_rewards.detach().to(self.device)
            if task_rewards is not None:
                self._last_task_reward = task_rewards.detach().to(self.device)
        super().process_env_step(obs, rewards, dones, extras)

    def update(self):
        if not hasattr(self, "amp_storage") or self.amp_storage.num_samples == 0:
            return super().update()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        num_updates = self.num_learning_epochs * self.num_mini_batches
        batch_size = max(1, self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches)
        generator = (
            self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            if self.actor.is_recurrent or self.critic.is_recurrent
            else self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        )
        policy_generator = self.amp_storage.feed_forward_generator(num_updates, batch_size)
        expert_generator = self.amp_data.feed_forward_generator(num_updates, batch_size)

        for batch, policy_batch, expert_batch in zip(generator, policy_generator, expert_generator):
            original_batch_size = batch.observations.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2)).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            policy_state, policy_next_state = policy_batch
            expert_state, expert_next_state = expert_batch
            raw_policy_state = policy_state
            raw_expert_state = expert_state
            raw_expert_next_state = expert_next_state
            with torch.no_grad():
                policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            expert_loss = torch.nn.functional.mse_loss(expert_d, torch.ones_like(expert_d))
            policy_loss = torch.nn.functional.mse_loss(policy_d, -torch.ones_like(policy_d))
            grad_pen = (
                self.discriminator.compute_grad_pen(raw_expert_state, raw_expert_next_state)
                if self.grad_penalty_coef > 0.0
                else torch.zeros((), device=self.device)
            )
            amp_loss = 0.5 * (expert_loss + policy_loss)
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            loss = loss + amp_loss + self.grad_penalty_coef * grad_pen

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.amp_normalizer.update(raw_policy_state)
            self.amp_normalizer.update(raw_expert_state)
            mean_value_loss += float(value_loss.detach().cpu())
            mean_surrogate_loss += float(surrogate_loss.detach().cpu())
            mean_entropy += float(entropy.mean().detach().cpu())
            mean_amp_loss += float(amp_loss.detach().cpu())
            mean_grad_pen += float(grad_pen.detach().cpu())
            mean_policy_pred += float(policy_d.mean().detach().cpu())
            mean_expert_pred += float(expert_d.mean().detach().cpu())

        self.storage.clear()
        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "amp": mean_amp_loss / num_updates,
            "amp_grad_pen": mean_grad_pen / num_updates,
            "amp_policy_pred": mean_policy_pred / num_updates,
            "amp_expert_pred": mean_expert_pred / num_updates,
        }
        if self._last_amp_reward is not None:
            loss_dict["amp_reward"] = float(self._last_amp_reward.mean().detach().cpu())
        if self._last_task_reward is not None:
            loss_dict["task_reward"] = float(self._last_task_reward.mean().detach().cpu())
        if self._last_combined_reward is not None:
            loss_dict["combined_reward"] = float(self._last_combined_reward.mean().detach().cpu())
        return loss_dict

    def train_mode(self):
        super().train_mode()
        if hasattr(self, "discriminator"):
            self.discriminator.train()

    def eval_mode(self):
        super().eval_mode()
        if hasattr(self, "discriminator"):
            self.discriminator.eval()

    def save(self):
        saved = super().save()
        if hasattr(self, "discriminator"):
            saved["amp_discriminator_state_dict"] = self.discriminator.state_dict()
            saved["amp_normalizer_state_dict"] = self.amp_normalizer.state_dict()
        return saved

    def load(self, loaded_dict, load_cfg, strict):
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if hasattr(self, "discriminator") and "amp_discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
        if hasattr(self, "amp_normalizer") and "amp_normalizer_state_dict" in loaded_dict:
            self.amp_normalizer.load_state_dict(loaded_dict["amp_normalizer_state_dict"])
        return load_iteration
