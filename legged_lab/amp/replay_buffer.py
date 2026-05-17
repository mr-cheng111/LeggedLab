# -*- coding: utf-8 -*-
"""AMP policy transition replay buffer。"""

import torch


class AMPReplayBuffer:
    def __init__(self, obs_dim: int, capacity: int, device: str):
        self.obs_dim = obs_dim
        self.capacity = int(capacity)
        self.device = device
        self.states = torch.zeros(self.capacity, obs_dim, device=device)
        self.next_states = torch.zeros(self.capacity, obs_dim, device=device)
        self.num_samples = 0
        self.position = 0

    def insert(self, states: torch.Tensor, next_states: torch.Tensor):
        states = states.detach().to(self.device)
        next_states = next_states.detach().to(self.device)
        batch = states.shape[0]
        for i in range(batch):
            self.states[self.position].copy_(states[i])
            self.next_states[self.position].copy_(next_states[i])
            self.position = (self.position + 1) % self.capacity
            self.num_samples = min(self.num_samples + 1, self.capacity)

    def feed_forward_generator(self, num_batches: int, batch_size: int):
        if self.num_samples == 0:
            raise RuntimeError("AMP replay buffer is empty.")
        for _ in range(num_batches):
            ids = torch.randint(0, self.num_samples, (batch_size,), device=self.device)
            yield self.states[ids], self.next_states[ids]
