from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


class ReplayBuffer:
    def __init__(self,
                 capacity,
                 observation_shape,
                 action_dim):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        
        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=bool)
        
        self.idx = 0
        self.is_filled = False
    
    def push(self, observation, action, done):
        self.observations[self.idx] = observation
        self.actions[self.idx] = action
        self.done[self.idx] = done
        
        if self.idx == self.capacity - 1:
            self.is_filled = True
        self.idx = (self.idx + 1) % self.capacity
    
    def sample(self, batch_size, seq_length):
        episode_borders = np.where(self.done)[0]
        sampled_indexes = []
        for _ in range(batch_size):
            cross_border = True
            while cross_border:
                initial_index = np.random.randint(len(self) - seq_length + 1)
                final_index = initial_index + seq_length - 1
                cross_border = np.logical_and(initial_index <= episode_borders,
                                              episode_borders < final_index).any()
            sampled_indexes += list(range(initial_index, final_index + 1))

        sampled_observations = self.observations[sampled_indexes].reshape(
            batch_size, seq_length, *self.observations.shape[1:])
        sampled_actions = self.actions[sampled_indexes].reshape(
            batch_size, seq_length, self.actions.shape[1])
        sampled_done = self.done[sampled_indexes].reshape(
            batch_size, seq_length, 1)
        return sampled_observations, sampled_actions, sampled_done

    def __len__(self):
        return self.capacity if self.is_filled else self.idx

    def save(self, dir):
        dir = Path(dir)
        np.save(dir / "observations", self.observations)
        np.save(dir / "actions", self.actions)
        np.save(dir / "done", self.done)

    def load(self, dir):
        dir = Path(dir)
        self.observations = np.load(dir / "observations")
        self.actions = np.load(dir / "actions")
        self.done = np.load(dir / "done")
