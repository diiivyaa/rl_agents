"""Uniform experience replay, with uint8 images to limit memory usage."""
from collections import deque
import random

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=20_000):
        self.transitions = deque(maxlen=capacity)

    def __len__(self):
        return len(self.transitions)

    def add(self, state, action, reward, next_state, terminated):
        # Copy observations so subsequent environment steps cannot change them.
        self.transitions.append(
            (state.copy(), action, reward, next_state.copy(), terminated)
        )

    def sample(self, batch_size):
        # TODO-Q3-4: Uniformly sample batch_size transitions without replacement.
        # Preserve the array formats and ordering of the return tuple below.
        # Runnable placeholder ONLY: use the oldest available transitions.
        # batch = list(self.transitions)[:batch_size]
        
        batch = random.sample(self.transitions, batch_size)
        states, actions, rewards, next_states, terminated = zip(*batch)
        return (
            np.stack(states), np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32), np.stack(next_states),
            np.asarray(terminated, dtype=np.float32),
        )
