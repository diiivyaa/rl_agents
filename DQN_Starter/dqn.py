"""Student starter for image-only DQN. Complete TODO-Q3-1 through TODO-Q3-6."""
from copy import deepcopy
import random

import torch
from torch import nn
from torch.nn import functional as F

from environment import FRAME_SIZE, STACK_SIZE, TORQUES, OBSERVATION_SPEC

ARCHITECTURE = "student_network_v1"  # Update this identifier if checkpoint architecture changes.


def select_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install the CUDA PyTorch build and check the driver.")
    return torch.device(name)


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO-Q3-1: Design a CNN and suitable linear layers for four images.
        # Input: [batch, 4, 42, 42]; output: [batch, 9] unrestricted Q-values.
        # Choose kernels, strides, channels, hidden dimensions and activations.
        # Runnable placeholder ONLY: identity features and one linear layer.
        # A linear-only network is not an acceptable final CNN solution.

        # Following CNN architecture from Mnih et al. (2013)
        self.cnn = nn.Sequential(
            nn.Conv2d(STACK_SIZE, 16, kernel_size=8, stride=4), # 4 x 42 x 42 --> 16 x 9 x 9
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), # 16 x 9 x 9 --> 32 x 3 x 3
            nn.ReLU(),
            nn.Flatten(), # 32 x 3 x 3 --> 288
            nn.Linear(32 * 3 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, len(TORQUES))
        )

    def forward(self, images):
        # TODO-Q3-2: Connect your CNN and linear layers in the correct order.
        # Normalize pixels, keep the batch dimension, and return nine Q-values.
        # Use only the image tensor; do not add numerical simulator state.
        # Runnable placeholder for the linear-only model above:
        # features = self.features(images.float() / 255.0)
        # return self.fc_q(features.flatten(start_dim=1))

        images = images.float() / 255.0 # normalize pixel
        out = self.cnn(images)
        return out


class DQN:
    def __init__(self, device="auto", learning_rate=2e-4, gamma=0.99):
        self.device = select_device(device)
        # Primary Network: updated by the optimizer.
        self.primary = QNetwork().to(self.device)
        # Target Network: used for Bellman targets; copied at sync intervals.
        self.target = deepcopy(self.primary).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.primary.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.updates = 0

    def choose_action(self, observation, epsilon=0.0):
        # TODO-Q3-3: Implement epsilon-greedy action selection.
        # Explore with probability epsilon; otherwise use the largest Q-value.
        # Use self.device and disable gradient tracking during inference.
        # Return one Python integer in [0, len(TORQUES)).
        # Runnable placeholder ONLY: always random, even when epsilon is zero.
        # return random.randrange(len(TORQUES))

        with torch.no_grad():
            decision = random.random()
            if decision < epsilon:
                return random.randrange(len(TORQUES))
            else:
                tensor = torch.tensor(observation, device=self.device).unsqueeze(0) # 4 x 42 x 42 --> 1 x 4 x 42 x 42
                q_values = self.primary(tensor) # 1 x 9
                return q_values.argmax(dim=1).item()
            
    def learn(self, batch):
        states, actions, rewards, next_states, terminated = (
            torch.as_tensor(array, device=self.device) for array in batch
        )
        # TODO-Q3-5: Compute the DQN minibatch loss.
        # Select Q-values from the Primary Network for the sampled actions.
        # Construct detached Bellman targets with the Target Network.
        # Only true termination stops
        # bootstrapping; time-limit truncation does not. Use mean squared TD-error loss.
        # Keep the optimizer/gradient-clipping code below as training boilerplate.
        # Runnable placeholder ONLY: connected zero loss produces zero gradients.
        # loss = self.primary(states).sum() * 0.0

        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values
            TD_target = rewards + self.gamma * (1 - terminated) * next_q
        
        predicted_q = self.primary(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(predicted_q, TD_target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.primary.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.updates += 1
        return float(loss.detach().item())

    def sync_target(self):
        # TODO-Q3-6: Copy Primary Network parameters into the Target Network.
        # The supplied loop calls this after the configured number of updates.
        # Runnable placeholder ONLY: leave target parameters unchanged.
        # return None

        self.target.load_state_dict(self.primary.state_dict())

    def save(self, path, *, step=0, config=None):
        torch.save({
            "format_version": 3, "observation": "four_grayscale_frames_only",
            "architecture": ARCHITECTURE,
            "preprocessing": OBSERVATION_SPEC,
            "frame_size": FRAME_SIZE, "stack_size": STACK_SIZE,
            "torques": TORQUES.tolist(), "step": step, "updates": self.updates,
            "model": self.primary.state_dict(), "config": config or {},
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint.get("observation") != "four_grayscale_frames_only":
            raise ValueError("Checkpoint is not compatible with the image-only DQN model.")
        if checkpoint.get("preprocessing") != OBSERVATION_SPEC:
            raise ValueError("Checkpoint preprocessing does not match the current arrow-free fixed crop. Retrain with the current environment.")
        if checkpoint.get("architecture") != ARCHITECTURE:
            raise ValueError("Checkpoint architecture does not match this model. Use a matching model definition.")
        if (checkpoint.get("frame_size"), checkpoint.get("stack_size"), checkpoint.get("torques")) != (
            FRAME_SIZE, STACK_SIZE, TORQUES.tolist(),
        ):
            raise ValueError("Checkpoint observation/action settings do not match this environment.")
        self.primary.load_state_dict(checkpoint["model"])
        self.sync_target()
        return checkpoint
