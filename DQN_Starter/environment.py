"""Pendulum: four grayscale images in, one of nine torques out.

All observation processing lives here; no custom wrapper chain is needed.
The simulator's numeric observation is deliberately discarded.
"""
from collections import deque

import gymnasium as gym
import numpy as np
import pygame
from pygame import gfxdraw
from PIL import Image

FRAME_SIZE = 42
STACK_SIZE = 4
TORQUES = np.linspace(-2.0, 2.0, 9, dtype=np.float32)
RENDER_SIZE = 500
# Fixed at the pivot, with margin around the rod's entire swept disk.
CROP_BOX = (120, 120, 380, 380)  # left, top, right, bottom: 260 x 260
OBSERVATION_SPEC = {
    "renderer": "legacy_rod_no_arrow_v1",
    "render_size": RENDER_SIZE,
    "crop_box": list(CROP_BOX),
    "grayscale": "legacy_weighted_uint8_truncate",
    "resize": "bilinear",
}


class ImagePendulum(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, episode_length=200):
        super().__init__()
        self.render_mode = "rgb_array"
        self.simulator = gym.make(
            "Pendulum-v1", g=9.81, render_mode="rgb_array",
            max_episode_steps=episode_length,
        )
        self.action_space = gym.spaces.Discrete(len(TORQUES))
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(STACK_SIZE, FRAME_SIZE, FRAME_SIZE), dtype=np.uint8,
        )
        self.frames = deque(maxlen=STACK_SIZE)
        self.observation_surface = pygame.Surface((RENDER_SIZE, RENDER_SIZE))

    def _observation_rgb(self):
        """Draw the pendulum rod and pivot without a torque indicator.

        Simulator angle is used here to DRAW the environment, never passed to
        the policy. Velocity and applied torque do not affect this image.
        """
        surface = self.observation_surface
        surface.fill((255, 255, 255))
        scale = RENDER_SIZE / 4.4
        offset = RENDER_SIZE // 2
        length, width = scale, 0.2 * scale
        angle = float(self.simulator.unwrapped.state[0]) + np.pi / 2
        corners = [(0, -width / 2), (0, width / 2),
                   (length, width / 2), (length, -width / 2)]
        corners = [pygame.math.Vector2(point).rotate_rad(angle) + (offset, offset)
                   for point in corners]
        color = (204, 77, 77)
        gfxdraw.aapolygon(surface, corners, color)
        gfxdraw.filled_polygon(surface, corners, color)
        end = pygame.math.Vector2(length, 0).rotate_rad(angle) + (offset, offset)
        for x, y in [(offset, offset), (int(end.x), int(end.y))]:
            gfxdraw.aacircle(surface, x, y, int(width / 2), color)
            gfxdraw.filled_circle(surface, x, y, int(width / 2), color)
        gfxdraw.aacircle(surface, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(surface, offset, offset, int(0.05 * scale), (0, 0, 0))
        surface = pygame.transform.flip(surface, False, True)
        return pygame.surfarray.array3d(surface).transpose(1, 0, 2)

    def _frame(self):
        left, top, right, bottom = CROP_BOX
        rgb = self._observation_rgb()[top:bottom, left:right].astype(np.float32)
        # Convert RGB to grayscale and truncate to unsigned 8-bit pixels.
        gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
                + 0.114 * rgb[:, :, 2]).astype(np.uint8)
        image = Image.fromarray(gray)
        image = image.resize((FRAME_SIZE, FRAME_SIZE), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.simulator.reset(seed=seed, options=options)
        frame = self._frame()
        self.frames.clear()
        self.frames.extend([frame] * STACK_SIZE)
        return np.stack(self.frames), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        torque = np.array([TORQUES[int(action)]], dtype=np.float32)
        _, reward, terminated, truncated, _ = self.simulator.step(torque)
        self.frames.append(self._frame())
        return np.stack(self.frames), float(reward), terminated, truncated, {}

    def render(self):
        # Human-facing animation/video retains Gymnasium's torque arrow.
        # This method is deliberately NOT used to generate agent observations.
        return self.simulator.render()

    def close(self):
        self.simulator.close()
