from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float


DEFAULT_WALLS: tuple[Rect, ...] = (
    Rect(24, 8, 5, 29),
    Rect(40, 27, 5, 29),
    Rect(11, 42, 23, 5),
)


class PocketWorldEnv(gym.Env[np.ndarray, int]):
    """A deterministic 64x64 world with inertia, walls, and a circular goal.

    The observation is an RGB uint8 image. A compact state is also exposed in
    ``info`` so experiments can measure position error without image matching.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}
    size = 64
    agent_radius = 3.0
    goal_radius = 4.0
    max_speed = 2.3
    acceleration = 0.75
    friction = 0.84

    def __init__(
        self,
        walls: Iterable[Rect] | None = None,
        agent_start: tuple[float, float] = (8.0, 8.0),
        goal: tuple[float, float] = (55.0, 55.0),
        agent_speed_scale: float = 1.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.walls = tuple(walls or DEFAULT_WALLS)
        self.agent_start = np.asarray(agent_start, dtype=np.float32)
        self.goal = np.asarray(goal, dtype=np.float32)
        self.agent_speed_scale = float(agent_speed_scale)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 255, (3, self.size, self.size), dtype=np.uint8)
        self.position = self.agent_start.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        self.position = np.asarray(options.get("agent_start", self.agent_start), dtype=np.float32).copy()
        self.goal = np.asarray(options.get("goal", self.goal), dtype=np.float32).copy()
        self.velocity.fill(0)
        self.steps = 0
        return self._observation(), self._info()

    def step(self, action: int):
        action = int(action)
        directions = np.asarray(((0, -1), (0, 1), (-1, 0), (1, 0)), dtype=np.float32)
        self.velocity *= self.friction
        self.velocity += directions[action] * self.acceleration * self.agent_speed_scale
        speed = float(np.linalg.norm(self.velocity))
        limit = self.max_speed * self.agent_speed_scale
        if speed > limit:
            self.velocity *= limit / speed

        proposed = self.position + self.velocity
        collided = self._collides(proposed)
        if collided:
            proposed = self.position.copy()
            self.velocity *= 0.0
        self.position = proposed
        self.steps += 1
        terminated = bool(np.linalg.norm(self.position - self.goal) <= self.goal_radius)
        truncated = self.steps >= 250
        reward = 1.0 if terminated else -0.01
        return self._observation(), reward, terminated, truncated, {**self._info(), "collision": collided}

    def render(self) -> np.ndarray:
        return self._observation()

    def _collides(self, point: np.ndarray) -> bool:
        edge = self.agent_radius
        if point[0] < edge or point[0] >= self.size - edge or point[1] < edge or point[1] >= self.size - edge:
            return True
        for wall in self.walls:
            if wall.x - edge <= point[0] <= wall.x + wall.width + edge and wall.y - edge <= point[1] <= wall.y + wall.height + edge:
                return True
        return False

    def _info(self) -> dict:
        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "goal": self.goal.copy(),
            "distance_to_goal": float(np.linalg.norm(self.position - self.goal)),
        }

    def _observation(self) -> np.ndarray:
        image = np.full((self.size, self.size, 3), (15, 22, 35), dtype=np.uint8)
        image[::8, :, :] = (17, 29, 44)
        image[:, ::8, :] = (17, 29, 44)
        for wall in self.walls:
            x0, y0 = int(wall.x), int(wall.y)
            x1, y1 = int(wall.x + wall.width), int(wall.y + wall.height)
            image[y0:y1, x0:x1] = (75, 93, 113)
            image[y0:min(y0 + 2, y1), x0:x1] = (115, 135, 156)
        _draw_disc(image, self.goal, self.goal_radius, (247, 190, 69))
        _draw_disc(image, self.position, self.agent_radius, (93, 224, 183))
        return np.transpose(image, (2, 0, 1))


def _draw_disc(image: np.ndarray, center: Sequence[float], radius: float, color: tuple[int, int, int]) -> None:
    cx, cy = float(center[0]), float(center[1])
    y0, y1 = max(0, int(cy - radius - 1)), min(image.shape[0], int(cy + radius + 2))
    x0, x1 = max(0, int(cx - radius - 1)), min(image.shape[1], int(cx + radius + 2))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    image[y0:y1, x0:x1][mask] = color

