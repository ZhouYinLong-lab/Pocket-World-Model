from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import PocketWorldEnv, Rect


@dataclass
class TransitionBatch:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray


def collect_random_transitions(
    episodes: int = 100,
    horizon: int = 80,
    seed: int = 7,
    map_variant: bool = False,
) -> TransitionBatch:
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    next_observations: list[np.ndarray] = []
    for _ in range(episodes):
        walls = _variant_walls(rng) if map_variant else None
        start = (float(rng.integers(6, 15)), float(rng.integers(6, 15)))
        goal = (float(rng.integers(49, 58)), float(rng.integers(49, 58)))
        env = PocketWorldEnv(walls=walls, agent_start=start, goal=goal, agent_speed_scale=1.0)
        observation, _ = env.reset()
        for _ in range(horizon):
            action = int(rng.integers(0, 4))
            next_observation, _, terminated, truncated, _ = env.step(action)
            observations.append(observation)
            actions.append(action)
            next_observations.append(next_observation)
            observation = next_observation
            if terminated or truncated:
                break
    return TransitionBatch(
        observations=np.stack(observations),
        actions=np.asarray(actions, dtype=np.int64),
        next_observations=np.stack(next_observations),
    )


def _variant_walls(rng: np.random.Generator) -> tuple[Rect, ...]:
    offset = int(rng.integers(-5, 6))
    return (
        Rect(24 + offset, 8, 5, 25),
        Rect(40 - offset // 2, 31, 5, 25),
        Rect(10, 40 + offset // 2, 20, 5),
    )

