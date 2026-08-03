from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import DEFAULT_WALLS, PocketWorldEnv, Rect


@dataclass
class TransitionBatch:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray


@dataclass
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    collisions: np.ndarray


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


def collect_random_rollouts(
    episodes: int = 100,
    horizon: int = 8,
    seed: int = 7,
    map_variant: bool = False,
    sticky_probability: float = 0.55,
    full_state_range: bool = False,
) -> RolloutBatch:
    """Collect contiguous trajectories for multi-step world-model training."""
    rng = np.random.default_rng(seed)
    all_observations = []
    all_actions = []
    all_positions = []
    all_velocities = []
    all_collisions = []
    for _ in range(episodes):
        walls = _variant_walls(rng) if map_variant else None
        start_low, start_high = (6, 58) if full_state_range else (6, 15)
        goal_low, goal_high = (6, 58) if full_state_range else (49, 58)
        sampling_walls = DEFAULT_WALLS if walls is None else walls
        start = _sample_free_point(rng, sampling_walls, start_low, start_high)
        goal = _sample_free_point(rng, sampling_walls, goal_low, goal_high)
        env = PocketWorldEnv(walls=walls, agent_start=start, goal=goal)
        observation, _ = env.reset()
        observations = [observation]
        positions = [env.position.copy()]
        velocities = [env.velocity.copy()]
        actions = []
        collisions = []
        previous_action: int | None = None
        for _ in range(horizon):
            action = previous_action if previous_action is not None and rng.random() < sticky_probability else int(rng.integers(0, 4))
            previous_action = action
            next_observation, _, terminated, truncated, step_info = env.step(action)
            collisions.append(float(step_info["collision"]))
            actions.append(action)
            observations.append(next_observation)
            positions.append(env.position.copy())
            velocities.append(env.velocity.copy())
            observation = next_observation
            if terminated or truncated:
                observation, _ = env.reset()
        all_observations.append(np.stack(observations))
        all_actions.append(np.asarray(actions, dtype=np.int64))
        all_positions.append(np.stack(positions).astype(np.float32))
        all_velocities.append(np.stack(velocities).astype(np.float32))
        all_collisions.append(np.asarray(collisions, dtype=np.float32))
    return RolloutBatch(
        observations=np.stack(all_observations),
        actions=np.stack(all_actions),
        positions=np.stack(all_positions),
        velocities=np.stack(all_velocities),
        collisions=np.stack(all_collisions),
    )


def _variant_walls(rng: np.random.Generator) -> tuple[Rect, ...]:
    offset = int(rng.integers(-5, 6))
    return (
        Rect(24 + offset, 8, 5, 25),
        Rect(40 - offset // 2, 31, 5, 25),
        Rect(10, 40 + offset // 2, 20, 5),
    )


def _sample_free_point(rng: np.random.Generator, walls: tuple[Rect, ...], low: int, high: int) -> tuple[float, float]:
    """Sample a point with enough clearance for the three-pixel agent."""
    for _ in range(100):
        point = np.asarray((rng.integers(low, high), rng.integers(low, high)), dtype=np.float32)
        if not any(
            wall.x - 3 <= point[0] <= wall.x + wall.width + 3
            and wall.y - 3 <= point[1] <= wall.y + wall.height + 3
            for wall in walls
        ):
            return float(point[0]), float(point[1])
    return float(low), float(low)
