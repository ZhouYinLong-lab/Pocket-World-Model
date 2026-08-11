"""Task generation on top of the named PocketWorld map suite."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import Rect
from .maps import get_map, map_names


@dataclass(frozen=True)
class NavigationTask:
    """A start state and one or more ordered goals on one map."""

    map_name: str
    start: tuple[float, float]
    goals: tuple[tuple[float, float], ...]

    @property
    def goal(self) -> tuple[float, float]:
        return self.goals[0]


def _is_free(point: np.ndarray, walls: tuple[Rect, ...], clearance: float = 3.0) -> bool:
    return not any(
        wall.x - clearance <= point[0] <= wall.x + wall.width + clearance
        and wall.y - clearance <= point[1] <= wall.y + wall.height + clearance
        for wall in walls
    )


def _sample_free_point(rng: np.random.Generator, walls: tuple[Rect, ...]) -> tuple[float, float]:
    for _ in range(500):
        point = rng.uniform(6.0, 58.0, size=2).astype(np.float32)
        if _is_free(point, walls):
            return float(point[0]), float(point[1])
    raise RuntimeError("could not sample a free PocketWorld task point")


def sample_navigation_task(
    rng: np.random.Generator,
    map_name: str,
    waypoint_count: int = 1,
    minimum_goal_spacing: float = 12.0,
) -> NavigationTask:
    """Sample a reachable-looking task without exposing privileged map labels.

    The environment remains deterministic after sampling. ``waypoint_count=1``
    is the ordinary navigation task; larger values create sequential-goal
    tasks that test whether a planner can finish more than one route.
    """

    if waypoint_count < 1:
        raise ValueError("waypoint_count must be positive")
    spec = get_map(map_name)
    start = _sample_free_point(rng, spec.walls)
    goals: list[tuple[float, float]] = []
    previous = np.asarray(start, dtype=np.float32)
    for _ in range(waypoint_count):
        for _ in range(500):
            candidate = np.asarray(_sample_free_point(rng, spec.walls), dtype=np.float32)
            if np.linalg.norm(candidate - previous) >= minimum_goal_spacing:
                goals.append((float(candidate[0]), float(candidate[1])))
                previous = candidate
                break
        else:
            raise RuntimeError("could not sample separated PocketWorld waypoints")
    return NavigationTask(map_name=map_name, start=start, goals=tuple(goals))


def sample_task_suite(
    rng: np.random.Generator,
    suite: str = "holdout",
    waypoint_count: int = 1,
) -> NavigationTask:
    """Sample a task from a named map suite for reproducible evaluation."""

    names = map_names(suite)
    map_name = str(rng.choice(names))
    return sample_navigation_task(rng, map_name, waypoint_count=waypoint_count)
