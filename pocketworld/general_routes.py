"""Deterministic non-vertical obstacle tasks for route-policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import PocketWorldEnv, Rect
from .planner import _astar_path, _dilate, extract_wall_mask


GENERAL_FAMILIES = ("staggered_blocks", "multi_channel", "staircase", "l_shapes")


@dataclass(frozen=True)
class GeneralRouteCase:
    """One reproducible route task with a non-specialized obstacle layout."""

    map_id: str
    walls: tuple[Rect, ...]
    start: tuple[float, float]
    goal: tuple[float, float]
    family: str
    obstacle_count: int
    channel_count: int


def _staggered_blocks(rng: np.random.Generator) -> tuple[tuple[Rect, ...], int]:
    walls: list[Rect] = []
    for index, x in enumerate((16.0, 27.0, 38.0, 49.0)):
        y = float(rng.integers(12, 43))
        width = float(rng.integers(5, 9))
        height = float(rng.integers(6, 12))
        walls.append(Rect(x, y, width, height))
        if index % 2 == 1:
            walls.append(Rect(x - 4.0, max(5.0, y - 7.0), 4.0, 5.0))
    return tuple(walls), 2


def _multi_channel(rng: np.random.Generator) -> tuple[tuple[Rect, ...], int]:
    """Build two broken walls, each with two visible route channels."""
    walls: list[Rect] = []
    for x in (22.0, 43.0):
        # A 6px half-gap leaves four free pixels after the 4px teacher
        # inflation, while the simulator's 3px circular footprint still has
        # a positive passage.  The previous 3px half-gap looked like two
        # channels in RGB but was disconnected after footprint inflation.
        centers = sorted(
            int(value) for value in (rng.integers(16, 21), rng.integers(43, 48))
        )
        gap_half = 6
        cursor = 4
        for center in centers:
            top = center - gap_half
            if top > cursor:
                walls.append(Rect(x, float(cursor), 4.0, float(top - cursor)))
            cursor = center + gap_half
        if cursor < 60:
            walls.append(Rect(x, float(cursor), 4.0, float(60 - cursor)))
    return tuple(walls), 2


def _staircase(rng: np.random.Generator) -> tuple[tuple[Rect, ...], int]:
    """Create a diagonal staircase of blocks with non-axis-aligned route flow."""
    walls: list[Rect] = []
    offset = int(rng.integers(-5, 6))
    for index, x in enumerate((14.0, 23.0, 32.0, 41.0, 50.0)):
        y = float(np.clip(10 + index * 8 + offset, 5, 48))
        walls.append(Rect(x, y, 6.0, 6.0))
        if index in (1, 3):
            walls.append(Rect(x - 3.0, y + 5.0, 3.0, 5.0))
    return tuple(walls), 2


def _l_shapes(rng: np.random.Generator) -> tuple[tuple[Rect, ...], int]:
    """Create two offset L-shaped obstacles from axis-aligned rectangles."""
    del rng
    walls = (
        Rect(15.0, 12.0, 5.0, 24.0),
        Rect(15.0, 31.0, 15.0, 5.0),
        Rect(39.0, 27.0, 5.0, 24.0),
        Rect(29.0, 27.0, 15.0, 5.0),
    )
    return walls, 2


def general_wall_layout(seed: int, family: str) -> tuple[tuple[Rect, ...], int]:
    """Return a deterministic obstacle family and its number of route channels."""
    if family not in GENERAL_FAMILIES:
        raise ValueError(f"family must be one of {GENERAL_FAMILIES}")
    rng = np.random.default_rng(seed)
    builders = {
        "staggered_blocks": _staggered_blocks,
        "multi_channel": _multi_channel,
        "staircase": _staircase,
        "l_shapes": _l_shapes,
    }
    return builders[family](rng)


def _sample_endpoints(rng: np.random.Generator, split: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if split == "train":
        start_y = float(rng.integers(7, 57))
        goal_y = float(rng.integers(7, 57))
        return (7.0, start_y), (57.0, goal_y)
    corners = ((7.0, 7.0), (7.0, 57.0), (57.0, 7.0), (57.0, 57.0))
    start = corners[int(rng.integers(0, len(corners)))]
    opposite = (57.0, 57.0 - start[1] + 7.0) if start[0] < 32 else (7.0, 57.0 - start[1] + 7.0)
    return start, opposite


def sample_general_route_cases(
    seed: int,
    episodes: int,
    split: str = "train",
) -> tuple[GeneralRouteCase, ...]:
    """Sample reachable train/holdout tasks with distinct shape families."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if split not in {"train", "holdout"}:
        raise ValueError("split must be train or holdout")
    rng = np.random.default_rng(seed)
    families = ("staggered_blocks", "multi_channel") if split == "train" else GENERAL_FAMILIES
    cases: list[GeneralRouteCase] = []
    for episode in range(episodes):
        for retry in range(48):
            family = str(rng.choice(families))
            layout_seed = int(rng.integers(0, 2**31 - 1))
            walls, channel_count = general_wall_layout(layout_seed, family)
            start, goal = _sample_endpoints(rng, split)
            env = PocketWorldEnv(walls=walls, agent_start=start, goal=goal)
            observation, _ = env.reset()
            if env._collides(np.asarray(start, dtype=np.float32)) or env._collides(np.asarray(goal, dtype=np.float32)):
                continue
            path = _astar_path(
                _dilate(extract_wall_mask(observation), 4),
                start,
                goal,
                allow_diagonal=False,
            )
            if not path or len(path) < 16:
                continue
            cases.append(
                GeneralRouteCase(
                    map_id=f"{split}-{seed}-{episode}-{family}-o{len(walls)}-r{retry}",
                    walls=walls,
                    start=start,
                    goal=goal,
                    family=family,
                    obstacle_count=len(walls),
                    channel_count=channel_count,
                )
            )
            break
        else:
            raise RuntimeError("could not sample a reachable general route")
    return tuple(cases)
