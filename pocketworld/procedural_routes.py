"""Procedurally generated obstacle layouts for route-policy generalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import PocketWorldEnv, Rect
from .planner import _astar_path, _dilate, extract_wall_mask


@dataclass(frozen=True)
class ProceduralRouteCase:
    """One reproducible left-to-right route task and its visible wall layout."""

    map_id: str
    walls: tuple[Rect, ...]
    start: tuple[float, float]
    goal: tuple[float, float]
    barrier_count: int
    gap_height: int


def procedural_wall_layout(
    seed: int,
    barrier_count: int = 3,
    gap_height: int = 16,
) -> tuple[Rect, ...]:
    """Create vertical barriers with traversable gaps at varied heights."""
    if barrier_count < 1 or barrier_count > 4:
        raise ValueError("barrier_count must be between one and four")
    if gap_height < 14 or gap_height > 24:
        raise ValueError("gap_height must be between 14 and 24 pixels")
    rng = np.random.default_rng(seed)
    # Keep a real corridor between neighboring inflated barriers.  The first
    # version used 10px spacing with 5px walls; after the 4px footprint
    # inflation those barriers touched and silently created disconnected
    # holdout tasks.  Three-pixel walls over this span leave measurable but
    # still non-trivial inter-barrier corridors.
    x_positions = np.rint(np.linspace(12, 49, barrier_count)).astype(np.int64)
    walls: list[Rect] = []
    half_gap = gap_height // 2
    for index, x_value in enumerate(x_positions):
        # Keep both wall segments at least three pixels tall (3x3 pixels is
        # the smallest component retained by the RGB connected-component
        # extractor).  Otherwise an edge-near gap loses one segment from the
        # observable map and becomes an artificial y=3/61 route.
        min_center = half_gap + 8
        max_center = 56 - half_gap
        gap_center = int(rng.integers(min_center, max_center + 1))
        top_end = max(4, gap_center - half_gap)
        bottom_start = min(60 - 4, gap_center + half_gap)
        top_height = max(0, top_end - 3)
        bottom_height = max(0, 61 - bottom_start)
        if top_height:
            walls.append(Rect(float(x_value), 3.0, 3.0, float(top_height)))
        if bottom_height:
            walls.append(Rect(float(x_value), float(bottom_start), 3.0, float(bottom_height)))
    return tuple(walls)


def sample_procedural_route_cases(
    seed: int,
    episodes: int,
    split: str = "train",
) -> tuple[ProceduralRouteCase, ...]:
    """Sample varied route tasks while keeping train/holdout geometry distinct."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if split not in {"train", "holdout"}:
        raise ValueError("split must be train or holdout")
    rng = np.random.default_rng(seed)
    cases: list[ProceduralRouteCase] = []
    for episode in range(episodes):
        if split == "train":
            barrier_count = 2 + int(rng.integers(0, 2))
            # Cover the full gap-width range during training so the holdout
            # isolates barrier-count generalization rather than trivial
            # numerical extrapolation from only 16/18/20px gaps.
            gap_height = int(rng.choice((14, 16, 18, 20, 22, 24)))
            layout_seed = int(rng.integers(0, 2**31 - 1))
            start_y = float(rng.integers(7, 58))
            goal_y = float(rng.integers(7, 58))
        else:
            barrier_count = int(rng.choice((3, 4)))
            gap_height = int(rng.choice((14, 22, 24)))
            layout_seed = int(rng.integers(0, 2**31 - 1))
            # Holdout tasks deliberately include more extreme endpoint bands.
            start_y = float(rng.choice(np.concatenate((np.arange(7, 18), np.arange(47, 58)))))
            goal_y = float(rng.choice(np.concatenate((np.arange(7, 18), np.arange(47, 58)))))
        walls = procedural_wall_layout(layout_seed, barrier_count, gap_height)
        case = ProceduralRouteCase(
            map_id=f"{split}-{seed}-{episode}-b{barrier_count}-g{gap_height}",
            walls=walls,
            start=(8.0, start_y),
            goal=(56.0, goal_y),
            barrier_count=barrier_count,
            gap_height=gap_height,
        )
        # A procedural sample is only useful if the footprint-inflated RGB
        # geometry has a route.  This is a teacher-side data-quality filter;
        # the learned student still receives only RGB and goal at evaluation.
        observation = PocketWorldEnv(walls=walls, agent_start=case.start, goal=case.goal).reset()[0]
        if _astar_path(_dilate(extract_wall_mask(observation), 4), case.start, case.goal, allow_diagonal=False):
            cases.append(case)
        else:
            # Keep deterministic output length while resampling a nearby
            # layout/task from the same RNG stream.
            for retry in range(32):
                retry_seed = int(rng.integers(0, 2**31 - 1))
                retry_walls = procedural_wall_layout(retry_seed, barrier_count, gap_height)
                retry_case = ProceduralRouteCase(
                    map_id=f"{split}-{seed}-{episode}-retry{retry}-b{barrier_count}-g{gap_height}",
                    walls=retry_walls,
                    start=case.start,
                    goal=case.goal,
                    barrier_count=barrier_count,
                    gap_height=gap_height,
                )
                env_observation = PocketWorldEnv(
                    walls=retry_walls, agent_start=retry_case.start, goal=retry_case.goal
                ).reset()[0]
                if _astar_path(_dilate(extract_wall_mask(env_observation), 4), retry_case.start, retry_case.goal, allow_diagonal=False):
                    cases.append(retry_case)
                    break
            else:
                raise RuntimeError("could not sample a reachable procedural route")
    return tuple(cases)
