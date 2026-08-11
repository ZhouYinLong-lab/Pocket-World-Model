"""Named PocketWorld layouts used for training and held-out evaluation.

The original ``DEFAULT_WALLS`` layout remains the compatibility baseline.  The
named layouts make map generalization explicit instead of hiding it behind one
random wall offset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import DEFAULT_WALLS, Rect


@dataclass(frozen=True)
class MapSpec:
    """A deterministic 64x64 wall layout with a short human description."""

    name: str
    walls: tuple[Rect, ...]
    description: str


MAPS: dict[str, MapSpec] = {
    "default": MapSpec("default", DEFAULT_WALLS, "three staggered walls; compatibility baseline"),
    "single_barrier": MapSpec(
        "single_barrier",
        (Rect(29, 10, 5, 44),),
        "one vertical barrier with top and bottom passages",
    ),
    "double_barrier": MapSpec(
        "double_barrier",
        (Rect(19, 7, 5, 30), Rect(40, 27, 5, 30)),
        "two offset vertical barriers requiring alternating detours",
    ),
    "cross": MapSpec(
        "cross",
        (Rect(29, 8, 6, 24), Rect(8, 29, 27, 6), Rect(35, 35, 6, 21)),
        "cross-shaped central obstruction with four approach directions",
    ),
    "zigzag": MapSpec(
        "zigzag",
        # Keep a > 2 * agent_radius opening between the first and third
        # segments.  The previous 6px gap was closed by the environment's
        # inclusive collision rule and made this holdout physically
        # disconnected for a 6px-diameter agent.
        (Rect(18, 7, 5, 24), Rect(41, 28, 5, 29), Rect(18, 43, 28, 5)),
        "zig-zag corridor with a physically traversable holdout passage",
    ),
    "open": MapSpec("open", (), "no interior walls; dynamics-only control reference"),
}

MAP_SUITES: dict[str, tuple[str, ...]] = {
    "baseline": ("default",),
    "train": ("default", "single_barrier", "double_barrier"),
    "holdout": ("cross", "zigzag"),
    "all": tuple(MAPS),
}


def get_map(name: str) -> MapSpec:
    """Return a named layout or raise a useful error for a typo."""

    try:
        return MAPS[name]
    except KeyError as exc:
        available = ", ".join(sorted(MAPS))
        raise ValueError(f"unknown map {name!r}; choose from {available}") from exc


def map_names(suite: str = "baseline") -> tuple[str, ...]:
    """Resolve a reproducible map suite name to ordered map names."""

    try:
        return MAP_SUITES[suite]
    except KeyError as exc:
        available = ", ".join(sorted(MAP_SUITES))
        raise ValueError(f"unknown map suite {suite!r}; choose from {available}") from exc


def sample_map_name(rng: np.random.Generator, suite: str = "baseline") -> str:
    """Sample one map without changing the suite's deterministic membership."""

    names = map_names(suite)
    return str(rng.choice(names))
