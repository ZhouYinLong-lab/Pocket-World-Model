"""Route-completion features for the general obstacle benchmark.

The gate predicts whether the learned route-field controller is likely to
finish a route before deciding whether to pay for an A* fallback.  Features
are computed from the current RGB observation, the learned field waypoints,
and the requested speed; labels are collected only by running the candidate
controller in a disjoint simulator split.
"""

from __future__ import annotations

import numpy as np

from .planner import _dilate, extract_agent_position, extract_wall_mask
from .route_completion import RouteCompletionPredictor
from .route_field import RouteFieldPolicy, field_waypoints


GENERAL_ROUTE_FEATURE_NAMES = (
    "direct_distance_norm",
    "field_route_length_norm",
    "field_route_ratio_norm",
    "field_route_distance_norm",
    "wall_fraction",
    "direct_wall_fraction",
    "direct_wall_blocked",
    "start_wall_clearance_norm",
    "agent_speed_scale_norm",
)


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _wall_clearance(mask: np.ndarray, point: np.ndarray) -> float:
    occupied = np.argwhere(mask)
    if occupied.size == 0:
        return float(np.hypot(*mask.shape))
    y, x = occupied[:, 0], occupied[:, 1]
    return float(np.sqrt(((x - point[0]) ** 2 + (y - point[1]) ** 2).min()))


def _direct_wall_fraction(
    mask: np.ndarray, start: np.ndarray, goal: np.ndarray
) -> tuple[float, float]:
    samples = max(2, int(np.ceil(np.linalg.norm(goal - start) * 2.0)))
    xs = np.rint(np.linspace(start[0], goal[0], samples)).astype(int)
    ys = np.rint(np.linspace(start[1], goal[1], samples)).astype(int)
    inside = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    blocked = np.zeros(samples, dtype=bool)
    blocked[inside] = mask[ys[inside], xs[inside]]
    return float(blocked.mean()), float(blocked.any())


def extract_general_route_features(
    observation: np.ndarray,
    goal: tuple[float, float] | np.ndarray,
    field_policy: RouteFieldPolicy,
    agent_speed_scale: float = 1.0,
) -> np.ndarray:
    """Return the nine-feature, planner-visible general-route contract."""
    frame = np.asarray(observation)
    start = extract_agent_position(frame).astype(np.float32)
    goal_array = np.asarray(goal, dtype=np.float32)
    if frame.shape != (3, 64, 64):
        raise ValueError("observation must have shape [3, 64, 64]")
    if start.shape != (2,) or not np.isfinite(start).all():
        raise ValueError("observation must contain a finite agent position")
    if goal_array.shape != (2,) or not np.isfinite(goal_array).all():
        raise ValueError("goal must have shape [2] and be finite")
    if agent_speed_scale <= 0.0 or not np.isfinite(agent_speed_scale):
        raise ValueError("agent_speed_scale must be finite and positive")

    field = field_policy.predict_field(frame, tuple(map(float, goal_array)))
    waypoints = field_waypoints(frame, tuple(map(float, goal_array)), field, rgb_guard=True, beam_width=4)
    route = np.asarray((tuple(start),) + tuple(waypoints), dtype=np.float32)
    route_length = _polyline_length(route)
    direct_distance = max(1.0, float(np.linalg.norm(goal_array - start)))
    route_distance = float(np.linalg.norm(route[-1] - goal_array))
    wall = extract_wall_mask(frame)
    occupied = _dilate(wall, radius=4)
    direct_fraction, direct_blocked = _direct_wall_fraction(occupied, start, goal_array)
    return np.asarray(
        (
            direct_distance / 64.0,
            route_length / 128.0,
            min(4.0, route_length / direct_distance) / 4.0,
            route_distance / 64.0,
            float(occupied.mean()),
            direct_fraction,
            direct_blocked,
            min(1.0, _wall_clearance(occupied, start) / 64.0),
            min(2.0, float(agent_speed_scale)) / 2.0,
        ),
        dtype=np.float32,
    )


def make_general_route_predictor() -> RouteCompletionPredictor:
    """Construct the predictor with the explicit general-route contract."""
    return RouteCompletionPredictor(feature_names=GENERAL_ROUTE_FEATURE_NAMES)


