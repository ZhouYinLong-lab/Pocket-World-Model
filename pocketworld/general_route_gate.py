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
    "field_route_turns_norm",
    "field_route_min_clearance_norm",
    "field_route_blocked_fraction",
    "field_route_waypoint_count_norm",
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


def _route_geometry(
    route: np.ndarray, occupied: np.ndarray
) -> tuple[float, float, float, float]:
    """Measure visible route geometry without simulator execution."""
    if len(route) < 2:
        return 0.0, 0.0, 1.0, 0.0
    samples: list[np.ndarray] = []
    for start, end in zip(route[:-1], route[1:]):
        count = max(2, int(np.ceil(np.linalg.norm(end - start) * 2.0)))
        samples.extend(np.linspace(start, end, count))
    points = np.asarray(samples, dtype=np.float32)
    xs = np.clip(np.rint(points[:, 0]).astype(int), 0, occupied.shape[1] - 1)
    ys = np.clip(np.rint(points[:, 1]).astype(int), 0, occupied.shape[0] - 1)
    blocked = occupied[ys, xs]
    clearances = np.asarray(
        [_wall_clearance(occupied, point) for point in points], dtype=np.float32
    )
    headings = np.diff(route, axis=0)
    angles = np.arctan2(headings[:, 1], headings[:, 0])
    turns = float(np.count_nonzero(np.abs(np.diff(np.unwrap(angles))) > 0.35))
    return (
        min(1.0, turns / 8.0),
        min(1.0, float(clearances.min()) / 16.0),
        float(blocked.mean()),
        min(1.0, len(route) / 32.0),
    )


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
    """Return the twelve-feature, planner-visible general-route contract."""
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
    wall = extract_wall_mask(frame)
    occupied = _dilate(wall, radius=4)
    direct_fraction, direct_blocked = _direct_wall_fraction(occupied, start, goal_array)
    turns, min_clearance, blocked_fraction, waypoint_count = _route_geometry(
        route, occupied
    )
    return np.asarray(
        (
            direct_distance / 64.0,
            route_length / 128.0,
            min(4.0, route_length / direct_distance) / 4.0,
            turns,
            min_clearance,
            blocked_fraction,
            waypoint_count,
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
