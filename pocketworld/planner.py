from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import heapq
from typing import Callable

import numpy as np
import torch

from .env import DEFAULT_WALLS
from .model import PocketWorldModel


@dataclass
class PlanResult:
    actions: np.ndarray
    imagined_positions: np.ndarray
    imagined_distance: float
    imagined_collision_risk: float = 0.0
    planning_score: float = 0.0
    route_score: float = 0.0
    route_progress: float = 0.0
    route_endpoint_distance: float = 0.0
    predicted_route_completion_probability: float = 0.0


@dataclass
class RecedingHorizonResult:
    actions: np.ndarray
    first_plan_distance: float
    first_plan_route_distance: float
    final_observation: np.ndarray
    final_info: dict
    collision_count: int = 0
    replans: int = 0
    route_alignment_error_px: float = 0.0
    max_route_alignment_error_px: float = 0.0
    mean_shift_score: float = 0.0
    max_shift_score: float = 0.0
    shift_detected_count: int = 0
    first_plan_route_completion_probability: float = 0.0
    alignment_fallback_trigger_count: int = 0
    fallback_steps: int = 0


def _rect_wall_mask() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=bool)
    for wall in DEFAULT_WALLS:
        x0, y0 = int(wall.x), int(wall.y)
        x1, y1 = int(wall.x + wall.width), int(wall.y + wall.height)
        mask[y0:y1, x0:x1] = True
    return mask


_DEFAULT_WALL_MASK = _rect_wall_mask()


def extract_wall_mask(frame: np.ndarray) -> np.ndarray:
    """Extract wall pixels from a CHW uint8 RGB observation."""
    if frame.dtype != np.uint8:
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError("expected a CHW RGB frame")
    red, green, blue = frame.astype(np.int16)
    return (red >= 60) & (red <= 140) & (green >= 70) & (green <= 160) & (blue >= 80) & (blue <= 180)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy:dy + height, dx:dx + width]
    return result


def _nearest_free_cell(occupied: np.ndarray, point: tuple[int, int], bounds: tuple[int, int, int, int]) -> tuple[int, int] | None:
    """Return the closest free grid cell when a noisy observation blocks a pose."""
    x_min, x_max, y_min, y_max = bounds
    x0 = int(np.clip(point[0], x_min, x_max))
    y0 = int(np.clip(point[1], y_min, y_max))
    if not occupied[y0, x0]:
        return x0, y0
    candidates = []
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            if not occupied[y, x]:
                candidates.append((abs(x - x0) + abs(y - y0), x, y))
    if not candidates:
        return None
    _, x, y = min(candidates)
    return x, y


def _astar_path(
    occupied: np.ndarray,
    start: tuple[float, float],
    goal: tuple[float, float],
    vertical_preference: str | None = None,
    bounds: tuple[int, int, int, int] = (3, 60, 3, 60),
) -> list[tuple[int, int]]:
    """Find a collision-free 8-connected route on a binary occupancy grid.

    This is intentionally a small global planner rather than an oracle: the
    occupancy grid is built from the current RGB observation.  Diagonal moves
    cannot cut through a blocked corner, which keeps the route valid for the
    circular agent footprint after wall inflation.
    """
    if occupied.ndim != 2 or occupied.shape[0] < 2 or occupied.shape[1] < 2:
        raise ValueError("occupied must be a 2D grid")
    x_min, x_max, y_min, y_max = bounds
    start_cell = _nearest_free_cell(
        occupied,
        (int(round(start[0])), int(round(start[1]))),
        (x_min, x_max, y_min, y_max),
    )
    goal_cell = _nearest_free_cell(
        occupied,
        (int(round(goal[0])), int(round(goal[1]))),
        (x_min, x_max, y_min, y_max),
    )
    if start_cell is None or goal_cell is None:
        return []
    if start_cell == goal_cell:
        return [start_cell]

    directions = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 2**0.5),
        (1, -1, 2**0.5),
        (-1, 1, 2**0.5),
        (1, 1, 2**0.5),
    )
    preference = vertical_preference.lower() if vertical_preference else None

    def heuristic(cell: tuple[int, int]) -> float:
        return float(np.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1]))

    def edge_cost(next_cell: tuple[int, int], step_cost: float) -> float:
        if preference == "top":
            return step_cost + 0.015 * next_cell[1]
        if preference == "bottom":
            return step_cost + 0.015 * (occupied.shape[0] - 1 - next_cell[1])
        return step_cost

    frontier: list[tuple[float, float, tuple[int, int]]] = [(heuristic(start_cell), 0.0, start_cell)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
    cost_so_far = {start_cell: 0.0}
    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current_cost > cost_so_far.get(current, float("inf")) + 1e-9:
            continue
        if current == goal_cell:
            path = []
            cursor: tuple[int, int] | None = current
            while cursor is not None:
                path.append(cursor)
                cursor = came_from[cursor]
            return path[::-1]
        for dx, dy, step_cost in directions:
            next_x, next_y = current[0] + dx, current[1] + dy
            if not (x_min <= next_x <= x_max and y_min <= next_y <= y_max):
                continue
            if occupied[next_y, next_x]:
                continue
            if dx and dy and (occupied[current[1], next_x] or occupied[next_y, current[0]]):
                continue
            next_cell = (next_x, next_y)
            new_cost = current_cost + edge_cost(next_cell, step_cost)
            if new_cost >= cost_so_far.get(next_cell, float("inf")):
                continue
            cost_so_far[next_cell] = new_cost
            came_from[next_cell] = current
            heapq.heappush(frontier, (new_cost + heuristic(next_cell), new_cost, next_cell))
    return []


def _path_waypoints(path: list[tuple[int, int]], spacing: int = 2) -> tuple[tuple[float, float], ...]:
    """Downsample a dense grid route while retaining every bend."""
    if not path:
        return ()
    waypoints: list[tuple[float, float]] = [(float(path[0][0]), float(path[0][1]))]
    last_direction: tuple[int, int] | None = None
    last_added = 0
    for index in range(1, len(path)):
        previous = path[index - 1]
        current = path[index]
        direction = (int(np.sign(current[0] - previous[0])), int(np.sign(current[1] - previous[1])))
        if last_direction is not None and direction != last_direction:
            bend = path[index - 1]
            if waypoints[-1] != (float(bend[0]), float(bend[1])):
                waypoints.append((float(bend[0]), float(bend[1])))
                last_added = index - 1
        if index - last_added >= spacing and waypoints[-1] != (float(current[0]), float(current[1])):
            waypoints.append((float(current[0]), float(current[1])))
            last_added = index
        last_direction = direction
    endpoint = (float(path[-1][0]), float(path[-1][1]))
    if waypoints[-1] != endpoint:
        waypoints.append(endpoint)
    return tuple(waypoints)


def extract_wall_boxes(mask: np.ndarray, min_area: int = 8) -> tuple[tuple[float, float, float, float], ...]:
    """Find connected wall components and return (x0, y0, x1, y1) boxes."""
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    boxes = []
    for y0, x0 in zip(*np.where(mask & ~visited)):
        if visited[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        visited[y0, x0] = True
        points = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= next_y < height and 0 <= next_x < width and mask[next_y, next_x] and not visited[next_y, next_x]:
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))
        if len(points) >= min_area:
            points_array = np.asarray(points)
            boxes.append((float(points_array[:, 1].min()), float(points_array[:, 0].min()), float(points_array[:, 1].max()), float(points_array[:, 0].max())))
    return tuple(boxes)


def _collision_prefix(positions: np.ndarray, wall_mask: np.ndarray, agent_radius: int = 3) -> np.ndarray:
    """Return whether each imagined step or an earlier step intersects a wall."""
    occupied = _dilate(wall_mask, agent_radius)
    finite = np.isfinite(positions).all(axis=-1)
    xs = np.rint(np.nan_to_num(positions[..., 0], nan=-1)).astype(np.int64)
    ys = np.rint(np.nan_to_num(positions[..., 1], nan=-1)).astype(np.int64)
    inside = (xs >= 0) & (xs < occupied.shape[1]) & (ys >= 0) & (ys < occupied.shape[0])
    clipped_xs = np.clip(xs, 0, occupied.shape[1] - 1)
    clipped_ys = np.clip(ys, 0, occupied.shape[0] - 1)
    collisions = (~finite) | (~inside) | occupied[clipped_ys, clipped_xs]
    return np.maximum.accumulate(collisions, axis=1)


def _detour_sequence(
    model: PocketWorldModel,
    start: torch.Tensor,
    start_position: np.ndarray,
    goal: tuple[float, float],
    waypoint: tuple[float, float],
    horizon: int,
) -> list[int]:
    """Use the compact dynamics to steer through a waypoint and then to goal."""
    state = model.state_from_latent(model.encode(start))
    targets = (waypoint, goal)
    target_index = 0
    actions = []
    for _ in range(horizon):
        position = state[0, :2].cpu().numpy() * 64.0
        target = np.asarray(targets[target_index], dtype=np.float32)
        delta = target - position
        if np.linalg.norm(delta) <= 5.0 and target_index < len(targets) - 1:
            target_index += 1
            target = np.asarray(targets[target_index], dtype=np.float32)
            delta = target - position
        if abs(delta[0]) >= abs(delta[1]):
            action = 3 if delta[0] >= 0 else 2
        else:
            action = 1 if delta[1] >= 0 else 0
        actions.append(action)
        state = model.state_transition(state, torch.tensor([action], device=state.device))
    return actions


def _route_sequence(
    model: PocketWorldModel,
    start: torch.Tensor,
    start_position: np.ndarray,
    targets: tuple[tuple[float, float], ...],
    horizon: int,
    tolerance: float = 1.5,
    start_velocity: np.ndarray | None = None,
) -> list[int]:
    """Track generic waypoints using the learned compact kinematics."""
    state = model.state_from_latent(model.encode(start))
    known_position = torch.as_tensor(start_position / 64.0, device=state.device, dtype=state.dtype)[None]
    state = torch.cat((known_position, state[..., 2:]), dim=-1)
    if start_velocity is not None:
        known_velocity = torch.as_tensor(start_velocity / 3.0, device=state.device, dtype=state.dtype)[None]
        state = torch.cat((state[..., :2], known_velocity), dim=-1)
    target_index = 0
    actions: list[int] = []
    for _ in range(horizon):
        position = state[0, :2].detach().cpu().numpy() * 64.0
        velocity = state[0, 2:].detach().cpu().numpy() * 3.0
        target = np.asarray(targets[target_index], dtype=np.float32)
        delta = target - position
        if np.linalg.norm(delta) <= tolerance and target_index < len(targets) - 1:
            target_index += 1
            target = np.asarray(targets[target_index], dtype=np.float32)
            delta = target - position
        control = delta - 3.0 * velocity
        if abs(control[0]) >= abs(control[1]):
            action = 3 if control[0] >= 0 else 2
        else:
            action = 1 if control[1] >= 0 else 0
        actions.append(action)
        state = model.state_transition(state, torch.tensor([action], device=state.device))
    return actions


def _learned_waypoint_templates(
    model: PocketWorldModel,
    start: torch.Tensor,
    start_position: np.ndarray,
    goal: tuple[float, float],
    horizon: int,
    start_velocity: np.ndarray | None = None,
) -> list[list[int]]:
    """Generate map-agnostic two-bend routes for learned risk scoring.

    The proposals never inspect wall pixels. They span lateral offsets around
    the direct route; the learned collision model decides which routes are safe.
    """
    goal_array = np.asarray(goal, dtype=np.float32)
    delta = goal_array - start_position
    length = float(np.linalg.norm(delta))
    if length < 1e-6:
        return []
    direction = delta / length
    perpendicular = np.asarray((-direction[1], direction[0]), dtype=np.float32)
    proposals = []
    for offset in (-32.0, -26.0, -20.0, -12.0, 12.0, 20.0, 26.0, 32.0):
        before = np.clip(start_position + 0.28 * delta + offset * perpendicular, 5.5, 58.5)
        after = np.clip(start_position + 0.72 * delta + offset * perpendicular, 5.5, 58.5)
        targets = (tuple(before.tolist()), tuple(after.tolist()), tuple(goal_array.tolist()))
        proposals.append(_route_sequence(model, start, start_position, targets, horizon, start_velocity=start_velocity))
    return proposals


def _detour_templates(
    model: PocketWorldModel,
    start: torch.Tensor,
    start_position: np.ndarray,
    goal: tuple[float, float],
    wall_mask: np.ndarray,
    horizon: int,
) -> list[list[int]]:
    """Generate top/bottom or left/right waypoint proposals for crossing walls."""
    boxes = extract_wall_boxes(wall_mask)
    proposals = []
    goal_array = np.asarray(goal, dtype=np.float32)
    for x0, y0, x1, y1 in boxes:
        if start_position[0] < x0 and goal_array[0] > x1 and y0 <= start_position[1] <= y1:
            proposals.extend([
                _detour_sequence(model, start, start_position, goal, (start_position[0], max(4.0, y0 - 5.0)), horizon),
                _detour_sequence(model, start, start_position, goal, (start_position[0], min(60.0, y1 + 5.0)), horizon),
            ])
        elif start_position[0] > x1 and goal_array[0] < x0 and y0 <= start_position[1] <= y1:
            proposals.extend([
                _detour_sequence(model, start, start_position, goal, (start_position[0], max(4.0, y0 - 5.0)), horizon),
                _detour_sequence(model, start, start_position, goal, (start_position[0], min(60.0, y1 + 5.0)), horizon),
            ])
        elif start_position[1] < y0 and goal_array[1] > y1 and x0 <= start_position[0] <= x1:
            proposals.extend([
                _detour_sequence(model, start, start_position, goal, (max(4.0, x0 - 5.0), start_position[1]), horizon),
                _detour_sequence(model, start, start_position, goal, (min(60.0, x1 + 5.0), start_position[1]), horizon),
            ])
        elif start_position[1] > y1 and goal_array[1] < y0 and x0 <= start_position[0] <= x1:
            proposals.extend([
                _detour_sequence(model, start, start_position, goal, (max(4.0, x0 - 5.0), start_position[1]), horizon),
                _detour_sequence(model, start, start_position, goal, (min(60.0, x1 + 5.0), start_position[1]), horizon),
            ])
    return proposals


def _wall_aware_route_templates(
    model: PocketWorldModel,
    start: torch.Tensor,
    start_position: np.ndarray,
    goal: tuple[float, float],
    wall_mask: np.ndarray,
    horizon: int,
    start_velocity: np.ndarray | None = None,
) -> list[list[int]]:
    """Build route-following proposals from the observed wall geometry.

    The wall mask is inflated by the agent radius plus one pixel of clearance,
    matching the environment's footprint collision rule.  Two biased A* runs
    provide top/bottom alternatives for a barrier; if both runs collapse to
    the same route, the duplicate is removed.
    """
    occupied = _dilate(wall_mask, radius=4)
    routes: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for preference in ("top", "bottom"):
        path = _astar_path(occupied, tuple(start_position), goal, vertical_preference=preference)
        if not path:
            continue
        # Keep only bends and endpoints.  Dense grid waypoints make the
        # inertial controller brake at every cell and waste the short horizon;
        # straight grid segments are safe to traverse as a single target.
        targets = _path_waypoints(path, spacing=len(path) + 1)
        actions = _route_sequence(
            model,
            start,
            start_position,
            targets,
            horizon,
            tolerance=3.5,
            start_velocity=start_velocity,
        )
        key = tuple(actions)
        if key not in seen:
            seen.add(key)
            routes.append(actions)
    return routes


def extract_agent_position(frame: np.ndarray) -> np.ndarray:
    """Locate the mint-green agent in a CHW uint8 or BCHW float frame."""
    if frame.dtype != np.uint8:
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
    frame = frame.astype(np.int16)
    batched = frame.ndim == 4
    if not batched:
        frame = frame[None]
    positions = []
    for sample in frame:
        mask = (sample[1] > sample[0] + 25) & (sample[1] > sample[2] + 15)
        ys, xs = np.where(mask)
        positions.append((xs.mean(), ys.mean()) if len(xs) else (np.nan, np.nan))
    result = np.asarray(positions, dtype=np.float32)
    return result if batched else result[0]


def estimate_agent_velocity(
    observation_history: Sequence[np.ndarray],
    max_speed: float = 2.3,
) -> np.ndarray:
    """Estimate current pixel velocity from recent observable agent positions."""
    if len(observation_history) < 2:
        return np.zeros(2, dtype=np.float32)
    positions = np.asarray([extract_agent_position(frame) for frame in observation_history[-4:]], dtype=np.float32)
    valid = np.isfinite(positions).all(axis=1)
    positions = positions[valid]
    if len(positions) < 2:
        return np.zeros(2, dtype=np.float32)
    differences = np.diff(positions, axis=0)
    if np.linalg.norm(differences[-1]) <= 0.15:
        return np.zeros(2, dtype=np.float32)
    weights = np.arange(1, len(differences) + 1, dtype=np.float32)
    velocity = np.average(differences, axis=0, weights=weights)
    speed = float(np.linalg.norm(velocity))
    if speed > max_speed:
        velocity *= max_speed / speed
    return velocity.astype(np.float32)


def wall_context_shift_score(observation: np.ndarray, mismatch_scale: float = 0.05) -> float:
    """Measure visible wall-layout mismatch against the training-map prior."""
    if mismatch_scale <= 0:
        raise ValueError("mismatch_scale must be positive")
    mismatch = float(np.mean(extract_wall_mask(observation) != _DEFAULT_WALL_MASK))
    return mismatch / mismatch_scale


def _route_completion_probabilities(
    goal_distances: np.ndarray,
    collision_prefix: np.ndarray,
    goal_radius: float = 4.0,
    distance_temperature: float = 1.5,
) -> np.ndarray:
    """Estimate route completion from endpoint reachability and predicted risk."""
    endpoint_probability = 1.0 / (
        1.0 + np.exp((goal_distances[:, -1] - goal_radius) / distance_temperature)
    )
    terminal_risk = np.clip(collision_prefix[:, -1], 0.0, 1.0)
    return endpoint_probability * (1.0 - terminal_risk)


@torch.no_grad()
def predictive_shift_score(
    model: PocketWorldModel,
    previous_observation: np.ndarray,
    action: int,
    next_observation: np.ndarray,
    observation_history: Sequence[np.ndarray] | None = None,
) -> float:
    """Score an observed transition by its calibrated standardized innovation.

    The score is available after the next RGB frame arrives and does not use
    the simulator state or an OOD label. A high value means the observed
    position/velocity transition is unlikely under the learned transition
    distribution. It is deliberately a scalar monitor, not a probability.
    """
    previous_position = extract_agent_position(previous_observation)
    next_position = extract_agent_position(next_observation)
    if not np.isfinite(previous_position).all() or not np.isfinite(next_position).all():
        return float("inf")
    history = list(observation_history) if observation_history is not None else [previous_observation]
    if not history or not np.array_equal(history[-1], previous_observation):
        history.append(previous_observation)
    frame = torch.from_numpy(previous_observation[None]).float() / 255.0
    latent = model.encode(frame)
    state = model.state_from_latent(latent)
    position = torch.as_tensor(previous_position / 64.0, dtype=state.dtype, device=state.device)[None]
    velocity = torch.as_tensor(
        estimate_agent_velocity(history) / 3.0,
        dtype=state.dtype,
        device=state.device,
    )[None]
    state = torch.cat((position, velocity), dim=-1)
    action_tensor = torch.tensor([int(action)], device=state.device)
    mean, std = model.transition_state_stats(latent, state, action_tensor)
    observed_velocity = (next_position - previous_position) / 3.0
    target = torch.as_tensor(
        np.concatenate((next_position / 64.0, observed_velocity)),
        dtype=state.dtype,
        device=state.device,
    )[None]
    standardized = (target - mean) / std.clamp_min(1e-6)
    transition_score = float(torch.sqrt(standardized.square().mean()).item())
    context_score = wall_context_shift_score(previous_observation)
    return max(transition_score, context_score)


@torch.no_grad()
def random_shooting(
    model: PocketWorldModel,
    observation: np.ndarray,
    goal: tuple[float, float],
    horizon: int = 12,
    candidates: int = 256,
    device: str = "cpu",
    guided_fraction: float = 0.35,
    collision_aware: bool = False,
    learned_collision: bool = False,
    hybrid_collision: bool = False,
    observation_history: Sequence[np.ndarray] | None = None,
    use_learned_velocity: bool = False,
    uncertainty_radius_px: float = 0.0,
    uncertainty_growth_px: float = 0.0,
    probabilistic_uncertainty: bool = False,
    uncertainty_samples: int = 16,
    robust_candidates: int = 64,
    route_objective: bool = False,
    route_execution_horizon: int | None = None,
    wall_aware_route: bool = False,
) -> PlanResult:
    model.eval()
    start = torch.from_numpy(observation[None]).float().to(device) / 255.0
    actions = torch.randint(0, 4, (candidates, horizon), device=device)
    delta = np.asarray(goal, dtype=np.float32) - extract_agent_position(observation)
    preferred_action = 3 if abs(delta[0]) >= abs(delta[1]) and delta[0] >= 0 else 2 if abs(delta[0]) >= abs(delta[1]) else 1 if delta[1] >= 0 else 0
    guided = torch.rand((candidates, horizon), device=device) < guided_fraction
    actions = torch.where(guided, torch.full_like(actions, preferred_action), actions)
    starts = start.expand(candidates, -1, -1, -1)
    start_position = extract_agent_position(observation).astype(np.float32)
    normalized_start_positions = torch.as_tensor(start_position / 64.0, device=device, dtype=start.dtype).expand(candidates, -1)
    normalized_start_velocities = None
    start_velocity = None
    if observation_history is not None and len(observation_history) >= 2:
        if use_learned_velocity:
            history_tensor = torch.from_numpy(np.stack(observation_history)).float().to(device) / 255.0
            learned_velocity, _ = model.temporal_velocity_stats(history_tensor[None])
            start_velocity = (learned_velocity[0].cpu().numpy() * 3.0).astype(np.float32)
        else:
            start_velocity = estimate_agent_velocity(observation_history)
        normalized_start_velocities = torch.as_tensor(start_velocity / 3.0, device=device, dtype=start.dtype).expand(candidates, -1)
    collision_response = learned_collision or hybrid_collision
    imagined_positions = model.imagine_positions(
        starts,
        actions,
        collision_response=collision_response,
        visual_collision_guard=hybrid_collision,
        initial_position=normalized_start_positions,
        initial_velocity=normalized_start_velocities,
        probabilistic_uncertainty=False,
        uncertainty_samples=uncertainty_samples,
    ).cpu().numpy() * 64.0
    positions = np.concatenate((np.broadcast_to(start_position, (candidates, 1, 2)), imagined_positions), axis=1)
    goal_distances = np.linalg.norm(positions - np.asarray(goal), axis=-1)
    if collision_aware:
        wall_mask = extract_wall_mask(observation)
        if wall_aware_route:
            templates = _wall_aware_route_templates(
                model,
                start,
                start_position,
                goal,
                wall_mask,
                horizon,
                start_velocity=start_velocity,
            )
        else:
            templates = (
                _learned_waypoint_templates(model, start, start_position, goal, horizon, start_velocity=start_velocity)
                if learned_collision
                else _detour_templates(model, start, start_position, goal, wall_mask, horizon)
            )
        count = 0
        if templates:
            template_tensor = torch.as_tensor(templates, device=device, dtype=torch.long)
            count = min(len(templates), candidates)
            actions[:count] = template_tensor[:count]
            imagined_positions[:count] = model.imagine_positions(
                starts[:count],
                actions[:count],
                collision_response=collision_response,
                visual_collision_guard=hybrid_collision,
                initial_position=normalized_start_positions[:count],
                initial_velocity=None if normalized_start_velocities is None else normalized_start_velocities[:count],
                probabilistic_uncertainty=False,
                uncertainty_samples=uncertainty_samples,
            ).cpu().numpy() * 64.0
            positions[:count, 1:] = imagined_positions[:count]
            goal_distances[:count] = np.linalg.norm(positions[:count] - np.asarray(goal), axis=-1)
        if learned_collision or hybrid_collision:
            collision_probabilities = model.imagine_collision_probabilities(
                starts,
                actions,
                visual_collision_guard=hybrid_collision,
                initial_position=normalized_start_positions,
                initial_velocity=normalized_start_velocities,
                probabilistic_uncertainty=False,
                uncertainty_samples=uncertainty_samples,
            ).cpu().numpy()
            eligible = np.ones(candidates, dtype=bool)
            uncertainty_active = probabilistic_uncertainty or uncertainty_radius_px > 0 or uncertainty_growth_px > 0
            if learned_collision and uncertainty_active:
                point_risk = np.maximum.accumulate(collision_probabilities, axis=1)
                point_risk = np.concatenate((np.zeros((candidates, 1), dtype=np.float32), point_risk), axis=1)
                preliminary_scores = goal_distances + point_risk * 64.0
                shortlist_size = min(candidates, max(count, robust_candidates))
                ranked = np.argsort(np.min(preliminary_scores, axis=1))[:shortlist_size]
                shortlist = np.unique(np.concatenate((np.arange(count), ranked)))
                shortlist_tensor = torch.as_tensor(shortlist, device=device, dtype=torch.long)
                robust_positions = model.imagine_positions(
                    starts[shortlist_tensor],
                    actions[shortlist_tensor],
                    collision_response=True,
                    initial_position=normalized_start_positions[shortlist_tensor],
                    initial_velocity=None if normalized_start_velocities is None else normalized_start_velocities[shortlist_tensor],
                    uncertainty_radius_px=uncertainty_radius_px,
                    uncertainty_growth_px=uncertainty_growth_px,
                    probabilistic_uncertainty=probabilistic_uncertainty,
                    uncertainty_samples=uncertainty_samples,
                ).cpu().numpy() * 64.0
                imagined_positions[shortlist] = robust_positions
                positions[shortlist, 1:] = robust_positions
                goal_distances[shortlist] = np.linalg.norm(positions[shortlist] - np.asarray(goal), axis=-1)
                robust_probabilities = model.imagine_collision_probabilities(
                    starts[shortlist_tensor],
                    actions[shortlist_tensor],
                    initial_position=normalized_start_positions[shortlist_tensor],
                    initial_velocity=None if normalized_start_velocities is None else normalized_start_velocities[shortlist_tensor],
                    uncertainty_radius_px=uncertainty_radius_px,
                    uncertainty_growth_px=uncertainty_growth_px,
                    probabilistic_uncertainty=probabilistic_uncertainty,
                    uncertainty_samples=uncertainty_samples,
                ).cpu().numpy()
                collision_probabilities[shortlist] = robust_probabilities
                eligible.fill(False)
                eligible[shortlist] = True
            peak_risk = np.maximum.accumulate(collision_probabilities, axis=1)
            collision_prefix = np.concatenate((np.zeros((candidates, 1), dtype=np.float32), peak_risk), axis=1)
        else:
            collision_prefix = _collision_prefix(positions, wall_mask)
        if hybrid_collision or wall_aware_route:
            collision_prefix = np.maximum(
                collision_prefix,
                _collision_prefix(positions, wall_mask).astype(np.float32),
            )
    else:
        collision_prefix = np.zeros_like(goal_distances, dtype=np.float32)
    planning_scores = goal_distances + collision_prefix * 64.0
    safe_scores = np.where(np.isfinite(planning_scores), planning_scores, 1e6)
    if collision_aware and learned_collision and (probabilistic_uncertainty or uncertainty_radius_px > 0 or uncertainty_growth_px > 0):
        safe_scores[~eligible] = 1e6
    route_completion_probabilities = _route_completion_probabilities(goal_distances, collision_prefix)
    if route_objective:
        route_regressions = np.maximum(0.0, np.diff(goal_distances, axis=1)).sum(axis=1)
        route_scores = (
            goal_distances[:, -1]
            + 0.35 * goal_distances[:, 1:].mean(axis=1)
            + 0.50 * route_regressions
            + 64.0 * collision_prefix[:, -1]
            - 10.0 * route_completion_probabilities
        )
        route_scores = np.where(np.isfinite(route_scores), route_scores, 1e6)
        if collision_aware and learned_collision and (probabilistic_uncertainty or uncertainty_radius_px > 0 or uncertainty_growth_px > 0):
            route_scores[~eligible] = 1e6
        best = int(np.argmin(route_scores))
        reached = np.flatnonzero(safe_scores[best] <= 4.0)
        best_step = int(reached[0]) if len(reached) else horizon
        selected_route_score = float(route_scores[best])
    else:
        best = int(np.argmin(np.min(safe_scores, axis=1)))
        best_step = int(np.argmin(safe_scores[best]))
        selected_route_score = float(safe_scores[best, best_step])
    initial_distance = float(goal_distances[best, 0])
    if route_objective and route_execution_horizon is not None:
        best_step = min(best_step, max(1, int(route_execution_horizon)))
    if best_step == 0 and initial_distance > 4.0:
        best_step = 1
    return PlanResult(
        actions=actions[best, :best_step].cpu().numpy(),
        imagined_positions=positions[best, :best_step + 1],
        imagined_distance=float(goal_distances[best, best_step]),
        imagined_collision_risk=float(collision_prefix[best, best_step]),
        planning_score=selected_route_score,
        route_score=selected_route_score,
        route_progress=initial_distance - float(goal_distances[best, -1]),
        route_endpoint_distance=float(goal_distances[best, -1]),
        predicted_route_completion_probability=float(route_completion_probabilities[best]),
    )


@torch.no_grad()
def receding_horizon_plan(
    model: PocketWorldModel,
    observation: np.ndarray,
    goal: tuple[float, float],
    step_fn: Callable[[int], tuple[np.ndarray, float, bool, bool, dict]],
    max_steps: int = 40,
    rollout_horizon: int = 16,
    candidates: int = 512,
    collision_aware: bool = True,
    commit_steps: int = 1,
    preserve_route: bool = False,
    route_tolerance: float = 6.0,
    learned_collision: bool = False,
    hybrid_collision: bool = False,
    use_history_velocity: bool = False,
    use_learned_velocity: bool = False,
    uncertainty_radius_px: float = 0.0,
    uncertainty_growth_px: float = 0.0,
    probabilistic_uncertainty: bool = False,
    uncertainty_samples: int = 16,
    route_objective: bool = False,
    shift_threshold: float | None = None,
    route_execution_horizon: int | None = None,
    alignment_fallback_threshold: float | None = None,
) -> RecedingHorizonResult:
    """Replan after every real action and return the closed-loop execution trace."""
    current_observation = observation
    executed_actions = []
    first_plan_distance = float("nan")
    first_plan_route_distance = float("nan")
    initial_position = extract_agent_position(observation)
    final_info: dict = {
        "position": initial_position.copy(),
        "goal": np.asarray(goal, dtype=np.float32),
        "distance_to_goal": float(np.linalg.norm(initial_position - np.asarray(goal))),
        "collision": False,
    }
    pending_plan: PlanResult | None = None
    pending_index = 0
    observation_history = [observation]
    collision_count = 0
    replans = 0
    alignment_errors: list[float] = []
    shift_scores: list[float] = []
    shift_detected_count = 0
    first_plan_route_completion_probability = float("nan")
    alignment_fallback_trigger_count = 0
    fallback_active = False
    fallback_steps = 0
    for step in range(max_steps):
        if pending_plan is None or pending_index >= len(pending_plan.actions):
            replans += 1
            pending_plan = random_shooting(
                model,
                current_observation,
                goal,
                horizon=min(rollout_horizon, max_steps - step),
                candidates=candidates,
                collision_aware=collision_aware,
                learned_collision=learned_collision and not fallback_active,
                hybrid_collision=hybrid_collision or fallback_active,
                observation_history=observation_history if use_history_velocity else None,
                use_learned_velocity=use_learned_velocity and not fallback_active,
                uncertainty_radius_px=uncertainty_radius_px,
                uncertainty_growth_px=uncertainty_growth_px,
                probabilistic_uncertainty=probabilistic_uncertainty,
                uncertainty_samples=uncertainty_samples,
                # Once alignment breaks, score complete wall-aware routes
                # instead of selecting a locally short collision-stop.
                route_objective=route_objective or fallback_active,
                route_execution_horizon=route_execution_horizon if not fallback_active else None,
                wall_aware_route=fallback_active,
            )
            pending_index = 0
        plan = pending_plan
        if step == 0:
            first_plan_distance = plan.imagined_distance
            first_plan_route_distance = plan.route_endpoint_distance
            first_plan_route_completion_probability = plan.predicted_route_completion_probability
        if len(plan.actions) == 0:
            break
        actions_to_execute = plan.actions[pending_index:pending_index + max(1, commit_steps)]
        for action_value in actions_to_execute:
            action = int(action_value)
            previous_observation = current_observation
            previous_history = list(observation_history)
            current_observation, _, terminated, truncated, final_info = step_fn(action)
            observation_history.append(current_observation)
            observation_history = observation_history[-4:]
            executed_actions.append(action)
            collision_count += int(final_info.get("collision", False))
            pending_index += 1
            expected_index = min(pending_index, len(plan.imagined_positions) - 1)
            actual_position = extract_agent_position(current_observation)
            expected_position = plan.imagined_positions[expected_index]
            if np.isfinite(actual_position).all() and np.isfinite(expected_position).all():
                alignment_error = float(np.linalg.norm(actual_position - expected_position))
                alignment_errors.append(alignment_error)
                if (
                    alignment_fallback_threshold is not None
                    and not fallback_active
                    and alignment_error >= alignment_fallback_threshold
                ):
                    alignment_fallback_trigger_count += 1
                    fallback_active = True
                    pending_plan = None
            if fallback_active:
                fallback_steps += 1
            if shift_threshold is not None:
                shift_score = predictive_shift_score(
                    model,
                    previous_observation,
                    action,
                    current_observation,
                    previous_history,
                )
                shift_scores.append(shift_score)
                if shift_score >= shift_threshold:
                    shift_detected_count += 1
                    pending_plan = None
            if terminated or truncated:
                break
            if preserve_route:
                deviated = not np.all(np.isfinite(actual_position)) or np.linalg.norm(actual_position - expected_position) > route_tolerance
                if final_info.get("collision", False) or deviated:
                    pending_plan = None
                    break
        if not preserve_route:
            pending_plan = None
        if final_info.get("distance_to_goal", float("inf")) <= 4.0 or terminated or truncated:
            break
    return RecedingHorizonResult(
        actions=np.asarray(executed_actions, dtype=np.int64),
        first_plan_distance=first_plan_distance,
        first_plan_route_distance=first_plan_route_distance,
        final_observation=current_observation,
        final_info=final_info,
        collision_count=collision_count,
        replans=replans,
        route_alignment_error_px=float(np.mean(alignment_errors)) if alignment_errors else 0.0,
        max_route_alignment_error_px=float(np.max(alignment_errors)) if alignment_errors else 0.0,
        mean_shift_score=float(np.mean(shift_scores)) if shift_scores else 0.0,
        max_shift_score=float(np.max(shift_scores)) if shift_scores else 0.0,
        shift_detected_count=shift_detected_count,
        first_plan_route_completion_probability=(
            first_plan_route_completion_probability
            if np.isfinite(first_plan_route_completion_probability)
            else 0.0
        ),
        alignment_fallback_trigger_count=alignment_fallback_trigger_count,
        fallback_steps=fallback_steps,
    )
