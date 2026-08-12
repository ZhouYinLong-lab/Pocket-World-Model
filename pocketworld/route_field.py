"""A learned coarse distance field for general route representation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .planner import _dilate, extract_agent_position, extract_wall_mask

FIELD_GRID = 16


def coarse_wall_signature(observation: np.ndarray, grid_size: int = FIELD_GRID) -> np.ndarray:
    """Encode only coarse RGB wall occupancy for layout-shift monitoring."""
    if grid_size < 1 or 64 % grid_size != 0:
        raise ValueError("grid_size must be a positive divisor of 64")
    mask = extract_wall_mask(observation)
    block = 64 // grid_size
    return mask.reshape(grid_size, block, grid_size, block).any(axis=(1, 3)).astype(np.float32).ravel()


def wall_layout_shift_score(
    observation: np.ndarray,
    reference_signatures: np.ndarray,
) -> float:
    """Return nearest-reference Hamming distance for the visible wall layout."""
    references = np.asarray(reference_signatures, dtype=np.float32)
    signature = coarse_wall_signature(observation)
    if references.ndim != 2 or references.shape[1] != signature.size or len(references) < 1:
        raise ValueError("reference_signatures must have shape [N, 256]")
    return float(np.min(np.mean(references != signature[None], axis=1)))


def estimate_action_velocity(
    observation_history: list[np.ndarray],
    action_history: list[int] | None = None,
    max_speed: float = 2.3,
) -> np.ndarray:
    """Fuse RGB finite differences with velocity reconstructed from own actions."""
    from .planner import estimate_agent_velocity

    rgb_velocity = estimate_agent_velocity(observation_history, max_speed=max_speed)
    if not action_history:
        return rgb_velocity
    if len(observation_history) >= 2:
        current = extract_agent_position(observation_history[-1]).astype(np.float32)
        previous = extract_agent_position(observation_history[-2]).astype(np.float32)
        if np.isfinite(current).all() and np.isfinite(previous).all():
            # A collision zeroes simulator velocity. RGB exposes this as a
            # repeated position even though the action history still contains
            # the old push; reset the latent action velocity before planning.
            if np.linalg.norm(current - previous) <= 0.15:
                return np.zeros(2, dtype=np.float32)
    directions = np.asarray(((0, -1), (0, 1), (-1, 0), (1, 0)), dtype=np.float32)
    predicted = np.zeros(2, dtype=np.float32)
    for action in action_history[-16:]:
        predicted *= 0.84
        predicted += 0.75 * directions[int(action)]
        speed = float(np.linalg.norm(predicted))
        if speed > max_speed:
            predicted *= max_speed / speed
    fused = 0.70 * predicted + 0.30 * rgb_velocity
    speed = float(np.linalg.norm(fused))
    if speed > max_speed:
        fused *= max_speed / speed
    return fused.astype(np.float32)


def _coarse_transition_is_safe(
    occupied: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    block: int,
) -> bool:
    """Check the whole center-to-center coarse edge, not only its endpoint."""
    start_xy = np.asarray(((start[0] + 0.5) * block, (start[1] + 0.5) * block), dtype=np.float32)
    end_xy = np.asarray(((end[0] + 0.5) * block, (end[1] + 0.5) * block), dtype=np.float32)
    samples = np.linspace(start_xy, end_xy, max(2, block * 2))
    xs = np.clip(np.rint(samples[:, 0]).astype(int), 0, occupied.shape[1] - 1)
    ys = np.clip(np.rint(samples[:, 1]).astype(int), 0, occupied.shape[0] - 1)
    return not bool(np.any(occupied[ys, xs]))


def _pixel_distance_field(occupied: np.ndarray, goal: tuple[float, float]) -> np.ndarray:
    """Compute a teacher distance field on the footprint-inflated RGB grid."""
    height, width = occupied.shape
    distances = np.full((height, width), np.inf, dtype=np.float32)
    gx = int(np.clip(round(goal[0]), 0, width - 1))
    gy = int(np.clip(round(goal[1]), 0, height - 1))
    if occupied[gy, gx]:
        free = np.argwhere(~occupied)
        if len(free) == 0:
            return distances
        nearest = np.argmin((free[:, 1] - gx) ** 2 + (free[:, 0] - gy) ** 2)
        gy, gx = (int(value) for value in free[nearest])
    queue = [(gy, gx)]
    distances[gy, gx] = 0.0
    cursor = 0
    while cursor < len(queue):
        y, x = queue[cursor]
        cursor += 1
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if not (0 <= next_y < height and 0 <= next_x < width):
                continue
            if occupied[next_y, next_x] or np.isfinite(distances[next_y, next_x]):
                continue
            distances[next_y, next_x] = distances[y, x] + 1.0
            queue.append((next_y, next_x))
    return distances


def _pixel_clearance_field(occupied: np.ndarray) -> np.ndarray:
    """Return Manhattan distance to the nearest inflated wall pixel."""
    height, width = occupied.shape
    distances = np.full((height, width), np.inf, dtype=np.float32)
    queue: list[tuple[int, int]] = []
    for y, x in zip(*np.where(occupied)):
        distances[y, x] = 0.0
        queue.append((int(y), int(x)))
    cursor = 0
    while cursor < len(queue):
        y, x = queue[cursor]
        cursor += 1
        for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if not (0 <= next_y < height and 0 <= next_x < width):
                continue
            if np.isfinite(distances[next_y, next_x]):
                continue
            distances[next_y, next_x] = distances[y, x] + 1.0
            queue.append((next_y, next_x))
    return distances


def _rgb_kinematic_landing(
    position: np.ndarray,
    velocity: np.ndarray,
    action: int,
    wall_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Predict one environment step with the exact observable collision rule.

    The RGB controller must not use a stricter square dilation than the
    simulator when comparing planner variants.  This helper mirrors
    ``PocketWorldEnv._collides`` in continuous coordinates while keeping the
    environment state private to the evaluator.
    """
    directions = np.asarray(((0, -1), (0, 1), (-1, 0), (1, 0)), dtype=np.float32)
    next_velocity = 0.84 * velocity + 0.75 * directions[int(action)]
    speed = float(np.linalg.norm(next_velocity))
    if speed > 2.3:
        next_velocity *= 2.3 / speed
    next_position = position + next_velocity
    wall_y, wall_x = np.where(wall_mask)
    x, y = next_position
    inside = 3 <= x < 61 and 3 <= y < 61
    intersects_wall = bool(
        np.any(
            (x >= wall_x - 3.0)
            & (x <= wall_x + 4.0)
            & (y >= wall_y - 3.0)
            & (y <= wall_y + 4.0)
        )
    )
    return next_position, next_velocity, bool(inside and not intersects_wall)


def route_field_targets(
    observations: np.ndarray,
    goals: np.ndarray,
    grid_size: int = FIELD_GRID,
    clearance_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized coarse distance fields and free-cell masks."""
    frames = np.asarray(observations)
    if frames.ndim == 3:
        frames = frames[None]
    goal_values = np.asarray(goals, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[1:] != (3, 64, 64):
        raise ValueError("observations must have shape [samples, 3, 64, 64]")
    if goal_values.shape != (len(frames), 2):
        raise ValueError("goals must have shape [samples, 2]")
    if 64 % grid_size != 0:
        raise ValueError("grid_size must divide 64")
    if clearance_weight < 0:
        raise ValueError("clearance_weight must be non-negative")
    block = 64 // grid_size
    fields = np.ones((len(frames), grid_size, grid_size), dtype=np.float32)
    valid = np.zeros_like(fields)
    for index, (frame, goal) in enumerate(zip(frames, goal_values)):
        occupied = _dilate(extract_wall_mask(frame), radius=4)
        distance = _pixel_distance_field(occupied, tuple(map(float, goal)))
        clearance = _pixel_clearance_field(occupied) if clearance_weight else None
        for row in range(grid_size):
            for column in range(grid_size):
                patch = distance[row * block:(row + 1) * block, column * block:(column + 1) * block]
                finite = patch[np.isfinite(patch)]
                if len(finite):
                    cost = finite
                    if clearance is not None:
                        clearance_patch = clearance[row * block:(row + 1) * block, column * block:(column + 1) * block]
                        clearance_values = clearance_patch[np.isfinite(patch)]
                        risk = np.maximum(0.0, 5.0 - clearance_values)
                        cost = finite + float(clearance_weight) * risk
                    fields[index, row, column] = float(np.clip(cost.min() / 128.0, 0.0, 1.0))
                    valid[index, row, column] = 1.0
    return torch.from_numpy(fields), torch.from_numpy(valid)


def _field_inputs(observations: np.ndarray, goals: np.ndarray) -> torch.Tensor:
    frames = np.asarray(observations)
    if frames.ndim == 3:
        frames = frames[None]
    goals = np.asarray(goals, dtype=np.float32)
    occupancy = []
    starts = []
    goal_maps = []
    for frame, goal in zip(frames, goals):
        wall = extract_wall_mask(frame)
        block = 64 // FIELD_GRID
        coarse = wall[:FIELD_GRID * block, :FIELD_GRID * block].reshape(FIELD_GRID, block, FIELD_GRID, block).max(axis=(1, 3))
        occupancy.append(coarse.astype(np.float32))
        start = extract_agent_position(frame).astype(np.float32)
        start_cell = np.rint(np.nan_to_num(start, nan=0.0) / block).astype(int).clip(0, FIELD_GRID - 1)
        goal_cell = np.rint(goal / block).astype(int).clip(0, FIELD_GRID - 1)
        start_map = np.zeros((FIELD_GRID, FIELD_GRID), dtype=np.float32)
        goal_map = np.zeros_like(start_map)
        start_map[start_cell[1], start_cell[0]] = 1.0
        goal_map[goal_cell[1], goal_cell[0]] = 1.0
        starts.append(start_map)
        goal_maps.append(goal_map)
    return torch.from_numpy(np.stack((occupancy, np.asarray(starts), np.asarray(goal_maps)), axis=1))


class RouteFieldPolicy(nn.Module):
    """Predict a coarse route distance field from RGB wall geometry and goal."""

    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        self.grid_size = FIELD_GRID
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs).squeeze(1)

    def fit(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        epochs: int = 240,
        seed: int = 7,
        clearance_weight: float = 0.0,
    ) -> dict[str, float | int]:
        frames = np.asarray(observations, dtype=np.uint8)
        inputs = _field_inputs(frames, goals)
        targets, valid = route_field_targets(
            frames, goals, clearance_weight=clearance_weight
        )
        if len(frames) < 8:
            raise ValueError("route field training data must contain at least eight samples")
        torch.manual_seed(seed)
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=1e-5)
        losses: list[float] = []
        for _ in range(max(1, int(epochs))):
            prediction = self(inputs)
            error = nn.functional.smooth_l1_loss(prediction, targets, reduction="none")
            loss = (error * valid).sum() / valid.sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            prediction = self(inputs)
            mae = (torch.abs(prediction - targets) * valid).sum() / valid.sum().clamp_min(1.0)
        return {
            "epochs": int(epochs),
            "samples": int(len(frames)),
            "final_loss": losses[-1],
            "mean_field_error_px": float(mae * 128.0),
            "clearance_weight": float(clearance_weight),
        }

    @torch.no_grad()
    def predict_field(self, observation: np.ndarray, goal: tuple[float, float]) -> np.ndarray:
        inputs = _field_inputs(
            np.asarray(observation)[None], np.asarray(goal, dtype=np.float32)[None]
        )
        return self(inputs)[0].cpu().numpy().astype(np.float32)

    def save(self, destination: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.state_dict(), "metadata": metadata or {}}, path)
        return path

    @classmethod
    def load(cls, checkpoint: str | Path) -> "RouteFieldPolicy":
        payload = torch.load(checkpoint, map_location="cpu")
        model = cls()
        model.load_state_dict(payload["model"], strict=True)
        return model


def field_waypoints(
    observation: np.ndarray,
    goal: tuple[float, float],
    field: np.ndarray,
    rgb_guard: bool = False,
    beam_width: int = 1,
) -> tuple[tuple[float, float], ...]:
    """Follow the predicted field without A* at evaluation time.

    ``beam_width=1`` is the greedy baseline.  A small fixed beam is a separate
    learned-planner ablation: it can recover from one locally overestimated
    neighbor without turning into a privileged global search.
    """
    values = np.asarray(field, dtype=np.float32)
    if values.shape != (FIELD_GRID, FIELD_GRID):
        raise ValueError("field must have shape [16, 16]")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    position = extract_agent_position(observation).astype(np.float32)
    block = 64 // FIELD_GRID
    current = np.rint(np.nan_to_num(position, nan=0.0) / block).astype(int).clip(0, FIELD_GRID - 1)
    target = np.rint(np.asarray(goal, dtype=np.float32) / block).astype(int).clip(0, FIELD_GRID - 1)
    occupied = _dilate(extract_wall_mask(observation), radius=4)
    beam: list[tuple[float, tuple[tuple[int, int], ...]]] = [(0.0, ((int(current[0]), int(current[1])),))]
    reached: tuple[tuple[int, int], ...] | None = None
    for _ in range(FIELD_GRID * FIELD_GRID * 2):
        expanded: list[tuple[float, tuple[tuple[int, int], ...]]] = []
        for score, path in beam:
            x, y = path[-1]
            if (x, y) == (int(target[0]), int(target[1])):
                reached = path
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < FIELD_GRID and 0 <= ny < FIELD_GRID):
                    continue
                if (nx, ny) in path:
                    continue
                if rgb_guard and not _coarse_transition_is_safe(
                    occupied, (x, y), (nx, ny), block
                ):
                    continue
                goal_distance = abs(nx - int(target[0])) + abs(ny - int(target[1]))
                expanded.append((
                    score + float(values[ny, nx]) + 0.002 * goal_distance,
                    path + ((nx, ny),),
                ))
        if reached is not None or not expanded:
            break
        expanded.sort(key=lambda item: (item[0], len(item[1])))
        beam = expanded[: int(beam_width)]
    if reached is None:
        reached = min(
            beam,
            key=lambda item: (
                abs(item[1][-1][0] - int(target[0]))
                + abs(item[1][-1][1] - int(target[1])),
                item[0],
            ),
        )[1] if beam else ((int(current[0]), int(current[1])),)
    route = [
        (
            float(np.clip((x + 0.5) * block, 5.0, 59.0)),
            float(np.clip((y + 0.5) * block, 5.0, 59.0)),
        )
        for x, y in reached[1:]
    ]
    route.append(tuple(map(float, np.clip(np.asarray(goal, dtype=np.float32), 5.0, 59.0))))
    return tuple(route)


def conservative_field_action(
    observation: np.ndarray,
    target: tuple[float, float],
    observation_history: list[np.ndarray] | None = None,
) -> int:
    """Choose a waypoint action with a one-step RGB collision shield.

    The ordinary waypoint controller estimates velocity from RGB history. This
    variant evaluates every action against a small velocity envelope around
    that estimate and only uses the target direction as a tie-breaker. It is a
    local safety ablation, not a global planner and does not call A*.
    """
    from .planner import estimate_agent_velocity

    position = extract_agent_position(observation).astype(np.float32)
    velocity = estimate_agent_velocity(observation_history or [observation], max_speed=2.5)
    target_array = np.asarray(target, dtype=np.float32)
    directions = np.asarray(((0, -1), (0, 1), (-1, 0), (1, 0)), dtype=np.float32)
    # The simulator uses an inclusive circular footprint around a 3px agent;
    # a 4px square dilation is the conservative observable approximation used
    # by the global planner and prevents boundary-touching false negatives.
    occupied = _dilate(extract_wall_mask(observation), radius=4)
    preferred = int(np.argmax(directions @ (target_array - position)))
    safe_actions: list[int] = []
    for action, direction in enumerate(directions):
        safe = True
        for velocity_scale in (0.75, 1.0, 1.25):
            next_velocity = 0.84 * velocity * velocity_scale + 0.75 * direction
            speed = float(np.linalg.norm(next_velocity))
            if speed > 2.3:
                next_velocity *= 2.3 / speed
            point = np.rint(position + next_velocity).astype(int)
            if not (3 <= point[0] < 61 and 3 <= point[1] < 61):
                safe = False
                break
            if occupied[point[1], point[0]]:
                safe = False
                break
        if safe:
            safe_actions.append(action)
    if preferred in safe_actions:
        return preferred
    return min(
        safe_actions,
        key=lambda action: float(
            np.linalg.norm(position + directions[action] - target_array)
        ),
    ) if safe_actions else preferred


def local_mpc_action(
    observation: np.ndarray,
    target: tuple[float, float],
    observation_history: list[np.ndarray] | None = None,
    horizon: int = 6,
    beam_width: int = 24,
    robust: bool = False,
    action_history: list[int] | None = None,
    velocity_source: str = "rgb",
) -> int:
    """Select the first action of a short RGB-only inertial rollout.

    The controller enumerates a bounded beam of discrete action sequences,
    simulates the known local kinematics (friction, acceleration, speed cap),
    rejects swept footprint-wall intersections from the current RGB mask, and
    scores distance-to-waypoint plus braking speed. It is deliberately local:
    it has no learned dynamics query and no A* call. The experiment therefore
    isolates whether the remaining v18 collisions are caused by waypoint
    tracking rather than by the learned route field.
    """
    if horizon < 1 or beam_width < 1:
        raise ValueError("horizon and beam_width must be positive")
    if velocity_source not in {"rgb", "action_fused"}:
        raise ValueError("velocity_source must be 'rgb' or 'action_fused'")
    position = extract_agent_position(observation).astype(np.float32)
    if velocity_source == "action_fused":
        velocity = estimate_action_velocity(
            observation_history or [observation], action_history, max_speed=2.3
        )
    else:
        # Match the baseline waypoint controller's observable contract. This
        # is the v19 primary condition; action-fused velocity is an explicit
        # representation ablation, not silently mixed into the main result.
        from .planner import estimate_agent_velocity

        velocity = estimate_agent_velocity(observation_history or [observation], max_speed=2.3)
    target_array = np.asarray(target, dtype=np.float32)
    wall_mask = extract_wall_mask(observation)
    directions = np.asarray(((0, -1), (0, 1), (-1, 0), (1, 0)), dtype=np.float32)
    # Keep a small action prior so equal safe trajectories prefer moving toward
    # the current waypoint and include an explicit braking action when needed.
    preferred = int(np.argmax(directions @ (target_array - position)))
    action_order = [preferred, 0, 1, 2, 3]
    action_order = list(dict.fromkeys(action_order))
    velocity_scales = (0.75, 1.0, 1.25) if robust else (1.0,)
    velocity_offsets = (
        (np.zeros(2, dtype=np.float32),)
        if not robust
        else (
            np.asarray((0.0, 0.0), dtype=np.float32),
            np.asarray((0.30, 0.0), dtype=np.float32),
            np.asarray((-0.30, 0.0), dtype=np.float32),
            np.asarray((0.0, 0.30), dtype=np.float32),
            np.asarray((0.0, -0.30), dtype=np.float32),
        )
    )
    candidates: list[tuple[float, np.ndarray, np.ndarray, tuple[int, ...]]] = [
        (0.0, position.copy(), velocity.copy(), ())
    ]
    for depth in range(horizon):
        expanded: list[tuple[float, np.ndarray, np.ndarray, tuple[int, ...]]] = []
        for _, pose, speed, actions in candidates:
            for action in action_order:
                scenario_speeds: list[np.ndarray] = []
                scenario_scores: list[float] = []
                nominal_pose: np.ndarray | None = None
                nominal_speed: np.ndarray | None = None
                safe = True
                for scale in velocity_scales:
                    for offset in velocity_offsets:
                        scenario_velocity = speed * scale + offset
                        next_pose, next_speed, landing_safe = _rgb_kinematic_landing(
                            pose, scenario_velocity, action, wall_mask
                        )
                        if not landing_safe:
                            safe = False
                            break
                        scenario_speeds.append(next_speed)
                        scenario_scores.append(float(np.linalg.norm(next_pose - target_array)))
                        if scale == 1.0 and float(np.linalg.norm(offset)) == 0.0:
                            nominal_pose = next_pose
                            nominal_speed = next_speed
                    if not safe:
                        break
                if not safe:
                    continue
                if nominal_pose is None or nominal_speed is None:
                    continue
                next_pose = nominal_pose
                next_speed = nominal_speed
                remaining = max(scenario_scores)
                speed_penalty = 0.25 * max(float(np.linalg.norm(value)) for value in scenario_speeds)
                action_penalty = 0.02 if action != preferred else 0.0
                score = remaining + speed_penalty + action_penalty
                expanded.append((score, next_pose, next_speed, actions + (action,)))
        if not expanded:
            break
        expanded.sort(key=lambda item: item[0])
        candidates = expanded[:beam_width]
    if not candidates or not candidates[0][3]:
        # When inertia has already carried the agent into a narrow corner,
        # every conservative sequence can be blocked. Evaluate the actual
        # inertial one-step proposals and choose the action with the largest
        # worst-case RGB clearance; this avoids repeating a target-facing
        # action that is already pushing into the wall.
        escape_scores: list[tuple[float, float, int]] = []
        for action, direction in enumerate(directions):
            scenario_clearances: list[float] = []
            valid = True
            for scale in ((0.75, 1.0, 1.25) if robust else (1.0,)):
                next_pose, _, landing_safe = _rgb_kinematic_landing(
                    position, velocity * scale, action, wall_mask
                )
                if not (np.isfinite(next_pose).all() and landing_safe):
                    valid = False
                    scenario_clearances.append(-100.0)
                    continue
                scenario_clearances.append(float(-np.linalg.norm(next_pose - target_array)))
            target_distance = float(np.linalg.norm(position + direction - target_array))
            escape_scores.append((min(scenario_clearances), -target_distance, action if valid else action))
        return max(escape_scores)[2]
    best_actions = min(candidates, key=lambda item: item[0])[3]
    return int(best_actions[0]) if best_actions else preferred


def rgb_action_is_safe(
    observation: np.ndarray,
    action: int,
    observation_history: list[np.ndarray] | None = None,
    action_history: list[int] | None = None,
    margin: int = 5,
) -> bool:
    """Check one action using the baseline RGB-only velocity contract."""
    if margin < 1:
        raise ValueError("margin must be positive")
    position = extract_agent_position(observation).astype(np.float32)
    from .planner import estimate_agent_velocity

    velocity = estimate_agent_velocity(observation_history or [observation], max_speed=2.3)
    directions = np.asarray(((0, -1), (0, 1), (-1, 0), (1, 0)), dtype=np.float32)
    next_velocity = 0.84 * velocity + 0.75 * directions[int(action)]
    speed = float(np.linalg.norm(next_velocity))
    if speed > 2.3:
        next_velocity *= 2.3 / speed
    next_position = position + next_velocity
    if not (margin <= next_position[0] <= 64 - margin and margin <= next_position[1] <= 64 - margin):
        return False
    _, _, safe = _rgb_kinematic_landing(
        position, velocity, int(action), extract_wall_mask(observation)
    )
    return safe


def guarded_mpc_action(
    observation: np.ndarray,
    target: tuple[float, float],
    baseline_action: int,
    observation_history: list[np.ndarray] | None = None,
    action_history: list[int] | None = None,
    horizon: int = 6,
    beam_width: int = 24,
) -> int:
    """Preserve a safe baseline action and invoke MPC only on unsafe steps."""
    if rgb_action_is_safe(
        observation, baseline_action, observation_history, action_history, margin=4
    ):
        return int(baseline_action)
    candidate = local_mpc_action(
        observation,
        target,
        observation_history,
        horizon=horizon,
        beam_width=beam_width,
        action_history=action_history,
    )
    if rgb_action_is_safe(
        observation, candidate, observation_history, action_history, margin=4
    ):
        return int(candidate)
    position = extract_agent_position(observation).astype(np.float32)
    directions = np.asarray(((0, -1), (0, 1), (-1, 0), (1, 0)), dtype=np.float32)
    safe_actions = [
        action
        for action in range(4)
        if rgb_action_is_safe(
            observation, action, observation_history, action_history, margin=4
        )
    ]
    if not safe_actions:
        return int(candidate)
    goal = np.asarray(target, dtype=np.float32)
    return min(
        safe_actions,
        key=lambda action: float(np.linalg.norm(position + directions[action] - goal)),
    )


def adaptive_mpc_risk_score(
    observation: np.ndarray,
    target: tuple[float, float],
    baseline_action: int,
    observation_history: list[np.ndarray] | None = None,
    action_history: list[int] | None = None,
) -> float:
    """Estimate local execution risk before choosing ordinary or robust MPC.

    The score uses only information available to the controller at the current
    step.  It is intentionally not a learned oracle: unsafe baseline landing,
    proximity to visible walls, velocity disagreement, and high speed are
    observable RGB/history signals.  A fixed threshold can therefore be
    evaluated without exposing shifted-map labels or future collisions.
    """
    history = observation_history or [observation]
    position = extract_agent_position(observation).astype(np.float32)
    from .planner import estimate_agent_velocity

    rgb_velocity = estimate_agent_velocity(history, max_speed=2.3)
    fused_velocity = estimate_action_velocity(history, action_history, max_speed=2.3)
    speed_risk = float(np.clip((np.linalg.norm(rgb_velocity) - 1.10) / 1.20, 0.0, 1.0))
    disagreement_risk = float(
        np.clip(np.linalg.norm(fused_velocity - rgb_velocity) / 1.75, 0.0, 1.0)
    )
    baseline_unsafe = not rgb_action_is_safe(
        observation, baseline_action, history, action_history, margin=4
    )

    wall_mask = extract_wall_mask(observation)
    wall_pixels = np.argwhere(wall_mask)
    if len(wall_pixels) == 0 or not np.isfinite(position).all():
        proximity_risk = 0.0
    else:
        distances = np.linalg.norm(
            wall_pixels[:, ::-1].astype(np.float32) - position[None], axis=1
        )
        proximity_risk = float(np.clip((10.0 - float(distances.min())) / 10.0, 0.0, 1.0))

    # Baseline safety dominates: an unsafe nominal landing should immediately
    # spend the robust-MPC budget, while benign open-space motion stays cheap.
    score = (
        0.55 * float(baseline_unsafe)
        + 0.20 * proximity_risk
        + 0.15 * disagreement_risk
        + 0.10 * speed_risk
    )
    return float(np.clip(score, 0.0, 1.0))


def adaptive_mpc_decision(
    observation: np.ndarray,
    target: tuple[float, float],
    baseline_action: int,
    observation_history: list[np.ndarray] | None = None,
    action_history: list[int] | None = None,
    horizon: int = 6,
    beam_width: int = 24,
    velocity_source: str = "rgb",
    risk_threshold: float = 0.45,
    risk_exit_threshold: float = 0.30,
    robust_active: bool = False,
) -> tuple[int, bool, float]:
    """Choose ordinary versus robust MPC from an online risk score.

    The lower exit threshold creates hysteresis, preventing robust MPC from
    flickering on and off when local risk is close to the entry boundary.
    """
    if not 0.0 <= risk_exit_threshold <= risk_threshold <= 1.0:
        raise ValueError(
            "risk_exit_threshold must be between 0 and risk_threshold, and risk_threshold <= 1"
        )
    risk_score = adaptive_mpc_risk_score(
        observation,
        target,
        baseline_action,
        observation_history,
        action_history,
    )
    use_robust = (
        risk_score >= risk_exit_threshold
        if robust_active
        else risk_score >= risk_threshold
    )
    action = local_mpc_action(
        observation,
        target,
        observation_history,
        horizon=horizon,
        beam_width=beam_width,
        robust=use_robust,
        action_history=action_history,
        velocity_source=velocity_source,
    )
    return int(action), bool(use_robust), float(risk_score)
