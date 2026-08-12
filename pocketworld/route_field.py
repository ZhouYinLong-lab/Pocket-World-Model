"""A learned coarse distance field for general route representation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .planner import _dilate, extract_agent_position, extract_wall_mask

FIELD_GRID = 16


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


def route_field_targets(
    observations: np.ndarray,
    goals: np.ndarray,
    grid_size: int = FIELD_GRID,
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
    block = 64 // grid_size
    fields = np.ones((len(frames), grid_size, grid_size), dtype=np.float32)
    valid = np.zeros_like(fields)
    for index, (frame, goal) in enumerate(zip(frames, goal_values)):
        occupied = _dilate(extract_wall_mask(frame), radius=4)
        distance = _pixel_distance_field(occupied, tuple(map(float, goal)))
        for row in range(grid_size):
            for column in range(grid_size):
                patch = distance[row * block:(row + 1) * block, column * block:(column + 1) * block]
                finite = patch[np.isfinite(patch)]
                if len(finite):
                    fields[index, row, column] = float(np.clip(finite.min() / 128.0, 0.0, 1.0))
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
    ) -> dict[str, float | int]:
        frames = np.asarray(observations, dtype=np.uint8)
        inputs = _field_inputs(frames, goals)
        targets, valid = route_field_targets(frames, goals)
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
    occupied = _dilate(extract_wall_mask(observation), radius=3)
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
