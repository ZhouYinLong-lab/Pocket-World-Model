"""A small RGB route policy distilled from the observable geometry teacher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .planner import (
    _astar_path,
    _dilate,
    estimate_agent_velocity,
    extract_agent_position,
    extract_wall_boxes,
    extract_wall_mask,
)

ROUTE_MODES = ("direct", "top", "bottom", "gap")


def wall_grid_features(observations: np.ndarray) -> torch.Tensor:
    """Downsample the visible RGB wall mask without using simulator geometry."""
    frames = np.asarray(observations)
    if frames.ndim == 3:
        frames = frames[None]
    if frames.ndim != 4 or frames.shape[1:] != (3, 64, 64):
        raise ValueError("observations must have shape [samples, 3, 64, 64]")
    masks = np.stack([extract_wall_mask(frame) for frame in frames], axis=0)
    tensor = torch.from_numpy(masks.astype(np.float32))[:, None]
    return F.avg_pool2d(tensor, kernel_size=8, stride=8).flatten(1)


def route_mode_label(observation: np.ndarray, goal: tuple[float, float]) -> int:
    """Infer a route-level teacher label from the visible RGB geometry.

    The label is generated from the same inflated visible wall mask used by
    the route teacher, but the learned policy only receives the resulting
    RGB-derived features.  It is deliberately a small task vocabulary: a
    direct route, a top/bottom detour, or a visible gap.
    """
    position = extract_agent_position(observation).astype(np.float32)
    if not np.isfinite(position).all():
        return 0
    mask = extract_wall_mask(observation)
    path = _astar_path(_dilate(mask, radius=4), tuple(position), goal, allow_diagonal=False)
    if not path:
        return 0
    points = np.asarray(path, dtype=np.float32)
    direct = float(np.linalg.norm(np.asarray(goal, dtype=np.float32) - position))
    if direct > 0 and len(path) <= direct * 1.25:
        return 0
    boxes = extract_wall_boxes(mask)
    vertical_boxes = [box for box in boxes if (box[3] - box[1]) >= (box[2] - box[0]) * 2.0]
    for first in vertical_boxes:
        for second in vertical_boxes:
            if second[1] <= first[3]:
                continue
            overlap = min(first[2], second[2]) - max(first[0], second[0])
            gap_mid = (first[3] + second[1]) / 2.0
            if overlap >= 3.0 and np.any(
                (points[:, 0] >= max(first[0], second[0]) - 5.0)
                & (points[:, 0] <= min(first[2], second[2]) + 5.0)
                & (np.abs(points[:, 1] - gap_mid) <= 5.0)
            ):
                return 3
    # For a single vertical barrier, classify the side by the extremum of the
    # actual shortest visible route rather than by the goal midpoint.  This
    # keeps one-wall top/bottom detours distinct from two-wall gap crossing.
    if vertical_boxes:
        wall_top = min(box[1] for box in vertical_boxes)
        wall_bottom = max(box[3] for box in vertical_boxes)
        if float(points[:, 1].min()) < wall_top - 2.0:
            return 1
        if float(points[:, 1].max()) > wall_bottom + 2.0:
            return 2
    midpoint_y = (float(position[1]) + float(goal[1])) / 2.0
    if float(points[:, 1].min()) < midpoint_y - 4.0:
        return 1
    return 2


def observable_route_waypoints(
    observation: np.ndarray,
    goal: tuple[float, float],
    mode: int,
    clearance: float = 6.0,
) -> tuple[tuple[float, float], ...]:
    """Build a short RGB-only waypoint sequence for a predicted route mode."""
    if mode not in range(len(ROUTE_MODES)):
        raise ValueError("mode must identify one of direct/top/bottom/gap")
    if mode == 0:
        return (tuple(map(float, goal)),)
    position = extract_agent_position(observation).astype(np.float32)
    boxes = extract_wall_boxes(extract_wall_mask(observation))
    if not boxes:
        return (tuple(map(float, goal)),)
    direction = 1.0 if goal[0] >= position[0] else -1.0
    blocking = [box for box in boxes if (box[0] - position[0]) * direction >= -4.0]
    if not blocking:
        blocking = list(boxes)
    wall = min(blocking, key=lambda box: abs(((box[0] + box[2]) / 2.0) - position[0]))
    left_x = float(wall[0] - clearance)
    right_x = float(wall[2] + clearance)
    if direction < 0:
        left_x, right_x = right_x, left_x
    if mode == 3:
        aligned = [box for box in boxes if abs(((box[0] + box[2]) / 2.0) - ((wall[0] + wall[2]) / 2.0)) <= 6.0]
        aligned.sort(key=lambda box: box[1])
        if len(aligned) >= 2:
            gap_y = (aligned[0][3] + aligned[1][1]) / 2.0
        else:
            gap_y = (wall[1] + wall[3]) / 2.0
    elif mode == 1:
        gap_y = min(float(wall[1]), float(position[1]), float(goal[1])) - clearance
    else:
        gap_y = max(float(wall[3]), float(position[1]), float(goal[1])) + clearance
    gap_y = float(np.clip(gap_y, clearance, 64.0 - clearance))
    return ((left_x, gap_y), (right_x, gap_y), tuple(map(float, goal)))


def observable_waypoint_action(
    observation: np.ndarray,
    target: tuple[float, float],
    observation_history: list[np.ndarray] | None = None,
    damping: float = 1.0,
) -> int:
    """Steer toward one RGB waypoint with a local visible-wall safety guard."""
    position = extract_agent_position(observation).astype(np.float32)
    velocity = estimate_agent_velocity(observation_history or [observation], max_speed=2.5)
    control = np.asarray(target, dtype=np.float32) - position - float(damping) * velocity
    if abs(control[0]) >= abs(control[1]):
        preferred = 3 if control[0] >= 0 else 2
    else:
        preferred = 1 if control[1] >= 0 else 0
    mask = extract_wall_mask(observation)
    wall_y, wall_x = np.where(mask)
    directions = np.asarray(((0, -1), (0, 1), (-1, 0), (1, 0)), dtype=np.float32)

    def safe(action: int) -> bool:
        next_velocity = 0.84 * velocity + 0.75 * directions[action]
        speed = float(np.linalg.norm(next_velocity))
        if speed > 2.3:
            next_velocity *= 2.3 / speed
        x, y = position + next_velocity
        if not (3 <= x < 61 and 3 <= y < 61):
            return False
        return not bool(np.any(
            (x >= wall_x - 3.0) & (x <= wall_x + 4.0)
            & (y >= wall_y - 3.0) & (y <= wall_y + 4.0)
        ))

    if safe(preferred):
        return preferred
    candidates = [action for action in range(4) if safe(action)]
    return min(candidates, key=lambda action: float(np.linalg.norm(
        position + directions[action] - np.asarray(target, dtype=np.float32)
    ))) if candidates else preferred


class RouteModePolicy(nn.Module):
    """Predict a route mode from the initial observable RGB geometry."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, len(ROUTE_MODES))

    def forward(self, goals: torch.Tensor, positions: torch.Tensor, wall_features: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(torch.cat((goals, positions, wall_features), dim=-1)))

    def fit(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        labels: np.ndarray,
        epochs: int = 80,
        seed: int = 7,
    ) -> dict[str, float | int | list[int]]:
        frames = np.asarray(observations, dtype=np.uint8)
        goal_values = torch.from_numpy(np.asarray(goals, dtype=np.float32)) / 64.0
        positions = extract_agent_position(frames).astype(np.float32)
        positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
        position_values = torch.from_numpy(positions) / 64.0
        wall_values = wall_grid_features(frames)
        targets = torch.from_numpy(np.asarray(labels, dtype=np.int64).reshape(-1))
        if frames.ndim != 4 or frames.shape[1:] != (3, 64, 64):
            raise ValueError("observations must have shape [samples, 3, 64, 64]")
        if goal_values.shape != position_values.shape or goal_values.shape != (len(targets), 2):
            raise ValueError("goals and route labels must have aligned sample dimensions")
        if len(targets) < 8 or not torch.all((targets >= 0) & (targets < len(ROUTE_MODES))):
            raise ValueError("route mode data must contain at least eight valid samples")
        torch.manual_seed(seed)
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=1e-5)
        counts = torch.bincount(targets, minlength=len(ROUTE_MODES)).float()
        weights = torch.zeros_like(counts)
        observed = counts > 0
        weights[observed] = counts[observed].sum() / (observed.sum() * counts[observed])
        for _ in range(max(1, int(epochs))):
            logits = self(goal_values, position_values, wall_values)
            loss = nn.functional.cross_entropy(logits, targets, weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predictions = self(goal_values, position_values, wall_values).argmax(dim=-1)
        return {
            "epochs": int(epochs),
            "samples": int(len(targets)),
            "mode_counts": [int(value) for value in counts.tolist()],
            "train_accuracy": float((predictions == targets).float().mean()),
        }

    @torch.no_grad()
    def predict_mode(self, observation: np.ndarray, goal: tuple[float, float]) -> int:
        position = extract_agent_position(observation).astype(np.float32)
        position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
        logits = self(
            torch.as_tensor(np.asarray(goal, dtype=np.float32) / 64.0)[None],
            torch.as_tensor(position / 64.0)[None],
            wall_grid_features(observation),
        )
        return int(logits.argmax(dim=-1).item())

    def save(self, destination: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.state_dict(), "metadata": metadata or {}}, path)
        return path

    @classmethod
    def load(cls, checkpoint: str | Path) -> "RouteModePolicy":
        payload = torch.load(checkpoint, map_location="cpu")
        model = cls()
        model.load_state_dict(payload["model"], strict=True)
        return model


class LearnedRoutePolicy(nn.Module):
    """Predict the next discrete action from RGB, goal, and recent velocity."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        # The wall grid is derived from RGB at the observation boundary. A
        # small MLP makes the policy's observable geometry contract explicit
        # and avoids asking a CNN to rediscover both the agent coordinate and
        # the wall mask before learning the teacher's local action rule.
        self.encoder = nn.Sequential(
            nn.Linear(64 + 6, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(
        self,
        observations: torch.Tensor,
        goals: torch.Tensor,
        velocities: torch.Tensor,
        positions: torch.Tensor | None = None,
        wall_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if positions is None:
            positions = torch.zeros_like(goals)
        if wall_features is None:
            wall_features = torch.zeros(
                (observations.shape[0], 64), dtype=observations.dtype, device=observations.device
            )
        features = torch.cat((goals, velocities, positions, wall_features), dim=-1)
        return self.head(self.encoder(features))

    @torch.no_grad()
    def predict_action(
        self,
        observation: np.ndarray,
        goal: tuple[float, float],
        observation_history: list[np.ndarray] | None = None,
    ) -> int:
        self.eval()
        history = observation_history or [observation]
        velocity = estimate_agent_velocity(history, max_speed=2.5)
        position = extract_agent_position(observation).astype(np.float32)
        if not np.isfinite(position).all():
            position = np.zeros(2, dtype=np.float32)
        frame = torch.from_numpy(np.asarray(observation)[None]).float() / 255.0
        goal_tensor = torch.as_tensor(np.asarray(goal, dtype=np.float32) / 64.0)[None]
        velocity_tensor = torch.as_tensor(velocity / 3.0, dtype=torch.float32)[None]
        position_tensor = torch.as_tensor(position / 64.0, dtype=torch.float32)[None]
        wall_features = wall_grid_features(np.asarray(observation))
        return int(
            self(frame, goal_tensor, velocity_tensor, position_tensor, wall_features)
            .argmax(dim=-1)
            .item()
        )

    def fit(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        velocities: np.ndarray,
        actions: np.ndarray,
        positions: np.ndarray | None = None,
        epochs: int = 80,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        seed: int = 7,
    ) -> dict[str, float | int | list[int]]:
        frames = torch.from_numpy(np.asarray(observations, dtype=np.float32)) / 255.0
        goal_values = torch.from_numpy(np.asarray(goals, dtype=np.float32)) / 64.0
        velocity_values = torch.from_numpy(np.asarray(velocities, dtype=np.float32)) / 3.0
        targets = torch.from_numpy(np.asarray(actions, dtype=np.int64).reshape(-1))
        if positions is None:
            positions = extract_agent_position(np.asarray(observations)).astype(np.float32)
            positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
        position_values = torch.from_numpy(np.asarray(positions, dtype=np.float32)) / 64.0
        wall_values = wall_grid_features(np.asarray(observations))
        if frames.ndim != 4 or frames.shape[1:] != (3, 64, 64):
            raise ValueError("observations must have shape [samples, 3, 64, 64]")
        if goal_values.shape != (frames.shape[0], 2) or velocity_values.shape != (frames.shape[0], 2):
            raise ValueError("goals and velocities must have shape [samples, 2]")
        if position_values.shape != (frames.shape[0], 2):
            raise ValueError("positions must have shape [samples, 2]")
        if targets.shape[0] != frames.shape[0] or not torch.all((targets >= 0) & (targets < 4)):
            raise ValueError("actions must contain one of four discrete action labels per sample")
        if frames.shape[0] < 8:
            raise ValueError("route policy training data must contain at least eight samples")
        torch.manual_seed(seed)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        generator = torch.Generator().manual_seed(seed)
        class_counts = torch.bincount(targets, minlength=4).float()
        losses: list[float] = []
        for _ in range(max(1, int(epochs))):
            order = torch.randperm(frames.shape[0], generator=generator)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, frames.shape[0], max(1, int(batch_size))):
                batch = order[start : start + max(1, int(batch_size))]
                logits = self(
                    frames[batch],
                    goal_values[batch],
                    velocity_values[batch],
                    position_values[batch],
                    wall_values[batch],
                )
                loss = nn.functional.cross_entropy(logits, targets[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach())
                batches += 1
            losses.append(epoch_loss / max(1, batches))
        with torch.no_grad():
            predictions = self(
                frames,
                goal_values,
                velocity_values,
                position_values,
                wall_values,
            ).argmax(dim=-1)
            accuracy = (predictions == targets).float().mean()
        return {
            "epochs": int(epochs),
            "samples": int(frames.shape[0]),
            "action_counts": [int(value) for value in class_counts.tolist()],
            "final_loss": losses[-1],
            "train_accuracy": float(accuracy),
        }

    def save(self, destination: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.state_dict(), "metadata": metadata or {}}, path)
        return path

    @classmethod
    def load(cls, checkpoint: str | Path) -> "LearnedRoutePolicy":
        payload = torch.load(checkpoint, map_location="cpu")
        model = cls()
        model.load_state_dict(payload["model"], strict=True)
        return model
