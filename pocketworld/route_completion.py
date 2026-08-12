"""Route-level completion prediction from imagined trajectories.

The collision head predicts local events. This module learns a different
target: whether an entire candidate route reaches its goal in the real
simulator. It is intentionally a small calibrator, so the planner comparison
can separate route-level supervision from changes to the world model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import heapq

import numpy as np
import torch
from torch import nn


ROUTE_FEATURE_NAMES = (
    "initial_distance_norm",
    "final_distance_norm",
    "minimum_distance_norm",
    "distance_progress_norm",
    "predicted_collision_risk",
    "first_risk_step_norm",
    "path_length_norm",
    "distance_regression_norm",
    "horizon_norm",
)

ROUTE_MAP_FEATURE_NAMES = (
    "wall_fraction",
    "wall_component_count_norm",
    "start_wall_clearance_norm",
    "goal_wall_clearance_norm",
    "direct_wall_fraction",
    "direct_wall_blocked",
    "top_detour_length_norm",
    "bottom_detour_length_norm",
)
MAP_AWARE_ROUTE_FEATURE_NAMES = ROUTE_FEATURE_NAMES + ROUTE_MAP_FEATURE_NAMES


def _dilate_wall_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    height, width = mask.shape
    result = np.zeros((height, width), dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy : dy + height, dx : dx + width]
    return result


def _component_count(mask: np.ndarray) -> int:
    visited = np.zeros_like(mask, dtype=bool)
    count = 0
    height, width = mask.shape
    for y, x in zip(*np.where(mask & ~visited)):
        if visited[y, x]:
            continue
        count += 1
        stack = [(int(y), int(x))]
        visited[y, x] = True
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
    return count


def _clearance(mask: np.ndarray, point: tuple[float, float]) -> float:
    occupied = np.argwhere(mask)
    if occupied.size == 0:
        return float(np.hypot(*mask.shape))
    y, x = occupied[:, 0], occupied[:, 1]
    return float(np.sqrt(((x - point[0]) ** 2 + (y - point[1]) ** 2).min()))


def _direct_wall_fraction(mask: np.ndarray, start: tuple[float, float], goal: tuple[float, float]) -> tuple[float, float]:
    samples = max(2, int(np.ceil(np.linalg.norm(np.asarray(goal) - np.asarray(start)) * 2.0)))
    xs = np.rint(np.linspace(start[0], goal[0], samples)).astype(int)
    ys = np.rint(np.linspace(start[1], goal[1], samples)).astype(int)
    inside = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    blocked = np.zeros(samples, dtype=bool)
    blocked[inside] = mask[ys[inside], xs[inside]]
    fraction = float(blocked.mean())
    return fraction, float(blocked.any())


def _detour_length(mask: np.ndarray, start: tuple[float, float], goal: tuple[float, float], preference: str) -> float:
    """Return a preference-biased shortest path length on the visible grid."""
    height, width = mask.shape
    start_cell = (int(np.clip(round(start[0]), 3, width - 4)), int(np.clip(round(start[1]), 3, height - 4)))
    goal_cell = (int(np.clip(round(goal[0]), 3, width - 4)), int(np.clip(round(goal[1]), 3, height - 4)))
    if mask[start_cell[1], start_cell[0]] or mask[goal_cell[1], goal_cell[0]]:
        return float("inf")
    frontier: list[tuple[float, float, int, int]] = [(0.0, 0.0, start_cell[0], start_cell[1])]
    best: dict[tuple[int, int], tuple[float, float]] = {start_cell: (0.0, 0.0)}
    while frontier:
        _, length, x, y = heapq.heappop(frontier)
        if (x, y) == goal_cell:
            return float(length)
        if best.get((x, y), (float("inf"),))[0] < length - 1e-6:
            continue
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if not (3 <= nx <= width - 4 and 3 <= ny <= height - 4) or mask[ny, nx]:
                continue
            next_length = length + 1.0
            bias = 0.01 * (ny if preference == "top" else height - 1 - ny)
            next_cost = next_length + bias
            previous = best.get((nx, ny))
            if previous is not None and previous[0] <= next_cost + 1e-6:
                continue
            best[(nx, ny)] = (next_cost, next_length)
            heapq.heappush(frontier, (next_cost, next_length, nx, ny))
    return float("inf")


def extract_map_context_features(
    start: tuple[float, float] | np.ndarray,
    goal: tuple[float, float] | np.ndarray,
    wall_mask: np.ndarray,
) -> np.ndarray:
    """Extract planner-visible geometry features shared by all candidates."""
    mask = np.asarray(wall_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("wall_mask must have shape [height, width]")
    start_tuple = (float(start[0]), float(start[1]))
    goal_tuple = (float(goal[0]), float(goal[1]))
    occupied = _dilate_wall_mask(mask)
    direct_fraction, direct_blocked = _direct_wall_fraction(occupied, start_tuple, goal_tuple)
    direct_distance = max(1.0, float(np.linalg.norm(np.asarray(goal_tuple) - np.asarray(start_tuple))))
    top_length = _detour_length(occupied, start_tuple, goal_tuple, "top")
    bottom_length = _detour_length(occupied, start_tuple, goal_tuple, "bottom")
    max_length = float(mask.shape[0] + mask.shape[1])
    return np.asarray(
        (
            float(occupied.mean()),
            min(1.0, _component_count(occupied) / 8.0),
            min(1.0, _clearance(occupied, start_tuple) / 64.0),
            min(1.0, _clearance(occupied, goal_tuple) / 64.0),
            direct_fraction,
            direct_blocked,
            min(4.0, top_length / max(direct_distance, max_length / 4.0)) if np.isfinite(top_length) else 4.0,
            min(4.0, bottom_length / max(direct_distance, max_length / 4.0)) if np.isfinite(bottom_length) else 4.0,
        ),
        dtype=np.float32,
    )


def extract_route_features(
    positions: np.ndarray,
    goal: tuple[float, float] | np.ndarray,
    collision_prefix: np.ndarray,
    actions: np.ndarray | None = None,
    map_context: np.ndarray | None = None,
) -> np.ndarray:
    """Convert imagined candidate trajectories into route-level features.

    ``positions`` and ``collision_prefix`` include the initial state. The
    output is deliberately composed of quantities available to the planner;
    it never reads simulator positions, collision flags, or map labels.
    """
    trajectory = np.asarray(positions, dtype=np.float32)
    if trajectory.ndim == 2:
        trajectory = trajectory[None]
    prefix = np.asarray(collision_prefix, dtype=np.float32)
    if prefix.ndim == 1:
        prefix = prefix[None]
    if trajectory.ndim != 3 or trajectory.shape[-1] != 2:
        raise ValueError("positions must have shape [candidates, time, 2]")
    if prefix.shape[:2] != trajectory.shape[:2]:
        raise ValueError("collision_prefix must align with positions")
    goal_array = np.asarray(goal, dtype=np.float32)
    distances = np.linalg.norm(trajectory - goal_array[None, None, :], axis=-1)
    initial = distances[:, 0]
    final = distances[:, -1]
    minimum = distances.min(axis=1)
    progress = initial - final
    path_length = np.linalg.norm(np.diff(trajectory, axis=1), axis=-1).sum(axis=1)
    regression = np.maximum(0.0, np.diff(distances, axis=1)).sum(axis=1)
    risk = np.clip(prefix[:, -1], 0.0, 1.0)
    risk_steps = np.argmax(prefix >= 0.5, axis=1).astype(np.float32)
    has_risk = (prefix >= 0.5).any(axis=1)
    horizon = max(1, trajectory.shape[1] - 1)
    first_risk_step = np.where(has_risk, risk_steps / horizon, 1.0)
    features = np.stack(
        (
            initial / 64.0,
            final / 64.0,
            minimum / 64.0,
            progress / 64.0,
            risk,
            first_risk_step,
            path_length / max(1.0, horizon * 3.0),
            regression / max(1.0, horizon * 3.0),
            np.full(trajectory.shape[0], min(1.0, horizon / 64.0), dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    if map_context is not None:
        context = np.asarray(map_context, dtype=np.float32)
        if context.ndim == 1:
            context = np.broadcast_to(context[None], (features.shape[0], context.shape[0]))
        if context.ndim != 2 or context.shape[0] != features.shape[0] or context.shape[1] != len(ROUTE_MAP_FEATURE_NAMES):
            raise ValueError("map_context must have shape [candidates, 8] or [8]")
        features = np.concatenate((features, context), axis=1)
    if features.shape[1] not in {len(ROUTE_FEATURE_NAMES), len(MAP_AWARE_ROUTE_FEATURE_NAMES)}:
        raise RuntimeError("route feature contract changed unexpectedly")
    del actions  # kept in the signature for future action-pattern ablations
    return features


class RouteCompletionPredictor(nn.Module):
    """Small route-level binary predictor with in-model feature scaling."""

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int = 32,
        feature_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.feature_names = tuple(feature_names or ROUTE_FEATURE_NAMES)
        input_dim = int(input_dim or len(self.feature_names))
        if input_dim != len(self.feature_names):
            raise ValueError("input_dim must match feature_names")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_scale", torch.ones(input_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale.clamp_min(1e-5)
        return self.network(normalized).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
        return torch.sigmoid(self(tensor)).cpu().numpy()

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 160,
        learning_rate: float = 5e-3,
        seed: int = 7,
    ) -> dict[str, float | int]:
        values = torch.from_numpy(np.asarray(features, dtype=np.float32))
        targets = torch.from_numpy(np.asarray(labels, dtype=np.float32).reshape(-1))
        if values.ndim != 2 or values.shape[1] != self.feature_mean.numel():
            raise ValueError("features do not match the route predictor contract")
        if values.shape[0] != targets.shape[0] or values.shape[0] < 4:
            raise ValueError("route training data must have at least four aligned examples")
        if not torch.isfinite(values).all() or not torch.isfinite(targets).all():
            raise ValueError("route training data must be finite")
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("route labels must be binary")
        if targets.min() == targets.max():
            raise ValueError("route training data must contain both success and failure labels")
        torch.manual_seed(seed)
        self.feature_mean.copy_(values.mean(dim=0))
        self.feature_scale.copy_(values.std(dim=0, unbiased=False).clamp_min(1e-3))
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        positives = targets.sum().clamp_min(1.0)
        negatives = (targets.numel() - targets.sum()).clamp_min(1.0)
        positive_weight = (negatives / positives).clamp(1.0, 20.0)
        loss = torch.zeros(())
        for _ in range(max(1, int(epochs))):
            logits = self(values)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=positive_weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            probabilities = torch.sigmoid(self(values))
            accuracy = ((probabilities >= 0.5) == (targets >= 0.5)).float().mean()
        return {
            "epochs": int(epochs),
            "samples": int(values.shape[0]),
            "positive_rate": float(targets.mean()),
            "final_loss": float(loss.detach()),
            "train_accuracy": float(accuracy),
        }

    def save(self, destination: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.state_dict(),
                "feature_names": list(self.feature_names),
                "metadata": metadata or {},
            },
            path,
        )
        return path

    @classmethod
    def load(cls, checkpoint: str | Path) -> "RouteCompletionPredictor":
        payload = torch.load(checkpoint, map_location="cpu")
        feature_names = tuple(payload.get("feature_names", ROUTE_FEATURE_NAMES))
        model = cls(feature_names=feature_names)
        model.load_state_dict(payload["model"], strict=True)
        return model
