"""Route-level completion prediction from imagined trajectories.

The collision head predicts local events. This module learns a different
target: whether an entire candidate route reaches its goal in the real
simulator. It is intentionally a small calibrator, so the planner comparison
can separate route-level supervision from changes to the world model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def extract_route_features(
    positions: np.ndarray,
    goal: tuple[float, float] | np.ndarray,
    collision_prefix: np.ndarray,
    actions: np.ndarray | None = None,
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
    if features.shape[1] != len(ROUTE_FEATURE_NAMES):
        raise RuntimeError("route feature contract changed unexpectedly")
    del actions  # kept in the signature for future action-pattern ablations
    return features


class RouteCompletionPredictor(nn.Module):
    """Small route-level binary predictor with in-model feature scaling."""

    def __init__(self, input_dim: int = len(ROUTE_FEATURE_NAMES), hidden_dim: int = 32) -> None:
        super().__init__()
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
                "feature_names": list(ROUTE_FEATURE_NAMES),
                "metadata": metadata or {},
            },
            path,
        )
        return path

    @classmethod
    def load(cls, checkpoint: str | Path) -> "RouteCompletionPredictor":
        payload = torch.load(checkpoint, map_location="cpu")
        model = cls()
        model.load_state_dict(payload["model"], strict=True)
        return model

