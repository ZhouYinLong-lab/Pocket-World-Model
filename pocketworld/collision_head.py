"""Learned short-horizon collision probability for route-conditioned MPC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .env import PocketWorldEnv, Rect
from .planner import extract_agent_position, extract_wall_mask
from .route_field import estimate_action_velocity

COLLISION_HORIZONS = (1, 2, 4)
COLLISION_CROP_RADIUS = 7


def collision_head_features(
    observation: np.ndarray,
    goal: tuple[float, float],
    target: tuple[float, float],
    velocity: np.ndarray,
    action: int,
    agent_speed_scale: float = 1.0,
) -> np.ndarray:
    """Encode only current RGB/history-observable route and action features."""
    position = extract_agent_position(observation).astype(np.float32)
    wall = extract_wall_mask(observation).astype(np.float32)
    radius = COLLISION_CROP_RADIUS
    padded = np.pad(wall, radius, mode="constant", constant_values=1.0)
    center = np.rint(np.nan_to_num(position, nan=32.0)).astype(int) + radius
    crop = padded[center[1] - radius:center[1] + radius + 1, center[0] - radius:center[0] + radius + 1]
    if crop.shape != (2 * radius + 1, 2 * radius + 1):
        fixed = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
        fixed[: crop.shape[0], : crop.shape[1]] = crop
        crop = fixed
    velocity = np.asarray(velocity, dtype=np.float32)
    goal_delta = (np.asarray(goal, dtype=np.float32) - position) / 64.0
    target_delta = (np.asarray(target, dtype=np.float32) - position) / 64.0
    action_one_hot = np.zeros(4, dtype=np.float32)
    action_one_hot[int(action)] = 1.0
    return np.concatenate(
        (
            crop.ravel(),
            np.nan_to_num(position / 64.0, nan=0.5),
            np.nan_to_num(goal_delta, nan=0.0),
            np.nan_to_num(target_delta, nan=0.0),
            np.nan_to_num(velocity / 2.3, nan=0.0).clip(-1.0, 1.0),
            action_one_hot,
            np.asarray((agent_speed_scale,), dtype=np.float32),
        )
    ).astype(np.float32)


def _rollout_collision_labels(
    walls: tuple[Rect, ...],
    goal: tuple[float, float],
    position: np.ndarray,
    velocity: np.ndarray,
    first_action: int,
    horizons: tuple[int, ...],
    continuation_samples: int,
    rng: np.random.Generator,
    agent_speed_scale: float = 1.0,
) -> np.ndarray:
    labels = np.zeros(len(horizons), dtype=np.float32)
    for _ in range(continuation_samples):
        max_horizon = max(horizons)
        env = PocketWorldEnv(
            walls=walls,
            agent_start=tuple(map(float, position)),
            goal=goal,
            agent_speed_scale=agent_speed_scale,
        )
        env.reset()
        env.position = np.asarray(position, dtype=np.float32).copy()
        env.velocity = np.asarray(velocity, dtype=np.float32).copy()
        hit = False
        for step in range(1, max_horizon + 1):
            action = int(first_action) if step == 1 else int(rng.integers(0, 4))
            _, _, terminated, truncated, info = env.step(action)
            hit = hit or bool(info.get("collision", False))
            for index, horizon in enumerate(horizons):
                if step == horizon and hit:
                    labels[index] += 1.0
            if terminated or truncated:
                break
    return labels / float(max(1, continuation_samples))


def collect_collision_head_dataset(
    seeds: tuple[int, ...] = (101, 103, 107),
    episodes: int = 24,
    max_steps: int = 96,
    horizons: tuple[int, ...] = COLLISION_HORIZONS,
    continuation_samples: int = 4,
    sample_stride: int = 2,
    families: tuple[str, ...] | None = None,
    balanced_families: bool = True,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Collect simulator-labelled risk examples from one map split."""
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive values")
    if continuation_samples < 1 or sample_stride < 1:
        raise ValueError("continuation_samples and sample_stride must be positive")
    if split not in {"train", "holdout"}:
        raise ValueError("split must be train or holdout")
    from .general_routes import sample_general_route_cases
    from .route_field import field_waypoints, route_field_targets

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for seed in seeds:
        cases = sample_general_route_cases(
            seed,
            episodes,
            split=split,
            families=families,
            balanced=balanced_families,
        )
        rng = np.random.default_rng(seed + 7919)
        for case in cases:
            env = PocketWorldEnv(walls=case.walls, agent_start=case.start, goal=case.goal)
            observation, _ = env.reset()
            teacher_field, _ = route_field_targets(
                np.asarray(observation)[None], np.asarray(case.goal, dtype=np.float32)[None]
            )
            waypoints = field_waypoints(
                observation,
                case.goal,
                teacher_field[0].numpy(),
                rgb_guard=True,
                beam_width=4,
            )
            waypoint_index = 0
            history = [observation]
            action_history: list[int] = []
            for step in range(max_steps):
                velocity = estimate_action_velocity(history, action_history, max_speed=2.3)
                position = env.position.astype(np.float32)
                if waypoint_index < len(waypoints) - 1 and np.linalg.norm(
                    position - np.asarray(waypoints[waypoint_index], dtype=np.float32)
                ) <= 5.0:
                    waypoint_index += 1
                target = waypoints[waypoint_index]
                if step % sample_stride == 0:
                    for action in range(4):
                        features.append(
                            collision_head_features(
                                observation,
                                case.goal,
                                target,
                                velocity,
                                action,
                            )
                        )
                        labels.append(
                            _rollout_collision_labels(
                                case.walls,
                                case.goal,
                                env.position,
                                env.velocity,
                                action,
                                tuple(horizons),
                                continuation_samples,
                                rng,
                            )
                        )
                action = int(rng.integers(0, 4))
                observation, _, terminated, truncated, _ = env.step(action)
                action_history.append(action)
                action_history = action_history[-16:]
                history.append(observation)
                history = history[-16:]
                if terminated or truncated:
                    break
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


class CollisionProbabilityHead(nn.Module):
    """Small MLP predicting collision probability at several horizons."""

    def __init__(self, input_dim: int = 238, hidden_dim: int = 128, horizons: tuple[int, ...] = COLLISION_HORIZONS) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.horizons = tuple(int(value) for value in horizons)
        # Temperatures are fitted only on a disjoint calibration split. Keep
        # them outside the network so raw weights remain available for an
        # auditable ablation; one value per horizon avoids conflating the
        # different base rates of 1-, 2-, and 4-step events.
        self.temperatures = np.ones(len(self.horizons), dtype=np.float32)
        self.temperature = 1.0
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(self.horizons)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 160,
        seed: int = 7,
    ) -> dict[str, float | int]:
        x = torch.from_numpy(np.asarray(features, dtype=np.float32))
        y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"features must have shape [N, {self.input_dim}]")
        if y.shape != (len(x), len(self.horizons)):
            raise ValueError("labels shape must match features and horizons")
        if len(x) < 8:
            raise ValueError("collision-head data must contain at least eight samples")
        torch.manual_seed(seed)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        losses: list[float] = []
        for _ in range(max(1, int(epochs))):
            logits = self(x)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        return {
            "epochs": int(epochs),
            "samples": int(len(x)),
            "final_loss": losses[-1],
            "positive_rate": float(y.mean()),
        }

    @torch.no_grad()
    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        """Return uncalibrated logits without applying the temperature."""
        values = torch.from_numpy(np.asarray(features, dtype=np.float32))
        if values.ndim == 1:
            values = values[None]
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"features must have shape [N, {self.input_dim}]")
        return self(values).cpu().numpy().astype(np.float32)

    def fit_temperature(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 200,
        seed: int = 17,
    ) -> dict[str, float | int]:
        """Fit one positive scalar temperature on held-out labels.

        This is post-hoc calibration only: network weights stay frozen and
        the final evaluation split must not be used here. Fractional labels
        are valid because the data collector averages continuation rollouts.
        """
        logits = torch.from_numpy(self.predict_logits(features))
        targets = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        if targets.shape != logits.shape:
            raise ValueError("labels shape must match features and horizons")
        if len(logits) < 8:
            raise ValueError("temperature calibration needs at least eight samples")
        torch.manual_seed(seed)
        log_temperature = torch.nn.Parameter(
            torch.zeros(len(self.horizons), dtype=torch.float32)
        )
        optimizer = torch.optim.Adam([log_temperature], lr=0.05)
        losses: list[float] = []
        for _ in range(max(1, int(epochs))):
            temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
            loss = nn.functional.binary_cross_entropy_with_logits(logits / temperature, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        fitted = torch.exp(log_temperature).clamp(0.05, 20.0).detach().cpu().numpy()
        self.temperatures = np.asarray(fitted, dtype=np.float32)
        self.temperature = float(self.temperatures.mean())
        return {
            "epochs": int(epochs),
            "samples": int(len(logits)),
            "temperature": self.temperature,
            "temperatures": self.temperatures.tolist(),
            "final_loss": losses[-1],
        }

    @torch.no_grad()
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = torch.from_numpy(self.predict_logits(features))
        temperatures = np.asarray(getattr(self, "temperatures", (self.temperature,)), dtype=np.float32)
        if temperatures.shape != (len(self.horizons),):
            temperatures = np.full(len(self.horizons), float(self.temperature), dtype=np.float32)
        temperature_tensor = torch.from_numpy(np.clip(temperatures, 0.05, 20.0))
        return torch.sigmoid(logits / temperature_tensor).cpu().numpy().astype(np.float32)

    def save(self, destination: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.state_dict(),
                "metadata": metadata or {},
                "input_dim": self.input_dim,
                "horizons": self.horizons,
                "temperature": float(self.temperature),
                "temperatures": np.asarray(self.temperatures, dtype=np.float32).tolist(),
            },
            path,
        )
        return path

    @classmethod
    def load(cls, checkpoint: str | Path) -> "CollisionProbabilityHead":
        payload = torch.load(checkpoint, map_location="cpu")
        model = cls(input_dim=int(payload.get("input_dim", 238)), horizons=tuple(payload.get("horizons", COLLISION_HORIZONS)))
        model.load_state_dict(payload["model"], strict=True)
        fallback_temperature = float(payload.get("temperature", payload.get("metadata", {}).get("temperature", 1.0)))
        values = payload.get("temperatures", payload.get("metadata", {}).get("temperatures"))
        if values is None:
            values = [fallback_temperature] * len(model.horizons)
        model.temperatures = np.asarray(values, dtype=np.float32)
        if model.temperatures.shape != (len(model.horizons),):
            model.temperatures = np.full(len(model.horizons), fallback_temperature, dtype=np.float32)
        model.temperature = float(model.temperatures.mean())
        return model

