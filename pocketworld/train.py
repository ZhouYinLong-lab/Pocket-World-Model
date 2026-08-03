from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import collect_random_rollouts
from .model import PocketWorldModel
from .planner import extract_agent_position


def _position_targets(frames: torch.Tensor) -> torch.Tensor:
    positions = []
    for frame in frames:
        position = extract_agent_position((frame.detach().cpu().numpy() * 255).astype(np.uint8))
        if not np.all(np.isfinite(position)):
            position = np.array([0.0, 0.0], dtype=np.float32)
        positions.append(position / 64.0)
    return torch.from_numpy(np.asarray(positions, dtype=np.float32))


def train(epochs: int = 5, episodes: int = 100, batch_size: int = 16, seed: int = 7, output: str = "artifacts/pocketworld.pt", unroll_horizon: int = 8) -> Path:
    torch.manual_seed(seed)
    batch = collect_random_rollouts(episodes=episodes, horizon=unroll_horizon, seed=seed)
    observations = torch.from_numpy(batch.observations).float() / 255.0
    actions = torch.from_numpy(batch.actions)
    position_targets = _position_targets(observations.reshape(-1, 3, 64, 64)).reshape(episodes, unroll_horizon + 1, 2)
    loader = DataLoader(TensorDataset(observations, actions, position_targets), batch_size=batch_size, shuffle=True)
    model = PocketWorldModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for epoch in range(epochs):
        losses = []
        for rollout_observations, rollout_actions, rollout_positions in loader:
            latent = model.encode(rollout_observations[:, 0])
            loss = torch.zeros((), dtype=rollout_observations.dtype)
            for step in range(rollout_actions.shape[1]):
                latent = model.transition(latent, rollout_actions[:, step])
                prediction = model.decode(latent)
                predicted_positions = model.position_head(latent)
                image_loss = loss_fn(prediction, rollout_observations[:, step + 1])
                position_loss = nn.functional.mse_loss(predicted_positions, rollout_positions[:, step + 1])
                loss = loss + image_loss + 1.0 * position_loss
            loss = loss / rollout_actions.shape[1]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        print(f"epoch {epoch + 1:02d}/{epochs:02d} loss={np.mean(losses):.5f}")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "seed": seed}, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PocketWorld one-step image dynamics model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--unroll-horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts/pocketworld.pt")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
