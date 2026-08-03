from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import collect_random_rollouts
from .model import PocketWorldModel


def train(epochs: int = 5, episodes: int = 100, batch_size: int = 16, seed: int = 7, output: str = "artifacts/pocketworld.pt", unroll_horizon: int = 8) -> Path:
    torch.manual_seed(seed)
    batch = collect_random_rollouts(episodes=episodes, horizon=unroll_horizon, seed=seed)
    observations = torch.from_numpy(batch.observations).float() / 255.0
    actions = torch.from_numpy(batch.actions)
    position_targets = torch.from_numpy(batch.positions / 64.0)
    velocity_targets = torch.from_numpy(batch.velocities / 3.0).clamp(-1.0, 1.0)
    loader = DataLoader(TensorDataset(observations, actions, position_targets, velocity_targets), batch_size=batch_size, shuffle=True)
    model = PocketWorldModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for epoch in range(epochs):
        losses = []
        for rollout_observations, rollout_actions, rollout_positions, rollout_velocities in loader:
            open_state = model.state_from_latent(model.encode(rollout_observations[:, 0]))
            initial_position, initial_velocity = model.kinematics(open_state)
            loss = 0.5 * nn.functional.mse_loss(initial_position, rollout_positions[:, 0])
            loss = loss + 0.2 * nn.functional.mse_loss(initial_velocity, rollout_velocities[:, 0])
            for step in range(rollout_actions.shape[1]):
                action = rollout_actions[:, step]
                teacher_latent = model.transition(model.encode(rollout_observations[:, step]), action)
                teacher_prediction = model.decode(teacher_latent)
                teacher_state = model.state_transition(model.state_from_latent(model.encode(rollout_observations[:, step])), action)
                teacher_positions, teacher_velocities = model.kinematics(teacher_state)
                target_frame = rollout_observations[:, step + 1]
                image_loss = loss_fn(teacher_prediction, target_frame)
                position_loss = nn.functional.mse_loss(teacher_positions, rollout_positions[:, step + 1])
                velocity_loss = nn.functional.mse_loss(teacher_velocities, rollout_velocities[:, step + 1])

                open_state = model.state_transition(open_state, action)
                open_positions, _ = model.kinematics(open_state)
                open_loss = nn.functional.mse_loss(open_positions, rollout_positions[:, step + 1])
                loss = loss + image_loss + position_loss + 0.2 * velocity_loss + 0.25 * open_loss
            loss = loss / (rollout_actions.shape[1] + 0.5)
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
