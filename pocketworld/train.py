from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import RolloutBatch, collect_random_rollouts
from .model import PocketWorldModel


def _make_loader(batch: RolloutBatch, batch_size: int, shuffle: bool) -> DataLoader:
    observations = torch.from_numpy(batch.observations).float() / 255.0
    actions = torch.from_numpy(batch.actions)
    position_targets = torch.from_numpy(batch.positions / 64.0)
    velocity_targets = torch.from_numpy(batch.velocities / 3.0).clamp(-1.0, 1.0)
    return DataLoader(TensorDataset(observations, actions, position_targets, velocity_targets), batch_size=batch_size, shuffle=shuffle)


def _run_epoch(model: PocketWorldModel, loader: DataLoader, optimizer: torch.optim.Optimizer | None, unroll_horizon: int) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []
    for rollout_observations, rollout_actions, rollout_positions, rollout_velocities in loader:
        open_state = model.state_from_latent(model.encode(rollout_observations[:, 0]))
        initial_position, initial_velocity = model.kinematics(open_state)
        loss = 0.5 * nn.functional.mse_loss(initial_position, rollout_positions[:, 0])
        loss = loss + 0.2 * nn.functional.mse_loss(initial_velocity, rollout_velocities[:, 0])
        for step in range(unroll_horizon):
            action = rollout_actions[:, step]
            teacher_latent = model.transition(model.encode(rollout_observations[:, step]), action)
            teacher_prediction = model.decode(teacher_latent)
            teacher_state = model.state_transition(model.state_from_latent(model.encode(rollout_observations[:, step])), action)
            teacher_positions, teacher_velocities = model.kinematics(teacher_state)
            target_frame = rollout_observations[:, step + 1]
            image_loss = nn.functional.smooth_l1_loss(teacher_prediction, target_frame)
            target_agent_signal = target_frame[:, 1:2] - target_frame[:, 0:1]
            predicted_agent_signal = teacher_prediction[:, 1:2] - teacher_prediction[:, 0:1]
            agent_color_loss = nn.functional.smooth_l1_loss(predicted_agent_signal, target_agent_signal)
            position_loss = nn.functional.mse_loss(teacher_positions, rollout_positions[:, step + 1])
            velocity_loss = nn.functional.mse_loss(teacher_velocities, rollout_velocities[:, step + 1])
            open_state = model.state_transition(open_state, action)
            open_positions, _ = model.kinematics(open_state)
            open_loss = nn.functional.mse_loss(open_positions, rollout_positions[:, step + 1])
            loss = loss + image_loss + 2.0 * agent_color_loss + position_loss + 0.2 * velocity_loss + 0.25 * open_loss
        loss = loss / (unroll_horizon + 0.5)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def train(
    epochs: int = 5,
    episodes: int = 100,
    validation_episodes: int = 25,
    batch_size: int = 16,
    seed: int = 7,
    output: str = "artifacts/pocketworld.pt",
    unroll_horizon: int = 8,
) -> Path:
    torch.manual_seed(seed)
    train_batch = collect_random_rollouts(episodes=episodes, horizon=unroll_horizon, seed=seed)
    validation_batch = collect_random_rollouts(episodes=validation_episodes, horizon=unroll_horizon, seed=seed + 10000)
    train_loader = _make_loader(train_batch, batch_size=batch_size, shuffle=True)
    validation_loader = _make_loader(validation_batch, batch_size=batch_size, shuffle=False)
    model = PocketWorldModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_loader, optimizer, unroll_horizon)
        with torch.no_grad():
            validation_loss = _run_epoch(model, validation_loader, None, unroll_horizon)
        print(f"epoch {epoch + 1:02d}/{epochs:02d} train={train_loss:.5f} val={validation_loss:.5f}")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "seed": seed, "epochs": epochs, "episodes": episodes, "validation_episodes": validation_episodes, "unroll_horizon": unroll_horizon}, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PocketWorld multi-step image and state dynamics model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--unroll-horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts/pocketworld.pt")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
