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
    collision_targets = torch.from_numpy(batch.collisions)
    return DataLoader(TensorDataset(observations, actions, position_targets, velocity_targets, collision_targets), batch_size=batch_size, shuffle=shuffle)


def _agent_mask_targets(positions: torch.Tensor, size: int = 64, radius: float = 4.0) -> torch.Tensor:
    coordinate = torch.linspace(0.0, 1.0, size, device=positions.device)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    distance = (xx[None] - positions[:, 0, None, None]) ** 2 + (yy[None] - positions[:, 1, None, None]) ** 2
    return (distance <= (radius / size) ** 2).float().unsqueeze(1)


def _run_epoch(model: PocketWorldModel, loader: DataLoader, optimizer: torch.optim.Optimizer | None, unroll_horizon: int) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []
    for rollout_observations, rollout_actions, rollout_positions, rollout_velocities, rollout_collisions in loader:
        open_state = model.state_from_latent(model.encode(rollout_observations[:, 0]))
        initial_position, initial_velocity = model.kinematics(open_state)
        loss = 0.5 * nn.functional.mse_loss(initial_position, rollout_positions[:, 0])
        loss = loss + 0.2 * nn.functional.mse_loss(initial_velocity, rollout_velocities[:, 0])
        for step in range(unroll_horizon):
            action = rollout_actions[:, step]
            teacher_latent = model.transition(model.encode(rollout_observations[:, step]), action)
            teacher_prediction = model.compose_agent_rgb(model.decode(teacher_latent), teacher_latent)
            teacher_state = model.state_transition(model.state_from_latent(model.encode(rollout_observations[:, step])), action)
            teacher_positions, teacher_velocities = model.kinematics(teacher_state)
            teacher_current_latent = model.encode(rollout_observations[:, step])
            teacher_current_state = model.state_from_latent(teacher_current_latent)
            collision_logits = model.collision_logits(teacher_current_latent, teacher_current_state, action, observation=rollout_observations[:, step])
            collision_loss = nn.functional.binary_cross_entropy_with_logits(
                collision_logits,
                rollout_collisions[:, step],
                pos_weight=torch.tensor(5.0, device=collision_logits.device),
            )
            agent_mask_logits = model.agent_mask_logits(teacher_latent.detach())
            agent_mask_loss = nn.functional.binary_cross_entropy_with_logits(
                agent_mask_logits,
                _agent_mask_targets(rollout_positions[:, step + 1]),
                pos_weight=torch.tensor(20.0, device=agent_mask_logits.device),
            )
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
            loss = loss + image_loss + 2.0 * agent_color_loss + position_loss + 0.2 * velocity_loss + 0.25 * open_loss + 0.5 * collision_loss + 0.1 * agent_mask_loss
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
    sticky_probability: float = 0.55,
    full_state_range: bool = False,
    barrier_probability: float = 0.0,
) -> Path:
    torch.manual_seed(seed)
    train_batch = collect_random_rollouts(episodes=episodes, horizon=unroll_horizon, seed=seed, sticky_probability=sticky_probability, full_state_range=full_state_range, barrier_probability=barrier_probability)
    validation_batch = collect_random_rollouts(episodes=validation_episodes, horizon=unroll_horizon, seed=seed + 10000, sticky_probability=sticky_probability, full_state_range=full_state_range, barrier_probability=barrier_probability)
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
    torch.save({"model": model.state_dict(), "seed": seed, "epochs": epochs, "episodes": episodes, "validation_episodes": validation_episodes, "unroll_horizon": unroll_horizon, "sticky_probability": sticky_probability, "full_state_range": full_state_range, "barrier_probability": barrier_probability, "collision_supervision": True, "agent_rendering": True, "dynamics": "learned_structured_kinematics"}, destination)
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
    parser.add_argument("--sticky-probability", type=float, default=0.55)
    parser.add_argument("--full-state-range", action="store_true", help="sample starts and goals across the whole free map")
    parser.add_argument("--barrier-probability", type=float, default=0.0, help="mix single-barrier maps into rollout training")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
