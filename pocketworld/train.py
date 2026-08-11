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


def _agent_mask_targets(positions: torch.Tensor, size: int = 64, radius: float = 3.0) -> torch.Tensor:
    coordinate = torch.linspace(0.0, 1.0, size, device=positions.device)
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    distance = (xx[None] - positions[:, 0, None, None]) ** 2 + (yy[None] - positions[:, 1, None, None]) ** 2
    return (distance <= (radius / size) ** 2).float().unsqueeze(1)


def _agent_mask_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weighted_bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=torch.tensor(100.0, device=logits.device),
    )
    probability = torch.sigmoid(logits)
    intersection = (probability * target).flatten(1).sum(dim=1)
    denominator = probability.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return weighted_bce + 0.5 * dice_loss


def _run_epoch(
    model: PocketWorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    unroll_horizon: int,
    collision_only: bool = False,
    kinematics_only: bool = False,
    temporal_only: bool = False,
) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []
    for rollout_observations, rollout_actions, rollout_positions, rollout_velocities, rollout_collisions in loader:
        if temporal_only:
            encoded_rollout = model.encode(
                rollout_observations.reshape(-1, *rollout_observations.shape[2:])
            ).reshape(rollout_observations.shape[0], rollout_observations.shape[1], -1)
            temporal_frame_features = model.encode_temporal_frames(rollout_observations)
            loss = torch.zeros((), dtype=rollout_observations.dtype)
            for step in range(unroll_horizon + 1):
                history_start = max(0, step + 1 - 4)
                history_latents = encoded_rollout[:, history_start:step + 1]
                predicted_velocity, predicted_std = model.temporal_velocity_stats_from_latents(
                    history_latents, temporal_frame_features[:, history_start:step + 1]
                )
                target_velocity = rollout_velocities[:, step]
                nll = 0.5 * (
                    ((target_velocity - predicted_velocity) / predicted_std).square()
                    + 2.0 * predicted_std.log()
                ).mean()
                position_prediction = model.temporal_position(temporal_frame_features[:, step])
                temporal_position_step_loss = nn.functional.mse_loss(position_prediction, rollout_positions[:, step])
                loss = loss + nn.functional.mse_loss(predicted_velocity, target_velocity) + 0.1 * nll + 0.5 * temporal_position_step_loss
            loss = loss / (unroll_horizon + 1)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach()))
            continue
        if kinematics_only:
            loss = torch.zeros((), dtype=rollout_observations.dtype)
            valid_steps = torch.zeros((), dtype=rollout_observations.dtype)
            for step in range(unroll_horizon):
                current_state = torch.cat((rollout_positions[:, step], rollout_velocities[:, step]), dim=-1)
                target_state = torch.cat((rollout_positions[:, step + 1], rollout_velocities[:, step + 1]), dim=-1)
                predicted_state = model.state_transition(current_state, rollout_actions[:, step])
                free_transition = 1.0 - rollout_collisions[:, step]
                per_sample_loss = nn.functional.mse_loss(predicted_state, target_state, reduction="none").mean(dim=-1)
                loss = loss + (per_sample_loss * free_transition).sum()
                valid_steps = valid_steps + free_transition.sum()
            loss = loss / valid_steps.clamp_min(1.0)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach()))
            continue
        if collision_only:
            loss = torch.zeros((), dtype=rollout_observations.dtype)
            for step in range(unroll_horizon):
                latent = model.encode(rollout_observations[:, step]).detach()
                state = model.state_from_latent(latent).detach()
                logits = model.collision_logits(latent, state, rollout_actions[:, step], observation=rollout_observations[:, step])
                targets = rollout_collisions[:, step]
                positives = targets.sum().clamp_min(1.0)
                negatives = (targets.numel() - targets.sum()).clamp_min(1.0)
                positive_weight = (negatives / positives).clamp(1.0, 12.0)
                loss = loss + nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)
            loss = loss / unroll_horizon
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach()))
            continue
        encoded_rollout = model.encode(
            rollout_observations.reshape(-1, *rollout_observations.shape[2:])
        ).reshape(rollout_observations.shape[0], rollout_observations.shape[1], -1)
        temporal_frame_features = model.encode_temporal_frames(rollout_observations)
        temporal_position_loss = nn.functional.mse_loss(
            model.temporal_position(temporal_frame_features), rollout_positions
        )
        open_latent = encoded_rollout[:, 0]
        open_state = model.state_from_latent(open_latent)
        initial_position, initial_velocity = model.kinematics(open_state)
        loss = 0.5 * nn.functional.mse_loss(initial_position, rollout_positions[:, 0]) + 0.25 * temporal_position_loss
        loss = loss + 0.2 * nn.functional.mse_loss(initial_velocity, rollout_velocities[:, 0])
        for step in range(unroll_horizon):
            action = rollout_actions[:, step]
            teacher_current_latent = encoded_rollout[:, step]
            teacher_current_state = model.state_from_latent(teacher_current_latent)
            teacher_latent = model.transition(teacher_current_latent, action)
            teacher_state = model.state_transition(teacher_current_state, action)
            teacher_prediction = model.compose_agent_rgb(model.decode(teacher_latent), teacher_latent, state=teacher_state)
            teacher_positions, teacher_velocities = model.kinematics(teacher_state)
            collision_logits = model.collision_logits(teacher_current_latent, teacher_current_state, action, observation=rollout_observations[:, step])
            collision_loss = nn.functional.binary_cross_entropy_with_logits(
                collision_logits,
                rollout_collisions[:, step],
                pos_weight=torch.tensor(5.0, device=collision_logits.device),
            )
            agent_mask_logits = model.agent_mask_logits(teacher_latent.detach(), state=teacher_state.detach())
            agent_mask_target = _agent_mask_targets(rollout_positions[:, step + 1])
            teacher_agent_mask_loss = _agent_mask_loss(agent_mask_logits, agent_mask_target)
            open_latent = model.transition(open_latent, action)
            open_state = model.state_transition(open_state, action)
            open_agent_mask_logits = model.agent_mask_logits(open_latent.detach(), state=open_state.detach())
            open_agent_mask_loss = _agent_mask_loss(open_agent_mask_logits, agent_mask_target)
            agent_mask_loss = 0.5 * (teacher_agent_mask_loss + open_agent_mask_loss)
            target_frame = rollout_observations[:, step + 1]
            image_loss = nn.functional.smooth_l1_loss(teacher_prediction, target_frame)
            target_agent_signal = target_frame[:, 1:2] - target_frame[:, 0:1]
            predicted_agent_signal = teacher_prediction[:, 1:2] - teacher_prediction[:, 0:1]
            agent_color_loss = nn.functional.smooth_l1_loss(predicted_agent_signal, target_agent_signal)
            position_loss = nn.functional.mse_loss(teacher_positions, rollout_positions[:, step + 1])
            velocity_loss = nn.functional.mse_loss(teacher_velocities, rollout_velocities[:, step + 1])
            open_positions, _ = model.kinematics(open_state)
            open_loss = nn.functional.mse_loss(open_positions, rollout_positions[:, step + 1])
            history_start = max(0, step + 1 - 4)
            history_latents = encoded_rollout[:, history_start:step + 1]
            temporal_velocity, temporal_velocity_std = model.temporal_velocity_stats_from_latents(
                history_latents, temporal_frame_features[:, history_start:step + 1]
            )
            temporal_velocity_target = rollout_velocities[:, step]
            temporal_velocity_nll = 0.5 * (
                ((temporal_velocity_target - temporal_velocity) / temporal_velocity_std).square()
                + 2.0 * temporal_velocity_std.log()
            ).mean()
            temporal_velocity_loss = nn.functional.mse_loss(temporal_velocity, temporal_velocity_target) + 0.1 * temporal_velocity_nll
            _, state_uncertainty_std = model.transition_state_stats(
                teacher_current_latent, teacher_current_state, action, next_state=teacher_state
            )
            target_state = torch.cat((rollout_positions[:, step + 1], rollout_velocities[:, step + 1]), dim=-1)
            state_uncertainty_nll = 0.5 * (
                ((target_state - teacher_state) / state_uncertainty_std).square()
                + 2.0 * state_uncertainty_std.log()
            ).mean()
            loss = loss + image_loss + 2.0 * agent_color_loss + position_loss + 0.2 * velocity_loss + 0.25 * open_loss + 0.5 * collision_loss + 0.1 * agent_mask_loss + 0.5 * temporal_velocity_loss + 0.05 * state_uncertainty_nll
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
    resume: str | None = None,
    agent_only: bool = False,
    collision_only: bool = False,
    collision_seek_probability: float = 0.0,
    kinematics_only: bool = False,
    temporal_only: bool = False,
    map_suite: str = "baseline",
) -> Path:
    torch.manual_seed(seed)
    train_batch = collect_random_rollouts(episodes=episodes, horizon=unroll_horizon, seed=seed, sticky_probability=sticky_probability, full_state_range=full_state_range, barrier_probability=barrier_probability, collision_seek_probability=collision_seek_probability, map_suite=map_suite)
    validation_batch = collect_random_rollouts(episodes=validation_episodes, horizon=unroll_horizon, seed=seed + 10000, sticky_probability=sticky_probability, full_state_range=full_state_range, barrier_probability=barrier_probability, collision_seek_probability=collision_seek_probability, map_suite=map_suite)
    train_loader = _make_loader(train_batch, batch_size=batch_size, shuffle=True)
    validation_loader = _make_loader(validation_batch, batch_size=batch_size, shuffle=False)
    model = PocketWorldModel()
    if resume:
        payload = torch.load(resume, map_location="cpu")
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        if missing:
            print(f"warning: checkpoint is missing {len(missing)} keys; newly initialized heads will be trained")
        if unexpected:
            print(f"warning: checkpoint has {len(unexpected)} legacy keys")
    if sum((agent_only, collision_only, kinematics_only, temporal_only)) > 1:
        raise ValueError("agent_only, collision_only, kinematics_only, and temporal_only are mutually exclusive")
    if agent_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("state_agent_renderer")
    elif collision_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("spatial_collision_head")
    elif kinematics_only:
        kinematic_parameters = {"action_acceleration_logit", "friction_logit", "max_speed_logit"}
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name in kinematic_parameters
    elif temporal_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (
                name.startswith("temporal_frame_encoder")
                or name.startswith("temporal_position_head")
                or name.startswith("temporal_velocity_encoder")
                or name.startswith("temporal_velocity_head")
            )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=1e-2 if kinematics_only else 2e-3)
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_loader, optimizer, unroll_horizon, collision_only=collision_only, kinematics_only=kinematics_only, temporal_only=temporal_only)
        with torch.no_grad():
            validation_loss = _run_epoch(model, validation_loader, None, unroll_horizon, collision_only=collision_only, kinematics_only=kinematics_only, temporal_only=temporal_only)
        print(f"epoch {epoch + 1:02d}/{epochs:02d} train={train_loss:.5f} val={validation_loss:.5f}")
    uncertainty_calibration = {}
    if not (agent_only or collision_only or kinematics_only or temporal_only):
        uncertainty_calibration = model.fit_uncertainty_calibration(
            torch.from_numpy(validation_batch.observations).float() / 255.0,
            torch.from_numpy(validation_batch.actions),
            torch.from_numpy(validation_batch.positions / 64.0),
            torch.from_numpy(validation_batch.velocities / 3.0).clamp(-1.0, 1.0),
        )
        print(f"uncertainty calibration: {uncertainty_calibration}")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "seed": seed, "epochs": epochs, "episodes": episodes, "validation_episodes": validation_episodes, "unroll_horizon": unroll_horizon, "sticky_probability": sticky_probability, "full_state_range": full_state_range, "barrier_probability": barrier_probability, "collision_seek_probability": collision_seek_probability, "map_suite": map_suite, "resume": resume, "agent_only": agent_only, "collision_only": collision_only, "kinematics_only": kinematics_only, "temporal_only": temporal_only, "collision_supervision": True, "agent_rendering": True, "temporal_velocity": True, "probabilistic_uncertainty": True, "uncertainty_calibration": uncertainty_calibration, "dynamics": "learned_structured_kinematics"}, destination)
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
    parser.add_argument("--resume", default=None, help="initialize from an existing checkpoint")
    parser.add_argument("--agent-only", action="store_true", help="freeze the world model and fine-tune only the state-conditioned agent renderer")
    parser.add_argument("--collision-only", action="store_true", help="freeze the world model and fine-tune only the wall-relative collision head")
    parser.add_argument("--collision-seek-probability", type=float, default=0.0, help="probability of collision-seeking actions in barrier curriculum episodes")
    parser.add_argument("--kinematics-only", action="store_true", help="identify acceleration, friction, and speed limit from collision-free transitions")
    parser.add_argument("--temporal-only", action="store_true", help="freeze the world model and fine-tune only the learned temporal velocity encoder")
    parser.add_argument("--map-suite", choices=("baseline", "train", "holdout", "all"), default="baseline", help="named map suite used for rollout collection")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
