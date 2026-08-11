from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def observable_velocity_from_frames(
    observation_history: np.ndarray,
    max_speed: float = 2.3,
) -> np.ndarray:
    """Estimate pixel velocity from the mint-green agent in RGB frames.

    This deliberately uses only information available to the deployed RGB
    agent.  It is kept next to the calibration code so the held-out scale fit
    and online evaluation cannot accidentally use simulator state.
    """
    frames = np.asarray(observation_history)
    if frames.ndim == 3:
        frames = frames[None]
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("expected RGB history with shape [time, channels, height, width]")
    if frames.dtype != np.uint8:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
    positions = []
    for frame in frames[-4:]:
        red, green, blue = frame.astype(np.int16)
        mask = (green > red + 25) & (green > blue + 15)
        ys, xs = np.where(mask)
        positions.append((xs.mean(), ys.mean()) if len(xs) else (np.nan, np.nan))
    positions_array = np.asarray(positions, dtype=np.float32)
    positions_array = positions_array[np.isfinite(positions_array).all(axis=1)]
    if len(positions_array) < 2:
        return np.zeros(2, dtype=np.float32)
    differences = np.diff(positions_array, axis=0)
    if np.linalg.norm(differences[-1]) <= 0.15:
        return np.zeros(2, dtype=np.float32)
    weights = np.arange(1, len(differences) + 1, dtype=np.float32)
    velocity = np.average(differences, axis=0, weights=weights)
    speed = float(np.linalg.norm(velocity))
    if speed > max_speed:
        velocity *= max_speed / speed
    return velocity.astype(np.float32)


class PocketWorldModel(nn.Module):
    """Deterministic image world model: encode -> dynamics -> decode."""

    def __init__(self, latent_dim: int = 64, action_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 8 * 8, latent_dim), nn.Tanh(),
        )
        self.action_embedding = nn.Embedding(4, action_dim)
        self.dynamics = nn.GRUCell(latent_dim + action_dim, latent_dim)
        self.state_encoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 4))
        self.temporal_frame_encoder = nn.Sequential(
            nn.Conv2d(3, 8, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(16 * 8 * 8, 16), nn.ReLU(),
        )
        self.temporal_position_head = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 2))
        self.temporal_velocity_encoder = nn.GRU(latent_dim * 2 + 16, 32, batch_first=True)
        self.temporal_velocity_head = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 4)
        )
        self.register_buffer(
            "action_directions",
            torch.tensor(((0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0))),
        )
        self.action_acceleration_logit = nn.Parameter(torch.tensor(-1.2))
        self.friction_logit = nn.Parameter(torch.tensor(0.8))
        self.max_speed_logit = nn.Parameter(torch.tensor(-0.2))
        self.state_dynamics = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 4))
        self.collision_head = nn.Sequential(
            nn.Linear(latent_dim + 4 + 4, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.spatial_collision_head = nn.Sequential(
            nn.Linear(latent_dim + 4 + 4 + 49, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.state_uncertainty_head = nn.Sequential(
            nn.Linear(latent_dim + 4 + 4, 64), nn.ReLU(), nn.Linear(64, 4)
        )
        self.register_buffer("uncertainty_scale", torch.ones(4))
        self.agent_renderer = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 64 * 64)
        )
        self.state_agent_renderer = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 64 * 64)
        )
        nn.init.zeros_(self.spatial_collision_head[-1].weight)
        nn.init.zeros_(self.spatial_collision_head[-1].bias)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8), nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder(observation)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def agent_mask_logits(self, latent: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """Decode an agent mask from the compact state, with a latent fallback for compatibility."""
        features = self.state_from_latent(latent) if state is None else state
        return self.state_agent_renderer(features).view(-1, 1, 64, 64)

    def agent_geometry_mask(self, state: torch.Tensor, radius: float = 3.0) -> torch.Tensor:
        """Render a soft circular agent mask from the normalized state position."""
        coordinate = torch.linspace(0.0, 1.0, 64, device=state.device, dtype=state.dtype)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        distance = torch.sqrt(
            (xx[None] - state[:, 0, None, None]) ** 2
            + (yy[None] - state[:, 1, None, None]) ** 2
            + 1e-8
        ) * 64.0
        return torch.sigmoid((radius + 0.5 - distance) / 0.7).unsqueeze(1)

    def compose_agent_rgb(
        self,
        frame: torch.Tensor,
        latent: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Overlay a state-conditioned RGB agent on a decoded background frame."""
        state = self.state_from_latent(latent) if state is None else state
        mask = self.agent_geometry_mask(state)
        color = frame.new_tensor((93.0 / 255.0, 224.0 / 255.0, 183.0 / 255.0))[None, :, None, None]
        hard_mask = (mask >= 0.5).to(frame.dtype)
        return frame * (1.0 - hard_mask) + color * hard_mask

    def transition(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat((latent, self.action_embedding(action)), dim=-1), latent)

    def predict_next(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        latent = self.transition(self.encode(observation), action)
        return self.decode(latent)

    def predict_next_state(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.transition(self.encode(observation), action)
        state = self.state_transition(self.state_from_latent(self.encode(observation)), action)
        position, _ = self.kinematics(state)
        return self.compose_agent_rgb(self.decode(latent), latent, state=state), position

    def state_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        raw = self.state_encoder(latent)
        return torch.cat((torch.sigmoid(raw[..., :2]), torch.tanh(raw[..., 2:])), dim=-1)

    def temporal_velocity_stats_from_latents(
        self,
        latent_history: torch.Tensor,
        frame_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict normalized velocity from a sequence of encoded observations.

        The representation is intentionally small: a GRU summarizes up to a few
        recent latent frames, while the head predicts both velocity mean and a
        bounded auxiliary scale. The mean is used by the planner; the scale is
        useful for diagnosing when the visual history is insufficient.
        """
        if latent_history.ndim != 3:
            raise ValueError("expected latent history with shape [batch, time, latent]")
        deltas = torch.cat((torch.zeros_like(latent_history[:, :1]), latent_history[:, 1:] - latent_history[:, :-1]), dim=1)
        if frame_features is None:
            frame_features = torch.zeros(
                (*latent_history.shape[:2], 16), device=latent_history.device, dtype=latent_history.dtype
            )
        if frame_features.shape[:2] != latent_history.shape[:2]:
            raise ValueError("frame feature history must align with latent history")
        temporal_features = torch.cat((latent_history, deltas, frame_features), dim=-1)
        _, hidden = self.temporal_velocity_encoder(temporal_features)
        raw = self.temporal_velocity_head(hidden[-1])
        mean = torch.tanh(raw[..., :2])
        std = (0.005 + 0.25 * F.softplus(raw[..., 2:])).clamp(0.005, 1.0)
        return mean, std

    def temporal_velocity_stats(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict velocity statistics from normalized BCHW or BTCHW RGB observations."""
        if observations.ndim == 4:
            observations = observations.unsqueeze(1)
        if observations.ndim != 5:
            raise ValueError("expected observations with shape [batch, time, channels, height, width]")
        batch, time = observations.shape[:2]
        latents = self.encode(observations.reshape(batch * time, *observations.shape[2:]))
        latents = latents.reshape(batch, time, -1)
        frame_features = self.temporal_frame_encoder(
            observations.reshape(batch * time, *observations.shape[2:])
        ).reshape(batch, time, -1)
        return self.temporal_velocity_stats_from_latents(latents, frame_features)

    def encode_temporal_frames(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode RGB frames for the learnable temporal motion pathway."""
        if observations.ndim == 4:
            observations = observations.unsqueeze(1)
        batch, time = observations.shape[:2]
        return self.temporal_frame_encoder(
            observations.reshape(batch * time, *observations.shape[2:])
        ).reshape(batch, time, -1)

    def temporal_position(self, frame_features: torch.Tensor) -> torch.Tensor:
        """Decode normalized agent position from the learned motion features."""
        return torch.sigmoid(self.temporal_position_head(frame_features))

    def state_from_history(self, observations: torch.Tensor) -> torch.Tensor:
        """Use the latest frame for position and learned temporal evidence for velocity."""
        if observations.ndim == 4:
            observations = observations.unsqueeze(1)
        latest_latent = self.encode(observations[:, -1])
        state = self.state_from_latent(latest_latent)
        velocity, _ = self.temporal_velocity_stats(observations)
        return torch.cat((state[..., :2], velocity), dim=-1)

    def state_transition(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Advance the compact state with a learned, interpretable kinematic prior.

        Position is normalized to [0, 1] and velocity to roughly [-1, 1].
        The action directions are known from the environment; acceleration,
        friction, and speed limit are learned from rollout supervision.
        """
        action_direction = F.embedding(action, self.action_directions)
        acceleration = 0.02 + 0.50 * torch.sigmoid(self.action_acceleration_logit)
        friction = 0.50 + 0.49 * torch.sigmoid(self.friction_logit)
        max_speed = 0.40 + 0.80 * torch.sigmoid(self.max_speed_logit)
        velocity = state[..., 2:] * friction + action_direction * acceleration
        speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True).clamp_min(1e-6)
        velocity = velocity * torch.minimum(torch.ones_like(speed), max_speed / speed)
        position = state[..., :2] + velocity * (3.0 / 64.0)
        position = position.clamp(3.0 / 64.0, 61.0 / 64.0)
        return torch.cat((position, velocity.clamp(-1.0, 1.0)), dim=-1)

    def transition_state_stats(
        self,
        latent: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the learned transition mean and calibrated diagonal stddev."""
        mean = self.state_transition(state, action) if next_state is None else next_state
        action_one_hot = F.one_hot(action, num_classes=4).float()
        features = torch.cat((latent, state, action_one_hot), dim=-1)
        raw_std = self.state_uncertainty_head(features)
        std = (0.002 + 0.25 * F.softplus(raw_std)).clamp(0.002, 1.0)
        std = std * self.uncertainty_scale.to(device=std.device, dtype=std.dtype)
        return mean, std

    @torch.no_grad()
    def fit_uncertainty_calibration(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        coverage: float = 0.90,
        observed_velocity_blend: float = 0.50,
    ) -> dict[str, object]:
        """Calibrate marginal Gaussian scales on a held-out rollout set.

        This is a lightweight split-calibration procedure: the model predicts
        transition means and raw scales, then a held-out residual quantile sets
        one scale per position/velocity coordinate. It is not a Bayesian
        posterior, but its empirical interval coverage is directly measurable.
        """
        if not 0.5 < coverage < 0.999:
            raise ValueError("coverage must be between 0.5 and 0.999")
        if not 0.0 <= observed_velocity_blend <= 1.0:
            raise ValueError("observed_velocity_blend must be between 0 and 1")
        was_training = self.training
        self.eval()
        self.uncertainty_scale.fill_(1.0)
        normalized_states = torch.cat((positions, velocities), dim=-1)
        ratios = []
        for step in range(actions.shape[1]):
            latent = self.encode(observations[:, step])
            history = observations[:, : step + 1]
            state = self.state_from_history(history)
            if observed_velocity_blend > 0.0:
                observed_velocity = torch.as_tensor(
                    np.stack([
                        observable_velocity_from_frames(history[index].detach().cpu().numpy())
                        for index in range(history.shape[0])
                    ]),
                    device=state.device,
                    dtype=state.dtype,
                ) / 3.0
                velocity = (
                    (1.0 - observed_velocity_blend) * state[..., 2:]
                    + observed_velocity_blend * observed_velocity
                )
                state = torch.cat((state[..., :2], velocity.clamp(-1.0, 1.0)), dim=-1)
            target = normalized_states[:, step + 1]
            mean, std = self.transition_state_stats(latent, state, actions[:, step])
            ratios.append(((target - mean).abs() / std.clamp_min(1e-6)).detach())
        ratio = torch.cat(ratios, dim=0)
        # RGB velocity is the least stationary coordinate under speed shifts.
        # Use a conservative tail for those two dimensions while retaining a
        # nominal split-conformal quantile for position.  This avoids a narrow
        # ID velocity interval becoming overconfident on fast dynamics.
        quantile = torch.quantile(ratio, coverage, dim=0)
        velocity_tail_quantile = torch.quantile(
            ratio[:, 2:], min(0.98, coverage + 0.05), dim=0
        )
        quantile[2:] = velocity_tail_quantile
        normal_quantile = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600}.get(round(coverage, 2), 1.6449)
        scale = (quantile / normal_quantile).clamp(0.25, 12.0)
        self.uncertainty_scale.copy_(scale)
        if was_training:
            self.train()
        return {
            "coverage": coverage,
            "normal_quantile": normal_quantile,
            "scale": [float(value) for value in scale],
            "samples": int(ratio.shape[0]),
            "state_representation": "learned_temporal_velocity+observable_rgb_velocity",
            "observed_velocity_blend": observed_velocity_blend,
            "velocity_tail_quantile": min(0.98, coverage + 0.05),
        }

    @torch.no_grad()
    def probabilistic_collision_probability(
        self,
        latent: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        observation: torch.Tensor,
        next_state: torch.Tensor,
        accumulated_std: torch.Tensor,
        samples: int = 32,
    ) -> torch.Tensor:
        """Estimate collision probability under calibrated state uncertainty."""
        samples = max(4, int(samples))
        batch = next_state.shape[0]
        noise = torch.randn(batch, samples, 2, device=next_state.device, dtype=next_state.dtype)
        sampled_position = next_state[:, None, :2] + noise * accumulated_std[:, None, :2]
        sampled_position = sampled_position.clamp(3.0 / 64.0, 61.0 / 64.0)
        sampled_state = torch.cat(
            (sampled_position, next_state[:, None, 2:].expand(-1, samples, -1)), dim=-1
        ).reshape(batch * samples, 4)
        logits = self.collision_logits(
            latent.repeat_interleave(samples, dim=0),
            state.repeat_interleave(samples, dim=0),
            action.repeat_interleave(samples, dim=0),
            observation.repeat_interleave(samples, dim=0),
            next_state=sampled_state,
        )
        return torch.sigmoid(logits).reshape(batch, samples).mean(dim=1)

    def wall_patch(self, observation: torch.Tensor, state: torch.Tensor, radius: int = 3) -> torch.Tensor:
        """Sample a 7x7 wall occupancy patch around the predicted next position."""
        red, green, blue = observation[:, 0], observation[:, 1], observation[:, 2]
        wall = ((red >= 60 / 255) & (red <= 140 / 255) & (green >= 70 / 255) & (green <= 160 / 255) & (blue >= 80 / 255) & (blue <= 180 / 255)).float()
        wall = wall.clone()
        wall[:, (0, -1), :] = 1.0
        wall[:, :, (0, -1)] = 1.0
        center = state[..., :2] * 2.0 - 1.0
        offsets = torch.arange(-radius, radius + 1, device=observation.device, dtype=observation.dtype) * (2.0 / 63.0)
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1)[None] + center[:, None, None, :]
        patch = F.grid_sample(wall[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return patch.flatten(1)

    def collision_logits(
        self,
        latent: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        observation: torch.Tensor | None = None,
        next_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action_one_hot = F.one_hot(action, num_classes=4).float()
        base_features = torch.cat((latent, state, action_one_hot), dim=-1)
        if observation is None:
            return self.collision_head(base_features).squeeze(-1)
        landing_state = self.state_transition(state, action) if next_state is None else next_state
        spatial_features = torch.cat((base_features, self.wall_patch(observation, landing_state)), dim=-1)
        return self.spatial_collision_head(spatial_features).squeeze(-1)

    def robust_collision_probability(
        self,
        latent: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        observation: torch.Tensor,
        next_state: torch.Tensor | None = None,
        uncertainty_radius_px: float = 0.0,
    ) -> torch.Tensor:
        """Return worst-case learned collision risk around the predicted landing state."""
        next_state = self.state_transition(state, action) if next_state is None else next_state
        if uncertainty_radius_px <= 0:
            return torch.sigmoid(self.collision_logits(latent, state, action, observation, next_state=next_state))
        radius = uncertainty_radius_px / 64.0
        offsets = next_state.new_tensor(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))) * radius
        scenario_count = offsets.shape[0]
        uncertain_positions = next_state[:, None, :2] + offsets[None]
        uncertain_positions = uncertain_positions.clamp(3.0 / 64.0, 61.0 / 64.0)
        uncertain_states = torch.cat(
            (
                uncertain_positions,
                next_state[:, None, 2:].expand(-1, scenario_count, -1),
            ),
            dim=-1,
        ).flatten(0, 1)
        repeated_latent = latent.repeat_interleave(scenario_count, dim=0)
        repeated_state = state.repeat_interleave(scenario_count, dim=0)
        repeated_action = action.repeat_interleave(scenario_count, dim=0)
        repeated_observation = observation.repeat_interleave(scenario_count, dim=0)
        logits = self.collision_logits(
            repeated_latent,
            repeated_state,
            repeated_action,
            repeated_observation,
            next_state=uncertain_states,
        )
        return torch.sigmoid(logits).view(-1, scenario_count).amax(dim=1)

    def kinematics(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return state[..., :2], state[..., 2:]

    @torch.no_grad()
    def imagine_positions(
        self,
        observation: torch.Tensor,
        actions: torch.Tensor,
        collision_response: bool = False,
        visual_collision_guard: bool = False,
        initial_position: torch.Tensor | None = None,
        initial_velocity: torch.Tensor | None = None,
        uncertainty_radius_px: float = 0.0,
        uncertainty_growth_px: float = 0.0,
        probabilistic_uncertainty: bool = False,
        uncertainty_samples: int = 32,
    ) -> torch.Tensor:
        """Return normalized [x, y] positions for every imagined future step.

        With ``collision_response=True``, a predicted collision freezes position
        and clears velocity before the next imagined action.
        """
        state = self.state_from_latent(self.encode(observation))
        if initial_position is not None:
            state = torch.cat((initial_position.to(state), state[..., 2:]), dim=-1)
        if initial_velocity is not None:
            state = torch.cat((state[..., :2], initial_velocity.to(state)), dim=-1)
        static_latent = self.encode(observation)
        accumulated_std = torch.zeros_like(state)
        positions = []
        for index in range(actions.shape[1]):
            action = actions[:, index]
            next_state = self.state_transition(state, action)
            if collision_response:
                if probabilistic_uncertainty:
                    _, step_std = self.transition_state_stats(static_latent, state, action, next_state=next_state)
                    accumulated_std = torch.sqrt(accumulated_std.square() + step_std.square())
                    collision_probability = self.probabilistic_collision_probability(
                        static_latent,
                        state,
                        action,
                        observation,
                        next_state,
                        accumulated_std,
                        samples=uncertainty_samples,
                    )
                else:
                    radius = uncertainty_radius_px + uncertainty_growth_px * (index + 1) ** 0.5
                    collision_probability = self.robust_collision_probability(
                        static_latent,
                        state,
                        action,
                        observation,
                        next_state=next_state,
                        uncertainty_radius_px=radius,
                    )
                if visual_collision_guard:
                    visual_probability = self.wall_patch(observation, next_state).amax(dim=1)
                    collision_probability = torch.maximum(collision_probability, visual_probability)
                collision = (collision_probability >= 0.5).unsqueeze(-1)
                stopped_state = torch.cat((state[..., :2], torch.zeros_like(state[..., 2:])), dim=-1)
                state = torch.where(collision, stopped_state, next_state)
                accumulated_std = torch.where(collision, torch.zeros_like(accumulated_std), accumulated_std)
            else:
                state = next_state
            positions.append(state[..., :2])
        return torch.stack(positions, dim=1)

    @torch.no_grad()
    def imagine_collision_probabilities(
        self,
        observation: torch.Tensor,
        actions: torch.Tensor,
        visual_collision_guard: bool = False,
        collision_response: bool = True,
        initial_position: torch.Tensor | None = None,
        initial_velocity: torch.Tensor | None = None,
        uncertainty_radius_px: float = 0.0,
        uncertainty_growth_px: float = 0.0,
        probabilistic_uncertainty: bool = False,
        uncertainty_samples: int = 32,
    ) -> torch.Tensor:
        """Predict collision probability for each imagined action step."""
        latent = self.encode(observation)
        state = self.state_from_latent(latent)
        if initial_position is not None:
            state = torch.cat((initial_position.to(state), state[..., 2:]), dim=-1)
        if initial_velocity is not None:
            state = torch.cat((state[..., :2], initial_velocity.to(state)), dim=-1)
        accumulated_std = torch.zeros_like(state)
        probabilities = []
        for index in range(actions.shape[1]):
            action = actions[:, index]
            next_state = self.state_transition(state, action)
            if probabilistic_uncertainty:
                _, step_std = self.transition_state_stats(latent, state, action, next_state=next_state)
                accumulated_std = torch.sqrt(accumulated_std.square() + step_std.square())
                probability = self.probabilistic_collision_probability(
                    latent,
                    state,
                    action,
                    observation,
                    next_state,
                    accumulated_std,
                    samples=uncertainty_samples,
                )
            else:
                radius = uncertainty_radius_px + uncertainty_growth_px * (index + 1) ** 0.5
                probability = self.robust_collision_probability(
                    latent,
                    state,
                    action,
                    observation,
                    next_state=next_state,
                    uncertainty_radius_px=radius,
                )
            if visual_collision_guard:
                probability = torch.maximum(probability, self.wall_patch(observation, next_state).amax(dim=1))
            probabilities.append(probability)
            if collision_response:
                collision = (probability >= 0.5).unsqueeze(-1)
                stopped_state = torch.cat((state[..., :2], torch.zeros_like(state[..., 2:])), dim=-1)
                state = torch.where(collision, stopped_state, next_state)
                accumulated_std = torch.where(collision, torch.zeros_like(accumulated_std), accumulated_std)
            else:
                state = next_state
        return torch.stack(probabilities, dim=1)

    @torch.no_grad()
    def imagine_agent_masks(self, observation: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return detached agent-renderer masks for each imagined step."""
        latent = self.encode(observation)
        state = self.state_from_latent(latent)
        masks = []
        for index in range(actions.shape[1]):
            latent = self.transition(latent, actions[:, index])
            state = self.state_transition(state, actions[:, index])
            masks.append(torch.sigmoid(self.agent_mask_logits(latent, state=state)))
        return torch.stack(masks, dim=1)

    @torch.no_grad()
    def imagine(self, observation: torch.Tensor, actions: torch.Tensor, compose_agent: bool = True) -> torch.Tensor:
        """Return the starting frame plus imagined frames for [batch, horizon] actions."""
        latent = self.encode(observation)
        state = self.state_from_latent(latent)
        frames = [observation]
        for index in range(actions.shape[1]):
            latent = self.transition(latent, actions[:, index])
            state = self.state_transition(state, actions[:, index])
            decoded = self.decode(latent)
            frames.append(self.compose_agent_rgb(decoded, latent, state=state) if compose_agent else decoded)
        return torch.stack(frames, dim=1)
