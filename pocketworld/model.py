from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


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
        self.agent_renderer = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 64 * 64)
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

    def agent_mask_logits(self, latent: torch.Tensor) -> torch.Tensor:
        return self.agent_renderer(latent).view(-1, 1, 64, 64)

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

    def compose_agent_rgb(self, frame: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Overlay a state-conditioned RGB agent on a decoded background frame."""
        state = self.state_from_latent(latent)
        mask = self.agent_geometry_mask(state)
        color = frame.new_tensor((93.0 / 255.0, 224.0 / 255.0, 183.0 / 255.0))[None, :, None, None]
        return frame * (1.0 - mask) + color * mask

    def transition(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat((latent, self.action_embedding(action)), dim=-1), latent)

    def predict_next(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        latent = self.transition(self.encode(observation), action)
        return self.decode(latent)

    def predict_next_state(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.transition(self.encode(observation), action)
        position, _ = self.kinematics(self.state_from_latent(latent))
        return self.compose_agent_rgb(self.decode(latent), latent), position

    def state_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        raw = self.state_encoder(latent)
        return torch.cat((torch.sigmoid(raw[..., :2]), torch.tanh(raw[..., 2:])), dim=-1)

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

    def wall_patch(self, observation: torch.Tensor, state: torch.Tensor, radius: int = 3) -> torch.Tensor:
        """Sample a 7x7 wall occupancy patch around the predicted next position."""
        red, green, blue = observation[:, 0], observation[:, 1], observation[:, 2]
        wall = ((red >= 60 / 255) & (red <= 140 / 255) & (green >= 70 / 255) & (green <= 160 / 255) & (blue >= 80 / 255) & (blue <= 180 / 255)).float()
        center = state[..., :2] * 2.0 - 1.0
        offsets = torch.arange(-radius, radius + 1, device=observation.device, dtype=observation.dtype) * (2.0 / 63.0)
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1)[None] + center[:, None, None, :]
        patch = F.grid_sample(wall[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return patch.flatten(1)

    def collision_logits(self, latent: torch.Tensor, state: torch.Tensor, action: torch.Tensor, observation: torch.Tensor | None = None) -> torch.Tensor:
        action_one_hot = F.one_hot(action, num_classes=4).float()
        base_features = torch.cat((latent, state, action_one_hot), dim=-1)
        logits = self.collision_head(base_features).squeeze(-1)
        if observation is not None:
            spatial_features = torch.cat((base_features, self.wall_patch(observation, self.state_transition(state, action))), dim=-1)
            logits = logits + self.spatial_collision_head(spatial_features).squeeze(-1)
        return logits

    def kinematics(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return state[..., :2], state[..., 2:]

    @torch.no_grad()
    def imagine_positions(self, observation: torch.Tensor, actions: torch.Tensor, collision_response: bool = False) -> torch.Tensor:
        """Return normalized [x, y] positions for every imagined future step.

        With ``collision_response=True``, a predicted collision freezes position
        and clears velocity before the next imagined action.
        """
        state = self.state_from_latent(self.encode(observation))
        static_latent = self.encode(observation)
        positions = []
        for index in range(actions.shape[1]):
            action = actions[:, index]
            next_state = self.state_transition(state, action)
            if collision_response:
                collision_probability = torch.sigmoid(self.collision_logits(static_latent, state, action, observation=observation))
                collision = (collision_probability >= 0.5).unsqueeze(-1)
                stopped_state = torch.cat((state[..., :2], torch.zeros_like(state[..., 2:])), dim=-1)
                state = torch.where(collision, stopped_state, next_state)
            else:
                state = next_state
            positions.append(state[..., :2])
        return torch.stack(positions, dim=1)

    @torch.no_grad()
    def imagine_collision_probabilities(self, observation: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict collision probability for each imagined action step."""
        latent = self.encode(observation)
        state = self.state_from_latent(latent)
        probabilities = []
        for index in range(actions.shape[1]):
            action = actions[:, index]
            probabilities.append(torch.sigmoid(self.collision_logits(latent, state, action, observation=observation)))
            state = self.state_transition(state, action)
        return torch.stack(probabilities, dim=1)

    @torch.no_grad()
    def imagine_agent_masks(self, observation: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return detached agent-renderer masks for each imagined step."""
        latent = self.encode(observation)
        masks = []
        for index in range(actions.shape[1]):
            latent = self.transition(latent, actions[:, index])
            masks.append(torch.sigmoid(self.agent_mask_logits(latent)))
        return torch.stack(masks, dim=1)

    @torch.no_grad()
    def imagine(self, observation: torch.Tensor, actions: torch.Tensor, compose_agent: bool = True) -> torch.Tensor:
        """Return the starting frame plus imagined frames for [batch, horizon] actions."""
        latent = self.encode(observation)
        frames = [observation]
        for index in range(actions.shape[1]):
            latent = self.transition(latent, actions[:, index])
            decoded = self.decode(latent)
            frames.append(self.compose_agent_rgb(decoded, latent) if compose_agent else decoded)
        return torch.stack(frames, dim=1)
