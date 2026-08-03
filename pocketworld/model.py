from __future__ import annotations

import torch
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

    def transition(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat((latent, self.action_embedding(action)), dim=-1), latent)

    def predict_next(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        latent = self.transition(self.encode(observation), action)
        return self.decode(latent)

    @torch.no_grad()
    def imagine(self, observation: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return the starting frame plus imagined frames for [batch, horizon] actions."""
        latent = self.encode(observation)
        frames = [observation]
        for index in range(actions.shape[1]):
            latent = self.transition(latent, actions[:, index])
            frames.append(self.decode(latent))
        return torch.stack(frames, dim=1)

