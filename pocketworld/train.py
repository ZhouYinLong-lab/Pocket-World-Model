from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import collect_random_transitions
from .model import PocketWorldModel


def train(epochs: int = 5, episodes: int = 100, batch_size: int = 64, seed: int = 7, output: str = "artifacts/pocketworld.pt") -> Path:
    torch.manual_seed(seed)
    batch = collect_random_transitions(episodes=episodes, seed=seed)
    x = torch.from_numpy(batch.observations).float() / 255.0
    y = torch.from_numpy(batch.next_observations).float() / 255.0
    a = torch.from_numpy(batch.actions)
    loader = DataLoader(TensorDataset(x, a, y), batch_size=batch_size, shuffle=True)
    model = PocketWorldModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for epoch in range(epochs):
        losses = []
        for observations, actions, targets in loader:
            prediction = model.predict_next(observations, actions)
            loss = loss_fn(prediction, targets)
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts/pocketworld.pt")
    args = parser.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()

