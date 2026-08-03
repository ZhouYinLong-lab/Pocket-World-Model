from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import PocketWorldModel


@dataclass
class PlanResult:
    actions: np.ndarray
    imagined_positions: np.ndarray
    imagined_distance: float


def extract_agent_position(frame: np.ndarray) -> np.ndarray:
    """Locate the mint-green agent in a CHW uint8 or BCHW float frame."""
    if frame.dtype != np.uint8:
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
    batched = frame.ndim == 4
    if not batched:
        frame = frame[None]
    positions = []
    for sample in frame:
        mask = (sample[1] > sample[0] + 25) & (sample[1] > sample[2] + 15)
        ys, xs = np.where(mask)
        positions.append((xs.mean(), ys.mean()) if len(xs) else (np.nan, np.nan))
    result = np.asarray(positions, dtype=np.float32)
    return result if batched else result[0]


@torch.no_grad()
def random_shooting(
    model: PocketWorldModel,
    observation: np.ndarray,
    goal: tuple[float, float],
    horizon: int = 12,
    candidates: int = 256,
    device: str = "cpu",
) -> PlanResult:
    model.eval()
    start = torch.from_numpy(observation[None]).float().to(device) / 255.0
    actions = torch.randint(0, 4, (candidates, horizon), device=device)
    starts = start.expand(candidates, -1, -1, -1)
    imagined = model.imagine(starts, actions)
    positions = np.stack([extract_agent_position(imagined[:, step].cpu().numpy()) for step in range(horizon + 1)])
    distances = np.linalg.norm(positions[-1] - np.asarray(goal), axis=-1) if positions.ndim == 3 else None
    if distances is None:
        # Image decoding is not guaranteed to produce a detectable color early in training.
        distances = np.full(candidates, 1e6)
    safe_distances = np.where(np.isfinite(distances), distances, 1e6)
    best = int(np.argmin(safe_distances))
    return PlanResult(actions=actions[best].cpu().numpy(), imagined_positions=positions[:, best], imagined_distance=float(safe_distances[best]))
