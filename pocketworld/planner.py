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
    guided_fraction: float = 0.35,
) -> PlanResult:
    model.eval()
    start = torch.from_numpy(observation[None]).float().to(device) / 255.0
    actions = torch.randint(0, 4, (candidates, horizon), device=device)
    delta = np.asarray(goal, dtype=np.float32) - extract_agent_position(observation)
    preferred_action = 3 if abs(delta[0]) >= abs(delta[1]) and delta[0] >= 0 else 2 if abs(delta[0]) >= abs(delta[1]) else 1 if delta[1] >= 0 else 0
    guided = torch.rand((candidates, horizon), device=device) < guided_fraction
    actions = torch.where(guided, torch.full_like(actions, preferred_action), actions)
    starts = start.expand(candidates, -1, -1, -1)
    start_position = extract_agent_position(observation).astype(np.float32)
    imagined_positions = model.imagine_positions(starts, actions).cpu().numpy() * 64.0
    positions = np.concatenate((np.broadcast_to(start_position, (candidates, 1, 2)), imagined_positions), axis=1)
    distances = np.linalg.norm(positions[:, -1] - np.asarray(goal), axis=-1)
    safe_distances = np.where(np.isfinite(distances), distances, 1e6)
    best = int(np.argmin(safe_distances))
    return PlanResult(actions=actions[best].cpu().numpy(), imagined_positions=positions[best], imagined_distance=float(safe_distances[best]))
