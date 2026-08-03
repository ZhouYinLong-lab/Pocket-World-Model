from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .model import PocketWorldModel


@dataclass
class PlanResult:
    actions: np.ndarray
    imagined_positions: np.ndarray
    imagined_distance: float


@dataclass
class RecedingHorizonResult:
    actions: np.ndarray
    first_plan_distance: float
    final_observation: np.ndarray
    final_info: dict


def extract_wall_mask(frame: np.ndarray) -> np.ndarray:
    """Extract wall pixels from a CHW uint8 RGB observation."""
    if frame.dtype != np.uint8:
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
    if frame.ndim != 3 or frame.shape[0] != 3:
        raise ValueError("expected a CHW RGB frame")
    red, green, blue = frame.astype(np.int16)
    return (red >= 60) & (red <= 140) & (green >= 70) & (green <= 160) & (blue >= 80) & (blue <= 180)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy:dy + height, dx:dx + width]
    return result


def extract_wall_boxes(mask: np.ndarray, min_area: int = 8) -> tuple[tuple[float, float, float, float], ...]:
    """Find connected wall components and return (x0, y0, x1, y1) boxes."""
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    boxes = []
    for y0, x0 in zip(*np.where(mask & ~visited)):
        if visited[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        visited[y0, x0] = True
        points = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= next_y < height and 0 <= next_x < width and mask[next_y, next_x] and not visited[next_y, next_x]:
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))
        if len(points) >= min_area:
            points_array = np.asarray(points)
            boxes.append((float(points_array[:, 1].min()), float(points_array[:, 0].min()), float(points_array[:, 1].max()), float(points_array[:, 0].max())))
    return tuple(boxes)


def _collision_prefix(positions: np.ndarray, wall_mask: np.ndarray, agent_radius: int = 3) -> np.ndarray:
    """Return whether each imagined step or an earlier step intersects a wall."""
    occupied = _dilate(wall_mask, agent_radius)
    finite = np.isfinite(positions).all(axis=-1)
    xs = np.rint(np.nan_to_num(positions[..., 0], nan=-1)).astype(np.int64)
    ys = np.rint(np.nan_to_num(positions[..., 1], nan=-1)).astype(np.int64)
    inside = (xs >= 0) & (xs < occupied.shape[1]) & (ys >= 0) & (ys < occupied.shape[0])
    clipped_xs = np.clip(xs, 0, occupied.shape[1] - 1)
    clipped_ys = np.clip(ys, 0, occupied.shape[0] - 1)
    collisions = (~finite) | (~inside) | occupied[clipped_ys, clipped_xs]
    return np.maximum.accumulate(collisions, axis=1)


def _detour_sequence(
    model: PocketWorldModel,
    start: torch.Tensor,
    start_position: np.ndarray,
    goal: tuple[float, float],
    waypoint: tuple[float, float],
    horizon: int,
) -> list[int]:
    """Use the compact dynamics to steer through a waypoint and then to goal."""
    state = model.state_from_latent(model.encode(start))
    targets = (waypoint, goal)
    target_index = 0
    actions = []
    for _ in range(horizon):
        position = state[0, :2].cpu().numpy() * 64.0
        target = np.asarray(targets[target_index], dtype=np.float32)
        delta = target - position
        if np.linalg.norm(delta) <= 5.0 and target_index < len(targets) - 1:
            target_index += 1
            target = np.asarray(targets[target_index], dtype=np.float32)
            delta = target - position
        if abs(delta[0]) >= abs(delta[1]):
            action = 3 if delta[0] >= 0 else 2
        else:
            action = 1 if delta[1] >= 0 else 0
        actions.append(action)
        state = model.state_transition(state, torch.tensor([action], device=state.device))
    return actions


def _detour_templates(
    model: PocketWorldModel,
    start: torch.Tensor,
    start_position: np.ndarray,
    goal: tuple[float, float],
    wall_mask: np.ndarray,
    horizon: int,
) -> list[list[int]]:
    """Generate top/bottom or left/right waypoint proposals for crossing walls."""
    boxes = extract_wall_boxes(wall_mask)
    proposals = []
    goal_array = np.asarray(goal, dtype=np.float32)
    for x0, y0, x1, y1 in boxes:
        if start_position[0] < x0 and goal_array[0] > x1 and y0 <= start_position[1] <= y1:
            proposals.extend([
                _detour_sequence(model, start, start_position, goal, (start_position[0], max(4.0, y0 - 5.0)), horizon),
                _detour_sequence(model, start, start_position, goal, (start_position[0], min(60.0, y1 + 5.0)), horizon),
            ])
        elif start_position[0] > x1 and goal_array[0] < x0 and y0 <= start_position[1] <= y1:
            proposals.extend([
                _detour_sequence(model, start, start_position, goal, (start_position[0], max(4.0, y0 - 5.0)), horizon),
                _detour_sequence(model, start, start_position, goal, (start_position[0], min(60.0, y1 + 5.0)), horizon),
            ])
        elif start_position[1] < y0 and goal_array[1] > y1 and x0 <= start_position[0] <= x1:
            proposals.extend([
                _detour_sequence(model, start, start_position, goal, (max(4.0, x0 - 5.0), start_position[1]), horizon),
                _detour_sequence(model, start, start_position, goal, (min(60.0, x1 + 5.0), start_position[1]), horizon),
            ])
        elif start_position[1] > y1 and goal_array[1] < y0 and x0 <= start_position[0] <= x1:
            proposals.extend([
                _detour_sequence(model, start, start_position, goal, (max(4.0, x0 - 5.0), start_position[1]), horizon),
                _detour_sequence(model, start, start_position, goal, (min(60.0, x1 + 5.0), start_position[1]), horizon),
            ])
    return proposals


def extract_agent_position(frame: np.ndarray) -> np.ndarray:
    """Locate the mint-green agent in a CHW uint8 or BCHW float frame."""
    if frame.dtype != np.uint8:
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
    frame = frame.astype(np.int16)
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
    collision_aware: bool = False,
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
    distances = np.linalg.norm(positions - np.asarray(goal), axis=-1)
    if collision_aware:
        wall_mask = extract_wall_mask(observation)
        templates = _detour_templates(model, start, start_position, goal, wall_mask, horizon)
        if templates:
            template_tensor = torch.as_tensor(templates, device=device, dtype=torch.long)
            count = min(len(templates), candidates)
            actions[:count] = template_tensor[:count]
            imagined_positions[:count] = model.imagine_positions(starts[:count], actions[:count]).cpu().numpy() * 64.0
            positions[:count, 1:] = imagined_positions[:count]
            distances[:count] = np.linalg.norm(positions[:count] - np.asarray(goal), axis=-1)
        collision_prefix = _collision_prefix(positions, wall_mask)
        distances = distances + collision_prefix * 64.0
    safe_distances = np.where(np.isfinite(distances), distances, 1e6)
    best = int(np.argmin(np.min(safe_distances, axis=1)))
    best_step = int(np.argmin(safe_distances[best]))
    return PlanResult(
        actions=actions[best, :best_step].cpu().numpy(),
        imagined_positions=positions[best, :best_step + 1],
        imagined_distance=float(safe_distances[best, best_step]),
    )


@torch.no_grad()
def receding_horizon_plan(
    model: PocketWorldModel,
    observation: np.ndarray,
    goal: tuple[float, float],
    step_fn: Callable[[int], tuple[np.ndarray, float, bool, bool, dict]],
    max_steps: int = 40,
    rollout_horizon: int = 16,
    candidates: int = 512,
    collision_aware: bool = True,
    commit_steps: int = 1,
) -> RecedingHorizonResult:
    """Replan after every real action and return the closed-loop execution trace."""
    current_observation = observation
    executed_actions = []
    first_plan_distance = float("nan")
    final_info: dict = {}
    for step in range(max_steps):
        plan = random_shooting(
            model,
            current_observation,
            goal,
            horizon=min(rollout_horizon, max_steps - step),
            candidates=candidates,
            collision_aware=collision_aware,
        )
        if step == 0:
            first_plan_distance = plan.imagined_distance
        if len(plan.actions) == 0:
            break
        for action_value in plan.actions[:max(1, commit_steps)]:
            action = int(action_value)
            current_observation, _, terminated, truncated, final_info = step_fn(action)
            executed_actions.append(action)
            if terminated or truncated:
                break
        if final_info.get("distance_to_goal", float("inf")) <= 4.0 or terminated or truncated:
            break
    return RecedingHorizonResult(
        actions=np.asarray(executed_actions, dtype=np.int64),
        first_plan_distance=first_plan_distance,
        final_observation=current_observation,
        final_info=final_info,
    )
