from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import _variant_walls
from .env import PocketWorldEnv, Rect
from .model import PocketWorldModel
from .planner import extract_agent_position, random_shooting, receding_horizon_plan


def evaluate_prediction(model: PocketWorldModel, episodes: int = 20, seed: int = 11, ood: bool = False) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    horizons = (1, 5, 10, 20)
    image_errors = {str(horizon): [] for horizon in horizons}
    composited_image_errors = {str(horizon): [] for horizon in horizons}
    position_errors = {str(horizon): [] for horizon in horizons}
    composited_position_errors = {str(horizon): [] for horizon in horizons}
    latent_position_errors = {str(horizon): [] for horizon in horizons}
    model.eval()
    for _ in range(episodes):
        walls = _variant_walls(rng) if ood else None
        env = PocketWorldEnv(
            walls=walls,
            agent_start=(float(rng.integers(6, 15)), float(rng.integers(6, 15))),
            goal=(float(rng.integers(49, 58)), float(rng.integers(49, 58))),
            agent_speed_scale=float(rng.choice((0.8, 1.0, 1.2))) if ood else 1.0,
        )
        observation, _ = env.reset()
        actions = rng.integers(0, 4, size=max(horizons), dtype=np.int64)
        actual = [observation]
        for action in actions:
            next_observation, _, terminated, truncated, _ = env.step(int(action))
            actual.append(next_observation)
            if terminated or truncated:
                actual.extend([next_observation] * (len(actions) - len(actual) + 1))
                break
        start = torch.from_numpy(observation[None]).float() / 255.0
        action_tensor = torch.from_numpy(actions[None])
        imagined = model.imagine(start, action_tensor, compose_agent=False)[0].cpu()
        composited = model.imagine(start, action_tensor, compose_agent=True)[0].cpu()
        imagined_positions = model.imagine_positions(start, action_tensor)[0].cpu().numpy() * 64.0
        for horizon in horizons:
            target = torch.from_numpy(actual[horizon]).float() / 255.0
            prediction = imagined[horizon]
            composited_prediction = composited[horizon]
            image_errors[str(horizon)].append(float(torch.abs(prediction - target).mean()))
            composited_image_errors[str(horizon)].append(float(torch.abs(composited_prediction - target).mean()))
            predicted_position = extract_agent_position(prediction.numpy())
            composited_position = extract_agent_position(composited_prediction.numpy())
            actual_position = extract_agent_position(actual[horizon])
            if np.all(np.isfinite(predicted_position)) and np.all(np.isfinite(actual_position)):
                position_errors[str(horizon)].append(float(np.linalg.norm(predicted_position - actual_position)))
            if np.all(np.isfinite(composited_position)) and np.all(np.isfinite(actual_position)):
                composited_position_errors[str(horizon)].append(float(np.linalg.norm(composited_position - actual_position)))
            if np.all(np.isfinite(imagined_positions[horizon - 1])) and np.all(np.isfinite(actual_position)):
                latent_position_errors[str(horizon)].append(float(np.linalg.norm(imagined_positions[horizon - 1] - actual_position)))
    return {
        "image_mae": {horizon: _mean(values) for horizon, values in image_errors.items()},
        "composited_image_mae": {horizon: _mean(values) for horizon, values in composited_image_errors.items()},
        "position_error_px": {horizon: _mean(values) for horizon, values in position_errors.items()},
        "composited_position_error_px": {horizon: _mean(values) for horizon, values in composited_position_errors.items()},
        "latent_position_error_px": {horizon: _mean(values) for horizon, values in latent_position_errors.items()},
        "position_coverage": {horizon: len(position_errors[horizon]) / episodes for horizon in position_errors},
        "composited_position_coverage": {horizon: len(composited_position_errors[horizon]) / episodes for horizon in composited_position_errors},
    }


def _agent_circle_mask(position: np.ndarray, radius: float = 3.0) -> np.ndarray:
    yy, xx = np.mgrid[:64, :64]
    return ((xx - position[0]) ** 2 + (yy - position[1]) ** 2 <= radius**2)


def evaluate_agent_rendering(model: PocketWorldModel, episodes: int = 20, seed: int = 31, ood: bool = False) -> dict[str, dict[str, float]]:
    """Measure the learned mask head separately from the RGB decoder."""
    rng = np.random.default_rng(seed)
    horizons = (1, 5, 10, 20)
    position_errors = {str(horizon): [] for horizon in horizons}
    mask_ious = {str(horizon): [] for horizon in horizons}
    coverage = {str(horizon): 0 for horizon in horizons}
    model.eval()
    for _ in range(episodes):
        walls = _variant_walls(rng) if ood else None
        env = PocketWorldEnv(
            walls=walls,
            agent_start=(float(rng.integers(6, 15)), float(rng.integers(6, 15))),
            goal=(float(rng.integers(49, 58)), float(rng.integers(49, 58))),
            agent_speed_scale=float(rng.choice((0.8, 1.0, 1.2))) if ood else 1.0,
        )
        observation, info = env.reset()
        actions = rng.integers(0, 4, size=max(horizons), dtype=np.int64)
        actual_positions = []
        for action in actions:
            _, _, terminated, truncated, info = env.step(int(action))
            actual_positions.append(np.asarray(info["position"], dtype=np.float32))
            if terminated or truncated:
                actual_positions.extend([actual_positions[-1]] * (len(actions) - len(actual_positions)))
                break
        start = torch.from_numpy(observation[None]).float() / 255.0
        action_tensor = torch.from_numpy(actions[None])
        predicted_masks = model.imagine_agent_masks(start, action_tensor)[0, :, 0].cpu().numpy()
        for horizon in horizons:
            target_position = actual_positions[horizon - 1]
            predicted_mask = predicted_masks[horizon - 1] >= 0.5
            target_mask = _agent_circle_mask(target_position)
            intersection = np.logical_and(predicted_mask, target_mask).sum()
            union = np.logical_or(predicted_mask, target_mask).sum()
            mask_ious[str(horizon)].append(float(intersection / max(1, union)))
            ys, xs = np.where(predicted_mask)
            if len(xs):
                coverage[str(horizon)] += 1
                predicted_position = np.asarray((xs.mean(), ys.mean()), dtype=np.float32)
                position_errors[str(horizon)].append(float(np.linalg.norm(predicted_position - target_position)))
    return {
        "mask_iou": {horizon: _mean(values) for horizon, values in mask_ious.items()},
        "mask_position_error_px": {horizon: _mean(values) for horizon, values in position_errors.items()},
        "mask_position_coverage": {horizon: coverage[horizon] / episodes for horizon in coverage},
    }


def evaluate_planning(model: PocketWorldModel, episodes: int = 20, horizon: int = 16, candidates: int = 256, seed: int = 17) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    imagined_successes = 0
    real_successes = 0
    imagined_distances = []
    real_distances = []
    for _ in range(episodes):
        env = PocketWorldEnv(
            agent_start=(float(rng.integers(7, 11)), float(rng.integers(7, 11))),
            goal=(float(rng.integers(16, 21)), float(rng.integers(28, 33))),
        )
        observation, info = env.reset()
        result = random_shooting(model, observation, tuple(info["goal"]), horizon=horizon, candidates=candidates)
        imagined_distances.append(result.imagined_distance)
        imagined_successes += int(result.imagined_distance <= env.goal_radius)
        for action in result.actions:
            _, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                break
        real_distances.append(float(info["distance_to_goal"]))
        real_successes += int(info["distance_to_goal"] <= env.goal_radius)
    return {
        "imagined_success_rate": imagined_successes / episodes,
        "real_success_rate": real_successes / episodes,
        "planning_gap": (imagined_successes - real_successes) / episodes,
        "mean_imagined_final_distance_px": _mean(imagined_distances),
        "mean_real_final_distance_px": _mean(real_distances),
    }


def evaluate_planning_sweep(model: PocketWorldModel, episodes: int = 20, horizons: tuple[int, ...] = (8, 16, 24, 32), candidates: int = 256, seed: int = 17) -> dict[str, dict[str, float]]:
    """Measure the imagined/real planning gap as the planning horizon grows."""
    return {str(horizon): evaluate_planning(model, episodes=episodes, horizon=horizon, candidates=candidates, seed=seed + index) for index, horizon in enumerate(horizons)}


def evaluate_obstacle_planning(
    model: PocketWorldModel,
    episodes: int = 20,
    horizon: int = 40,
    candidates: int = 512,
    seed: int = 71,
    learned_collision: bool = False,
    hybrid_collision: bool = False,
) -> dict[str, dict[str, float]]:
    """Compare unconstrained and wall-aware planning on a single barrier task."""
    rng = np.random.default_rng(seed)
    walls = (Rect(29, 10, 5, 44),)
    reports = {"unconstrained": [], "collision_aware": [], "collision_aware_receding": [], "collision_aware_chunked": [], "collision_aware_route": []}
    if learned_collision:
        reports["collision_aware_learned"] = []
        reports["collision_aware_learned_route"] = []
    if hybrid_collision:
        reports["collision_aware_hybrid"] = []
    for _ in range(episodes):
        start = (float(rng.integers(7, 13)), float(rng.integers(25, 39)))
        goal = (float(rng.integers(51, 57)), float(rng.integers(25, 39)))
        planners = (("unconstrained", False, False), ("collision_aware", True, False))
        if learned_collision:
            planners += (("collision_aware_learned", True, True),)
        if hybrid_collision:
            planners += (("collision_aware_hybrid", True, False),)
        for label, collision_aware, learned_collision_mode in planners:
            env = PocketWorldEnv(walls=walls, agent_start=start, goal=goal)
            observation, info = env.reset()
            result = random_shooting(
                model,
                observation,
                tuple(info["goal"]),
                horizon=horizon,
                candidates=candidates,
                collision_aware=collision_aware,
                learned_collision=learned_collision_mode,
                hybrid_collision=label == "collision_aware_hybrid",
            )
            imagined_success = result.imagined_distance <= env.goal_radius
            for action in result.actions:
                _, _, terminated, truncated, info = env.step(int(action))
                if terminated or truncated:
                    break
            reports[label].append((imagined_success, info["distance_to_goal"] <= env.goal_radius, info["distance_to_goal"]))
        env = PocketWorldEnv(walls=walls, agent_start=start, goal=goal)
        observation, info = env.reset()
        for label, commit_steps, preserve_route in (
            ("collision_aware_receding", 1, False),
            ("collision_aware_chunked", 4, False),
            ("collision_aware_route", 1, True),
        ):
            env = PocketWorldEnv(walls=walls, agent_start=start, goal=goal)
            observation, info = env.reset()
            result = receding_horizon_plan(
                model,
                observation,
                tuple(info["goal"]),
                env.step,
                max_steps=horizon,
                rollout_horizon=min(16, horizon),
                candidates=candidates,
                collision_aware=True,
                commit_steps=commit_steps,
                preserve_route=preserve_route,
            )
            reports[label].append((result.first_plan_distance <= env.goal_radius, result.final_info.get("distance_to_goal", float("inf")) <= env.goal_radius, result.final_info.get("distance_to_goal", float("inf"))))
        if learned_collision:
            env = PocketWorldEnv(walls=walls, agent_start=start, goal=goal)
            observation, info = env.reset()
            result = receding_horizon_plan(
                model,
                observation,
                tuple(info["goal"]),
                env.step,
                max_steps=horizon,
                rollout_horizon=horizon,
                candidates=candidates,
                collision_aware=True,
                preserve_route=True,
                route_tolerance=6.0,
                learned_collision=True,
            )
            reports["collision_aware_learned_route"].append((result.first_plan_distance <= env.goal_radius, result.final_info.get("distance_to_goal", float("inf")) <= env.goal_radius, result.final_info.get("distance_to_goal", float("inf"))))
    return {
        label: {
            "imagined_success_rate": float(np.mean([row[0] for row in rows])),
            "real_success_rate": float(np.mean([row[1] for row in rows])),
            "mean_real_final_distance_px": float(np.mean([row[2] for row in rows])),
        }
        for label, rows in reports.items()
    }


def evaluate_collision_prediction(model: PocketWorldModel, episodes: int = 20, horizon: int = 16, seed: int = 83, ood: bool = False) -> dict[str, float]:
    """Measure learned collision-event probabilities against simulator events."""
    rng = np.random.default_rng(seed)
    probabilities = []
    targets = []
    for _ in range(episodes):
        walls = _variant_walls(rng) if ood else None
        env = PocketWorldEnv(
            walls=walls,
            agent_start=(float(rng.integers(6, 58)), float(rng.integers(6, 58))),
            goal=(float(rng.integers(6, 58)), float(rng.integers(6, 58))),
        )
        observation, _ = env.reset()
        start_observation = observation.copy()
        actions = rng.integers(0, 4, size=horizon, dtype=np.int64)
        actual_collisions = []
        for action in actions:
            next_observation, _, terminated, truncated, info = env.step(int(action))
            actual_collisions.append(float(info["collision"]))
            observation = next_observation
            if terminated or truncated:
                break
        start = torch.from_numpy(start_observation[None]).float() / 255.0
        action_tensor = torch.from_numpy(actions[None])
        predicted = model.imagine_collision_probabilities(start, action_tensor)[0].cpu().numpy()
        probabilities.extend(predicted[:len(actual_collisions)].tolist())
        targets.extend(actual_collisions)
    predicted_labels = np.asarray(probabilities) >= 0.5
    target_array = np.asarray(targets, dtype=bool)
    true_positive = np.logical_and(predicted_labels, target_array).sum()
    return {
        "accuracy": float(np.mean(predicted_labels == target_array)),
        "positive_rate": float(np.mean(target_array)),
        "predicted_positive_rate": float(np.mean(predicted_labels)),
        "precision": float(true_positive / max(1, predicted_labels.sum())),
        "recall": float(true_positive / max(1, target_array.sum())),
    }


def evaluate_action_effects(model: PocketWorldModel, repeat: int = 8) -> dict[str, dict[str, list[float] | float]]:
    """Compare predicted versus real displacement for each repeated action."""
    env = PocketWorldEnv(walls=(), agent_start=(32.0, 16.0), goal=(55.0, 55.0))
    observation, info = env.reset()
    start = torch.from_numpy(observation[None]).float() / 255.0
    result = {}
    for action, label in enumerate(("up", "down", "left", "right")):
        actions = torch.full((1, repeat), action, dtype=torch.long)
        predicted = model.imagine_positions(start, actions)[0, -1].cpu().numpy() * 64.0
        actual_env = PocketWorldEnv(walls=(), agent_start=(32.0, 16.0), goal=(55.0, 55.0))
        actual_env.reset()
        for _ in range(repeat):
            actual_env.step(action)
        actual = actual_env.position.copy()
        result[label] = {
            "predicted_position": predicted.tolist(),
            "actual_position": actual.tolist(),
            "position_error_px": float(np.linalg.norm(predicted - actual)),
        }
    return result


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _summarize(reports: list[dict]) -> dict:
    """Recursively summarize numeric report leaves across random seeds."""
    summary = {}
    for key in reports[0]:
        values = [report[key] for report in reports]
        if isinstance(values[0], dict):
            summary[key] = _summarize(values)
        elif isinstance(values[0], list):
            array = np.asarray(values, dtype=float)
            summary[key] = {"mean": array.mean(axis=0).tolist(), "std": array.std(axis=0).tolist()}
        elif values[0] is None:
            summary[key] = None
        else:
            summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-step prediction, OOD generalization, and imagined planning")
    parser.add_argument("checkpoint", nargs="?", default="artifacts/pocketworld.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--seeds", default="11,23,41", help="comma-separated evaluation seeds")
    parser.add_argument("--output", default="artifacts/evaluation.json")
    args = parser.parse_args()
    model = PocketWorldModel()
    payload = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if missing:
        print(f"warning: checkpoint is missing {len(missing)} structured-dynamics keys; use a freshly trained checkpoint for planning")
    if unexpected:
        print(f"warning: checkpoint has {len(unexpected)} legacy keys")
    collision_supervision = bool(payload.get("collision_supervision", False))
    agent_rendering = bool(payload.get("agent_rendering", False))
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    runs = []
    for seed in seeds:
        runs.append({
            "seed": seed,
            "in_distribution": evaluate_prediction(model, episodes=args.episodes, seed=seed),
            "out_of_distribution": evaluate_prediction(model, episodes=args.episodes, seed=seed + 1000, ood=True),
            "planning": evaluate_planning(model, episodes=args.episodes, candidates=args.candidates, seed=seed + 2000),
            "planning_sweep": evaluate_planning_sweep(model, episodes=args.episodes, candidates=args.candidates, seed=seed + 3000),
            "obstacle_planning": evaluate_obstacle_planning(
                model,
                episodes=args.episodes,
                candidates=args.candidates,
                seed=seed + 4000,
                learned_collision=collision_supervision,
                hybrid_collision=collision_supervision,
            ),
            "collision_prediction": {
                "in_distribution": evaluate_collision_prediction(model, episodes=args.episodes, seed=seed + 5000),
                "out_of_distribution": evaluate_collision_prediction(model, episodes=args.episodes, seed=seed + 6000, ood=True),
            } if collision_supervision else None,
            "agent_rendering": {
                "in_distribution": evaluate_agent_rendering(model, episodes=args.episodes, seed=seed + 7000),
                "out_of_distribution": evaluate_agent_rendering(model, episodes=args.episodes, seed=seed + 8000, ood=True),
            } if agent_rendering else None,
            "action_effects": evaluate_action_effects(model),
        })
    numeric_runs = [{key: value for key, value in run.items() if key != "seed"} for run in runs]
    report = {"config": {"episodes": args.episodes, "candidates": args.candidates, "seeds": seeds}, "runs": runs, "summary": _summarize(numeric_runs)}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
