from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import _variant_walls
from .env import PocketWorldEnv
from .model import PocketWorldModel
from .planner import extract_agent_position, random_shooting


def evaluate_prediction(model: PocketWorldModel, episodes: int = 20, seed: int = 11, ood: bool = False) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    horizons = (1, 5, 10, 20)
    image_errors = {str(horizon): [] for horizon in horizons}
    position_errors = {str(horizon): [] for horizon in horizons}
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
        imagined = model.imagine(start, action_tensor)[0].cpu()
        imagined_positions = model.imagine_positions(start, action_tensor)[0].cpu().numpy() * 64.0
        for horizon in horizons:
            target = torch.from_numpy(actual[horizon]).float() / 255.0
            prediction = imagined[horizon]
            image_errors[str(horizon)].append(float(torch.abs(prediction - target).mean()))
            predicted_position = extract_agent_position(prediction.numpy())
            actual_position = extract_agent_position(actual[horizon])
            if np.all(np.isfinite(predicted_position)) and np.all(np.isfinite(actual_position)):
                position_errors[str(horizon)].append(float(np.linalg.norm(predicted_position - actual_position)))
            if np.all(np.isfinite(imagined_positions[horizon - 1])) and np.all(np.isfinite(actual_position)):
                latent_position_errors[str(horizon)].append(float(np.linalg.norm(imagined_positions[horizon - 1] - actual_position)))
    return {
        "image_mae": {horizon: _mean(values) for horizon, values in image_errors.items()},
        "position_error_px": {horizon: _mean(values) for horizon, values in position_errors.items()},
        "latent_position_error_px": {horizon: _mean(values) for horizon, values in latent_position_errors.items()},
        "position_coverage": {horizon: len(position_errors[horizon]) / episodes for horizon in position_errors},
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
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    runs = []
    for seed in seeds:
        runs.append({
            "seed": seed,
            "in_distribution": evaluate_prediction(model, episodes=args.episodes, seed=seed),
            "out_of_distribution": evaluate_prediction(model, episodes=args.episodes, seed=seed + 1000, ood=True),
            "planning": evaluate_planning(model, episodes=args.episodes, candidates=args.candidates, seed=seed + 2000),
            "planning_sweep": evaluate_planning_sweep(model, episodes=args.episodes, candidates=args.candidates, seed=seed + 3000),
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
