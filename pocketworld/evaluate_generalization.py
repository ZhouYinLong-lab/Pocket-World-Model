"""Map- and task-generalization evaluation for PocketWorld."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import PocketWorldEnv
from .maps import MAP_SUITES, get_map, map_names
from .model import PocketWorldModel
from .planner import receding_horizon_plan
from .tasks import sample_navigation_task


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def evaluate_map_prediction(
    model: PocketWorldModel,
    map_name: str,
    episodes: int = 10,
    horizon: int = 20,
    seed: int = 101,
) -> dict[str, object]:
    """Measure multi-step RGB-model position error on one named layout."""

    rng = np.random.default_rng(seed)
    horizons = tuple(sorted({value for value in (1, 5, 10, 20) if value <= horizon}))
    position_errors = {str(value): [] for value in horizons}
    image_errors = {str(value): [] for value in horizons}
    spec = get_map(map_name)
    model.eval()
    for _ in range(episodes):
        task = sample_navigation_task(rng, map_name, waypoint_count=1)
        env = PocketWorldEnv(walls=spec.walls, agent_start=task.start, goal=task.goal, map_name=map_name)
        observation, _ = env.reset()
        actions = rng.integers(0, 4, size=horizon, dtype=np.int64)
        actual_frames = [observation]
        actual_positions = [env.position.copy()]
        for action in actions:
            next_observation, _, terminated, truncated, _ = env.step(int(action))
            actual_frames.append(next_observation)
            actual_positions.append(env.position.copy())
            if terminated or truncated:
                actual_frames.extend([next_observation] * (horizon - len(actual_frames) + 1))
                actual_positions.extend([env.position.copy()] * (horizon - len(actual_positions) + 1))
                break
        start = torch.from_numpy(observation[None]).float() / 255.0
        action_tensor = torch.from_numpy(actions[None])
        predicted_frames = model.imagine(start, action_tensor, compose_agent=True)[0].cpu().numpy()
        predicted_positions = model.imagine_positions(start, action_tensor)[0].cpu().numpy() * 64.0
        for step in horizons:
            actual_frame = actual_frames[step].astype(np.float32) / 255.0
            image_errors[str(step)].append(float(np.abs(predicted_frames[step] - actual_frame).mean()))
            position_errors[str(step)].append(
                float(np.linalg.norm(predicted_positions[step - 1] - actual_positions[step]))
            )
    return {
        "map_name": map_name,
        "map_description": spec.description,
        "episodes": episodes,
        "position_error_px": {key: _mean(value) for key, value in position_errors.items()},
        "image_mae": {key: _mean(value) for key, value in image_errors.items()},
    }


def evaluate_waypoint_tasks(
    model: PocketWorldModel,
    map_name: str,
    episodes: int = 5,
    waypoint_count: int = 2,
    horizon: int = 32,
    candidates: int = 128,
    seed: int = 201,
) -> dict[str, float | int | str]:
    """Evaluate sequential-goal tasks that require multiple route completions."""

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    spec = get_map(map_name)
    completed_tasks = 0
    completed_waypoints = 0
    total_waypoints = episodes * waypoint_count
    collisions = []
    final_distances = []
    actions = []
    for _ in range(episodes):
        task = sample_navigation_task(rng, map_name, waypoint_count=waypoint_count)
        env = PocketWorldEnv(walls=spec.walls, agent_start=task.start, goal=task.goal, map_name=map_name)
        observation, _ = env.reset()
        task_completed = True
        for goal in task.goals:
            env.set_goal(goal)
            observation = env.render()
            result = receding_horizon_plan(
                model,
                observation,
                goal,
                env.step,
                max_steps=horizon,
                rollout_horizon=min(16, horizon),
                candidates=candidates,
                collision_aware=True,
                preserve_route=True,
                learned_collision=True,
                hybrid_collision=True,
                use_history_velocity=True,
                use_learned_velocity=True,
                route_objective=True,
                alignment_fallback_threshold=4.0,
                wall_aware_route=True,
            )
            reached = result.final_info.get("distance_to_goal", float("inf")) <= env.goal_radius
            completed_waypoints += int(reached)
            collisions.append(float(result.collision_count))
            final_distances.append(float(result.final_info.get("distance_to_goal", float("inf"))))
            actions.append(float(len(result.actions)))
            if not reached:
                task_completed = False
                break
        completed_tasks += int(task_completed)
    return {
        "map_name": map_name,
        "episodes": episodes,
        "waypoint_count": waypoint_count,
        "task_success_rate": completed_tasks / episodes,
        "waypoint_completion_rate": completed_waypoints / total_waypoints,
        "mean_final_distance_px": _mean(final_distances),
        "mean_collisions": _mean(collisions),
        "mean_actions_per_leg": _mean(actions),
    }


def evaluate_generalization(
    model: PocketWorldModel,
    episodes: int = 10,
    horizon: int = 20,
    candidates: int = 128,
    seed: int = 101,
    train_suite: str = "train",
    holdout_suite: str = "holdout",
    waypoint_count: int = 2,
) -> dict[str, object]:
    """Return train-map, unseen-map, and sequential-task reports."""

    torch.manual_seed(seed)
    train_maps = map_names(train_suite)
    holdout_maps = map_names(holdout_suite)
    prediction = {
        "train": [evaluate_map_prediction(model, name, episodes, horizon, seed + index * 100) for index, name in enumerate(train_maps)],
        "unseen": [evaluate_map_prediction(model, name, episodes, horizon, seed + 1000 + index * 100) for index, name in enumerate(holdout_maps)],
    }
    waypoint_tasks = {
        # Keep the requested execution budget intact.  Capping this at 32
        # steps made the evaluator silently truncate routes whose remaining
        # geometric distance was valid but longer than the old demo budget.
        "train": [evaluate_waypoint_tasks(model, name, max(1, episodes // 2), waypoint_count, max(8, horizon), candidates, seed + 2000 + index * 100) for index, name in enumerate(train_maps)],
        "unseen": [evaluate_waypoint_tasks(model, name, max(1, episodes // 2), waypoint_count, max(8, horizon), candidates, seed + 3000 + index * 100) for index, name in enumerate(holdout_maps)],
    }
    prediction_summary = {
        split: {
            "mean_position_error_px": {
                horizon_key: _mean([
                    float(report["position_error_px"][horizon_key])
                    for report in reports
                ])
                for horizon_key in reports[0]["position_error_px"]
            },
            "mean_image_mae": {
                horizon_key: _mean([
                    float(report["image_mae"][horizon_key])
                    for report in reports
                ])
                for horizon_key in reports[0]["image_mae"]
            },
        }
        for split, reports in prediction.items()
    }
    waypoint_summary = {
        split: {
            "mean_task_success_rate": _mean([float(report["task_success_rate"]) for report in reports]),
            "mean_waypoint_completion_rate": _mean([float(report["waypoint_completion_rate"]) for report in reports]),
            "mean_final_distance_px": _mean([float(report["mean_final_distance_px"]) for report in reports]),
        }
        for split, reports in waypoint_tasks.items()
    }
    return {
        "config": {
            "episodes": episodes,
            "horizon": horizon,
            "candidates": candidates,
            "seed": seed,
            "train_suite": train_suite,
            "holdout_suite": holdout_suite,
            "waypoint_count": waypoint_count,
        },
        "map_suites": {key: list(value) for key, value in MAP_SUITES.items()},
        "prediction": prediction,
        "prediction_summary": prediction_summary,
        "waypoint_tasks": waypoint_tasks,
        "waypoint_summary": waypoint_summary,
    }


def evaluate_generalization_seeds(
    model: PocketWorldModel,
    seeds: tuple[int, ...] = (11, 23, 41),
    episodes: int = 20,
    horizon: int = 64,
    candidates: int = 64,
    train_suite: str = "train",
    holdout_suite: str = "holdout",
    waypoint_count: int = 2,
) -> dict[str, object]:
    """Run the same map/task protocol over independent evaluation seeds."""

    runs = [
        evaluate_generalization(
            model,
            episodes=episodes,
            horizon=horizon,
            candidates=candidates,
            seed=seed,
            train_suite=train_suite,
            holdout_suite=holdout_suite,
            waypoint_count=waypoint_count,
        )
        for seed in seeds
    ]
    summary = {
        "prediction_20_step_position_error_px": {
            split: {
                "mean": float(np.mean([
                    run["prediction_summary"][split]["mean_position_error_px"].get("20", float("nan"))
                    for run in runs
                ])),
                "std": float(np.std([
                    run["prediction_summary"][split]["mean_position_error_px"].get("20", float("nan"))
                    for run in runs
                ])),
            }
            for split in ("train", "unseen")
        },
        "waypoint_task_success_rate": {
            split: {
                "mean": float(np.mean([
                    run["waypoint_summary"][split]["mean_task_success_rate"] for run in runs
                ])),
                "std": float(np.std([
                    run["waypoint_summary"][split]["mean_task_success_rate"] for run in runs
                ])),
            }
            for split in ("train", "unseen")
        },
        "waypoint_completion_rate": {
            split: {
                "mean": float(np.mean([
                    run["waypoint_summary"][split]["mean_waypoint_completion_rate"] for run in runs
                ])),
                "std": float(np.std([
                    run["waypoint_summary"][split]["mean_waypoint_completion_rate"] for run in runs
                ])),
            }
            for split in ("train", "unseen")
        },
    }
    return {
        "config": {
            "episodes": episodes,
            "horizon": horizon,
            "candidates": candidates,
            "seeds": list(seeds),
            "train_suite": train_suite,
            "holdout_suite": holdout_suite,
            "waypoint_count": waypoint_count,
        },
        "runs": runs,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PocketWorld on named maps and sequential-goal tasks")
    parser.add_argument("checkpoint", nargs="?", default="artifacts/pocketworld.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--seeds", default=None, help="comma-separated seeds for a formal multi-seed report")
    parser.add_argument("--train-suite", choices=tuple(MAP_SUITES), default="train")
    parser.add_argument("--holdout-suite", choices=tuple(MAP_SUITES), default="holdout")
    parser.add_argument("--waypoints", type=int, default=2)
    parser.add_argument("--output", default="artifacts/evaluation-generalization.json")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = PocketWorldModel()
    model.load_state_dict(payload["model"], strict=False)
    if args.seeds:
        seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
        report = evaluate_generalization_seeds(
            model,
            seeds=seeds,
            episodes=args.episodes,
            horizon=args.horizon,
            candidates=args.candidates,
            train_suite=args.train_suite,
            holdout_suite=args.holdout_suite,
            waypoint_count=args.waypoints,
        )
    else:
        report = evaluate_generalization(
            model,
            episodes=args.episodes,
            horizon=args.horizon,
            candidates=args.candidates,
            seed=args.seed,
            train_suite=args.train_suite,
            holdout_suite=args.holdout_suite,
            waypoint_count=args.waypoints,
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
