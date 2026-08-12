"""Evaluate continuous RGB route sketches on procedural multi-barrier maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .env import PocketWorldEnv
from .procedural_routes import ProceduralRouteCase, sample_procedural_route_cases
from .planner import extract_agent_position
from .route_policy import (
    ROUTE_SKETCH_POINTS,
    RouteSketchPolicy,
    observable_route_sketch_waypoints,
    observable_waypoint_action,
    route_sketch_targets,
)


def _initial_observation(case: ProceduralRouteCase) -> np.ndarray:
    env = PocketWorldEnv(walls=case.walls, agent_start=case.start, goal=case.goal)
    observation, _ = env.reset()
    return observation


def collect_route_sketch_data(
    seeds: tuple[int, ...],
    episodes: int,
    split: str,
    points: int = ROUTE_SKETCH_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for seed in seeds:
        for case in sample_procedural_route_cases(seed, episodes, split=split):
            observation = _initial_observation(case)
            observations.append(observation)
            goals.append(np.asarray(case.goal, dtype=np.float32))
            targets.append(route_sketch_targets(observation, case.goal, points=points))
    return np.asarray(observations, dtype=np.uint8), np.asarray(goals, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def evaluate_route_sketch_policy(
    policy: RouteSketchPolicy,
    seeds: tuple[int, ...],
    episodes: int,
    max_steps: int = 96,
    split: str = "holdout",
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for case in sample_procedural_route_cases(seed, episodes, split=split):
            env = PocketWorldEnv(walls=case.walls, agent_start=case.start, goal=case.goal)
            observation, info = env.reset()
            history = [observation]
            predicted = policy.predict_points(observation, case.goal)
            waypoints = observable_route_sketch_waypoints(observation, case.goal, predicted)
            waypoint_index = 0
            collisions = 0
            for _ in range(max_steps):
                target = waypoints[min(waypoint_index, len(waypoints) - 1)]
                position = np.asarray(extract_agent_position(observation), dtype=np.float32)
                if np.isfinite(position).all() and np.linalg.norm(position - np.asarray(target)) <= 5.0:
                    waypoint_index = min(waypoint_index + 1, len(waypoints) - 1)
                    target = waypoints[waypoint_index]
                action = observable_waypoint_action(observation, target, history, damping=1.0)
                observation, _, terminated, truncated, info = env.step(action)
                history.append(observation)
                history = history[-16:]
                collisions += int(info.get("collision", False))
                if terminated or truncated:
                    break
            rows.append(
                {
                    "map_id": case.map_id,
                    "seed": seed,
                    "split": split,
                    "predicted_waypoint_count": len(waypoints),
                    "real_success": float(info["distance_to_goal"] <= env.goal_radius),
                    "final_distance_px": float(info["distance_to_goal"]),
                    "collision_count": collisions,
                    "executed_actions": int(env.steps),
                }
            )
    values = {
        key: np.asarray([row[key] for row in rows], dtype=np.float64)
        for key in ("real_success", "final_distance_px", "collision_count", "executed_actions")
    }
    return {
        "rows": rows,
        "summary": {key: {"mean": float(value.mean()), "std": float(value.std())} for key, value in values.items()},
    }


def train_and_evaluate_route_sketch(
    train_seeds: tuple[int, ...] = (101, 103, 107),
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    train_episodes: int = 80,
    evaluation_episodes: int = 40,
    max_steps: int = 96,
    epochs: int = 240,
    predictor_output: str | Path = "artifacts/route-sketch-policy-v16.pt",
    points: int = ROUTE_SKETCH_POINTS,
) -> dict[str, object]:
    observations, goals, targets = collect_route_sketch_data(
        train_seeds, train_episodes, split="train", points=points
    )
    policy = RouteSketchPolicy(points=points)
    training = policy.fit(observations, goals, targets, epochs=epochs)
    validation = collect_route_sketch_data(
        (109,), max(8, train_episodes // 4), split="train", points=points
    )
    validation_prediction = policy.predict_points(validation[0][0], tuple(validation[1][0]))
    validation_error = float(np.linalg.norm(validation_prediction - validation[2][0], axis=-1).mean())
    policy.save(predictor_output, metadata={"points": points, "student_evaluation_uses_astar": False})
    evaluation = evaluate_route_sketch_policy(
        policy, evaluation_seeds, evaluation_episodes, max_steps=max_steps, split="holdout"
    )
    return {
        "protocol": {
            "teacher": "visible RGB wall mask with A* route labels",
            "student_inputs": ["initial RGB observation", "goal coordinates"],
            "student_output": f"{points} continuous route points plus goal",
            "student_evaluation_uses_astar": False,
            "train_seeds": list(train_seeds),
            "evaluation_seeds": list(evaluation_seeds),
            "train_episodes_per_seed": train_episodes,
            "evaluation_episodes_per_seed": evaluation_episodes,
            "max_steps": max_steps,
            "points": points,
        },
        "training": training,
        "validation": {"samples": int(len(validation[0])), "mean_point_error_px": validation_error},
        "evaluation": evaluation,
        "comparison": {
            "fixed_route_mode_v15": "4-category direct/top/bottom/gap student",
            "route_sketch_v16": "continuous multi-obstacle route student",
            "geometry_teacher": "used only for labels, not at student evaluation",
        },
        "checkpoint": str(predictor_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a continuous route sketch policy")
    parser.add_argument("--train-seeds", default="101,103,107")
    parser.add_argument("--evaluation-seeds", default="11,23,41")
    parser.add_argument("--train-episodes", type=int, default=80)
    parser.add_argument("--evaluation-episodes", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--points", type=int, default=ROUTE_SKETCH_POINTS)
    parser.add_argument("--predictor-output", default="artifacts/route-sketch-policy-v16.pt")
    parser.add_argument("--output", default="artifacts/evaluation-route-sketch-policy-v16.json")
    args = parser.parse_args()
    report = train_and_evaluate_route_sketch(
        train_seeds=tuple(int(item) for item in args.train_seeds.split(",") if item.strip()),
        evaluation_seeds=tuple(int(item) for item in args.evaluation_seeds.split(",") if item.strip()),
        train_episodes=args.train_episodes,
        evaluation_episodes=args.evaluation_episodes,
        max_steps=args.max_steps,
        epochs=args.epochs,
        predictor_output=args.predictor_output,
        points=args.points,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
