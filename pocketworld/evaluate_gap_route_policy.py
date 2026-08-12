"""Evaluate per-barrier gap prediction as a structured route policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .env import PocketWorldEnv
from .planner import extract_agent_position
from .procedural_routes import sample_procedural_route_cases
from .route_policy import GapRoutePolicy, observable_gap_route_waypoints
from .route_policy import observable_waypoint_action


def collect_gap_data(seeds: tuple[int, ...], episodes: int, split: str) -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    for seed in seeds:
        for case in sample_procedural_route_cases(seed, episodes, split=split):
            observation, _ = PocketWorldEnv(walls=case.walls, agent_start=case.start, goal=case.goal).reset()
            observations.append(observation)
            goals.append(np.asarray(case.goal, dtype=np.float32))
    return np.asarray(observations, dtype=np.uint8), np.asarray(goals, dtype=np.float32)


def evaluate_gap_policy(
    policy: GapRoutePolicy,
    seeds: tuple[int, ...],
    episodes: int,
    max_steps: int,
    split: str,
    project_to_visible_gap: bool = False,
    report_suffix: str = "",
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for case in sample_procedural_route_cases(seed, episodes, split=split):
            env = PocketWorldEnv(walls=case.walls, agent_start=case.start, goal=case.goal)
            observation, info = env.reset()
            history = [observation]
            gap_centers = policy.predict_gap_centers(observation, case.goal)
            waypoints = observable_gap_route_waypoints(
                observation,
                case.goal,
                gap_centers,
                project_to_visible_gap=project_to_visible_gap,
            )
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
            rows.append({
                "map_id": case.map_id,
                "variant": report_suffix or ("projected" if project_to_visible_gap else "raw"),
                "seed": seed,
                "barrier_count": case.barrier_count,
                "gap_height": case.gap_height,
                "real_success": float(info["distance_to_goal"] <= env.goal_radius),
                "final_distance_px": float(info["distance_to_goal"]),
                "collision_count": collisions,
                "executed_actions": int(env.steps),
            })
    values = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in ("real_success", "final_distance_px", "collision_count", "executed_actions")}
    return {"rows": rows, "summary": {key: {"mean": float(value.mean()), "std": float(value.std())} for key, value in values.items()}}


def train_and_evaluate_gap_policy(
    train_seeds: tuple[int, ...] = (101, 103, 107),
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    train_episodes: int = 200,
    evaluation_episodes: int = 20,
    max_steps: int = 140,
    epochs: int = 320,
    predictor_output: str | Path = "artifacts/gap-route-policy-v17.pt",
) -> dict[str, object]:
    observations, goals = collect_gap_data(train_seeds, train_episodes, "train")
    policy = GapRoutePolicy()
    training = policy.fit(observations, goals, epochs=epochs)
    raw_evaluation = evaluate_gap_policy(
        policy, evaluation_seeds, evaluation_episodes, max_steps, "holdout",
        project_to_visible_gap=False, report_suffix="raw_prediction",
    )
    projected_evaluation = evaluate_gap_policy(
        policy, evaluation_seeds, evaluation_episodes, max_steps, "holdout",
        project_to_visible_gap=True, report_suffix="visible_gap_projection",
    )
    policy.save(predictor_output, metadata={"student_evaluation_uses_astar": False})
    return {
        "protocol": {
            "teacher": "visible RGB vertical-barrier gap geometry",
            "teacher_uses_astar_for_labels": False,
            "data_quality_filter_uses_astar": True,
            "student_inputs": ["initial RGB observation", "goal coordinates"],
            "student_input_gap_center": False,
            "student_output": "one normalized gap center per visible vertical barrier",
            "student_evaluation_uses_astar": False,
            "visible_gap_projection_uses_astar": False,
            "train_seeds": list(train_seeds),
            "evaluation_seeds": list(evaluation_seeds),
            "train_episodes_per_seed": train_episodes,
            "evaluation_episodes_per_seed": evaluation_episodes,
            "max_steps": max_steps,
            "project_to_visible_gap": True,
        },
        "training": training,
        "evaluation": projected_evaluation,
        "ablation": {
            "raw_prediction": raw_evaluation,
            "visible_gap_projection": projected_evaluation,
        },
        "comparison": {
            "route_mode_v15": "fixed four-class mode",
            "route_sketch_v16": "fixed continuous route points",
            "gap_route_v17": "per-barrier structured gap centers",
        },
        "checkpoint": str(predictor_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a per-barrier gap route policy")
    parser.add_argument("--train-seeds", default="101,103,107")
    parser.add_argument("--evaluation-seeds", default="11,23,41")
    parser.add_argument("--train-episodes", type=int, default=200)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--predictor-output", default="artifacts/gap-route-policy-v17.pt")
    parser.add_argument("--output", default="artifacts/evaluation-gap-route-policy-v17.json")
    args = parser.parse_args()
    report = train_and_evaluate_gap_policy(
        train_seeds=tuple(int(item) for item in args.train_seeds.split(",") if item.strip()),
        evaluation_seeds=tuple(int(item) for item in args.evaluation_seeds.split(",") if item.strip()),
        train_episodes=args.train_episodes,
        evaluation_episodes=args.evaluation_episodes,
        max_steps=args.max_steps,
        epochs=args.epochs,
        predictor_output=args.predictor_output,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
