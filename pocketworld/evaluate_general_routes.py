"""Compare learned route sketches with RGB projection and A* fallback methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .env import PocketWorldEnv
from .general_routes import GENERAL_FAMILIES, sample_general_route_cases
from .planner import _astar_path, _dilate, extract_agent_position, extract_wall_mask
from .route_policy import (
    RouteSketchPolicy,
    observable_route_sketch_waypoints,
    observable_waypoint_action,
    route_sketch_targets,
)
from .route_field import RouteFieldPolicy, conservative_field_action, field_waypoints

GENERAL_METHODS = (
    "learned",
    "rgb_projection",
    "hybrid_astar",
    "rgb_astar",
    "distance_field",
    "distance_field_rgb_projection",
    "distance_field_beam_rgb_projection",
    "distance_field_beam_conservative",
)


def _segment_hits_wall(
    observation: np.ndarray,
    start: np.ndarray,
    target: tuple[float, float],
    radius: int = 4,
) -> bool:
    occupied = _dilate(extract_wall_mask(observation), radius)
    end = np.asarray(target, dtype=np.float32)
    distance = float(np.linalg.norm(end - start))
    samples = max(2, int(np.ceil(distance * 2.0)))
    line = np.linspace(start, end, samples)
    xs = np.clip(np.rint(line[:, 0]).astype(int), 0, 63)
    ys = np.clip(np.rint(line[:, 1]).astype(int), 0, 63)
    return bool(np.any(occupied[ys, xs]))


def _project_target(
    observation: np.ndarray,
    start: np.ndarray,
    target: tuple[float, float],
) -> tuple[float, float]:
    """Project a learned waypoint onto a nearby RGB-visible line-safe point."""
    target_array = np.asarray(target, dtype=np.float32)
    if not _segment_hits_wall(observation, start, target):
        return tuple(map(float, target))
    candidates: list[tuple[float, float, float]] = []
    offsets = np.arange(-16.0, 17.0, 2.0)
    for dx in offsets:
        for dy in offsets:
            point = np.clip(target_array + (dx, dy), 5.0, 59.0)
            candidate = (float(point[0]), float(point[1]))
            if not _segment_hits_wall(observation, start, candidate):
                candidates.append((float(np.linalg.norm(point - target_array)), candidate[0], candidate[1]))
    if not candidates:
        return tuple(map(float, np.clip(target_array, 5.0, 59.0)))
    _, x, y = min(candidates)
    return x, y


def _astar_waypoints(
    observation: np.ndarray,
    goal: tuple[float, float],
    start: np.ndarray,
    points: int,
) -> tuple[tuple[float, float], ...]:
    """Create route waypoints from a current RGB wall mask."""
    path = _astar_path(
        _dilate(extract_wall_mask(observation), 4),
        tuple(map(float, start)),
        goal,
        allow_diagonal=False,
    )
    if not path:
        return (tuple(map(float, goal)),)
    dense = np.asarray(path, dtype=np.float32)
    indices = np.linspace(0, len(dense) - 1, max(1, points) + 1).round().astype(int)
    return tuple(tuple(map(float, point)) for point in dense[indices[1:]])


def collect_general_route_data(
    seeds: tuple[int, ...],
    episodes: int,
    split: str,
    points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for seed in seeds:
        for case in sample_general_route_cases(seed, episodes, split=split):
            observation, _ = PocketWorldEnv(
                walls=case.walls, agent_start=case.start, goal=case.goal
            ).reset()
            observations.append(observation)
            goals.append(np.asarray(case.goal, dtype=np.float32))
            targets.append(route_sketch_targets(observation, case.goal, points=points))
    return (
        np.asarray(observations, dtype=np.uint8),
        np.asarray(goals, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def evaluate_general_policy(
    policy: RouteSketchPolicy | RouteFieldPolicy,
    seeds: tuple[int, ...],
    episodes: int,
    max_steps: int,
    points: int,
    method: str,
) -> dict[str, object]:
    if method not in GENERAL_METHODS:
        raise ValueError(f"method must be one of {GENERAL_METHODS}")
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for case in sample_general_route_cases(seed, episodes, split="holdout"):
            env = PocketWorldEnv(walls=case.walls, agent_start=case.start, goal=case.goal)
            observation, info = env.reset()
            history = [observation]
            position = extract_agent_position(observation).astype(np.float32)
            astar_calls = 0
            fallback_triggered = False
            if method == "rgb_astar":
                waypoints = _astar_waypoints(observation, case.goal, position, points)
                astar_calls += 1
            elif method in {
                "distance_field",
                "distance_field_rgb_projection",
                "distance_field_beam_rgb_projection",
                "distance_field_beam_conservative",
            }:
                field = policy.predict_field(observation, case.goal)
                waypoints = field_waypoints(
                    observation,
                    case.goal,
                    field,
                    rgb_guard=method != "distance_field",
                    beam_width=4
                    if method in {
                        "distance_field_beam_rgb_projection",
                        "distance_field_beam_conservative",
                    }
                    else 1,
                )
            else:
                predicted = policy.predict_points(observation, case.goal)
                waypoints = observable_route_sketch_waypoints(observation, case.goal, predicted)
            waypoint_index = 0
            collisions = 0
            target_dirty = True
            planned_target: tuple[float, float] | None = None
            last_route_check = -100
            for _ in range(max_steps):
                step_index = env.steps
                position = extract_agent_position(observation).astype(np.float32)
                target = waypoints[min(waypoint_index, len(waypoints) - 1)]
                if np.isfinite(position).all() and np.linalg.norm(position - np.asarray(target)) <= 5.0:
                    waypoint_index = min(waypoint_index + 1, len(waypoints) - 1)
                    target = waypoints[waypoint_index]
                    target_dirty = True
                if method == "rgb_projection" and target_dirty:
                    target = _project_target(observation, position, target)
                    planned_target = target
                    target_dirty = False
                elif method == "rgb_projection" and planned_target is not None:
                    target = planned_target
                route_check_due = target_dirty or step_index - last_route_check >= 8
                if method == "hybrid_astar" and route_check_due and _segment_hits_wall(observation, position, target):
                    waypoints = _astar_waypoints(observation, case.goal, position, points)
                    waypoint_index = 0
                    target = waypoints[0]
                    astar_calls += 1
                    fallback_triggered = True
                    target_dirty = True
                    planned_target = None
                    last_route_check = step_index
                elif method == "rgb_astar" and route_check_due and _segment_hits_wall(observation, position, target):
                    waypoints = _astar_waypoints(observation, case.goal, position, points)
                    waypoint_index = 0
                    target = waypoints[0]
                    astar_calls += 1
                    target_dirty = True
                    last_route_check = step_index
                elif method in {"hybrid_astar", "rgb_astar"} and route_check_due:
                    last_route_check = step_index
                action = observable_waypoint_action(observation, target, history, damping=1.0)
                if method == "distance_field_beam_conservative":
                    action = conservative_field_action(observation, target, history)
                observation, _, terminated, truncated, info = env.step(action)
                history.append(observation)
                history = history[-16:]
                collisions += int(info.get("collision", False))
                if terminated or truncated:
                    break
            rows.append(
                {
                    "map_id": case.map_id,
                    "method": method,
                    "seed": seed,
                    "family": case.family,
                    "obstacle_count": case.obstacle_count,
                    "channel_count": case.channel_count,
                    "real_success": float(info["distance_to_goal"] <= env.goal_radius),
                    "final_distance_px": float(info["distance_to_goal"]),
                    "collision_count": collisions,
                    "executed_actions": int(env.steps),
                    "astar_calls": astar_calls,
                    "fallback_triggered": fallback_triggered,
                }
            )
    metrics = ("real_success", "final_distance_px", "collision_count", "executed_actions", "astar_calls", "fallback_triggered")
    values = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in metrics}
    return {
        "rows": rows,
        "summary": {
            key: {"mean": float(value.mean()), "std": float(value.std())}
            for key, value in values.items()
        },
    }


def train_and_evaluate_general_routes(
    train_seeds: tuple[int, ...] = (101, 103, 107),
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    train_episodes: int = 200,
    evaluation_episodes: int = 20,
    max_steps: int = 160,
    points: int = 13,
    epochs: int = 360,
    predictor_output: str | Path = "artifacts/general-route-sketch-v18.pt",
) -> dict[str, object]:
    observations, goals, targets = collect_general_route_data(
        train_seeds, train_episodes, "train", points
    )
    policy = RouteSketchPolicy(points=points)
    training = {"route_sketch": policy.fit(observations, goals, targets, epochs=epochs)}
    field_policy = RouteFieldPolicy()
    training["distance_field"] = field_policy.fit(observations, goals, epochs=epochs)
    field_output = Path(predictor_output).with_name(
        f"{Path(predictor_output).stem}-distance-field{Path(predictor_output).suffix}"
    )
    evaluations = {
        method: evaluate_general_policy(
            field_policy if method.startswith("distance_field") else policy,
            evaluation_seeds,
            evaluation_episodes,
            max_steps,
            points,
            method,
        )
        for method in GENERAL_METHODS
    }
    policy.save(
        predictor_output,
        metadata={
            "student_evaluation_uses_astar": False,
            "teacher_labels_use_astar": True,
            "rgb_projection_uses_astar": False,
        },
    )
    field_policy.save(
        field_output,
        metadata={
            "student_evaluation_uses_astar": False,
            "teacher_labels_use_astar": True,
            "rgb_projection_uses_astar": False,
        },
    )
    return {
        "protocol": {
            "train_seeds": list(train_seeds),
            "evaluation_seeds": list(evaluation_seeds),
            "train_episodes_per_seed": train_episodes,
            "evaluation_episodes_per_seed": evaluation_episodes,
            "max_steps": max_steps,
            "route_points": points,
            "families_train": ["staggered_blocks", "multi_channel"],
            "families_holdout": list(GENERAL_FAMILIES),
            "teacher_labels_use_astar": True,
            "data_quality_filter_uses_astar": True,
            "student_evaluation_uses_astar": False,
            "rgb_projection_uses_astar": False,
            "hybrid_astar_is_fallback_only": True,
            "method_astar_contract": {
                "learned": False,
                "rgb_projection": False,
                "hybrid_astar": True,
                "rgb_astar": True,
                "distance_field": False,
                "distance_field_rgb_projection": False,
                "distance_field_beam_rgb_projection": False,
                "distance_field_beam_conservative": False,
            },
            "representation_comparison": ["route_sketch", "coarse_distance_field"],
        },
        "training": training,
        "evaluation": evaluations,
        "comparison": {
            "learned": "fixed continuous route sketch + RGB waypoint controller",
            "rgb_projection": "learned route sketch + local RGB line-safe waypoint projection",
            "hybrid_astar": "learned route sketch with RGB-triggered A* fallback",
            "rgb_astar": "RGB/A* reference with closed-loop route refresh",
            "distance_field": "learned 16x16 coarse route distance field",
            "distance_field_rgb_projection": "learned distance field with RGB-occupied-cell guard",
            "distance_field_beam_rgb_projection": "fixed-width learned-field beam with RGB-occupied-cell guard",
            "distance_field_beam_conservative": "learned-field beam with RGB guard and conservative local action shield",
        },
        "checkpoint": str(predictor_output),
        "distance_field_checkpoint": str(field_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate general obstacle route representations")
    parser.add_argument("--train-seeds", default="101,103,107")
    parser.add_argument("--evaluation-seeds", default="11,23,41")
    parser.add_argument("--train-episodes", type=int, default=200)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--points", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=360)
    parser.add_argument("--predictor-output", default="artifacts/general-route-sketch-v18.pt")
    parser.add_argument("--output", default="artifacts/evaluation-general-routes-v18.json")
    args = parser.parse_args()
    report = train_and_evaluate_general_routes(
        train_seeds=tuple(int(value) for value in args.train_seeds.split(",") if value.strip()),
        evaluation_seeds=tuple(int(value) for value in args.evaluation_seeds.split(",") if value.strip()),
        train_episodes=args.train_episodes,
        evaluation_episodes=args.evaluation_episodes,
        max_steps=args.max_steps,
        points=args.points,
        epochs=args.epochs,
        predictor_output=args.predictor_output,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
