"""Distill and evaluate a learned RGB route policy without A* at test time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import PocketWorldEnv
from .evaluate_planners import _episode_cases, _scenario_walls
from .planner import extract_agent_position, route_controller_parameters, route_following_action, estimate_agent_velocity
from .route_policy import (
    ROUTE_MODES,
    LearnedRoutePolicy,
    RouteModePolicy,
    observable_route_waypoints,
    observable_waypoint_action,
    route_mode_label,
    wall_grid_features,
)


def collect_route_mode_data(
    seeds: tuple[int, ...],
    scenarios: tuple[str, ...],
    episodes: int = 20,
    agent_speed_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect one RGB state per task with a visible-geometry route label."""
    observations: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    labels: list[int] = []
    for seed in seeds:
        for scenario in scenarios:
            walls = _scenario_walls(scenario)
            for start, goal in _episode_cases(episodes, seed, scenario):
                env = PocketWorldEnv(
                    walls=walls,
                    agent_start=start,
                    goal=goal,
                    agent_speed_scale=agent_speed_scale,
                )
                observation, _ = env.reset()
                observations.append(observation.copy())
                goals.append(np.asarray(goal, dtype=np.float32))
                labels.append(route_mode_label(observation, goal))
    return (
        np.asarray(observations, dtype=np.uint8),
        np.asarray(goals, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def evaluate_route_mode_policy(
    policy: RouteModePolicy,
    seeds: tuple[int, ...],
    scenarios: tuple[str, ...],
    episodes: int = 20,
    max_steps: int = 64,
    agent_speed_scale: float = 1.0,
) -> dict[str, object]:
    """Execute a predicted route mode with RGB-only waypoint control."""
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        walls = _scenario_walls(scenario)
        for seed in seeds:
            for episode, (start, goal) in enumerate(_episode_cases(episodes, seed, scenario)):
                env = PocketWorldEnv(
                    walls=walls,
                    agent_start=start,
                    goal=goal,
                    agent_speed_scale=agent_speed_scale,
                )
                observation, info = env.reset()
                history = [observation]
                mode = policy.predict_mode(observation, goal)
                waypoints = observable_route_waypoints(observation, goal, mode)
                waypoint_index = 0
                collisions = 0
                actions: list[int] = []
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
                    actions.append(action)
                    collisions += int(info.get("collision", False))
                    if terminated or truncated:
                        break
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "episode": episode,
                        "predicted_mode": ROUTE_MODES[mode],
                        "real_success": float(info["distance_to_goal"] <= env.goal_radius),
                        "final_distance_px": float(info["distance_to_goal"]),
                        "collision_count": collisions,
                        "executed_actions": len(actions),
                        "waypoint_count": len(waypoints),
                    }
                )
    values = {
        key: np.asarray([row[key] for row in rows], dtype=np.float64)
        for key in ("real_success", "final_distance_px", "collision_count", "executed_actions")
    }
    return {
        "rows": rows,
        "summary": {
            key: {"mean": float(value.mean()), "std": float(value.std())}
            for key, value in values.items()
        },
    }


def train_and_evaluate_route_mode(
    train_seeds: tuple[int, ...] = (101, 103),
    train_scenarios: tuple[str, ...] = ("single_barrier", "barrier_narrow_gap"),
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    evaluation_scenarios: tuple[str, ...] = ("single_barrier", "barrier_shifted", "barrier_narrow_gap", "barrier_wide_gap"),
    train_episodes: int = 20,
    evaluation_episodes: int = 20,
    max_steps: int = 64,
    epochs: int = 80,
    predictor_output: str | Path = "artifacts/route-mode-policy-v15.pt",
    validation_seeds: tuple[int, ...] = (107,),
    validation_scenarios: tuple[str, ...] = ("single_barrier", "barrier_narrow_gap"),
) -> dict[str, object]:
    observations, goals, labels = collect_route_mode_data(
        train_seeds, train_scenarios, train_episodes
    )
    policy = RouteModePolicy()
    training = policy.fit(observations, goals, labels, epochs=epochs)
    validation_data = collect_route_mode_data(
        validation_seeds, validation_scenarios, max(1, train_episodes // 2)
    )
    with torch.no_grad():
        positions = extract_agent_position(validation_data[0]).astype(np.float32)
        positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
        predictions = policy(
            torch.from_numpy(validation_data[1]) / 64.0,
            torch.from_numpy(positions) / 64.0,
            wall_grid_features(validation_data[0]),
        ).argmax(dim=-1).numpy()
    policy.save(predictor_output, metadata={"route_modes": list(ROUTE_MODES)})
    evaluation = evaluate_route_mode_policy(
        policy, evaluation_seeds, evaluation_scenarios, evaluation_episodes, max_steps
    )
    return {
        "protocol": {
            "teacher": "visible RGB geometry route mode",
            "student_inputs": ["initial RGB observation", "goal coordinates"],
            "student_evaluation_uses_astar": False,
            "route_modes": list(ROUTE_MODES),
            "train_seeds": list(train_seeds),
            "train_scenarios": list(train_scenarios),
            "evaluation_seeds": list(evaluation_seeds),
            "evaluation_scenarios": list(evaluation_scenarios),
            "train_episodes": train_episodes,
            "evaluation_episodes": evaluation_episodes,
            "max_steps": max_steps,
        },
        "training": training,
        "validation": {
            "samples": int(len(validation_data[2])),
            "mode_accuracy": float(np.mean(predictions == validation_data[2])),
        },
        "evaluation": evaluation,
        "comparison": {
            "route_mode_student": {
                "uses_astar_at_evaluation": False,
                "uses_rgb_waypoint_controller": True,
                "route_level_prediction": True,
            },
            "action_level_student": {
                "uses_astar_at_evaluation": False,
                "route_level_prediction": False,
                "known_smoke_failure": "action imitation had 0% real success under covariate shift",
            },
            "geometry_locked_hybrid_upper_bound": {
                "uses_astar_at_evaluation": True,
                "uses_rgb_geometry": True,
                "role": "observable-geometry upper bound, not a pure learned policy",
            },
        },
        "checkpoint": str(predictor_output),
    }


def collect_teacher_data(
    seeds: tuple[int, ...],
    scenarios: tuple[str, ...],
    episodes: int = 20,
    max_steps: int = 64,
    agent_speed_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    actions: list[int] = []
    for seed in seeds:
        for scenario in scenarios:
            walls = _scenario_walls(scenario)
            for start, goal in _episode_cases(episodes, seed, scenario):
                env = PocketWorldEnv(
                    walls=walls,
                    agent_start=start,
                    goal=goal,
                    agent_speed_scale=agent_speed_scale,
                )
                observation, _ = env.reset()
                history = [observation]
                teacher_config = route_controller_parameters(observation, goal)
                for _ in range(max_steps):
                    observations.append(observation.copy())
                    goals.append(np.asarray(goal, dtype=np.float32))
                    velocities.append(estimate_agent_velocity(history, max_speed=2.5))
                    positions.append(extract_agent_position(observation).astype(np.float32))
                    action, _ = route_following_action(
                        observation,
                        goal,
                        observation_history=history,
                        **teacher_config,
                    )
                    actions.append(action)
                    observation, _, terminated, truncated, _ = env.step(action)
                    history.append(observation)
                    history = history[-16:]
                    if terminated or truncated:
                        break
    return (
        np.asarray(observations, dtype=np.uint8),
        np.asarray(goals, dtype=np.float32),
        np.asarray(velocities, dtype=np.float32),
        np.asarray(positions, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
    )


def collect_dagger_data(
    policy: LearnedRoutePolicy,
    seeds: tuple[int, ...],
    scenarios: tuple[str, ...],
    episodes: int = 20,
    max_steps: int = 64,
    agent_speed_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Label states visited by the student with the observable teacher action."""
    observations: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    actions: list[int] = []
    student_actions: list[int] = []
    for seed in seeds:
        for scenario in scenarios:
            walls = _scenario_walls(scenario)
            for start, goal in _episode_cases(episodes, seed, scenario):
                env = PocketWorldEnv(
                    walls=walls,
                    agent_start=start,
                    goal=goal,
                    agent_speed_scale=agent_speed_scale,
                )
                observation, _ = env.reset()
                history = [observation]
                teacher_config = route_controller_parameters(observation, goal)
                for _ in range(max_steps):
                    observations.append(observation.copy())
                    goals.append(np.asarray(goal, dtype=np.float32))
                    velocities.append(estimate_agent_velocity(history, max_speed=2.5))
                    positions.append(extract_agent_position(observation).astype(np.float32))
                    teacher_action, _ = route_following_action(
                        observation,
                        goal,
                        observation_history=history,
                        **teacher_config,
                    )
                    actions.append(teacher_action)
                    student_action = policy.predict_action(observation, goal, history)
                    student_actions.append(student_action)
                    observation, _, terminated, truncated, _ = env.step(student_action)
                    history.append(observation)
                    history = history[-16:]
                    if terminated or truncated:
                        break
    return (
        np.asarray(observations, dtype=np.uint8),
        np.asarray(goals, dtype=np.float32),
        np.asarray(velocities, dtype=np.float32),
        np.asarray(positions, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        np.asarray(student_actions, dtype=np.int64),
    )


def evaluate_policy(
    policy: LearnedRoutePolicy,
    seeds: tuple[int, ...],
    scenarios: tuple[str, ...],
    episodes: int = 20,
    max_steps: int = 64,
    agent_speed_scale: float = 1.0,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        walls = _scenario_walls(scenario)
        for seed in seeds:
            for episode, (start, goal) in enumerate(_episode_cases(episodes, seed, scenario)):
                env = PocketWorldEnv(
                    walls=walls,
                    agent_start=start,
                    goal=goal,
                    agent_speed_scale=agent_speed_scale,
                )
                observation, info = env.reset()
                history = [observation]
                collisions = 0
                actions = []
                for _ in range(max_steps):
                    action = policy.predict_action(observation, goal, history)
                    observation, _, terminated, truncated, info = env.step(action)
                    history.append(observation)
                    history = history[-16:]
                    actions.append(action)
                    collisions += int(info.get("collision", False))
                    if terminated or truncated:
                        break
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "episode": episode,
                        "real_success": float(info["distance_to_goal"] <= env.goal_radius),
                        "final_distance_px": float(info["distance_to_goal"]),
                        "collision_count": collisions,
                        "executed_actions": len(actions),
                    }
                )
    values = {
        key: np.asarray([row[key] for row in rows], dtype=np.float64)
        for key in ("real_success", "final_distance_px", "collision_count", "executed_actions")
    }
    return {
        "rows": rows,
        "summary": {
            key: {"mean": float(value.mean()), "std": float(value.std())}
            for key, value in values.items()
        },
    }


def evaluate_action_imitation(
    policy: LearnedRoutePolicy,
    observations: np.ndarray,
    goals: np.ndarray,
    velocities: np.ndarray,
    actions: np.ndarray,
    positions: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Measure offline teacher-action agreement without executing the student."""
    frame_tensor = np.asarray(observations, dtype=np.uint8)
    goal_values = np.asarray(goals, dtype=np.float32)
    velocity_values = np.asarray(velocities, dtype=np.float32)
    target_values = np.asarray(actions, dtype=np.int64).reshape(-1)
    if positions is None:
        positions = extract_agent_position(frame_tensor).astype(np.float32)
        positions = np.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
    if frame_tensor.ndim != 4 or frame_tensor.shape[0] != target_values.shape[0]:
        raise ValueError("imitation arrays must have aligned sample dimensions")
    with torch.no_grad():
        logits = policy(
            torch.from_numpy(frame_tensor).float() / 255.0,
            torch.from_numpy(goal_values).float() / 64.0,
            torch.from_numpy(velocity_values).float() / 3.0,
            torch.from_numpy(np.asarray(positions, dtype=np.float32)).float() / 64.0,
            wall_grid_features(frame_tensor),
        )
        predicted = logits.argmax(dim=-1).cpu().numpy()
    return {
        "samples": int(target_values.size),
        "action_accuracy": float(np.mean(predicted == target_values)),
        "action_macro_recall": float(
            np.mean(
                [
                    np.mean(predicted[target_values == action] == action)
                    if np.any(target_values == action)
                    else float("nan")
                    for action in range(4)
                ]
            )
        ),
    }


def train_and_evaluate(
    train_seeds: tuple[int, ...] = (101, 103),
    train_scenarios: tuple[str, ...] = ("single_barrier", "barrier_narrow_gap"),
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    evaluation_scenarios: tuple[str, ...] = ("single_barrier", "barrier_shifted", "barrier_narrow_gap", "barrier_wide_gap"),
    train_episodes: int = 20,
    evaluation_episodes: int = 20,
    max_steps: int = 64,
    epochs: int = 80,
    predictor_output: str | Path = "artifacts/learned-route-policy-v14.pt",
    validation_seeds: tuple[int, ...] = (107,),
    validation_scenarios: tuple[str, ...] = ("single_barrier", "barrier_narrow_gap"),
    dagger_rounds: int = 2,
) -> dict[str, object]:
    observations, goals, velocities, positions, actions = collect_teacher_data(
        train_seeds, train_scenarios, train_episodes, max_steps
    )
    policy = LearnedRoutePolicy()
    training = policy.fit(
        observations, goals, velocities, actions, positions=positions, epochs=epochs
    )
    dagger_history: list[dict[str, object]] = []
    for round_index in range(max(0, int(dagger_rounds))):
        dagger_data = collect_dagger_data(
            policy,
            train_seeds,
            train_scenarios,
            episodes=train_episodes,
            max_steps=max_steps,
        )
        observations = np.concatenate((observations, dagger_data[0]), axis=0)
        goals = np.concatenate((goals, dagger_data[1]), axis=0)
        velocities = np.concatenate((velocities, dagger_data[2]), axis=0)
        positions = np.concatenate((positions, dagger_data[3]), axis=0)
        actions = np.concatenate((actions, dagger_data[4]), axis=0)
        training = policy.fit(
            observations,
            goals,
            velocities,
            actions,
            positions=positions,
            epochs=epochs,
            seed=7 + round_index + 1,
        )
        dagger_history.append(
            {
                "round": round_index + 1,
                "aggregated_samples": int(observations.shape[0]),
                "student_rollout_samples": int(dagger_data[0].shape[0]),
                "student_rollout_teacher_action_rate": float(
                    np.mean(dagger_data[4] == dagger_data[5])
                ),
            }
        )
    validation_data = collect_teacher_data(
        validation_seeds,
        validation_scenarios,
        episodes=max(1, train_episodes // 2),
        max_steps=max_steps,
    )
    validation_imitation = evaluate_action_imitation(
        policy,
        validation_data[0],
        validation_data[1],
        validation_data[2],
        validation_data[4],
        validation_data[3],
    )
    policy.save(
        predictor_output,
        metadata={
            "teacher": "geometry-locked RGB/A* route controller",
            "train_seeds": list(train_seeds),
            "train_scenarios": list(train_scenarios),
            "train_episodes": train_episodes,
            "max_steps": max_steps,
        },
    )
    evaluation = evaluate_policy(policy, evaluation_seeds, evaluation_scenarios, evaluation_episodes, max_steps)
    return {
        "protocol": {
            "teacher": "geometry-locked RGB/A* controller",
            "student_inputs": ["RGB observation", "goal coordinates", "RGB history velocity"],
            "student_evaluation_uses_astar": False,
            "train_seeds": list(train_seeds),
            "train_scenarios": list(train_scenarios),
            "evaluation_seeds": list(evaluation_seeds),
            "evaluation_scenarios": list(evaluation_scenarios),
            "train_episodes": train_episodes,
            "evaluation_episodes": evaluation_episodes,
            "max_steps": max_steps,
            "validation_seeds": list(validation_seeds),
            "validation_scenarios": list(validation_scenarios),
        },
        "training": training,
        "dagger": dagger_history,
        "teacher_imitation": {
            "train": evaluate_action_imitation(
                policy, observations, goals, velocities, actions, positions
            ),
            "validation": validation_imitation,
        },
        "evaluation": evaluation,
        "checkpoint": str(predictor_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a learned RGB route policy")
    parser.add_argument("--route-level", action="store_true", help="run the route-mode waypoint study")
    parser.add_argument("--train-seeds", default="101,103")
    parser.add_argument("--train-scenarios", default="single_barrier,barrier_narrow_gap")
    parser.add_argument("--evaluation-seeds", default="11,23,41")
    parser.add_argument("--evaluation-scenarios", default="single_barrier,barrier_shifted,barrier_narrow_gap,barrier_wide_gap")
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--validation-seeds", default="107")
    parser.add_argument("--validation-scenarios", default="single_barrier,barrier_narrow_gap")
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--predictor-output", default="artifacts/learned-route-policy-v14.pt")
    parser.add_argument("--output", default="artifacts/evaluation-learned-route-policy-v14.json")
    args = parser.parse_args()
    parse_strings = lambda value: tuple(item.strip() for item in value.split(",") if item.strip())
    if args.route_level:
        report = train_and_evaluate_route_mode(
            train_seeds=tuple(int(item) for item in args.train_seeds.split(",") if item.strip()),
            train_scenarios=parse_strings(args.train_scenarios),
            evaluation_seeds=tuple(int(item) for item in args.evaluation_seeds.split(",") if item.strip()),
            evaluation_scenarios=parse_strings(args.evaluation_scenarios),
            train_episodes=args.train_episodes,
            evaluation_episodes=args.evaluation_episodes,
            max_steps=args.max_steps,
            epochs=args.epochs,
            predictor_output=args.predictor_output,
            validation_seeds=tuple(int(item) for item in args.validation_seeds.split(",") if item.strip()),
            validation_scenarios=parse_strings(args.validation_scenarios),
        )
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    report = train_and_evaluate(
        train_seeds=tuple(int(item) for item in args.train_seeds.split(",") if item.strip()),
        train_scenarios=parse_strings(args.train_scenarios),
        evaluation_seeds=tuple(int(item) for item in args.evaluation_seeds.split(",") if item.strip()),
        evaluation_scenarios=parse_strings(args.evaluation_scenarios),
        train_episodes=args.train_episodes,
        evaluation_episodes=args.evaluation_episodes,
        max_steps=args.max_steps,
        epochs=args.epochs,
        predictor_output=args.predictor_output,
        validation_seeds=tuple(int(item) for item in args.validation_seeds.split(",") if item.strip()),
        validation_scenarios=parse_strings(args.validation_scenarios),
        dagger_rounds=args.dagger_rounds,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
