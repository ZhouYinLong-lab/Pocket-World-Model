"""Compare learned route sketches with RGB projection and A* fallback methods."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .env import PocketWorldEnv
from .general_routes import GENERAL_FAMILIES, GeneralRouteCase, sample_general_route_cases
from .planner import _astar_path, _dilate, extract_agent_position, extract_wall_mask
from .route_policy import (
    RouteSketchPolicy,
    observable_route_sketch_waypoints,
    observable_waypoint_action,
    route_sketch_targets,
)
from .route_field import (
    RouteFieldPolicy,
    conservative_field_action,
    field_waypoints,
    guarded_mpc_action,
    local_mpc_action,
    rgb_action_is_safe,
    route_progress_metrics,
)

GENERAL_METHODS = (
    "learned",
    "rgb_projection",
    "hybrid_astar",
    "rgb_astar",
    "distance_field",
    "distance_field_rgb_projection",
    "distance_field_beam_rgb_projection",
    "distance_field_beam_conservative",
    "distance_field_beam_mpc",
    "distance_field_beam_robust_mpc",
    "distance_field_beam_adaptive_mpc",
    "distance_field_beam_collision_head_mpc",
    "distance_field_clearance_beam_rgb_projection",
    "distance_field_beam_guarded_mpc",
    "distance_field_mpc_shift_fallback",
    "distance_field_budgeted_hybrid_mpc",
)
GENERAL_DEFAULT_METHODS = tuple(
    method
    for method in GENERAL_METHODS[:-2]
    if method
    not in {
        "distance_field_beam_adaptive_mpc",
        "distance_field_beam_collision_head_mpc",
        "distance_field_budgeted_hybrid_mpc",
    }
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
    families: tuple[str, ...] | None = None,
    balanced_families: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for seed in seeds:
        for case in sample_general_route_cases(
            seed,
            episodes,
            split=split,
            families=families,
            balanced=balanced_families,
        ):
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
    mpc_horizon: int = 6,
    mpc_beam_width: int = 24,
    mpc_velocity_source: str = "rgb",
    cases_by_seed: dict[int, tuple[GeneralRouteCase, ...]] | None = None,
    agent_speed_scale: float = 1.0,
    reference_signatures: np.ndarray | None = None,
    shift_threshold: float = 0.0,
    adaptive_risk_threshold: float = 0.45,
    adaptive_risk_exit_threshold: float = 0.30,
    collision_head: object | None = None,
    collision_head_risk_threshold: float = 0.35,
    collision_head_risk_exit_threshold: float = 0.25,
    collision_head_horizon_index: int = 1,
    route_budget_margin: float = 1.05,
    route_progress_tolerance: float = 1.5,
) -> dict[str, object]:
    if method not in GENERAL_METHODS:
        raise ValueError(f"method must be one of {GENERAL_METHODS}")
    rows: list[dict[str, object]] = []
    if agent_speed_scale <= 0.0 or not np.isfinite(agent_speed_scale):
        raise ValueError("agent_speed_scale must be finite and positive")
    if route_budget_margin < 0.0 or not np.isfinite(route_budget_margin):
        raise ValueError("route_budget_margin must be finite and non-negative")
    if route_progress_tolerance < 0.0 or not np.isfinite(route_progress_tolerance):
        raise ValueError("route_progress_tolerance must be finite and non-negative")
    for seed in seeds:
        cases = (
            cases_by_seed[seed]
            if cases_by_seed is not None
            else sample_general_route_cases(seed, episodes, split="holdout")
        )
        for case in cases:
            env = PocketWorldEnv(
                walls=case.walls,
                agent_start=case.start,
                goal=case.goal,
                agent_speed_scale=agent_speed_scale,
            )
            observation, info = env.reset()
            history = [observation]
            action_history: list[int] = []
            position = extract_agent_position(observation).astype(np.float32)
            astar_calls = 0
            fallback_triggered = False
            mpc_calls = 0
            robust_mpc_calls = 0
            mpc_override_count = 0
            adaptive_switches = 0
            adaptive_risk_sum = 0.0
            adaptive_risk_max = 0.0
            adaptive_robust_active = False
            collision_head_active = False
            collision_head_calls = 0
            collision_head_robust_calls = 0
            collision_head_switches = 0
            collision_head_risk_sum = 0.0
            collision_head_risk_max = 0.0
            planning_time_ms = 0.0
            route_points = np.asarray((tuple(position),), dtype=np.float32)
            layout_shift_score = 0.0
            shift_detected = False
            if method == "rgb_astar":
                waypoints = _astar_waypoints(observation, case.goal, position, points)
                astar_calls += 1
            elif method in {
                "distance_field",
                "distance_field_rgb_projection",
                "distance_field_beam_rgb_projection",
                "distance_field_beam_conservative",
                "distance_field_beam_mpc",
                "distance_field_beam_robust_mpc",
                "distance_field_beam_adaptive_mpc",
                "distance_field_beam_collision_head_mpc",
                "distance_field_clearance_beam_rgb_projection",
                "distance_field_beam_guarded_mpc",
                "distance_field_mpc_shift_fallback",
                "distance_field_budgeted_hybrid_mpc",
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
                        "distance_field_beam_mpc",
                        "distance_field_beam_robust_mpc",
                        "distance_field_beam_adaptive_mpc",
                        "distance_field_beam_collision_head_mpc",
                        "distance_field_clearance_beam_rgb_projection",
                        "distance_field_mpc_shift_fallback",
                        "distance_field_budgeted_hybrid_mpc",
                    }
                    else 1,
                )
                route_points = np.asarray((tuple(position),) + tuple(waypoints), dtype=np.float32)
                if method == "distance_field_mpc_shift_fallback":
                    if reference_signatures is None or shift_threshold <= 0.0:
                        raise ValueError(
                            "shift fallback requires reference_signatures and positive shift_threshold"
                        )
                    from .route_field import wall_layout_shift_score

                    layout_shift_score = wall_layout_shift_score(
                        observation, reference_signatures
                    )
                    if layout_shift_score > shift_threshold:
                        waypoints = _astar_waypoints(
                            observation, case.goal, position, points
                        )
                        astar_calls += 1
                        fallback_triggered = True
                        shift_detected = True
                        route_points = np.asarray((tuple(position),) + tuple(waypoints), dtype=np.float32)
            else:
                predicted = policy.predict_points(observation, case.goal)
                waypoints = observable_route_sketch_waypoints(observation, case.goal, predicted)
            route_points = np.asarray((tuple(position),) + tuple(waypoints), dtype=np.float32)
            waypoint_index = 0
            collisions = 0
            target_dirty = True
            planned_target: tuple[float, float] | None = None
            last_route_check = -100
            route_progress = route_progress_metrics(position, route_points)
            last_route_progress = float(route_progress["progress_px"])
            progress_regression_streak = 0
            budget_fallbacks = 0
            budget_infeasible_events = 0
            wall_block_events = 0
            progress_regression_events = 0
            wall_block_streak = 0
            last_budget_fallback_step = -1000
            budget_slack_sum = 0.0
            budget_slack_min = float("inf")
            for _ in range(max_steps):
                step_index = env.steps
                position = extract_agent_position(observation).astype(np.float32)
                route_progress = route_progress_metrics(position, route_points)
                progress_delta = float(route_progress["progress_px"] - last_route_progress)
                if method == "distance_field_budgeted_hybrid_mpc":
                    if progress_delta < -route_progress_tolerance:
                        progress_regression_streak += 1
                        progress_regression_events += 1
                    else:
                        progress_regression_streak = max(0, progress_regression_streak - 1)
                    remaining_steps = max(1, max_steps - step_index)
                    reachable_distance = (
                        remaining_steps * PocketWorldEnv.max_speed * agent_speed_scale
                    )
                    budget_slack = reachable_distance - (
                        route_budget_margin * float(route_progress["remaining_px"])
                    )
                    budget_slack_sum += budget_slack
                    budget_slack_min = min(budget_slack_min, budget_slack)
                else:
                    budget_slack = float("nan")
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
                if method == "distance_field_budgeted_hybrid_mpc" and (
                    target_dirty or step_index - last_route_check >= 4
                ):
                    route_check_due = True
                    route_is_blocked = _segment_hits_wall(observation, position, target)
                    budget_is_infeasible = budget_slack < 0.0
                    progress_is_stalled = progress_regression_streak >= 2
                    wall_block_events += int(route_is_blocked)
                    budget_infeasible_events += int(budget_is_infeasible)
                    wall_block_streak = wall_block_streak + 1 if route_is_blocked else 0
                    fallback_cooldown_elapsed = step_index - last_budget_fallback_step >= 12
                    fallback_reason_ready = (
                        wall_block_streak >= 2 or budget_is_infeasible or progress_is_stalled
                    )
                    if fallback_reason_ready and fallback_cooldown_elapsed:
                        waypoints = _astar_waypoints(observation, case.goal, position, points)
                        waypoint_index = 0
                        target = waypoints[0]
                        route_points = np.asarray(
                            (tuple(position),) + tuple(waypoints), dtype=np.float32
                        )
                        route_progress = route_progress_metrics(position, route_points)
                        last_route_progress = float(route_progress["progress_px"])
                        progress_regression_streak = 0
                        wall_block_streak = 0
                        astar_calls += 1
                        budget_fallbacks += 1
                        last_budget_fallback_step = step_index
                        fallback_triggered = True
                        target_dirty = True
                        planned_target = None
                        last_route_check = step_index
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
                planning_start = time.perf_counter()
                action = observable_waypoint_action(observation, target, history, damping=1.0)
                if method == "distance_field_beam_conservative":
                    action = conservative_field_action(observation, target, history)
                if method == "distance_field_beam_mpc":
                    mpc_calls += 1
                    action = local_mpc_action(
                        observation,
                        target,
                        history,
                        horizon=mpc_horizon,
                        beam_width=mpc_beam_width,
                        action_history=action_history,
                        velocity_source=mpc_velocity_source,
                    )
                if method == "distance_field_mpc_shift_fallback":
                    mpc_calls += 1
                    action = local_mpc_action(
                        observation,
                        target,
                        history,
                        horizon=mpc_horizon,
                        beam_width=mpc_beam_width,
                        action_history=action_history,
                        velocity_source=mpc_velocity_source,
                    )
                if method == "distance_field_beam_robust_mpc":
                    mpc_calls += 1
                    action = local_mpc_action(
                        observation,
                        target,
                        history,
                        horizon=mpc_horizon,
                        beam_width=mpc_beam_width,
                        robust=True,
                        action_history=action_history,
                        velocity_source=mpc_velocity_source,
                    )
                    robust_mpc_calls += 1
                if method == "distance_field_beam_adaptive_mpc":
                    from .route_field import adaptive_mpc_decision

                    mpc_calls += 1
                    action, use_robust, risk_score = adaptive_mpc_decision(
                        observation,
                        target,
                        action,
                        history,
                        action_history,
                        horizon=mpc_horizon,
                        beam_width=mpc_beam_width,
                        velocity_source=mpc_velocity_source,
                        risk_threshold=adaptive_risk_threshold,
                        risk_exit_threshold=adaptive_risk_exit_threshold,
                        robust_active=adaptive_robust_active,
                    )
                    adaptive_risk_sum += risk_score
                    adaptive_risk_max = max(adaptive_risk_max, risk_score)
                    if use_robust:
                        # The decision first evaluates an ordinary MPC
                        # candidate, then runs robust MPC when the risk gate is
                        # active. Count both actual planner calls.
                        mpc_calls += 1
                    if use_robust != adaptive_robust_active:
                        adaptive_switches += 1
                    adaptive_robust_active = use_robust
                    if use_robust:
                        robust_mpc_calls += 1
                if method == "distance_field_beam_collision_head_mpc":
                    if collision_head is None:
                        raise ValueError(
                            "collision_head method requires a trained collision_head"
                        )
                    from .route_field import collision_head_mpc_decision

                    collision_head_calls += 1
                    mpc_calls += 1
                    action, use_robust, risk_score = collision_head_mpc_decision(
                        collision_head,
                        observation,
                        case.goal,
                        target,
                        history,
                        action_history,
                        horizon=mpc_horizon,
                        beam_width=mpc_beam_width,
                        velocity_source=mpc_velocity_source,
                        risk_threshold=collision_head_risk_threshold,
                        risk_exit_threshold=collision_head_risk_exit_threshold,
                        robust_active=collision_head_active,
                        probability_horizon_index=collision_head_horizon_index,
                    )
                    collision_head_risk_sum += risk_score
                    collision_head_risk_max = max(collision_head_risk_max, risk_score)
                    if use_robust:
                        mpc_calls += 1
                        collision_head_robust_calls += 1
                    if use_robust != collision_head_active:
                        collision_head_switches += 1
                    collision_head_active = use_robust
                if method == "distance_field_beam_guarded_mpc":
                    mpc_calls += 1
                    baseline_action = action
                    action = guarded_mpc_action(
                        observation,
                        target,
                        action,
                        history,
                        action_history,
                        horizon=mpc_horizon,
                        beam_width=mpc_beam_width,
                    )
                    if action != baseline_action:
                        mpc_override_count += 1
                if method == "distance_field_budgeted_hybrid_mpc":
                    mpc_calls += 1
                    action = local_mpc_action(
                        observation,
                        target,
                        history,
                        horizon=mpc_horizon,
                        beam_width=mpc_beam_width,
                        robust=True,
                        action_history=action_history,
                        velocity_source=mpc_velocity_source,
                    )
                    robust_mpc_calls += 1
                planning_time_ms += (time.perf_counter() - planning_start) * 1000.0
                observation, _, terminated, truncated, info = env.step(action)
                action_history.append(int(action))
                action_history = action_history[-16:]
                history.append(observation)
                history = history[-16:]
                collisions += int(info.get("collision", False))
                route_progress = route_progress_metrics(
                    extract_agent_position(observation).astype(np.float32), route_points
                )
                last_route_progress = max(
                    last_route_progress,
                    float(route_progress["progress_px"]),
                )
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
                    "mpc_calls": mpc_calls,
                    "robust_mpc_calls": robust_mpc_calls,
                    "mpc_override_count": mpc_override_count,
                    "adaptive_switches": adaptive_switches,
                    "adaptive_risk_mean": adaptive_risk_sum / max(1, int(env.steps)),
                    "adaptive_risk_max": adaptive_risk_max,
                    "collision_head_calls": collision_head_calls,
                    "collision_head_robust_calls": collision_head_robust_calls,
                    "collision_head_switches": collision_head_switches,
                    "collision_head_risk_mean": collision_head_risk_sum / max(1, int(env.steps)),
                    "collision_head_risk_max": collision_head_risk_max,
                    "planning_time_ms": planning_time_ms,
                    "layout_shift_score": layout_shift_score,
                    "shift_detected": shift_detected,
                    "budget_fallbacks": budget_fallbacks,
                    "budget_infeasible_events": budget_infeasible_events,
                    "wall_block_events": wall_block_events,
                    "progress_regression_events": progress_regression_events,
                    "progress_regressions": int(progress_regression_streak),
                    "route_progress_norm": float(route_progress["progress_norm"]),
                    "route_remaining_px": float(route_progress["remaining_px"]),
                    "route_budget_slack_mean": budget_slack_sum / max(1, int(env.steps)),
                    "route_budget_slack_min": (
                        float(budget_slack_min) if np.isfinite(budget_slack_min) else 0.0
                    ),
                }
            )
    metrics = (
        "real_success",
        "final_distance_px",
        "collision_count",
        "executed_actions",
        "astar_calls",
        "fallback_triggered",
        "mpc_calls",
        "robust_mpc_calls",
        "mpc_override_count",
        "adaptive_switches",
        "adaptive_risk_mean",
        "adaptive_risk_max",
        "collision_head_calls",
        "collision_head_robust_calls",
        "collision_head_switches",
        "collision_head_risk_mean",
        "collision_head_risk_max",
        "planning_time_ms",
        "layout_shift_score",
        "shift_detected",
        "budget_fallbacks",
        "budget_infeasible_events",
        "wall_block_events",
        "progress_regression_events",
        "progress_regressions",
        "route_progress_norm",
        "route_remaining_px",
        "route_budget_slack_mean",
        "route_budget_slack_min",
    )
    values = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in metrics}
    by_family: dict[str, dict[str, dict[str, float]]] = {}
    for family in sorted({str(row["family"]) for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        family_values = {
            key: np.asarray([row[key] for row in family_rows], dtype=np.float64)
            for key in metrics
        }
        by_family[family] = {
            key: {"mean": float(value.mean()), "std": float(value.std())}
            for key, value in family_values.items()
        }
    return {
        "rows": rows,
        "summary": {
            key: {"mean": float(value.mean()), "std": float(value.std())}
            for key, value in values.items()
        },
        "by_family": by_family,
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
    mpc_horizon: int = 6,
    mpc_beam_width: int = 24,
    mpc_velocity_source: str = "rgb",
    methods: tuple[str, ...] | None = None,
    training_families: tuple[str, ...] | None = None,
    balanced_training_families: bool = False,
    adaptive_risk_threshold: float = 0.45,
    adaptive_risk_exit_threshold: float = 0.30,
    route_budget_margin: float = 1.05,
    route_progress_tolerance: float = 1.5,
) -> dict[str, object]:
    selected_methods = tuple(GENERAL_DEFAULT_METHODS if methods is None else methods)
    unknown_methods = set(selected_methods) - set(GENERAL_METHODS)
    if unknown_methods:
        raise ValueError(f"unknown general route methods: {sorted(unknown_methods)}")
    observations, goals, targets = collect_general_route_data(
        train_seeds,
        train_episodes,
        "train",
        points,
        families=training_families,
        balanced_families=balanced_training_families,
    )
    policy = RouteSketchPolicy(points=points)
    training = {"route_sketch": policy.fit(observations, goals, targets, epochs=epochs)}
    field_policy = RouteFieldPolicy()
    training["distance_field"] = field_policy.fit(observations, goals, epochs=epochs)
    clearance_policy = RouteFieldPolicy()
    training["distance_field_clearance"] = clearance_policy.fit(
        observations, goals, epochs=epochs, clearance_weight=8.0
    )
    field_output = Path(predictor_output).with_name(
        f"{Path(predictor_output).stem}-distance-field{Path(predictor_output).suffix}"
    )
    evaluations = {
        method: evaluate_general_policy(
            clearance_policy
            if method == "distance_field_clearance_beam_rgb_projection"
            else field_policy
            if method.startswith("distance_field")
            else policy,
            evaluation_seeds,
            evaluation_episodes,
            max_steps,
            points,
            method,
            mpc_horizon,
                mpc_beam_width,
                mpc_velocity_source,
                adaptive_risk_threshold=adaptive_risk_threshold,
                adaptive_risk_exit_threshold=adaptive_risk_exit_threshold,
                route_budget_margin=route_budget_margin,
                route_progress_tolerance=route_progress_tolerance,
            )
        for method in selected_methods
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
    clearance_output = Path(predictor_output).with_name(
        f"{Path(predictor_output).stem}-clearance-field{Path(predictor_output).suffix}"
    )
    clearance_policy.save(
        clearance_output,
        metadata={
            "student_evaluation_uses_astar": False,
            "teacher_labels_use_astar": True,
            "rgb_projection_uses_astar": False,
            "clearance_weight": 8.0,
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
            "mpc_horizon": mpc_horizon,
            "mpc_beam_width": mpc_beam_width,
            "mpc_velocity_source": mpc_velocity_source,
            "adaptive_risk_threshold": adaptive_risk_threshold,
            "adaptive_risk_exit_threshold": adaptive_risk_exit_threshold,
            "route_budget_margin": route_budget_margin,
            "route_progress_tolerance": route_progress_tolerance,
            "methods": list(selected_methods),
            "training_families": list(
                training_families
                if training_families is not None
                else ("staggered_blocks", "multi_channel")
            ),
            "balanced_training_families": balanced_training_families,
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
                "distance_field_beam_mpc": False,
                "distance_field_beam_robust_mpc": False,
                "distance_field_beam_adaptive_mpc": False,
                "distance_field_beam_collision_head_mpc": False,
                "distance_field_clearance_beam_rgb_projection": False,
                "distance_field_beam_guarded_mpc": False,
                "distance_field_mpc_shift_fallback": True,
                "distance_field_budgeted_hybrid_mpc": True,
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
            "distance_field_beam_mpc": "learned-field beam with RGB-only short-horizon inertial MPC",
            "distance_field_beam_robust_mpc": "learned-field beam with velocity-scale robust RGB-only MPC",
            "distance_field_beam_adaptive_mpc": "learned-field beam with fixed-threshold online ordinary/robust MPC switching",
            "distance_field_beam_collision_head_mpc": "learned-field beam with simulator-labelled short-horizon collision probability gate",
            "distance_field_clearance_beam_rgb_projection": "clearance-penalized learned field with beam and RGB guard",
            "distance_field_beam_guarded_mpc": "baseline waypoint controller with RGB-triggered local MPC safety override",
            "distance_field_mpc_shift_fallback": "learned field with coarse RGB shift detector and one A* fallback",
            "distance_field_budgeted_hybrid_mpc": "learned field with route-progress/remaining-budget gate and explicit RGB/A* hybrid fallback",
        },
        "checkpoint": str(predictor_output),
        "distance_field_checkpoint": str(field_output),
        "clearance_field_checkpoint": str(clearance_output),
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
    parser.add_argument("--mpc-horizon", type=int, default=6)
    parser.add_argument("--mpc-beam-width", type=int, default=24)
    parser.add_argument("--mpc-velocity-source", choices=("rgb", "action_fused"), default="rgb")
    parser.add_argument("--output", default="artifacts/evaluation-general-routes-v18.json")
    parser.add_argument(
        "--methods",
        default=",".join(GENERAL_DEFAULT_METHODS),
        help="comma-separated subset of methods for focused ablations",
    )
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
        mpc_horizon=args.mpc_horizon,
        mpc_beam_width=args.mpc_beam_width,
        mpc_velocity_source=args.mpc_velocity_source,
        methods=tuple(value.strip() for value in args.methods.split(",") if value.strip()),
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
