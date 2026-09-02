"""Evaluate calibrated adaptive imagination horizons under paired budgets.

The evaluator has three explicitly separated tracks:

* ``pure_learning`` uses the learned world model and never calls A*;
* ``astar_fallback`` adds the existing visible-geometry A* route proposals;
* ``solver_reference`` compares the existing fixed-horizon ordinary/robust
  solver gate with adaptive horizon + robust MPC.  These solver references do
  not claim an imagined-versus-real model gap because their local MPC is not
  the learned world-model rollout.

All tracks reset the same case for every method and reuse the same candidate
action bank.  Calibration seeds are used only for transition/risk diagnostics;
final holdout seeds are never used to choose policy thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .adaptive_horizon import (
    DEFAULT_HORIZONS,
    AdaptiveHorizonPolicy,
    HorizonDecision,
    validate_horizon_candidates,
)
from .env import PocketWorldEnv
from .evaluate_general_ood import _shifted_cases_with_exclusions
from .general_routes import GeneralRouteCase, sample_general_route_cases
from .model import PocketWorldModel
from .planner import (
    estimate_agent_velocity,
    extract_agent_position,
    predictive_shift_score,
    random_shooting,
)
from .route_field import (
    RouteFieldPolicy,
    adaptive_mpc_decision,
    field_waypoints,
    local_mpc_action,
)


METHOD_NAMES = (
    "fixed_horizon_8",
    "fixed_horizon_16",
    "fixed_horizon_24",
    "fixed_horizon_32",
    "existing_adaptive_solver_gate",
    "adaptive_horizon",
    "adaptive_horizon_robust_mpc",
)
WORLD_MODEL_METHODS = frozenset(
    {"fixed_horizon_8", "fixed_horizon_16", "fixed_horizon_24", "fixed_horizon_32", "adaptive_horizon"}
)
SOLVER_REFERENCE_METHODS = frozenset(
    {"existing_adaptive_solver_gate", "adaptive_horizon_robust_mpc"}
)
DEFAULT_CONDITIONS = (
    {"name": "id", "map_shift": "nominal", "speed_scale": 1.0},
    {"name": "ood_map_minus2", "map_shift": "walls_x_minus2", "speed_scale": 1.0},
    {"name": "ood_speed_fast", "map_shift": "nominal", "speed_scale": 1.25},
    {"name": "ood_joint", "map_shift": "walls_x_minus2", "speed_scale": 1.25},
)


def _strict_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def validate_seed_splits(
    train_seeds: tuple[int, ...],
    calibration_seeds: tuple[int, ...],
    final_seeds: tuple[int, ...],
) -> None:
    """Reject any overlap between training, calibration, and final holdout."""
    groups = {
        "train": set(train_seeds),
        "calibration": set(calibration_seeds),
        "final_holdout": set(final_seeds),
    }
    for left_name, left in groups.items():
        for right_name, right in groups.items():
            if left_name < right_name and left.intersection(right):
                raise ValueError(f"seed splits overlap: {left_name} and {right_name}")


def _case_ids(cases_by_seed: dict[int, tuple[GeneralRouteCase, ...]]) -> list[str]:
    return [case.map_id for seed in sorted(cases_by_seed) for case in cases_by_seed[seed]]


def _case_hash(cases_by_seed: dict[int, tuple[GeneralRouteCase, ...]]) -> str:
    joined = "\n".join(_case_ids(cases_by_seed)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def _load_world_model(checkpoint: str | Path) -> PocketWorldModel:
    payload = torch.load(checkpoint, map_location="cpu")
    model = PocketWorldModel()
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model


def _candidate_bank(
    seed: int,
    episode_index: int,
    replan_step: int,
    start: tuple[float, float],
    goal: tuple[float, float],
    candidates: int,
    max_horizon: int,
    guided_fraction: float = 0.35,
) -> np.ndarray:
    """Generate a deterministic bank shared by every method on one case."""
    rng = np.random.default_rng(np.random.SeedSequence((seed, episode_index, replan_step, 991)))
    actions = rng.integers(0, 4, size=(candidates, max_horizon), dtype=np.int64)
    delta = np.asarray(goal, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    preferred = (
        3 if abs(delta[0]) >= abs(delta[1]) and delta[0] >= 0
        else 2 if abs(delta[0]) >= abs(delta[1])
        else 1 if delta[1] >= 0
        else 0
    )
    guided = rng.random((candidates, max_horizon)) < guided_fraction
    actions[guided] = preferred
    return actions


def _initial_state_inputs(
    model: PocketWorldModel,
    observation: np.ndarray,
    history: list[np.ndarray],
    candidates: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    frame = torch.from_numpy(observation[None]).float() / 255.0
    latent = model.encode(frame)
    state = model.state_from_latent(latent)
    position = extract_agent_position(observation).astype(np.float32)
    initial_position = torch.as_tensor(position / 64.0, dtype=frame.dtype).expand(candidates, -1)
    initial_velocity: torch.Tensor | None = None
    if len(history) >= 2:
        velocity = estimate_agent_velocity(history, max_speed=2.3)
        initial_velocity = torch.as_tensor(velocity / 3.0, dtype=frame.dtype).expand(candidates, -1)
        state = torch.cat((state[..., :2], initial_velocity[:1]), dim=-1)
    else:
        state = torch.cat((initial_position[:1], state[..., 2:]), dim=-1)
    return frame, latent, state.expand(candidates, -1), initial_velocity


@torch.no_grad()
def _horizon_diagnostics(
    model: PocketWorldModel,
    observation: np.ndarray,
    history: list[np.ndarray],
    action_bank: np.ndarray,
    horizons: tuple[int, ...],
    alignment_error: float,
    ood_score: float,
    recent_risk: float,
    uncertainty_budget_px: float,
    uncertainty_samples: int,
) -> tuple[dict[int, float], dict[int, float], dict[str, float]]:
    """Estimate horizon curves from model uncertainty and learned collision risk."""
    candidates = int(action_bank.shape[0])
    max_horizon = max(horizons)
    frame, latent_one, state, initial_velocity = _initial_state_inputs(
        model, observation, history, candidates
    )
    starts = frame.expand(candidates, -1, -1, -1)
    actions = torch.as_tensor(action_bank[:, :max_horizon], dtype=torch.long)
    probabilities = model.imagine_collision_probabilities(
        starts,
        actions,
        initial_position=torch.as_tensor(
            extract_agent_position(observation) / 64.0, dtype=frame.dtype
        ).expand(candidates, -1),
        initial_velocity=initial_velocity,
        probabilistic_uncertainty=True,
        uncertainty_samples=uncertainty_samples,
    ).cpu().numpy()
    collision_curve: dict[int, float] = {}
    uncertainty_curve: dict[int, float] = {}
    state = state.to(frame)
    latent = latent_one.expand(candidates, -1)
    accumulated_square = torch.zeros_like(state)
    uncertainty_by_step: dict[int, float] = {}
    for index in range(max_horizon):
        action = actions[:, index]
        next_state = model.state_transition(state, action)
        _, step_std = model.transition_state_stats(
            latent, state, action, next_state=next_state
        )
        accumulated_square = accumulated_square + step_std.square()
        position_std_px = torch.linalg.vector_norm(
            accumulated_square[:, :2].sqrt(), dim=-1
        ).cpu().numpy() * 64.0
        uncertainty_by_step[index + 1] = float(np.quantile(position_std_px, 0.90))
        state = next_state
    for horizon in horizons:
        cumulative_collision = np.maximum.accumulate(probabilities[:, :horizon], axis=1)[:, -1]
        collision_curve[horizon] = float(np.clip(np.quantile(cumulative_collision, 0.10), 0.0, 1.0))
        uncertainty_curve[horizon] = float(
            np.clip(uncertainty_by_step[horizon] / max(1e-6, uncertainty_budget_px), 0.0, 1.0)
        )
    diagnostics = {
        "uncertainty_budget_px": float(uncertainty_budget_px),
        "uncertainty_p90_px_at_max_horizon": float(uncertainty_by_step[max_horizon]),
        "alignment_error_px": float(alignment_error),
        "ood_score": float(ood_score),
        "recent_risk": float(recent_risk),
    }
    return uncertainty_curve, collision_curve, diagnostics


def _finite_shift_score(score: float, ood_budget: float) -> float:
    if not np.isfinite(score):
        return float(ood_budget)
    return float(np.clip(score, 0.0, 4.0 * ood_budget))


def _decision_fixed(horizon: int, previous_horizon: int | None) -> HorizonDecision:
    return HorizonDecision(
        horizon=int(horizon),
        uncertainty_score=0.0,
        collision_risk=0.0,
        alignment_error=0.0,
        reason="fixed_horizon_baseline",
        previous_horizon=previous_horizon,
        switched=previous_horizon is not None and previous_horizon != horizon,
        candidate_risks={int(horizon): 0.0},
    )


def _decision_log(decision: HorizonDecision, step: int, method: str) -> dict[str, Any]:
    payload = decision.as_dict()
    payload["step"] = int(step)
    payload["method"] = method
    return payload


def _safe_distance(info: dict[str, Any]) -> float:
    value = float(info.get("distance_to_goal", 0.0))
    return value if np.isfinite(value) else 64.0


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "imagined_success",
        "real_success",
        "imagination_real_success_gap",
        "collision_count_per_episode",
        "final_distance_px",
        "route_completion",
        "replanning_count",
        "planning_latency_ms",
        "model_queries",
        "horizon_switches",
        "alignment_error_px",
        "max_alignment_error_px",
        "astar_fallback_calls",
    )
    summary: dict[str, Any] = {"episodes": len(rows)}
    for name in metric_names:
        values = [float(row[name]) for row in rows if row.get(name) is not None and np.isfinite(row[name])]
        summary[name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "samples": len(values),
        }
    horizons = Counter()
    reasons = Counter()
    for row in rows:
        for value in row.get("selected_horizons", []):
            horizons[str(int(value))] += 1
        for decision in row.get("horizon_decisions", []):
            reasons[str(decision["reason"])] += 1
    summary["selected_horizon_distribution"] = dict(sorted(horizons.items(), key=lambda item: int(item[0])))
    summary["horizon_decision_reasons"] = dict(sorted(reasons.items()))
    return summary


def _summarize_by_seed(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seed in sorted({int(row["seed"]) for row in rows}):
        result[str(seed)] = _summarize_rows([row for row in rows if int(row["seed"]) == seed])
    return result


def _report_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "summary": _summarize_rows(rows),
        "by_seed": _summarize_by_seed(rows),
        "rows": rows,
    }


def _run_world_model_episode(
    model: PocketWorldModel,
    case: GeneralRouteCase,
    seed: int,
    episode_index: int,
    method: str,
    speed_scale: float,
    max_steps: int,
    candidates: int,
    horizons: tuple[int, ...],
    commit_steps: int,
    fallback: bool,
    horizon_policy: AdaptiveHorizonPolicy,
    uncertainty_budget_px: float,
    uncertainty_samples: int,
) -> dict[str, Any]:
    env = PocketWorldEnv(
        walls=case.walls,
        agent_start=case.start,
        goal=case.goal,
        agent_speed_scale=speed_scale,
    )
    observation, info = env.reset()
    history = [observation]
    action_history: list[int] = []
    expected_position: np.ndarray | None = None
    alignment_errors: list[float] = []
    shift_scores: list[float] = []
    selected_horizons: list[int] = []
    horizon_decisions: list[dict[str, Any]] = []
    previous_horizon: int | None = None
    planned_actions = np.zeros(0, dtype=np.int64)
    planned_positions = np.zeros((0, 2), dtype=np.float32)
    plan_offset = 0
    imagined_success = None
    collisions = 0
    planning_latency_ms = 0.0
    model_queries = 0
    planning_calls = 0
    astar_fallback_calls = 0
    last_shift_score = 0.0
    recent_risk = 0.0
    for step in range(max_steps):
        position = extract_agent_position(observation).astype(np.float32)
        if expected_position is not None and np.isfinite(position).all():
            alignment_errors.append(float(np.linalg.norm(position - expected_position)))
        if len(planned_actions) == 0 or plan_offset >= len(planned_actions) or step % max(1, commit_steps) == 0:
            action_bank = _candidate_bank(
                seed,
                episode_index,
                step,
                tuple(map(float, position)),
                tuple(map(float, case.goal)),
                candidates,
                max(horizons),
            )
            if method == "adaptive_horizon":
                uncertainty_curve, collision_curve, _ = _horizon_diagnostics(
                    model,
                    observation,
                    history,
                    action_bank,
                    horizons,
                    alignment_errors[-1] if alignment_errors else 0.0,
                    last_shift_score,
                    recent_risk,
                    uncertainty_budget_px,
                    uncertainty_samples,
                )
                decision = horizon_policy.select_horizon(
                    uncertainty_curve,
                    collision_curve,
                    alignment_errors[-1] if alignment_errors else 0.0,
                    ood_score=last_shift_score,
                    recent_risk=recent_risk,
                    previous_horizon=previous_horizon,
                )
            else:
                horizon = int(method.rsplit("_", 1)[-1])
                decision = _decision_fixed(horizon, previous_horizon)
            selected_horizon = decision.horizon
            selected_horizons.append(selected_horizon)
            horizon_decisions.append(_decision_log(decision, step, method))
            previous_horizon = selected_horizon
            planning_seed = int(seed * 100_000 + episode_index * 100 + step)
            torch.manual_seed(planning_seed)
            planning_start = torch.get_default_dtype()  # keep the branch explicit for audit logs
            del planning_start
            import time

            started = time.perf_counter()
            result = random_shooting(
                model,
                observation,
                tuple(map(float, case.goal)),
                horizon=selected_horizon,
                candidates=candidates,
                collision_aware=True,
                learned_collision=True,
                probabilistic_uncertainty=True,
                uncertainty_samples=uncertainty_samples,
                collision_risk_budget=horizon_policy.risk_budget,
                observation_history=history,
                use_learned_velocity=True,
                candidate_actions=action_bank,
                wall_aware_route=fallback,
                hybrid_collision=fallback,
            )
            planning_latency_ms += (time.perf_counter() - started) * 1000.0
            planning_calls += 1
            model_queries += candidates * selected_horizon
            astar_fallback_calls += int(fallback)
            planned_actions = np.asarray(result.actions, dtype=np.int64)
            planned_positions = np.asarray(result.imagined_positions, dtype=np.float32)
            plan_offset = 0
            if imagined_success is None:
                imagined_success = float(result.imagined_distance <= env.goal_radius)
            if len(planned_actions) == 0:
                break
        action = int(planned_actions[plan_offset])
        if len(planned_positions) > plan_offset + 1:
            expected_position = planned_positions[plan_offset + 1].copy()
        else:
            expected_position = None
        previous_observation = observation
        observation, _, terminated, truncated, info = env.step(action)
        action_history.append(action)
        action_history = action_history[-16:]
        history.append(observation)
        history = history[-16:]
        collisions += int(info.get("collision", False))
        last_shift_score = _finite_shift_score(
            predictive_shift_score(
                model,
                previous_observation,
                action,
                observation,
                history[:-1],
                action_history[:-1],
            ),
            horizon_policy.ood_budget,
        )
        shift_scores.append(last_shift_score)
        if alignment_errors or shift_scores:
            recent_alignment = float(np.mean(alignment_errors[-4:])) / horizon_policy.alignment_budget_px if alignment_errors else 0.0
            recent_shift = float(np.mean(shift_scores[-4:])) / horizon_policy.ood_budget if shift_scores else 0.0
            recent_risk = float(np.clip(max(recent_alignment, recent_shift), 0.0, 1.0))
        plan_offset += 1
        if terminated or truncated:
            break
    final_distance = _safe_distance(info)
    real_success = float(final_distance <= env.goal_radius)
    imagined_value = float(imagined_success) if imagined_success is not None else 0.0
    return {
        "seed": int(seed),
        "map_id": case.map_id,
        "family": case.family,
        "method": method,
        "mode": "astar_fallback" if fallback else "pure_learning",
        "speed_scale": float(speed_scale),
        "imagined_success": imagined_value,
        "real_success": real_success,
        "imagination_real_success_gap": imagined_value - real_success,
        "collision_count_per_episode": float(collisions),
        "final_distance_px": final_distance,
        "route_completion": real_success,
        "replanning_count": float(planning_calls),
        "planning_latency_ms": float(planning_latency_ms),
        "model_queries": float(model_queries),
        "horizon_switches": float(sum(int(item["switched"]) for item in horizon_decisions)),
        "alignment_error_px": float(np.mean(alignment_errors)) if alignment_errors else 0.0,
        "max_alignment_error_px": float(max(alignment_errors)) if alignment_errors else 0.0,
        "astar_fallback_calls": float(astar_fallback_calls),
        "selected_horizons": [int(value) for value in selected_horizons],
        "horizon_decisions": horizon_decisions,
    }


def _run_solver_episode(
    world_model: PocketWorldModel,
    field_policy: RouteFieldPolicy,
    case: GeneralRouteCase,
    seed: int,
    episode_index: int,
    method: str,
    speed_scale: float,
    max_steps: int,
    candidates: int,
    horizons: tuple[int, ...],
    horizon_policy: AdaptiveHorizonPolicy,
    uncertainty_budget_px: float,
    uncertainty_samples: int,
    solver_horizon: int,
    risk_threshold: float,
    risk_exit_threshold: float,
) -> dict[str, Any]:
    """Run the solver-only references on the same paired initial case."""
    env = PocketWorldEnv(
        walls=case.walls,
        agent_start=case.start,
        goal=case.goal,
        agent_speed_scale=speed_scale,
    )
    observation, _ = env.reset()
    history = [observation]
    action_history: list[int] = []
    waypoints = field_waypoints(observation, case.goal, field_policy.predict_field(observation, case.goal), rgb_guard=True, beam_width=4)
    waypoint_index = 0
    previous_horizon: int | None = None
    selected_horizons: list[int] = []
    decisions: list[dict[str, Any]] = []
    robust_active = False
    switches = 0
    robust_calls = 0
    risks: list[float] = []
    collisions = 0
    planning_latency_ms = 0.0
    planning_calls = 0
    alignment_errors: list[float] = []
    expected_position: np.ndarray | None = None
    last_shift_score = 0.0
    recent_risk = 0.0
    import time

    for step in range(max_steps):
        position = extract_agent_position(observation).astype(np.float32)
        if expected_position is not None and np.isfinite(position).all():
            alignment_errors.append(float(np.linalg.norm(position - expected_position)))
        target = waypoints[min(waypoint_index, len(waypoints) - 1)]
        if np.linalg.norm(position - np.asarray(target, dtype=np.float32)) <= 5.0:
            waypoint_index = min(waypoint_index + 1, len(waypoints) - 1)
            target = waypoints[waypoint_index]
        action_bank = _candidate_bank(seed, episode_index, step, tuple(position), case.goal, candidates, max(horizons))
        if method == "adaptive_horizon_robust_mpc":
            uncertainty_curve, collision_curve, _ = _horizon_diagnostics(
                world_model,
                observation,
                history,
                action_bank,
                horizons,
                alignment_errors[-1] if alignment_errors else 0.0,
                last_shift_score,
                recent_risk,
                uncertainty_budget_px,
                uncertainty_samples,
            )
            decision = horizon_policy.select_horizon(
                uncertainty_curve,
                collision_curve,
                alignment_errors[-1] if alignment_errors else 0.0,
                ood_score=last_shift_score,
                recent_risk=recent_risk,
                previous_horizon=previous_horizon,
            )
            selected_horizon = decision.horizon
            started = time.perf_counter()
            action = local_mpc_action(
                observation,
                target,
                history,
                horizon=selected_horizon,
                beam_width=max(4, candidates // max(1, selected_horizon)),
                robust=True,
                action_history=action_history,
            )
            planning_latency_ms += (time.perf_counter() - started) * 1000.0
        else:
            decision = _decision_fixed(solver_horizon, previous_horizon)
            selected_horizon = solver_horizon
            baseline_action = local_mpc_action(
                observation,
                target,
                history,
                horizon=solver_horizon,
                beam_width=max(4, candidates // max(1, solver_horizon)),
                action_history=action_history,
            )
            started = time.perf_counter()
            action, use_robust, risk = adaptive_mpc_decision(
                observation,
                target,
                baseline_action,
                history,
                action_history,
                horizon=solver_horizon,
                beam_width=max(4, candidates // max(1, solver_horizon)),
                risk_threshold=risk_threshold,
                risk_exit_threshold=risk_exit_threshold,
                robust_active=robust_active,
            )
            planning_latency_ms += (time.perf_counter() - started) * 1000.0
            robust_calls += int(use_robust)
            risks.append(float(risk))
            if bool(use_robust) != robust_active:
                switches += 1
            robust_active = bool(use_robust)
        selected_horizons.append(selected_horizon)
        decisions.append(_decision_log(decision, step, method))
        previous_horizon = selected_horizon
        expected_position = None
        observation_before = observation
        observation, _, terminated, truncated, info = env.step(int(action))
        action_history.append(int(action))
        action_history = action_history[-16:]
        history.append(observation)
        history = history[-16:]
        collisions += int(info.get("collision", False))
        last_shift_score = _finite_shift_score(
            predictive_shift_score(
                world_model,
                observation_before,
                int(action),
                observation,
                history[:-1],
                action_history[:-1],
            ),
            horizon_policy.ood_budget,
        )
        recent_risk = float(np.clip(last_shift_score / horizon_policy.ood_budget, 0.0, 1.0))
        planning_calls += 1
        if terminated or truncated:
            break
    final_distance = _safe_distance(info)
    real_success = float(final_distance <= env.goal_radius)
    return {
        "seed": int(seed),
        "map_id": case.map_id,
        "family": case.family,
        "method": method,
        "mode": "solver_reference",
        "speed_scale": float(speed_scale),
        "imagined_success": None,
        "real_success": real_success,
        "imagination_real_success_gap": None,
        "collision_count_per_episode": float(collisions),
        "final_distance_px": final_distance,
        "route_completion": real_success,
        "replanning_count": float(planning_calls),
        "planning_latency_ms": float(planning_latency_ms),
        "model_queries": 0.0,
        "horizon_switches": float(switches),
        "alignment_error_px": float(np.mean(alignment_errors)) if alignment_errors else 0.0,
        "max_alignment_error_px": float(max(alignment_errors)) if alignment_errors else 0.0,
        "astar_fallback_calls": 0.0,
        "selected_horizons": [int(value) for value in selected_horizons],
        "horizon_decisions": decisions,
        "robust_mpc_calls": int(robust_calls),
        "risk_mean": float(np.mean(risks)) if risks else 0.0,
        "risk_max": float(max(risks)) if risks else 0.0,
    }


@torch.no_grad()
def _calibration_metrics(
    model: PocketWorldModel,
    cases_by_seed: dict[int, tuple[GeneralRouteCase, ...]],
    episodes_per_seed: int,
    horizon: int,
    uncertainty_samples: int,
) -> dict[str, Any]:
    """Measure calibrated transition coverage and collision Brier on calibration cases."""
    residuals: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    collision_probs: list[float] = []
    collision_labels: list[float] = []
    for seed in sorted(cases_by_seed):
        for episode_index, case in enumerate(cases_by_seed[seed][:episodes_per_seed]):
            rng = np.random.default_rng(np.random.SeedSequence((seed, episode_index, 7331)))
            actions = rng.integers(0, 4, size=horizon, dtype=np.int64)
            env = PocketWorldEnv(walls=case.walls, agent_start=case.start, goal=case.goal)
            observation, _ = env.reset()
            observations = [observation]
            infos: list[dict[str, Any]] = []
            for action in actions:
                next_observation, _, terminated, truncated, info = env.step(int(action))
                observations.append(next_observation)
                infos.append(info)
                if terminated or truncated:
                    break
            actual_actions = actions[: len(infos)]
            if not len(actual_actions):
                continue
            model_start = torch.from_numpy(observations[0][None]).float() / 255.0
            action_tensor = torch.from_numpy(actual_actions[None])
            probabilities = model.imagine_collision_probabilities(
                model_start.expand(1, -1, -1, -1),
                action_tensor,
                probabilistic_uncertainty=True,
                uncertainty_samples=uncertainty_samples,
            )[0].cpu().numpy()
            collision_probs.extend(probabilities.tolist())
            collision_labels.extend([float(info.get("collision", False)) for info in infos])
            for index, action in enumerate(actual_actions):
                previous = observations[index]
                current = observations[index + 1]
                previous_position = extract_agent_position(previous)
                current_position = extract_agent_position(current)
                if not np.isfinite(previous_position).all() or not np.isfinite(current_position).all():
                    continue
                frame = torch.from_numpy(previous[None]).float() / 255.0
                latent = model.encode(frame)
                state = model.state_from_latent(latent)
                velocity = estimate_agent_velocity(observations[: index + 1], max_speed=2.3)
                state = torch.cat(
                    (
                        torch.as_tensor(previous_position / 64.0, dtype=state.dtype)[None],
                        torch.as_tensor(velocity / 3.0, dtype=state.dtype)[None],
                    ),
                    dim=-1,
                )
                mean, std = model.transition_state_stats(
                    latent, state, torch.as_tensor([int(action)])
                )
                observed_velocity = (current_position - previous_position) / 3.0
                target = torch.as_tensor(
                    np.concatenate((current_position / 64.0, observed_velocity)),
                    dtype=state.dtype,
                )[None]
                residuals.append((target - mean).abs().cpu().numpy()[0])
                scales.append(std.cpu().numpy()[0])
    residual = np.asarray(residuals, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64).clip(1e-6, None)
    z90 = 1.6449
    covered = residual <= z90 * scale if len(residual) else np.zeros((0, 4), dtype=bool)
    probabilities = np.asarray(collision_probs, dtype=np.float64)
    labels = np.asarray(collision_labels, dtype=np.float64)
    return {
        "seeds": sorted(int(seed) for seed in cases_by_seed),
        "samples": int(len(residual)),
        "uncertainty_coverage_90_position": float(covered[:, :2].mean()) if len(covered) else None,
        "uncertainty_coverage_90_velocity": float(covered[:, 2:].mean()) if len(covered) else None,
        "state_gaussian_nll": float(
            0.5 * np.mean((residual / scale) ** 2 + 2.0 * np.log(scale))
        ) if len(residual) else None,
        "collision_brier": float(np.mean((probabilities - labels) ** 2)) if len(probabilities) else None,
        "collision_samples": int(len(probabilities)),
    }


@torch.no_grad()
def _fixed_horizon_error_curve(
    model: PocketWorldModel,
    cases_by_seed: dict[int, tuple[GeneralRouteCase, ...]],
    speed_scale: float,
    horizons: tuple[int, ...],
) -> dict[str, dict[str, float | int | None]]:
    """Measure open-loop position error at each fixed horizon.

    This is a diagnostic curve, not a planning policy.  It uses one paired,
    deterministic action trace per case and compares model rollouts with the
    simulator only after the trace has been generated.  The simulator state is
    therefore evaluation-only and never enters ``AdaptiveHorizonPolicy``.
    """
    max_horizon = max(horizons)
    errors: dict[int, list[float]] = {horizon: [] for horizon in horizons}
    for seed in sorted(cases_by_seed):
        for episode_index, case in enumerate(cases_by_seed[seed]):
            action_bank = _candidate_bank(
                seed,
                episode_index,
                0,
                case.start,
                case.goal,
                1,
                max_horizon,
            )
            actions = action_bank[0]
            env = PocketWorldEnv(
                walls=case.walls,
                agent_start=case.start,
                goal=case.goal,
                agent_speed_scale=speed_scale,
            )
            observation, _ = env.reset()
            actual_positions = []
            for action in actions:
                observation, _, _, _, info = env.step(int(action))
                actual_positions.append(np.asarray(info["position"], dtype=np.float32))
            start = torch.from_numpy(observation[None]).float() / 255.0
            # The final observation is not the initial frame.  Recreate the
            # initial frame so the learned rollout starts from the same case.
            initial_env = PocketWorldEnv(
                walls=case.walls,
                agent_start=case.start,
                goal=case.goal,
                agent_speed_scale=speed_scale,
            )
            initial_observation, _ = initial_env.reset()
            start = torch.from_numpy(initial_observation[None]).float() / 255.0
            imagined = model.imagine_positions(
                start,
                torch.from_numpy(actions[None]).long(),
                collision_response=False,
                initial_position=torch.as_tensor(
                    np.asarray(case.start, dtype=np.float32) / 64.0
                )[None],
            )[0].cpu().numpy() * 64.0
            actual = np.asarray(actual_positions, dtype=np.float32)
            for horizon in horizons:
                errors[horizon].append(
                    float(np.linalg.norm(imagined[horizon - 1] - actual[horizon - 1]))
                )
    return {
        str(horizon): {
            "mean_position_error_px": float(np.mean(values)) if values else None,
            "std_position_error_px": float(np.std(values)) if values else None,
            "samples": int(len(values)),
        }
        for horizon, values in errors.items()
    }


def _build_cases(
    seeds: tuple[int, ...], episodes: int, split: str = "holdout"
) -> dict[int, tuple[GeneralRouteCase, ...]]:
    return {
        int(seed): sample_general_route_cases(int(seed), episodes, split=split)
        for seed in seeds
    }


def _calibration_rollout_tensors(
    cases_by_seed: dict[int, tuple[GeneralRouteCase, ...]],
    episodes_per_seed: int,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create held-out RGB/state rollouts used only for scale calibration.

    The rollout labels stay inside the calibration split.  In particular, this
    helper is called before any final-holdout episode is evaluated and its
    output is never mixed with final reports when selecting a horizon.
    """
    observations: list[np.ndarray] = []
    actions_batch: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    for seed in sorted(cases_by_seed):
        for episode_index, case in enumerate(cases_by_seed[seed][:episodes_per_seed]):
            rng = np.random.default_rng(np.random.SeedSequence((seed, episode_index, 4401)))
            actions = rng.integers(0, 4, size=horizon, dtype=np.int64)
            env = PocketWorldEnv(walls=case.walls, agent_start=case.start, goal=case.goal)
            observation, info = env.reset()
            episode_observations = [observation]
            episode_positions = [np.asarray(info["position"], dtype=np.float32)]
            episode_velocities = [np.asarray(info["velocity"], dtype=np.float32)]
            for action in actions:
                observation, _, _, _, info = env.step(int(action))
                episode_observations.append(observation)
                episode_positions.append(np.asarray(info["position"], dtype=np.float32))
                episode_velocities.append(np.asarray(info["velocity"], dtype=np.float32))
            observations.append(np.stack(episode_observations))
            actions_batch.append(actions)
            positions.append(np.stack(episode_positions) / 64.0)
            velocities.append(np.clip(np.stack(episode_velocities) / 3.0, -1.0, 1.0))
    return (
        torch.from_numpy(np.stack(observations)).float() / 255.0,
        torch.from_numpy(np.stack(actions_batch)).long(),
        torch.from_numpy(np.stack(positions)).float(),
        torch.from_numpy(np.stack(velocities)).float(),
    )


def _condition_cases(
    base_cases: dict[int, tuple[GeneralRouteCase, ...]], map_shift: str
) -> tuple[dict[int, tuple[GeneralRouteCase, ...]], dict[str, list[str]]]:
    if map_shift == "nominal":
        return base_cases, {}
    result: dict[int, tuple[GeneralRouteCase, ...]] = {}
    excluded: dict[str, list[str]] = {}
    for seed, cases in base_cases.items():
        shifted, dropped = _shifted_cases_with_exclusions(cases, map_shift)
        result[seed] = shifted
        if dropped:
            excluded[str(seed)] = list(dropped)
    return result, excluded


def run_adaptive_horizon_evaluation(
    world_model_checkpoint: str | Path,
    route_field_checkpoint: str | Path | None = None,
    train_seeds: tuple[int, ...] = (101, 103, 107),
    calibration_seeds: tuple[int, ...] = (53, 67),
    final_seeds: tuple[int, ...] = (11, 23, 41),
    episodes_per_seed: int = 20,
    calibration_episodes: int = 12,
    max_steps: int = 48,
    candidates: int = 256,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    commit_steps: int = 4,
    conditions: tuple[dict[str, Any], ...] = DEFAULT_CONDITIONS,
    uncertainty_budget_px: float = 6.0,
    uncertainty_samples: int = 8,
    risk_budget: float = 0.45,
    entry_threshold: float = 0.55,
    exit_threshold: float = 0.35,
    solver_horizon: int = 16,
    solver_risk_threshold: float = 0.45,
    solver_risk_exit_threshold: float = 0.30,
) -> dict[str, Any]:
    horizons = validate_horizon_candidates(horizons)
    if episodes_per_seed < 1 or calibration_episodes < 1 or max_steps < 1 or candidates < 4:
        raise ValueError("episodes, max_steps, and candidates must be positive")
    if commit_steps < 1:
        raise ValueError("commit_steps must be positive")
    validate_seed_splits(train_seeds, calibration_seeds, final_seeds)
    model = _load_world_model(world_model_checkpoint)
    field_policy = RouteFieldPolicy.load(route_field_checkpoint) if route_field_checkpoint else None
    if field_policy is None:
        raise ValueError("route_field_checkpoint is required for solver references")
    calibration_cases = _build_cases(calibration_seeds, calibration_episodes)
    final_base_cases = _build_cases(final_seeds, episodes_per_seed)
    horizon_policy = AdaptiveHorizonPolicy(
        horizons=horizons,
        risk_budget=risk_budget,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )
    calibration_horizon = max(1, min(max(horizons), 8))
    calibration_rollouts = _calibration_rollout_tensors(
        calibration_cases,
        calibration_episodes,
        calibration_horizon,
    )
    fitted_calibration = model.fit_uncertainty_calibration(
        *calibration_rollouts,
        coverage=0.90,
        observed_velocity_blend=0.50,
    )
    calibration = _calibration_metrics(
        model,
        calibration_cases,
        calibration_episodes,
        calibration_horizon,
        uncertainty_samples,
    )
    calibration["fit"] = fitted_calibration
    by_condition: dict[str, Any] = {}
    for condition in conditions:
        name = str(condition["name"])
        map_shift = str(condition.get("map_shift", "nominal"))
        speed_scale = float(condition.get("speed_scale", 1.0))
        cases, excluded = _condition_cases(final_base_cases, map_shift)
        fixed_error_curve = _fixed_horizon_error_curve(
            model,
            cases,
            speed_scale,
            horizons,
        )
        pure_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in WORLD_MODEL_METHODS}
        fallback_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in WORLD_MODEL_METHODS}
        solver_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in SOLVER_REFERENCE_METHODS}
        for seed in sorted(cases):
            for episode_index, case in enumerate(cases[seed]):
                for method in ("fixed_horizon_8", "fixed_horizon_16", "fixed_horizon_24", "fixed_horizon_32", "adaptive_horizon"):
                    pure_rows[method].append(
                        _run_world_model_episode(
                            model, case, seed, episode_index, method, speed_scale, max_steps,
                            candidates, horizons, commit_steps, False, horizon_policy,
                            uncertainty_budget_px, uncertainty_samples,
                        )
                    )
                    fallback_rows[method].append(
                        _run_world_model_episode(
                            model, case, seed, episode_index, method, speed_scale, max_steps,
                            candidates, horizons, commit_steps, True, horizon_policy,
                            uncertainty_budget_px, uncertainty_samples,
                        )
                    )
                for method in ("existing_adaptive_solver_gate", "adaptive_horizon_robust_mpc"):
                    solver_rows[method].append(
                        _run_solver_episode(
                            model, field_policy, case, seed, episode_index, method, speed_scale,
                            max_steps, candidates, horizons, horizon_policy, uncertainty_budget_px,
                            uncertainty_samples, solver_horizon, solver_risk_threshold,
                            solver_risk_exit_threshold,
                        )
                    )
        by_condition[name] = {
            "protocol": {
                "map_shift": map_shift,
                "speed_scale": speed_scale,
                "paired_episode_count": sum(len(items) for items in cases.values()),
                "excluded_unreachable_by_seed": excluded,
                "case_hash": _case_hash(cases),
                "case_ids": _case_ids(cases),
                "fixed_horizon_error_curve": fixed_error_curve,
            },
            "pure_learning": {method: _report_method(rows) for method, rows in pure_rows.items()},
            "astar_fallback": {method: _report_method(rows) for method, rows in fallback_rows.items()},
            "solver_reference": {method: _report_method(rows) for method, rows in solver_rows.items()},
        }
    return {
        "protocol": {
            "world_model_checkpoint": str(world_model_checkpoint),
            "route_field_checkpoint": str(route_field_checkpoint),
            "train_seeds": list(train_seeds),
            "calibration_seeds": list(calibration_seeds),
            "final_holdout_seeds": list(final_seeds),
            "episodes_per_seed": int(episodes_per_seed),
            "calibration_episodes_per_seed": int(calibration_episodes),
            "max_steps": int(max_steps),
            "candidate_count": int(candidates),
            "horizons": list(horizons),
            "commit_steps": int(commit_steps),
            "conditions": [dict(condition) for condition in conditions],
            "uncertainty_budget_px": float(uncertainty_budget_px),
            "uncertainty_samples": int(uncertainty_samples),
            "risk_budget": float(risk_budget),
            "entry_threshold": float(entry_threshold),
            "exit_threshold": float(exit_threshold),
            "solver_horizon": int(solver_horizon),
            "solver_risk_threshold": float(solver_risk_threshold),
            "solver_risk_exit_threshold": float(solver_risk_exit_threshold),
            "method_semantics": {
                "fixed_horizon_*": "pure learned world-model random shooting with fixed imagination horizon",
                "existing_adaptive_solver_gate": "fixed solver horizon; ordinary/robust MPC switch only",
                "adaptive_horizon": "calibrated horizon selection; ordinary learned world-model rollout",
            "adaptive_horizon_robust_mpc": "same horizon policy with robust local MPC, independent ablation",
            "astar_fallback": "explicit RGB footprint geometry plus A* route proposals; not pure learning",
        },
        "imagined_success_definition": (
            "success of the first selected plan's predicted endpoint; "
            "not closed-loop episode success"
        ),
        "imagined_real_gap_comparable_at_policy_level": False,
        "thresholds_selected_on_final_holdout": False,
            "calibration_and_final_disjoint": True,
            "student_pure_learning_calls_astar": False,
            "student_pure_learning_reads_true_collision_labels": False,
            "calibration_fit_uses_final_holdout": False,
        },
        "calibration": calibration,
        "conditions": by_condition,
    }


def load_protocol(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("protocol JSON must contain an object")
    return payload


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/adaptive-horizon-v1.json")
    parser.add_argument("--world-model", default="")
    parser.add_argument("--route-field", default="")
    parser.add_argument("--output", default="artifacts/evaluation-adaptive-horizon-v1.json")
    parser.add_argument("--episodes-per-seed", type=int, default=0)
    parser.add_argument("--calibration-episodes", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=0)
    parser.add_argument("--horizons", default="")
    parser.add_argument("--commit-steps", type=int, default=0)
    parser.add_argument("--uncertainty-samples", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="run a one-seed, one-episode smoke protocol")
    args = parser.parse_args()
    config = load_protocol(args.protocol)
    world_model = args.world_model or str(config["world_model_checkpoint"])
    route_field = args.route_field or str(config["route_field_checkpoint"])
    train_seeds = tuple(int(value) for value in config["train_seeds"])
    calibration_seeds = tuple(int(value) for value in config["calibration_seeds"])
    final_seeds = tuple(int(value) for value in config["final_holdout_seeds"])
    conditions = tuple(dict(value) for value in config["conditions"])
    episodes = args.episodes_per_seed or int(config["episodes_per_seed"])
    calibration_episodes = args.calibration_episodes or int(config["calibration_episodes_per_seed"])
    max_steps = args.max_steps or int(config["max_steps"])
    candidates = args.candidates or int(config["candidate_count"])
    horizons = validate_horizon_candidates(
        _parse_ints(args.horizons) if args.horizons else tuple(int(value) for value in config["horizons"])
    )
    commit_steps = args.commit_steps or int(config["commit_steps"])
    uncertainty_samples = args.uncertainty_samples or int(config["uncertainty_samples"])
    if args.smoke:
        final_seeds = final_seeds[:1]
        calibration_seeds = calibration_seeds[:1]
        episodes = min(episodes, 1)
        calibration_episodes = min(calibration_episodes, 1)
        max_steps = min(max_steps, 8)
        candidates = min(candidates, 16)
        # Keep all four fixed baselines in the smoke protocol.  The episode is
        # shortened, but removing 24/32 here would make the paired comparison
        # incomplete and would leave their candidate-action banks undersized.
        conditions = (conditions[0],)
    report = run_adaptive_horizon_evaluation(
        world_model_checkpoint=world_model,
        route_field_checkpoint=route_field,
        train_seeds=train_seeds,
        calibration_seeds=calibration_seeds,
        final_seeds=final_seeds,
        episodes_per_seed=episodes,
        calibration_episodes=calibration_episodes,
        max_steps=max_steps,
        candidates=candidates,
        horizons=horizons,
        commit_steps=commit_steps,
        conditions=conditions,
        uncertainty_budget_px=float(config["uncertainty_budget_px"]),
        uncertainty_samples=uncertainty_samples,
        risk_budget=float(config["risk_budget"]),
        entry_threshold=float(config["entry_threshold"]),
        exit_threshold=float(config["exit_threshold"]),
        solver_horizon=int(config["solver_horizon"]),
        solver_risk_threshold=float(config["solver_risk_threshold"]),
        solver_risk_exit_threshold=float(config["solver_risk_exit_threshold"]),
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
