"""Fixed-budget comparisons between model-based planning algorithms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import PocketWorldEnv, Rect
from .evaluate import _summarize
from .model import PocketWorldModel
from .planner import beam_search, cem_shooting, random_shooting, receding_horizon_plan


SINGLE_BARRIER_WALLS = (Rect(29, 10, 5, 44),)
SHIFTED_BARRIER_WALLS = (Rect(34, 10, 5, 44),)
NARROW_GAP_WALLS = (Rect(29, 3, 5, 22), Rect(29, 39, 5, 22))
WIDE_GAP_WALLS = (Rect(29, 3, 5, 15), Rect(29, 46, 5, 15))
COLLISION_RISK_BUDGET = 0.15
ROUTE_COMPLETION_WEIGHT = 64.0
ROUTE_MPC_COMPLETION_WEIGHT = 10.0
DEFAULT_PLANNERS = (
    "random_shooting",
    "cem",
    "beam_search",
    "learned_collision",
    "cem_collision",
    "route_aware_hybrid",
)
SUPPORTED_PLANNERS = DEFAULT_PLANNERS + (
    "ensemble_collision",
    "conformal_collision",
    "route_completion",
    "route_completion_safe_gate",
    "route_completion_rgb_only",
    "route_completion_soft",
    "route_completion_mpc",
)
PLANNER_SEED_OFFSETS = {
    # Keep the historical route-study offsets stable even when a tournament
    # evaluates only a subset of methods. This makes parameter sweeps
    # comparable to the full seven-method report.
    "learned_collision": 0,
    "route_completion": 1,
    "route_completion_safe_gate": 2,
    "route_completion_rgb_only": 3,
    "route_completion_soft": 4,
    "route_completion_mpc": 5,
    "route_aware_hybrid": 6,
    "random_shooting": 7,
    "cem": 8,
    "beam_search": 9,
    "cem_collision": 10,
    "ensemble_collision": 11,
    "conformal_collision": 12,
}


def _nominal_queries(planner: str, horizon: int, candidates: int) -> int:
    if planner == "beam_search":
        width = max(1, candidates // (4 * horizon))
        return 4 * horizon * width
    return candidates


def _episode_cases(episodes: int, seed: int, scenario: str) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    if scenario == "open":
        return [
            (
                (float(rng.integers(7, 11)), float(rng.integers(7, 11))),
                (float(rng.integers(16, 21)), float(rng.integers(28, 33))),
            )
            for _ in range(episodes)
        ]
    if scenario in {"single_barrier", "barrier_shifted"}:
        return [
            (
                (float(rng.integers(7, 13)), float(rng.integers(25, 39))),
                (float(rng.integers(51, 57)), float(rng.integers(25, 39))),
            )
            for _ in range(episodes)
        ]
    if scenario == "barrier_narrow_gap":
        # Start and goal lie in the wall-covered bands, not in the 14px gap;
        # a successful route must visibly detour through that gap.
        bands = np.concatenate((np.arange(8, 23), np.arange(42, 57)))
        return [
            (
                (float(rng.integers(7, 13)), float(rng.choice(bands))),
                (float(rng.integers(51, 57)), float(rng.choice(bands))),
            )
            for _ in range(episodes)
        ]
    if scenario == "barrier_wide_gap":
        # Start and goal lie in the wall-covered bands, not in the 28px gap.
        bands = np.concatenate((np.arange(7, 17), np.arange(49, 59)))
        return [
            (
                (float(rng.integers(7, 13)), float(rng.choice(bands))),
                (float(rng.integers(51, 57)), float(rng.choice(bands))),
            )
            for _ in range(episodes)
        ]
    raise ValueError(f"unsupported scenario: {scenario}")


def _scenario_walls(scenario: str) -> tuple[Rect, ...]:
    if scenario == "open":
        return ()
    if scenario == "single_barrier":
        return SINGLE_BARRIER_WALLS
    if scenario == "barrier_shifted":
        return SHIFTED_BARRIER_WALLS
    if scenario == "barrier_narrow_gap":
        return NARROW_GAP_WALLS
    if scenario == "barrier_wide_gap":
        return WIDE_GAP_WALLS
    raise ValueError(f"unsupported scenario: {scenario}")


def _planner_result(
    model: PocketWorldModel,
    planner: str,
    observation: np.ndarray,
    goal: tuple[float, float],
    horizon: int,
    candidates: int,
    collision_models: dict[str, object] | None = None,
    route_models: dict[str, object] | None = None,
    soft_rgb_penalty: float = 64.0,
) -> object:
    collision_model = None if collision_models is None else collision_models.get(planner)
    route_model = None if route_models is None else route_models.get(planner)
    if planner == "random_shooting":
        return random_shooting(model, observation, goal, horizon=horizon, candidates=candidates)
    if planner == "cem":
        return cem_shooting(model, observation, goal, horizon=horizon, candidates=candidates)
    if planner == "beam_search":
        return beam_search(model, observation, goal, horizon=horizon, candidates=candidates)
    if planner == "learned_collision":
        return random_shooting(
            model,
            observation,
            goal,
            horizon=horizon,
            candidates=candidates,
            collision_aware=True,
            learned_collision=True,
            probabilistic_uncertainty=True,
            uncertainty_samples=8,
            robust_candidates=min(32, candidates),
            observation_history=[observation],
        )
    if planner == "cem_collision":
        return cem_shooting(
            model,
            observation,
            goal,
            horizon=horizon,
            candidates=candidates,
            collision_aware=True,
            learned_collision=True,
            collision_risk_budget=COLLISION_RISK_BUDGET,
            observation_history=[observation],
        )
    if planner in {"ensemble_collision", "conformal_collision"}:
        return random_shooting(
            model,
            observation,
            goal,
            horizon=horizon,
            candidates=candidates,
            collision_aware=True,
            learned_collision=True,
            collision_model=collision_model,
            collision_risk_budget=COLLISION_RISK_BUDGET,
            observation_history=[observation],
        )
    if planner == "route_completion":
        if route_model is None:
            raise ValueError("route_completion requires a route_models['route_completion'] predictor")
        return random_shooting(
            model,
            observation,
            goal,
            horizon=horizon,
            candidates=candidates,
            collision_aware=True,
            learned_collision=True,
            probabilistic_uncertainty=True,
            uncertainty_samples=8,
            robust_candidates=min(32, candidates),
            route_objective=True,
            route_execution_horizon=horizon,
            route_completion_model=route_model,
            route_completion_weight=ROUTE_COMPLETION_WEIGHT,
            collision_risk_budget=COLLISION_RISK_BUDGET,
            observation_history=[observation],
        )
    if planner == "route_completion_safe_gate":
        if route_model is None:
            raise ValueError("route_completion_safe_gate requires a route_models['route_completion_safe_gate'] predictor")
        return random_shooting(
            model,
            observation,
            goal,
            horizon=horizon,
            candidates=candidates,
            collision_aware=True,
            learned_collision=True,
            probabilistic_uncertainty=True,
            uncertainty_samples=8,
            robust_candidates=min(32, candidates),
            route_objective=True,
            route_execution_horizon=horizon,
            route_completion_model=route_model,
            route_completion_weight=ROUTE_COMPLETION_WEIGHT,
            collision_risk_budget=COLLISION_RISK_BUDGET,
            visual_safety_gate=True,
            observation_history=[observation],
        )
    if planner in {"route_completion_rgb_only", "route_completion_soft"}:
        if route_model is None:
            raise ValueError(f"{planner} requires a route_models['{planner}'] predictor")
        safety_mode = "rgb_only" if planner == "route_completion_rgb_only" else "soft"
        return random_shooting(
            model,
            observation,
            goal,
            horizon=horizon,
            candidates=candidates,
            collision_aware=True,
            learned_collision=True,
            probabilistic_uncertainty=True,
            uncertainty_samples=8,
            robust_candidates=min(32, candidates),
            route_objective=True,
            route_execution_horizon=horizon,
            route_completion_model=route_model,
            route_completion_weight=ROUTE_COMPLETION_WEIGHT,
            collision_risk_budget=COLLISION_RISK_BUDGET,
            visual_safety_mode=safety_mode,
            visual_safety_penalty=soft_rgb_penalty,
            observation_history=[observation],
        )
    if planner == "route_aware_hybrid":
        return random_shooting(
            model,
            observation,
            goal,
            horizon=horizon,
            candidates=candidates,
            collision_aware=True,
            hybrid_collision=True,
            route_objective=True,
            route_execution_horizon=min(12, horizon),
            wall_aware_route=True,
            observation_history=[observation],
        )
    raise ValueError(f"unsupported planner: {planner}")


def evaluate_planner_tournament(
    model: PocketWorldModel,
    seeds: tuple[int, ...] = (11, 23, 41),
    episodes: int = 50,
    horizon: int = 16,
    candidates: int = 256,
    scenario: str = "open",
    planners: tuple[str, ...] = DEFAULT_PLANNERS,
    collision_models: dict[str, object] | None = None,
    route_models: dict[str, object] | None = None,
    risk_model_metadata: dict[str, object] | None = None,
    include_rows: bool = False,
    agent_speed_scale: float = 1.0,
    soft_rgb_penalty: float = 64.0,
) -> dict[str, object]:
    """Compare planners on paired tasks under one rollout-query budget.

    Every planner receives the same start/goal cases for a seed. ``candidates``
    is the declared model-query budget; CEM divides it across its iterations.
    The output keeps per-seed values so improvements cannot be attributed to a
    lucky task sample.
    """
    unknown = set(planners) - set(SUPPORTED_PLANNERS)
    if unknown:
        raise ValueError(f"unsupported planners: {sorted(unknown)}")
    runs = []
    retained_rows: dict[str, dict[str, list[dict[str, float | None]]]] = {}
    for seed in seeds:
        rows = {planner: [] for planner in planners}
        for episode, (start, goal) in enumerate(_episode_cases(episodes, seed, scenario)):
            for planner in planners:
                torch.manual_seed(
                    seed * 100_000
                    + episode * 100
                    + PLANNER_SEED_OFFSETS[planner]
                )
                route_model = None if route_models is None else route_models.get(planner)
                walls = _scenario_walls(scenario)
                env = PocketWorldEnv(
                    walls=walls,
                    agent_start=start,
                    goal=goal,
                    agent_speed_scale=agent_speed_scale,
                )
                observation, info = env.reset()
                if planner in {"route_aware_hybrid", "route_completion_mpc"}:
                    if planner == "route_completion_mpc" and route_model is None:
                        raise ValueError("route_completion_mpc requires a route_models['route_completion_mpc'] predictor")
                    route_mpc = planner == "route_completion_mpc"
                    closed = receding_horizon_plan(
                        model,
                        observation,
                        goal,
                        env.step,
                        max_steps=horizon,
                        rollout_horizon=min(horizon, 16),
                        candidates=candidates,
                        collision_aware=True,
                        preserve_route=True,
                        route_tolerance=6.0,
                        learned_collision=True,
                        hybrid_collision=not route_mpc,
                        use_history_velocity=True,
                        use_learned_velocity=True,
                        probabilistic_uncertainty=True,
                        uncertainty_samples=8,
                        route_objective=True,
                        route_execution_horizon=min(12, horizon),
                        alignment_fallback_threshold=4.0,
                        wall_aware_route=not route_mpc,
                        collision_risk_budget=COLLISION_RISK_BUDGET if route_mpc else None,
                        route_completion_model=route_model if route_mpc else None,
                        route_completion_weight=ROUTE_MPC_COMPLETION_WEIGHT if route_mpc else ROUTE_COMPLETION_WEIGHT,
                    )
                    rows[planner].append(
                        {
                            # Closed-loop route control does not expose one
                            # open-loop imagined trajectory comparable to the
                            # other planners; keep those fields null instead
                            # of manufacturing an invalid imagination gap.
                            "imagined_success": None,
                            "real_success": float(closed.final_info["distance_to_goal"] <= env.goal_radius),
                            "imagined_final_distance_px": None,
                            "real_final_distance_px": float(closed.final_info["distance_to_goal"]),
                            "collision_count": float(closed.collision_count),
                            "planning_score": float(closed.first_plan_route_distance),
                            "executed_actions": float(len(closed.actions)),
                            "imagined_collision_risk": None,
                            "rgb_route_collision": None,
                            "predicted_route_completion_probability": float(closed.first_plan_route_completion_probability),
                            "planning_calls": float(max(1, closed.replans)),
                            "estimated_model_queries": float(max(1, closed.replans) * candidates),
                        }
                    )
                    continue
                result = _planner_result(
                    model,
                    planner,
                    observation,
                    goal,
                    horizon,
                    candidates,
                    collision_models=collision_models,
                    route_models=route_models,
                    soft_rgb_penalty=soft_rgb_penalty,
                )
                collision_count = 0
                for action in result.actions:
                    _, _, terminated, truncated, info = env.step(int(action))
                    collision_count += int(info.get("collision", False))
                    if terminated or truncated:
                        break
                rows[planner].append(
                    {
                        "imagined_success": float(result.imagined_distance <= env.goal_radius),
                        "real_success": float(info["distance_to_goal"] <= env.goal_radius),
                        "imagined_final_distance_px": float(result.imagined_distance),
                        "real_final_distance_px": float(info["distance_to_goal"]),
                        "collision_count": float(collision_count),
                        "planning_score": float(result.planning_score),
                        "executed_actions": float(len(result.actions)),
                        "imagined_collision_risk": float(result.imagined_collision_risk),
                        "rgb_route_collision": float(result.rgb_route_collision),
                        "predicted_route_completion_probability": float(result.predicted_route_completion_probability),
                        "planning_calls": 1.0,
                        "estimated_model_queries": float(_nominal_queries(planner, horizon, candidates)),
                    }
                )
        runs.append(
            {
                "seed": seed,
                "planners": {
                    planner: {
                        metric: (
                            None
                            if all(row[metric] is None for row in planner_rows)
                            else float(np.mean([row[metric] for row in planner_rows]))
                        )
                        for metric in planner_rows[0]
                    }
                    for planner, planner_rows in rows.items()
                },
            }
        )
        if include_rows:
            retained_rows[str(seed)] = rows
    summary = _summarize([run["planners"] for run in runs])
    report = {
        "config": {
            "seeds": list(seeds),
            "episodes": episodes,
            "horizon": horizon,
            "candidates": candidates,
            "scenario": scenario,
            "agent_speed_scale": agent_speed_scale,
            "planners": list(planners),
            "paired_tasks": True,
            "planner_seed_offset_policy": "fixed by planner name; subset tournaments preserve full-tournament random streams",
            "budget_scope": "nominal candidate budget per planning call; closed-loop totals are reported separately",
            "cem_budget_policy": "candidates split evenly across categorical CEM iterations",
            "beam_budget_policy": "beam width is floor(candidates / (4 * horizon)); one full four-action branch is retained for small budgets",
            "collision_risk_budget": COLLISION_RISK_BUDGET,
            "route_completion_weight": ROUTE_COMPLETION_WEIGHT,
            "visual_safety_gate": "route_completion_safe_gate",
            "soft_rgb_penalty": soft_rgb_penalty,
            "route_mpc_completion_weight": ROUTE_MPC_COMPLETION_WEIGHT,
            "risk_model_metadata": risk_model_metadata or {},
        },
        "runs": runs,
        "summary": summary,
    }
    if include_rows:
        report["rows"] = retained_rows
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PocketWorld planners under a fixed model-query budget")
    parser.add_argument("checkpoint", nargs="?", default="artifacts/pocketworld.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--seeds", default="11,23,41")
    parser.add_argument("--scenario", choices=("open", "single_barrier"), default="open")
    parser.add_argument("--planners", default=",".join(DEFAULT_PLANNERS))
    parser.add_argument("--output", default="artifacts/evaluation-planner-tournament.json")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = PocketWorldModel()
    model.load_state_dict(payload["model"], strict=False)
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    planners = tuple(value.strip() for value in args.planners.split(",") if value.strip())
    report = evaluate_planner_tournament(
        model,
        seeds=seeds,
        episodes=args.episodes,
        horizon=args.horizon,
        candidates=args.candidates,
        scenario=args.scenario,
        planners=planners,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
