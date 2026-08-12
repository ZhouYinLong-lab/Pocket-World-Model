"""Evaluate general-route execution under controlled map and speed shifts.

The learned route-field checkpoint is fixed.  Holdout cases are generated once
and then transformed deterministically; no shifted observation is used for
training or configuration selection.  This isolates OOD execution robustness
from route-field retraining variance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .env import PocketWorldEnv, Rect
from .evaluate_general_routes import evaluate_general_policy
from .general_routes import GeneralRouteCase, sample_general_route_cases
from .planner import _astar_path, _dilate, extract_wall_mask
from .route_field import RouteFieldPolicy, coarse_wall_signature, wall_layout_shift_score


DEFAULT_METHODS = (
    "distance_field_beam_rgb_projection",
    "distance_field_beam_mpc",
    "distance_field_mpc_shift_fallback",
    "distance_field_budgeted_hybrid_mpc",
)
DEFAULT_MAP_SHIFTS = ("nominal", "walls_x_minus2", "walls_x_plus1")


def _shift_walls(walls: tuple[Rect, ...], dx: float) -> tuple[Rect, ...]:
    return tuple(Rect(wall.x + dx, wall.y, wall.width, wall.height) for wall in walls)


def _shifted_cases(
    cases: tuple[GeneralRouteCase, ...], map_shift: str
) -> tuple[GeneralRouteCase, ...]:
    shifted, excluded = _shifted_cases_with_exclusions(cases, map_shift)
    if excluded:
        raise RuntimeError(
            "map shift made tasks unreachable: " + ", ".join(excluded)
        )
    return shifted


def _shifted_cases_with_exclusions(
    cases: tuple[GeneralRouteCase, ...], map_shift: str
) -> tuple[tuple[GeneralRouteCase, ...], tuple[str, ...]]:
    offsets = {
        "nominal": 0.0,
        "walls_x_minus2": -2.0,
        "walls_x_plus1": 1.0,
        "walls_x_plus2": 2.0,
        "walls_x_plus4": 4.0,
    }
    if map_shift not in offsets:
        raise ValueError(f"map_shift must be one of {tuple(offsets)}")
    dx = offsets[map_shift]
    shifted: list[GeneralRouteCase] = []
    excluded: list[str] = []
    for case in cases:
        walls = _shift_walls(case.walls, dx)
        env = PocketWorldEnv(walls=walls, agent_start=case.start, goal=case.goal)
        observation, _ = env.reset()
        path = _astar_path(
            _dilate(extract_wall_mask(observation), 4),
            case.start,
            case.goal,
            allow_diagonal=False,
        )
        if not path or len(path) < 16:
            excluded.append(case.map_id)
            continue
        shifted.append(
            GeneralRouteCase(
                map_id=f"{case.map_id}-{map_shift}",
                walls=walls,
                start=case.start,
                goal=case.goal,
                family=case.family,
                obstacle_count=len(walls),
                channel_count=case.channel_count,
            )
        )
    return tuple(shifted), tuple(excluded)


def _case_signatures(
    cases_by_seed: dict[int, tuple[GeneralRouteCase, ...]],
) -> np.ndarray:
    signatures: list[np.ndarray] = []
    for cases in cases_by_seed.values():
        for case in cases:
            observation, _ = PocketWorldEnv(
                walls=case.walls, agent_start=case.start, goal=case.goal
            ).reset()
            signatures.append(coarse_wall_signature(observation))
    return np.asarray(signatures, dtype=np.float32)


def _leave_one_out_threshold(signatures: np.ndarray, quantile: float = 0.99) -> float:
    if signatures.ndim != 2 or len(signatures) < 2:
        raise ValueError("at least two reference signatures are required")
    distances = []
    for index, signature in enumerate(signatures):
        others = np.delete(signatures, index, axis=0)
        distances.append(float(np.min(np.mean(others != signature[None], axis=1))))
    return float(np.quantile(np.asarray(distances, dtype=np.float32), quantile))


def run_general_ood(
    checkpoint: str | Path,
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    evaluation_episodes: int = 20,
    max_steps: int = 160,
    points: int = 13,
    map_shifts: tuple[str, ...] = DEFAULT_MAP_SHIFTS,
    speed_scales: tuple[float, ...] = (0.75, 1.0, 1.25),
    methods: tuple[str, ...] = DEFAULT_METHODS,
    mpc_horizon: int = 4,
    mpc_beam_width: int = 8,
    mpc_velocity_source: str = "rgb",
    train_seeds: tuple[int, ...] = (101, 103, 107),
    train_episodes: int = 20,
    shift_threshold_quantile: float = 0.99,
    adaptive_risk_threshold: float = 0.45,
    adaptive_risk_exit_threshold: float = 0.30,
    collision_head: object | None = None,
    collision_head_risk_threshold: float = 0.35,
    collision_head_risk_exit_threshold: float = 0.25,
    collision_head_horizon_index: int = 1,
    route_budget_margin: float = 1.05,
    route_progress_tolerance: float = 1.5,
    rgb_shield_margin: int = 4,
    route_gate_model: object | None = None,
    route_gate_threshold: float = 0.5,
) -> dict[str, object]:
    policy = RouteFieldPolicy.load(checkpoint)
    nominal_cases = {
        seed: sample_general_route_cases(seed, evaluation_episodes, split="holdout")
        for seed in evaluation_seeds
    }
    train_cases = {
        seed: sample_general_route_cases(seed, train_episodes, split="train")
        for seed in train_seeds
    }
    reference_signatures = _case_signatures(train_cases)
    shift_threshold = _leave_one_out_threshold(
        reference_signatures, quantile=shift_threshold_quantile
    )
    results: dict[str, object] = {}
    for map_shift in map_shifts:
        cases: dict[int, tuple[GeneralRouteCase, ...]] = {}
        excluded_by_seed: dict[str, list[str]] = {}
        for seed, items in nominal_cases.items():
            shifted, excluded = _shifted_cases_with_exclusions(items, map_shift)
            cases[seed] = shifted
            if excluded:
                excluded_by_seed[str(seed)] = list(excluded)
        for speed_scale in speed_scales:
            condition = f"{map_shift}@speed{speed_scale:g}"
            condition_results: dict[str, object] = {}
            for method in methods:
                condition_results[method] = evaluate_general_policy(
                    policy,
                    evaluation_seeds,
                    evaluation_episodes,
                    max_steps,
                    points,
                    method,
                    mpc_horizon=mpc_horizon,
                    mpc_beam_width=mpc_beam_width,
                    mpc_velocity_source=mpc_velocity_source,
                    cases_by_seed=cases,
                    agent_speed_scale=speed_scale,
                    reference_signatures=reference_signatures,
                    shift_threshold=shift_threshold,
                    adaptive_risk_threshold=adaptive_risk_threshold,
                    adaptive_risk_exit_threshold=adaptive_risk_exit_threshold,
                    collision_head=collision_head,
                    collision_head_risk_threshold=collision_head_risk_threshold,
                    collision_head_risk_exit_threshold=collision_head_risk_exit_threshold,
                    collision_head_horizon_index=collision_head_horizon_index,
                    route_budget_margin=route_budget_margin,
                    route_progress_tolerance=route_progress_tolerance,
                    rgb_shield_margin=rgb_shield_margin,
                    route_gate_model=route_gate_model,
                    route_gate_threshold=route_gate_threshold,
                )
            results[condition] = {
                "map_shift": map_shift,
                "agent_speed_scale": speed_scale,
                "paired_episode_count": int(sum(len(items) for items in cases.values())),
                "excluded_unreachable_by_seed": excluded_by_seed,
                "methods": condition_results,
            }
    return {
        "protocol": {
            "checkpoint": str(checkpoint),
            "evaluation_seeds": list(evaluation_seeds),
            "evaluation_episodes_per_seed": evaluation_episodes,
            "max_steps": max_steps,
            "route_points": points,
            "map_shifts": list(map_shifts),
            "speed_scales": list(speed_scales),
            "methods": list(methods),
            "mpc_horizon": mpc_horizon,
            "mpc_beam_width": mpc_beam_width,
            "mpc_velocity_source": mpc_velocity_source,
            "train_seeds_for_shift_calibration": list(train_seeds),
            "train_episodes_for_shift_calibration": train_episodes,
            "shift_threshold_quantile": shift_threshold_quantile,
            "shift_threshold": shift_threshold,
            "adaptive_risk_threshold": adaptive_risk_threshold,
            "adaptive_risk_exit_threshold": adaptive_risk_exit_threshold,
            "collision_head_risk_threshold": collision_head_risk_threshold,
            "collision_head_risk_exit_threshold": collision_head_risk_exit_threshold,
            "collision_head_horizon_index": collision_head_horizon_index,
            "route_budget_margin": route_budget_margin,
            "route_progress_tolerance": route_progress_tolerance,
            "rgb_shield_margin": rgb_shield_margin,
            "route_gate_threshold": route_gate_threshold,
            "reference_signature_count": len(reference_signatures),
            "task_generation": "one nominal holdout sample per seed, deterministic wall translation, reachability checked teacher-side",
            "student_evaluation_uses_astar": False,
            "fallback_method_uses_astar": any(
                method in {
                    "distance_field_mpc_shift_fallback",
                    "distance_field_budgeted_hybrid_mpc",
                    "distance_field_budgeted_hybrid_fast_mpc",
                    "distance_field_budgeted_hybrid_gated_mpc",
                }
                for method in methods
            ),
            "shift_labels_visible_to_planner": False,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate general-route MPC under OOD shifts")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/general-route-sketch-v19-mpc-distance-field.pt",
    )
    parser.add_argument("--evaluation-seeds", default="11,23,41")
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--points", type=int, default=13)
    parser.add_argument("--map-shifts", default=",".join(DEFAULT_MAP_SHIFTS))
    parser.add_argument("--speed-scales", default="0.75,1.0,1.25")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--mpc-horizon", type=int, default=4)
    parser.add_argument("--mpc-beam-width", type=int, default=8)
    parser.add_argument("--mpc-velocity-source", choices=("rgb", "action_fused"), default="rgb")
    parser.add_argument("--train-seeds", default="101,103,107")
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--adaptive-risk-threshold", type=float, default=0.45)
    parser.add_argument("--adaptive-risk-exit-threshold", type=float, default=0.30)
    parser.add_argument("--collision-head-checkpoint", default="")
    parser.add_argument("--collision-head-risk-threshold", type=float, default=0.35)
    parser.add_argument("--collision-head-risk-exit-threshold", type=float, default=0.25)
    parser.add_argument("--collision-head-horizon-index", type=int, default=1)
    parser.add_argument("--shift-threshold-quantile", type=float, default=0.99)
    parser.add_argument("--route-budget-margin", type=float, default=1.05)
    parser.add_argument("--route-progress-tolerance", type=float, default=1.5)
    parser.add_argument("--rgb-shield-margin", type=int, default=4)
    parser.add_argument("--route-gate-checkpoint", default="")
    parser.add_argument("--route-gate-threshold", type=float, default=0.25)
    parser.add_argument("--output", default="artifacts/evaluation-general-ood-v21.json")
    args = parser.parse_args()
    from .collision_head import CollisionProbabilityHead
    from .route_completion import RouteCompletionPredictor

    collision_head = (
        CollisionProbabilityHead.load(args.collision_head_checkpoint)
        if args.collision_head_checkpoint
        else None
    )
    route_gate_model = (
        RouteCompletionPredictor.load(args.route_gate_checkpoint)
        if args.route_gate_checkpoint
        else None
    )
    report = run_general_ood(
        checkpoint=args.checkpoint,
        evaluation_seeds=tuple(int(item) for item in args.evaluation_seeds.split(",") if item.strip()),
        evaluation_episodes=args.evaluation_episodes,
        max_steps=args.max_steps,
        points=args.points,
        map_shifts=tuple(item.strip() for item in args.map_shifts.split(",") if item.strip()),
        speed_scales=tuple(float(item) for item in args.speed_scales.split(",") if item.strip()),
        methods=tuple(item.strip() for item in args.methods.split(",") if item.strip()),
        mpc_horizon=args.mpc_horizon,
        mpc_beam_width=args.mpc_beam_width,
        mpc_velocity_source=args.mpc_velocity_source,
        train_seeds=tuple(int(item) for item in args.train_seeds.split(",") if item.strip()),
        train_episodes=args.train_episodes,
        adaptive_risk_threshold=args.adaptive_risk_threshold,
        adaptive_risk_exit_threshold=args.adaptive_risk_exit_threshold,
        collision_head=collision_head,
        collision_head_risk_threshold=args.collision_head_risk_threshold,
        collision_head_risk_exit_threshold=args.collision_head_risk_exit_threshold,
        collision_head_horizon_index=args.collision_head_horizon_index,
        shift_threshold_quantile=args.shift_threshold_quantile,
        route_budget_margin=args.route_budget_margin,
        route_progress_tolerance=args.route_progress_tolerance,
        rgb_shield_margin=args.rgb_shield_margin,
        route_gate_model=route_gate_model,
        route_gate_threshold=args.route_gate_threshold,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
