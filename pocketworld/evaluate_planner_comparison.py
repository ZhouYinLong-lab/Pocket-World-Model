"""Compare route-field planners on one shared, paired holdout task set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate_general_routes import evaluate_general_policy
from .general_routes import sample_general_route_cases
from .route_field import RouteFieldPolicy


DEFAULT_METHODS = (
    "rgb_astar",
    "distance_field_beam_rgb_projection",
    "distance_field_clearance_beam_rgb_projection",
    "distance_field_beam_mpc",
    "distance_field_beam_robust_mpc",
    "distance_field_beam_adaptive_mpc",
    "distance_field_beam_guarded_mpc",
)


def run_planner_comparison(
    checkpoint: str | Path,
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    evaluation_episodes: int = 20,
    max_steps: int = 160,
    points: int = 13,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    mpc_horizon: int = 4,
    mpc_beam_width: int = 8,
    mpc_velocity_source: str = "rgb",
    adaptive_risk_threshold: float = 0.45,
    adaptive_risk_exit_threshold: float = 0.30,
) -> dict[str, object]:
    policy = RouteFieldPolicy.load(checkpoint)
    cases_by_seed = {
        seed: sample_general_route_cases(seed, evaluation_episodes, split="holdout")
        for seed in evaluation_seeds
    }
    evaluations = {
        method: evaluate_general_policy(
            policy,
            evaluation_seeds,
            evaluation_episodes,
            max_steps,
            points,
            method,
            mpc_horizon=mpc_horizon,
            mpc_beam_width=mpc_beam_width,
            mpc_velocity_source=mpc_velocity_source,
            cases_by_seed=cases_by_seed,
            adaptive_risk_threshold=adaptive_risk_threshold,
            adaptive_risk_exit_threshold=adaptive_risk_exit_threshold,
        )
        for method in methods
    }
    return {
        "protocol": {
            "checkpoint": str(checkpoint),
            "evaluation_seeds": list(evaluation_seeds),
            "evaluation_episodes_per_seed": evaluation_episodes,
            "max_steps": max_steps,
            "route_points": points,
            "methods": list(methods),
            "mpc_horizon": mpc_horizon,
            "mpc_beam_width": mpc_beam_width,
            "mpc_velocity_source": mpc_velocity_source,
            "adaptive_risk_threshold": adaptive_risk_threshold,
            "adaptive_risk_exit_threshold": adaptive_risk_exit_threshold,
            "shared_holdout_cases": True,
            "student_evaluation_uses_astar": False,
            "rgb_astar_is_geometric_reference": True,
            "selection_on_holdout": False,
        },
        "evaluation": evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare route-field planner variants")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/coverage-study-v23-four_family_600-distance-field.pt",
    )
    parser.add_argument("--evaluation-seeds", default="11,23,41")
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--points", type=int, default=13)
    parser.add_argument("--methods", default=','.join(DEFAULT_METHODS))
    parser.add_argument("--mpc-horizon", type=int, default=4)
    parser.add_argument("--mpc-beam-width", type=int, default=8)
    parser.add_argument("--mpc-velocity-source", choices=("rgb", "action_fused"), default="rgb")
    parser.add_argument("--adaptive-risk-threshold", type=float, default=0.45)
    parser.add_argument("--adaptive-risk-exit-threshold", type=float, default=0.30)
    parser.add_argument("--output", default="artifacts/evaluation-planner-comparison-v23.json")
    args = parser.parse_args()
    report = run_planner_comparison(
        checkpoint=args.checkpoint,
        evaluation_seeds=tuple(int(value) for value in args.evaluation_seeds.split(",") if value.strip()),
        evaluation_episodes=args.evaluation_episodes,
        max_steps=args.max_steps,
        points=args.points,
        methods=tuple(value.strip() for value in args.methods.split(",") if value.strip()),
        mpc_horizon=args.mpc_horizon,
        mpc_beam_width=args.mpc_beam_width,
        mpc_velocity_source=args.mpc_velocity_source,
        adaptive_risk_threshold=args.adaptive_risk_threshold,
        adaptive_risk_exit_threshold=args.adaptive_risk_exit_threshold,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
