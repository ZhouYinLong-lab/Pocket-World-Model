"""Compare route-field training coverage under one fixed MPC evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate_general_routes import train_and_evaluate_general_routes
from .general_routes import GENERAL_FAMILIES


DEFAULT_CONDITIONS = {
    "two_family_600": {
        "families": ("staggered_blocks", "multi_channel"),
        "episode_multiplier": 1,
    },
    "four_family_600": {
        "families": GENERAL_FAMILIES,
        "episode_multiplier": 1,
    },
    "four_family_1200": {
        "families": GENERAL_FAMILIES,
        "episode_multiplier": 2,
    },
}


def run_coverage_study(
    train_seeds: tuple[int, ...] = (101, 103, 107),
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    train_episodes: int = 200,
    evaluation_episodes: int = 20,
    max_steps: int = 160,
    points: int = 13,
    epochs: int = 360,
    mpc_horizon: int = 4,
    mpc_beam_width: int = 8,
    predictor_root: str | Path = "artifacts/coverage-study-v22",
) -> dict[str, object]:
    methods = (
        "distance_field_beam_rgb_projection",
        "distance_field_beam_mpc",
    )
    reports: dict[str, object] = {}
    root = Path(predictor_root)
    for name, condition in DEFAULT_CONDITIONS.items():
        families = condition["families"]
        condition_train_episodes = int(train_episodes * condition["episode_multiplier"])
        report = train_and_evaluate_general_routes(
            train_seeds=train_seeds,
            evaluation_seeds=evaluation_seeds,
            train_episodes=condition_train_episodes,
            evaluation_episodes=evaluation_episodes,
            max_steps=max_steps,
            points=points,
            epochs=epochs,
            predictor_output=root.with_name(f"{root.name}-{name}.pt"),
            mpc_horizon=mpc_horizon,
            mpc_beam_width=mpc_beam_width,
            methods=methods,
            training_families=families,
            balanced_training_families=True,
        )
        reports[name] = {
            "training_families": list(families),
            "train_episodes_per_seed": condition_train_episodes,
            "samples_per_family": condition_train_episodes // len(families),
            "report": report,
        }
    return {
        "protocol": {
            "train_seeds": list(train_seeds),
            "evaluation_seeds": list(evaluation_seeds),
            "train_episodes_per_seed": train_episodes,
            "evaluation_episodes_per_seed": evaluation_episodes,
            "max_steps": max_steps,
            "route_points": points,
            "epochs": epochs,
            "mpc_horizon": mpc_horizon,
            "mpc_beam_width": mpc_beam_width,
            "methods": list(methods),
            "holdout_is_shared": True,
            "conditions": {
                name: {
                    "families": list(config["families"]),
                    "train_episodes_per_seed": int(train_episodes * config["episode_multiplier"]),
                    "total_samples": len(train_seeds) * int(train_episodes * config["episode_multiplier"]),
                    "samples_per_family_per_seed": int(train_episodes * config["episode_multiplier"]) // len(config["families"]),
                }
                for name, config in DEFAULT_CONDITIONS.items()
            },
            "balanced_training_families": True,
            "coverage_density_are_explicit_variables": True,
        },
        "conditions": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare route-field training family coverage")
    parser.add_argument("--train-seeds", default="101,103,107")
    parser.add_argument("--evaluation-seeds", default="11,23,41")
    parser.add_argument("--train-episodes", type=int, default=200)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--points", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=360)
    parser.add_argument("--mpc-horizon", type=int, default=4)
    parser.add_argument("--mpc-beam-width", type=int, default=8)
    parser.add_argument("--predictor-root", default="artifacts/coverage-study-v23")
    parser.add_argument("--output", default="artifacts/evaluation-coverage-study-v23.json")
    args = parser.parse_args()
    report = run_coverage_study(
        train_seeds=tuple(int(item) for item in args.train_seeds.split(",") if item.strip()),
        evaluation_seeds=tuple(int(item) for item in args.evaluation_seeds.split(",") if item.strip()),
        train_episodes=args.train_episodes,
        evaluation_episodes=args.evaluation_episodes,
        max_steps=args.max_steps,
        points=args.points,
        epochs=args.epochs,
        mpc_horizon=args.mpc_horizon,
        mpc_beam_width=args.mpc_beam_width,
        predictor_root=args.predictor_root,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
