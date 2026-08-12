"""Calibrate the adaptive MPC risk gate on a disjoint calibration split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate_general_routes import evaluate_general_policy
from .general_routes import GeneralRouteCase, sample_general_route_cases
from .route_field import RouteFieldPolicy


DEFAULT_THRESHOLDS = (0.25, 0.35, 0.45, 0.55, 0.65)


def _select_threshold(
    candidates: list[dict[str, object]],
    minimum_success: float,
) -> dict[str, object]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["summary"]["real_success"]["mean"] >= minimum_success
    ]
    if eligible:
        return min(
            eligible,
            key=lambda candidate: (
                candidate["summary"]["collision_count"]["mean"],
                candidate["summary"]["planning_time_ms"]["mean"],
                -candidate["summary"]["real_success"]["mean"],
            ),
        )
    return max(
        candidates,
        key=lambda candidate: (
            candidate["summary"]["real_success"]["mean"],
            -candidate["summary"]["collision_count"]["mean"],
            -candidate["summary"]["planning_time_ms"]["mean"],
        ),
    )


def run_adaptive_calibration(
    checkpoint: str | Path,
    calibration_seeds: tuple[int, ...] = (53, 67),
    calibration_episodes: int = 12,
    max_steps: int = 160,
    points: int = 13,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    exit_ratio: float = 2.0 / 3.0,
    minimum_success: float = 0.90,
    mpc_horizon: int = 4,
    mpc_beam_width: int = 8,
    mpc_velocity_source: str = "rgb",
) -> dict[str, object]:
    if not thresholds or any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("thresholds must be non-empty values in (0, 1]")
    if not 0.0 < exit_ratio <= 1.0:
        raise ValueError("exit_ratio must be in (0, 1]")
    if not 0.0 <= minimum_success <= 1.0:
        raise ValueError("minimum_success must be between 0 and 1")
    policy = RouteFieldPolicy.load(checkpoint)
    cases_by_seed: dict[int, tuple[GeneralRouteCase, ...]] = {
        seed: sample_general_route_cases(seed, calibration_episodes, split="holdout")
        for seed in calibration_seeds
    }
    candidates: list[dict[str, object]] = []
    for threshold in thresholds:
        exit_threshold = float(threshold * exit_ratio)
        evaluation = evaluate_general_policy(
            policy,
            calibration_seeds,
            calibration_episodes,
            max_steps,
            points,
            "distance_field_beam_adaptive_mpc",
            mpc_horizon=mpc_horizon,
            mpc_beam_width=mpc_beam_width,
            mpc_velocity_source=mpc_velocity_source,
            cases_by_seed=cases_by_seed,
            adaptive_risk_threshold=float(threshold),
            adaptive_risk_exit_threshold=exit_threshold,
        )
        candidates.append(
            {
                "entry_threshold": float(threshold),
                "exit_threshold": exit_threshold,
                "summary": evaluation["summary"],
                "by_family": evaluation["by_family"],
            }
        )
    selected = _select_threshold(candidates, minimum_success)
    return {
        "protocol": {
            "checkpoint": str(checkpoint),
            "calibration_seeds": list(calibration_seeds),
            "calibration_episodes_per_seed": calibration_episodes,
            "max_steps": max_steps,
            "route_points": points,
            "threshold_grid": [float(value) for value in thresholds],
            "exit_ratio": exit_ratio,
            "minimum_success": minimum_success,
            "mpc_horizon": mpc_horizon,
            "mpc_beam_width": mpc_beam_width,
            "mpc_velocity_source": mpc_velocity_source,
            "selection_split_is_disjoint_from_final_holdout": True,
            "selection_rule": "success floor, then collision mean, then planning time",
            "student_evaluation_uses_astar": False,
        },
        "candidates": candidates,
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the adaptive MPC risk gate")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/coverage-study-v23-four_family_600-distance-field.pt",
    )
    parser.add_argument("--calibration-seeds", default="53,67")
    parser.add_argument("--calibration-episodes", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--points", type=int, default=13)
    parser.add_argument("--thresholds", default=",".join(str(v) for v in DEFAULT_THRESHOLDS))
    parser.add_argument("--exit-ratio", type=float, default=2.0 / 3.0)
    parser.add_argument("--minimum-success", type=float, default=0.90)
    parser.add_argument("--mpc-horizon", type=int, default=4)
    parser.add_argument("--mpc-beam-width", type=int, default=8)
    parser.add_argument("--mpc-velocity-source", choices=("rgb", "action_fused"), default="rgb")
    parser.add_argument("--output", default="artifacts/evaluation-adaptive-calibration-v25.json")
    args = parser.parse_args()
    report = run_adaptive_calibration(
        checkpoint=args.checkpoint,
        calibration_seeds=tuple(int(value) for value in args.calibration_seeds.split(",") if value.strip()),
        calibration_episodes=args.calibration_episodes,
        max_steps=args.max_steps,
        points=args.points,
        thresholds=tuple(float(value) for value in args.thresholds.split(",") if value.strip()),
        exit_ratio=args.exit_ratio,
        minimum_success=args.minimum_success,
        mpc_horizon=args.mpc_horizon,
        mpc_beam_width=args.mpc_beam_width,
        mpc_velocity_source=args.mpc_velocity_source,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
