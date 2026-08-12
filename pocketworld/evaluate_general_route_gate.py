"""Train and evaluate a route-completion gate on general obstacle maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate_general_routes import evaluate_general_policy
from .general_route_gate import (
    GENERAL_ROUTE_FEATURE_NAMES,
    extract_general_route_features,
    make_general_route_predictor,
)
from .general_routes import sample_general_route_cases
from .route_field import RouteFieldPolicy


def _features_for_cases(
    field_policy: RouteFieldPolicy,
    cases_by_seed: dict[int, tuple[object, ...]],
    agent_speed_scale: float = 1.0,
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    features: list[np.ndarray] = []
    keys: list[tuple[int, str]] = []
    from .env import PocketWorldEnv

    for seed, cases in cases_by_seed.items():
        for case in cases:
            observation, _ = PocketWorldEnv(
                walls=case.walls, agent_start=case.start, goal=case.goal,
                agent_speed_scale=agent_speed_scale,
            ).reset()
            features.append(
                extract_general_route_features(
                    observation, case.goal, field_policy, agent_speed_scale
                )
            )
            keys.append((seed, case.map_id))
    return np.asarray(features, dtype=np.float32), keys


def _labels_from_report(
    report: dict[str, object], keys: list[tuple[int, str]]
) -> np.ndarray:
    rows = {
        (int(row["seed"]), str(row["map_id"])): float(row["real_success"])
        for row in report["rows"]
    }
    missing = [key for key in keys if key not in rows]
    if missing:
        raise RuntimeError(f"missing route labels for {len(missing)} cases")
    return np.asarray([rows[key] for key in keys], dtype=np.float32)


def _select_threshold(
    predictor: object,
    field_policy: RouteFieldPolicy,
    cases_by_seed: dict[int, tuple[object, ...]],
    thresholds: tuple[float, ...],
    max_steps: int,
    points: int,
    mpc_horizon: int,
    mpc_beam_width: int,
) -> tuple[float, list[dict[str, float]]]:
    audits: list[dict[str, float]] = []
    for threshold in thresholds:
        report = evaluate_general_policy(
            field_policy,
            tuple(cases_by_seed),
            max(1, max(len(items) for items in cases_by_seed.values())),
            max_steps,
            points,
            "distance_field_predicted_gate_hybrid",
            mpc_horizon=mpc_horizon,
            mpc_beam_width=mpc_beam_width,
            cases_by_seed=cases_by_seed,
            route_gate_model=predictor,
            route_gate_threshold=threshold,
        )
        summary = report["summary"]
        audits.append(
            {
                "threshold": float(threshold),
                "real_success": float(summary["real_success"]["mean"]),
                "collision_count": float(summary["collision_count"]["mean"]),
                "astar_calls": float(summary["astar_calls"]["mean"]),
                "planning_time_ms": float(summary["planning_time_ms"]["mean"]),
            }
        )
    selected = max(
        audits,
        key=lambda item: (
            item["real_success"],
            -item["astar_calls"],
            -item["collision_count"],
            -item["threshold"],
        ),
    )
    return float(selected["threshold"]), audits


def train_and_evaluate_general_route_gate(
    field_checkpoint: str | Path,
    predictor_output: str | Path = "artifacts/general-route-gate-v1.pt",
    train_seeds: tuple[int, ...] = (101, 103, 107),
    calibration_seeds: tuple[int, ...] = (53, 67),
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    train_episodes: int = 20,
    calibration_episodes: int = 12,
    evaluation_episodes: int = 20,
    max_steps: int = 160,
    points: int = 13,
    epochs: int = 240,
    mpc_horizon: int = 6,
    mpc_beam_width: int = 24,
    thresholds: tuple[float, ...] = (0.25, 0.4, 0.55, 0.7, 0.85),
) -> dict[str, object]:
    field_policy = RouteFieldPolicy.load(field_checkpoint)
    train_cases = {
        seed: sample_general_route_cases(seed, train_episodes, split="train")
        for seed in train_seeds
    }
    calibration_cases = {
        seed: sample_general_route_cases(seed, calibration_episodes, split="holdout")
        for seed in calibration_seeds
    }
    holdout_cases = {
        seed: sample_general_route_cases(seed, evaluation_episodes, split="holdout")
        for seed in evaluation_seeds
    }

    train_features, train_keys = _features_for_cases(field_policy, train_cases)
    train_labels = _labels_from_report(
        evaluate_general_policy(
            field_policy,
            train_seeds,
            train_episodes,
            max_steps,
            points,
            "distance_field_beam_mpc",
            mpc_horizon=mpc_horizon,
            mpc_beam_width=mpc_beam_width,
            cases_by_seed=train_cases,
        ),
        train_keys,
    )
    predictor = make_general_route_predictor()
    training = predictor.fit(train_features, train_labels, epochs=epochs)
    predictor.save(
        predictor_output,
        metadata={
            "feature_names": list(GENERAL_ROUTE_FEATURE_NAMES),
            "label_method": "distance_field_beam_mpc",
            "student_evaluation_uses_astar": False,
            "train_seeds": list(train_seeds),
        },
    )
    calibration_threshold, calibration_audit = _select_threshold(
        predictor,
        field_policy,
        calibration_cases,
        thresholds,
        max_steps,
        points,
        mpc_horizon,
        mpc_beam_width,
    )
    final_reports = {}
    for method in (
        "distance_field_beam_mpc",
        "distance_field_budgeted_hybrid_fast_mpc",
        "distance_field_predicted_gate_hybrid",
    ):
        final_reports[method] = evaluate_general_policy(
            field_policy,
            evaluation_seeds,
            evaluation_episodes,
            max_steps,
            points,
            method,
            mpc_horizon=mpc_horizon,
            mpc_beam_width=mpc_beam_width,
            cases_by_seed=holdout_cases,
            route_gate_model=predictor,
            route_gate_threshold=calibration_threshold,
        )
    return {
        "protocol": {
            "field_checkpoint": str(field_checkpoint),
            "predictor_output": str(predictor_output),
            "train_seeds": list(train_seeds),
            "calibration_seeds": list(calibration_seeds),
            "evaluation_seeds": list(evaluation_seeds),
            "episodes": {"train": train_episodes, "calibration": calibration_episodes, "evaluation": evaluation_episodes},
            "max_steps": max_steps,
            "feature_names": list(GENERAL_ROUTE_FEATURE_NAMES),
            "label_method": "distance_field_beam_mpc",
            "gate_method": "initial RGB/field probability only; A* fallback before execution",
            "student_evaluation_uses_astar": False,
            "fallback_method_uses_astar": True,
        },
        "training": training,
        "calibration": {
            "thresholds": list(thresholds),
            "selected_threshold": calibration_threshold,
            "selection_rule": "maximize success, then minimize A* calls, then collisions",
            "audit": calibration_audit,
        },
        "evaluation": {
            method: {
                "summary": report["summary"],
                "by_family": report["by_family"],
            }
            for method, report in final_reports.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field_checkpoint")
    parser.add_argument("--predictor-output", default="artifacts/general-route-gate-v1.pt")
    parser.add_argument("--output", default="artifacts/evaluation-general-route-gate-v1.json")
    args = parser.parse_args()
    report = train_and_evaluate_general_route_gate(
        field_checkpoint=args.field_checkpoint,
        predictor_output=args.predictor_output,
    )
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

