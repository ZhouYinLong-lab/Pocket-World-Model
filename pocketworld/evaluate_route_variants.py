"""Compare route-completion feature contracts under in- and out-of-distribution maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .evaluate_planners import (
    COLLISION_RISK_BUDGET,
    ROUTE_COMPLETION_WEIGHT,
    evaluate_planner_tournament,
)
from .evaluate_route_completion import (
    _load_model,
    collect_route_examples,
)
from .route_completion import (
    MAP_AWARE_ROUTE_FEATURE_NAMES,
    ROUTE_FEATURE_NAMES,
    RouteCompletionPredictor,
)


def _paired_success_delta(rows: dict[str, dict[str, list[dict[str, float | None]]]], baseline: str, challenger: str) -> dict[str, object]:
    deltas: list[float] = []
    per_seed: list[float] = []
    wins = losses = ties = 0
    for seed_rows in rows.values():
        seed_deltas = [
            float(candidate["real_success"]) - float(reference["real_success"])
            for reference, candidate in zip(seed_rows[baseline], seed_rows[challenger])
        ]
        deltas.extend(seed_deltas)
        per_seed.append(float(np.mean(seed_deltas) * 100.0))
        wins += sum(delta > 0.0 for delta in seed_deltas)
        losses += sum(delta < 0.0 for delta in seed_deltas)
        ties += sum(delta == 0.0 for delta in seed_deltas)
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "baseline": baseline,
        "challenger": challenger,
        "episodes": int(values.size),
        "mean_delta_pp": float(values.mean() * 100.0),
        "per_seed_mean_delta_pp": per_seed,
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(ties),
    }


def _fit_variant(
    model,
    train_seeds: tuple[int, ...],
    train_scenarios: tuple[str, ...],
    train_episodes: int,
    calibration_candidates: int,
    horizon: int,
    epochs: int,
    map_aware: bool,
    output: str | Path,
) -> tuple[RouteCompletionPredictor, dict[str, object]]:
    features, labels = collect_route_examples(
        model,
        train_seeds,
        episodes=train_episodes,
        candidates=calibration_candidates,
        horizon=horizon,
        scenarios=train_scenarios,
        map_aware=map_aware,
    )
    names = MAP_AWARE_ROUTE_FEATURE_NAMES if map_aware else ROUTE_FEATURE_NAMES
    predictor = RouteCompletionPredictor(feature_names=names)
    training = predictor.fit(features, labels, epochs=epochs)
    predictor.save(
        output,
        metadata={
            "protocol": "route feature contract comparison",
            "map_aware": map_aware,
            "train_seeds": list(train_seeds),
            "train_scenarios": list(train_scenarios),
            "train_episodes": train_episodes,
            "calibration_candidates": calibration_candidates,
            "horizon": horizon,
            "feature_names": list(names),
        },
    )
    return predictor, {
        "features": list(names),
        "samples": int(features.shape[0]),
        "positive_rate": float(labels.mean()),
        "training": training,
        "output": str(output),
    }


def _evaluate_variant(
    model,
    predictor: RouteCompletionPredictor,
    evaluation_seeds: tuple[int, ...],
    evaluation_episodes: int,
    candidates: int,
    horizon: int,
    scenarios: tuple[str, ...],
    speed_scales: tuple[float, ...],
    on_condition: Callable[[dict[str, object]], None] | None = None,
    initial_conditions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    conditions: list[dict[str, object]] = list(initial_conditions or [])
    skip = len(conditions)
    condition_index = 0
    for scenario in scenarios:
        for speed_scale in speed_scales:
            if condition_index < skip:
                condition_index += 1
                continue
            report = evaluate_planner_tournament(
                model,
                seeds=evaluation_seeds,
                episodes=evaluation_episodes,
                horizon=horizon,
                candidates=candidates,
                scenario=scenario,
                agent_speed_scale=speed_scale,
                planners=("learned_collision", "route_completion_soft"),
                route_models={"route_completion_soft": predictor},
                risk_model_metadata={
                    "protocol": "route feature contract comparison",
                    "route_features": list(predictor.feature_names),
                    "scenario": scenario,
                    "agent_speed_scale": speed_scale,
                    "collision_risk_budget": COLLISION_RISK_BUDGET,
                    "route_completion_weight": ROUTE_COMPLETION_WEIGHT,
                },
                include_rows=True,
            )
            rows = report["rows"]
            condition = {
                    "scenario": scenario,
                    "agent_speed_scale": speed_scale,
                    "summary": report["summary"],
                    "paired_soft_comparison": _paired_success_delta(
                        rows, "learned_collision", "route_completion_soft"
                    ),
                }
            conditions.append(condition)
            condition_index += 1
            if on_condition is not None:
                on_condition({"conditions": conditions})
    return {"conditions": conditions}


def evaluate_route_variants(
    checkpoint: str | Path,
    baseline_output: str | Path = "artifacts/route-completion-v9-baseline.pt",
    map_aware_output: str | Path = "artifacts/route-completion-v9-map-aware.pt",
    train_seeds: tuple[int, ...] = (101, 103),
    train_scenarios: tuple[str, ...] = ("single_barrier", "barrier_shifted"),
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    evaluation_scenarios: tuple[str, ...] = (
        "single_barrier",
        "barrier_shifted",
        "barrier_narrow_gap",
        "barrier_wide_gap",
    ),
    speed_scales: tuple[float, ...] = (0.8, 1.0, 1.2),
    train_episodes: int = 12,
    evaluation_episodes: int = 20,
    calibration_candidates: int = 32,
    candidates: int = 256,
    horizon: int = 48,
    epochs: int = 160,
    progress_output: str | Path | None = None,
    resume: bool = False,
) -> dict[str, object]:
    model = _load_model(checkpoint)
    progress_path = Path(progress_output) if progress_output is not None else None
    previous = {}
    if resume and progress_path is not None and progress_path.exists():
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
    if resume and previous.get("baseline_9d", {}).get("training", {}).get("output") == str(baseline_output):
        baseline = RouteCompletionPredictor.load(baseline_output)
        baseline_training = previous["baseline_9d"]["training"]
    else:
        baseline, baseline_training = _fit_variant(
            model, train_seeds, train_scenarios, train_episodes,
            calibration_candidates, horizon, epochs, False, baseline_output,
        )
    if resume and previous.get("map_aware_17d", {}).get("training", {}).get("output") == str(map_aware_output):
        map_aware = RouteCompletionPredictor.load(map_aware_output)
        map_aware_training = previous["map_aware_17d"]["training"]
    else:
        map_aware, map_aware_training = _fit_variant(
            model, train_seeds, train_scenarios, train_episodes,
            calibration_candidates, horizon, epochs, True, map_aware_output,
        )
    result: dict[str, object] = {
        "protocol": {
            "train_seeds": list(train_seeds),
            "train_scenarios": list(train_scenarios),
            "evaluation_seeds": list(evaluation_seeds),
            "evaluation_scenarios": list(evaluation_scenarios),
            "speed_scales": list(speed_scales),
            "paired_planner": "route_completion_soft vs learned_collision",
            "fixed_planner_seed_offsets": True,
        },
        "baseline_9d": {
            "training": baseline_training,
            "evaluation": previous.get("baseline_9d", {}).get("evaluation", {"conditions": []})
            if resume else {"conditions": []},
        },
        "map_aware_17d": {
            "training": map_aware_training,
            "evaluation": previous.get("map_aware_17d", {}).get("evaluation", {"conditions": []})
            if resume else {"conditions": []},
        },
    }
    def save_progress() -> None:
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    baseline_done = len(result["baseline_9d"]["evaluation"]["conditions"])
    expected_conditions = len(evaluation_scenarios) * len(speed_scales)
    if baseline_done < expected_conditions:
        result["baseline_9d"]["evaluation"] = _evaluate_variant(
            model, baseline, evaluation_seeds, evaluation_episodes, candidates,
            horizon, evaluation_scenarios, speed_scales,
            on_condition=lambda value: (result["baseline_9d"].update({"evaluation": value}), save_progress()),
            initial_conditions=result["baseline_9d"]["evaluation"]["conditions"],
        )
        save_progress()
    map_done = len(result["map_aware_17d"]["evaluation"]["conditions"])
    if map_done < expected_conditions:
        result["map_aware_17d"]["evaluation"] = _evaluate_variant(
            model, map_aware, evaluation_seeds, evaluation_episodes, candidates,
            horizon, evaluation_scenarios, speed_scales,
            on_condition=lambda value: (result["map_aware_17d"].update({"evaluation": value}), save_progress()),
            initial_conditions=result["map_aware_17d"]["evaluation"]["conditions"],
        )
        save_progress()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and map-aware route predictors")
    parser.add_argument("checkpoint", nargs="?", default="artifacts/pocketworld-map-suite-v3-final.pt")
    parser.add_argument("--baseline-output", default="artifacts/route-completion-v9-baseline.pt")
    parser.add_argument("--map-aware-output", default="artifacts/route-completion-v9-map-aware.pt")
    parser.add_argument("--train-seeds", default="101,103")
    parser.add_argument("--train-scenarios", default="single_barrier,barrier_shifted")
    parser.add_argument("--evaluation-seeds", default="11,23,41")
    parser.add_argument("--evaluation-scenarios", default="single_barrier,barrier_shifted,barrier_narrow_gap,barrier_wide_gap")
    parser.add_argument("--speed-scales", default="0.8,1.0,1.2")
    parser.add_argument("--train-episodes", type=int, default=12)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--calibration-candidates", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--output", default="artifacts/evaluation-route-variants-v9.json")
    parser.add_argument("--progress-output", default="artifacts/evaluation-route-variants-v9-progress.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    parse_ints = lambda value: tuple(int(item) for item in value.split(",") if item.strip())
    parse_strings = lambda value: tuple(item.strip() for item in value.split(",") if item.strip())
    report = evaluate_route_variants(
        checkpoint=args.checkpoint,
        baseline_output=args.baseline_output,
        map_aware_output=args.map_aware_output,
        train_seeds=parse_ints(args.train_seeds),
        train_scenarios=parse_strings(args.train_scenarios),
        evaluation_seeds=parse_ints(args.evaluation_seeds),
        evaluation_scenarios=parse_strings(args.evaluation_scenarios),
        speed_scales=tuple(float(item) for item in args.speed_scales.split(",") if item.strip()),
        train_episodes=args.train_episodes,
        evaluation_episodes=args.evaluation_episodes,
        calibration_candidates=args.calibration_candidates,
        candidates=args.candidates,
        horizon=args.horizon,
        epochs=args.epochs,
        progress_output=args.progress_output,
        resume=args.resume,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
