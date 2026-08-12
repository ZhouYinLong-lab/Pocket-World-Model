"""Train, calibrate, and evaluate the route-conditioned collision head."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .collision_head import COLLISION_HORIZONS, CollisionProbabilityHead, collect_collision_head_dataset
from .evaluate_general_routes import evaluate_general_policy
from .general_routes import sample_general_route_cases
from .route_field import RouteFieldPolicy
from .evaluate_general_ood import run_general_ood


def _binary_auc(probabilities: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute weighted-free AUROC without a scikit-learn dependency."""
    scores = np.asarray(probabilities, dtype=np.float64).ravel()
    positives = (np.asarray(labels).ravel() > 0.0).astype(np.int64)
    positive_count = int(positives.sum())
    negative_count = int(len(positives) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(ranks[positives == 1].sum())
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _average_precision(probabilities: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute binary average precision for rare collision events."""
    scores = np.asarray(probabilities, dtype=np.float64).ravel()
    positives = np.asarray(labels).ravel() > 0.0
    positive_count = int(positives.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    hits = positives[order].astype(np.float64)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1, dtype=np.float64)
    return float((precision * hits).sum() / positive_count)


def probability_calibration_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
    bins: int = 10,
) -> dict[str, object]:
    """Summarize calibration and discrimination independently per horizon."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probabilities.ndim != 2 or labels.shape != probabilities.shape:
        raise ValueError("probabilities and labels must have matching [N, H] shapes")
    if bins < 2:
        raise ValueError("bins must be at least two")
    results: list[dict[str, object]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for horizon_index in range(probabilities.shape[1]):
        probability = np.clip(probabilities[:, horizon_index], 0.0, 1.0)
        label = labels[:, horizon_index]
        rows: list[dict[str, float | int]] = []
        ece = 0.0
        for index in range(bins):
            lower, upper = float(edges[index]), float(edges[index + 1])
            mask = (probability >= lower) & (
                probability <= upper if index == bins - 1 else probability < upper
            )
            count = int(mask.sum())
            if count == 0:
                continue
            mean_probability = float(probability[mask].mean())
            observed_rate = float(label[mask].mean())
            ece += count / len(probability) * abs(mean_probability - observed_rate)
            rows.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": count,
                    "mean_probability": mean_probability,
                    "observed_rate": observed_rate,
                }
            )
        metrics: dict[str, object] = {
            "horizon_index": horizon_index,
            "brier": float(np.mean((probability - label) ** 2)),
            "ece": float(ece),
            "auroc": _binary_auc(probability, label),
            "average_precision": _average_precision(probability, label),
            "positive_rate": float(np.mean(label > 0.0)),
            "mean_label": float(label.mean()),
            "reliability_bins": rows,
        }
        if threshold is not None:
            predicted = probability >= float(threshold)
            positives = label > 0.0
            true_positive = int((predicted & positives).sum())
            false_positive = int((predicted & ~positives).sum())
            metrics["threshold"] = float(threshold)
            metrics["collision_recall"] = float(true_positive / max(1, int(positives.sum())))
            metrics["precision"] = float(true_positive / max(1, int(predicted.sum())))
            metrics["false_positive_rate"] = float(false_positive / max(1, int((~positives).sum())))
            metrics["predicted_positive_rate"] = float(predicted.mean())
        results.append(metrics)
    return {"sample_count": int(len(probabilities)), "bin_count": bins, "horizons": results}


def _quantile_threshold(probabilities: np.ndarray, labels: np.ndarray, target_coverage: float) -> float:
    """Choose a threshold with empirical collision recall at the target level."""
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1]")
    candidates = np.unique(np.concatenate((np.asarray(probabilities).ravel(), np.asarray((0.0, 1.0)))))
    feasible: list[float] = []
    for threshold in candidates:
        predicted = np.asarray(probabilities) >= threshold
        positives = np.asarray(labels) > 0.0
        recall = float((predicted & positives).sum() / max(1, positives.sum()))
        if recall >= target_coverage:
            feasible.append(float(threshold))
    # Use the highest threshold that still meets recall: this preserves the
    # safety target while minimizing unnecessary robust-MPC activation.
    return max(feasible) if feasible else 0.0


def run_collision_head_study(
    route_checkpoint: str | Path,
    head_output: str | Path = "artifacts/collision-head-v26.pt",
    train_seeds: tuple[int, ...] = (101, 103, 107),
    train_episodes: int = 24,
    calibration_seeds: tuple[int, ...] = (53, 67),
    calibration_episodes: int = 12,
    calibration_holdout_seeds: tuple[int, ...] = (29, 31, 37),
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    evaluation_episodes: int = 20,
    max_steps: int = 160,
    points: int = 13,
    train_max_steps: int = 96,
    continuation_samples: int = 4,
    sample_stride: int = 2,
    epochs: int = 160,
    mpc_horizon: int = 4,
    mpc_beam_width: int = 8,
    probability_horizon_index: int = 1,
    target_recall: float = 0.80,
    calibrate_temperature: bool = True,
    train_families: tuple[str, ...] | None = None,
    run_ood: bool = False,
    ood_map_shifts: tuple[str, ...] = ("nominal", "walls_x_plus1"),
    ood_speed_scales: tuple[float, ...] = (1.0, 1.25),
) -> dict[str, object]:
    features, labels = collect_collision_head_dataset(
        seeds=train_seeds,
        episodes=train_episodes,
        max_steps=train_max_steps,
        continuation_samples=continuation_samples,
        sample_stride=sample_stride,
        families=train_families,
        balanced_families=True,
    )
    head = CollisionProbabilityHead(input_dim=features.shape[1])
    training = head.fit(features, labels, epochs=epochs)
    raw_head = copy.deepcopy(head)
    raw_head.temperature = 1.0
    calibration_features, calibration_labels = collect_collision_head_dataset(
        seeds=calibration_seeds,
        episodes=calibration_episodes,
        max_steps=train_max_steps,
        continuation_samples=continuation_samples,
        sample_stride=sample_stride,
        families=train_families,
        balanced_families=True,
    )
    raw_calibration_probabilities = raw_head.predict_proba(calibration_features)
    temperature_fit = None
    if calibrate_temperature:
        temperature_fit = head.fit_temperature(calibration_features, calibration_labels)
    calibrated_calibration_probabilities = head.predict_proba(calibration_features)
    raw_threshold = _quantile_threshold(
        raw_calibration_probabilities[:, probability_horizon_index],
        calibration_labels[:, probability_horizon_index],
        target_recall,
    )
    threshold = _quantile_threshold(
        calibrated_calibration_probabilities[:, probability_horizon_index],
        calibration_labels[:, probability_horizon_index],
        target_recall,
    )
    head_path = head.save(
        head_output,
        metadata={
            "teacher_labels_use_simulator_rollout": True,
            "future_state_hidden_from_features": True,
            "train_seeds": list(train_seeds),
            "temperature": float(head.temperature),
            "temperatures": head.temperatures.tolist(),
            "calibrated_on_disjoint_calibration_split": True,
            "raw_threshold": raw_threshold,
            "calibrated_threshold": threshold,
        },
    )
    route_policy = RouteFieldPolicy.load(route_checkpoint)
    cases_by_seed = {
        seed: sample_general_route_cases(seed, evaluation_episodes, split="holdout")
        for seed in evaluation_seeds
    }
    holdout_features, holdout_labels = collect_collision_head_dataset(
        seeds=calibration_holdout_seeds,
        episodes=calibration_episodes,
        max_steps=train_max_steps,
        continuation_samples=continuation_samples,
        sample_stride=sample_stride,
        families=train_families,
        balanced_families=True,
        split="holdout",
    )
    raw_holdout_probabilities = raw_head.predict_proba(holdout_features)
    calibrated_holdout_probabilities = head.predict_proba(holdout_features)
    raw_holdout_metrics = probability_calibration_metrics(
        raw_holdout_probabilities,
        holdout_labels,
        threshold=raw_threshold,
    )
    calibrated_holdout_metrics = probability_calibration_metrics(
        calibrated_holdout_probabilities,
        holdout_labels,
        threshold=threshold,
    )
    raw_evaluation = evaluate_general_policy(
        route_policy,
        evaluation_seeds,
        evaluation_episodes,
        max_steps,
        points,
        "distance_field_beam_collision_head_mpc",
        mpc_horizon=mpc_horizon,
        mpc_beam_width=mpc_beam_width,
        cases_by_seed=cases_by_seed,
        collision_head=raw_head,
        collision_head_risk_threshold=raw_threshold,
        collision_head_risk_exit_threshold=raw_threshold * 2.0 / 3.0,
        collision_head_horizon_index=probability_horizon_index,
    )
    evaluation = evaluate_general_policy(
        route_policy,
        evaluation_seeds,
        evaluation_episodes,
        max_steps,
        points,
        "distance_field_beam_collision_head_mpc",
        mpc_horizon=mpc_horizon,
        mpc_beam_width=mpc_beam_width,
        cases_by_seed=cases_by_seed,
        collision_head=head,
        collision_head_risk_threshold=threshold,
        collision_head_risk_exit_threshold=threshold * 2.0 / 3.0,
        collision_head_horizon_index=probability_horizon_index,
    )
    ood = None
    if run_ood:
        ood = run_general_ood(
            route_checkpoint,
            evaluation_seeds=evaluation_seeds,
            evaluation_episodes=evaluation_episodes,
            max_steps=max_steps,
            points=points,
            map_shifts=ood_map_shifts,
            speed_scales=ood_speed_scales,
            methods=("distance_field_beam_collision_head_mpc",),
            mpc_horizon=mpc_horizon,
            mpc_beam_width=mpc_beam_width,
            collision_head=head,
            collision_head_risk_threshold=threshold,
            collision_head_risk_exit_threshold=threshold * 2.0 / 3.0,
            collision_head_horizon_index=probability_horizon_index,
        )
    return {
        "protocol": {
            "route_checkpoint": str(route_checkpoint),
            "head_checkpoint": str(head_path),
            "train_seeds": list(train_seeds),
            "train_episodes_per_seed": train_episodes,
            "calibration_seeds": list(calibration_seeds),
            "calibration_episodes_per_seed": calibration_episodes,
            "calibration_holdout_seeds": list(calibration_holdout_seeds),
            "calibration_holdout_uses_distinct_maps": True,
            "evaluation_seeds": list(evaluation_seeds),
            "evaluation_episodes_per_seed": evaluation_episodes,
            "train_max_steps": train_max_steps,
            "continuation_samples": continuation_samples,
            "sample_stride": sample_stride,
            "epochs": epochs,
            "horizons": list(COLLISION_HORIZONS),
            "probability_horizon_index": probability_horizon_index,
            "target_recall": target_recall,
            "calibrated_threshold": threshold,
            "raw_threshold": raw_threshold,
            "temperature_calibration_enabled": calibrate_temperature,
            "threshold_selection_is_disjoint_from_final_holdout": True,
            "student_evaluation_uses_astar": False,
        },
        "training": training,
        "dataset": {
            "train_features": int(len(features)),
            "train_positive_rate": labels.mean(axis=0).tolist(),
            "calibration_features": int(len(calibration_features)),
            "calibration_positive_rate": calibration_labels.mean(axis=0).tolist(),
            "calibration_holdout_features": int(len(holdout_features)),
            "calibration_holdout_positive_rate": holdout_labels.mean(axis=0).tolist(),
        },
        "calibration": {
            "threshold": threshold,
            "raw_probability_mean": float(raw_calibration_probabilities.mean()),
            "raw_probability_std": float(raw_calibration_probabilities.std()),
            "calibrated_probability_mean": float(calibrated_calibration_probabilities.mean()),
            "calibrated_probability_std": float(calibrated_calibration_probabilities.std()),
            "temperature_fit": temperature_fit,
            "raw": probability_calibration_metrics(
                raw_calibration_probabilities,
                calibration_labels,
                threshold=raw_threshold,
            ),
            "calibrated": probability_calibration_metrics(
                calibrated_calibration_probabilities,
                calibration_labels,
                threshold=threshold,
            ),
            "calibration_holdout_is_not_used_for_temperature_or_threshold": True,
            "calibration_holdout_raw": raw_holdout_metrics,
            "calibration_holdout_calibrated": calibrated_holdout_metrics,
        },
        "raw_evaluation": raw_evaluation,
        "evaluation": evaluation,
        "ood": ood,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a learned collision probability head")
    parser.add_argument("--route-checkpoint", default="artifacts/coverage-study-v23-four_family_600-distance-field.pt")
    parser.add_argument("--head-output", default="artifacts/collision-head-v26.pt")
    parser.add_argument("--train-seeds", default="101,103,107")
    parser.add_argument("--train-episodes", type=int, default=24)
    parser.add_argument("--calibration-seeds", default="53,67")
    parser.add_argument("--calibration-episodes", type=int, default=12)
    parser.add_argument("--calibration-holdout-seeds", default="29,31,37")
    parser.add_argument("--evaluation-seeds", default="11,23,41")
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--points", type=int, default=13)
    parser.add_argument("--train-max-steps", type=int, default=96)
    parser.add_argument("--continuation-samples", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--mpc-horizon", type=int, default=4)
    parser.add_argument("--mpc-beam-width", type=int, default=8)
    parser.add_argument("--probability-horizon-index", type=int, default=1)
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--no-temperature-calibration", action="store_true")
    parser.add_argument("--run-ood", action="store_true")
    parser.add_argument("--ood-map-shifts", default="nominal,walls_x_plus1")
    parser.add_argument("--ood-speed-scales", default="1.0,1.25")
    parser.add_argument("--output", default="artifacts/evaluation-collision-head-v26.json")
    args = parser.parse_args()
    parse_ints = lambda value: tuple(int(item) for item in value.split(",") if item.strip())
    report = run_collision_head_study(
        route_checkpoint=args.route_checkpoint,
        head_output=args.head_output,
        train_seeds=parse_ints(args.train_seeds),
        train_episodes=args.train_episodes,
        calibration_seeds=parse_ints(args.calibration_seeds),
        calibration_episodes=args.calibration_episodes,
        calibration_holdout_seeds=parse_ints(args.calibration_holdout_seeds),
        evaluation_seeds=parse_ints(args.evaluation_seeds),
        evaluation_episodes=args.evaluation_episodes,
        max_steps=args.max_steps,
        points=args.points,
        train_max_steps=args.train_max_steps,
        continuation_samples=args.continuation_samples,
        sample_stride=args.sample_stride,
        epochs=args.epochs,
        mpc_horizon=args.mpc_horizon,
        mpc_beam_width=args.mpc_beam_width,
        probability_horizon_index=args.probability_horizon_index,
        target_recall=args.target_recall,
        calibrate_temperature=not args.no_temperature_calibration,
        run_ood=args.run_ood,
        ood_map_shifts=tuple(value.strip() for value in args.ood_map_shifts.split(",") if value.strip()),
        ood_speed_scales=tuple(float(value) for value in args.ood_speed_scales.split(",") if value.strip()),
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
