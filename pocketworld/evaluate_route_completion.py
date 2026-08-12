"""Train and evaluate an explicit route-completion predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import PocketWorldEnv
from .evaluate_planners import (
    COLLISION_RISK_BUDGET,
    ROUTE_COMPLETION_WEIGHT,
    SINGLE_BARRIER_WALLS,
    _scenario_walls,
    _episode_cases,
    evaluate_planner_tournament,
)
from .model import PocketWorldModel
from .planner import extract_agent_position, extract_wall_mask, random_shooting, receding_horizon_plan
from .route_completion import (
    MAP_AWARE_ROUTE_FEATURE_NAMES,
    ROUTE_FEATURE_NAMES,
    RouteCompletionPredictor,
    extract_map_context_features,
    extract_route_features,
)


def _load_model(checkpoint: str | Path) -> PocketWorldModel:
    payload = torch.load(checkpoint, map_location="cpu")
    model = PocketWorldModel()
    model.load_state_dict(payload["model"], strict=False)
    return model


def _preferred_action(start: tuple[float, float], goal: tuple[float, float]) -> int:
    delta = np.asarray(goal, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    if abs(delta[0]) >= abs(delta[1]):
        return 3 if delta[0] >= 0 else 2
    return 1 if delta[1] >= 0 else 0


def _sample_action_sequences(
    rng: np.random.Generator,
    start: tuple[float, float],
    goal: tuple[float, float],
    candidates: int,
    horizon: int,
) -> np.ndarray:
    """Mix random, goal-biased, and alternating candidates for calibration."""
    actions = rng.integers(0, 4, size=(candidates, horizon), dtype=np.int64)
    preferred = _preferred_action(start, goal)
    if candidates:
        actions[0] = preferred
    if candidates > 1:
        actions[1, : max(1, horizon // 2)] = preferred
    if candidates > 2:
        actions[2, ::2] = preferred
    return actions


def _imagined_route_features(
    model: PocketWorldModel,
    observation: np.ndarray,
    goal: tuple[float, float],
    actions: np.ndarray,
    map_aware: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    start_position = extract_agent_position(observation).astype(np.float32)
    if not np.isfinite(start_position).all():
        raise ValueError("agent could not be located in the RGB observation")
    action_tensor = torch.from_numpy(actions)
    start = torch.from_numpy(observation[None]).float() / 255.0
    starts = start.expand(actions.shape[0], -1, -1, -1)
    normalized_start = torch.from_numpy(start_position / 64.0).float().expand(actions.shape[0], -1)
    positions = model.imagine_positions(
        starts,
        action_tensor,
        collision_response=True,
        initial_position=normalized_start,
    ).cpu().numpy() * 64.0
    positions = np.concatenate(
        (np.broadcast_to(start_position, (actions.shape[0], 1, 2)), positions),
        axis=1,
    )
    risks = model.imagine_collision_probabilities(
        starts,
        action_tensor,
        initial_position=normalized_start,
    ).cpu().numpy()
    prefix = np.concatenate(
        (np.zeros((actions.shape[0], 1), dtype=np.float32), np.maximum.accumulate(risks, axis=1)),
        axis=1,
    )
    map_context = None
    if map_aware:
        map_context = extract_map_context_features(
            start_position,
            goal,
            extract_wall_mask(observation),
        )
    return extract_route_features(
        positions,
        goal,
        prefix,
        actions=actions,
        map_context=map_context,
    ), prefix


def _execute_route(
    start: tuple[float, float],
    goal: tuple[float, float],
    actions: np.ndarray,
    walls=SINGLE_BARRIER_WALLS,
) -> float:
    env = PocketWorldEnv(walls=walls, agent_start=start, goal=goal)
    _, info = env.reset()
    for action in actions:
        _, _, terminated, truncated, info = env.step(int(action))
        if terminated or truncated:
            break
    return float(info["distance_to_goal"] <= env.goal_radius)


def collect_route_examples(
    model: PocketWorldModel,
    seeds: tuple[int, ...],
    episodes: int = 12,
    candidates: int = 32,
    horizon: int = 48,
    scenarios: tuple[str, ...] = ("single_barrier",),
    map_aware: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect imagined-route features with real simulator completion labels."""
    all_features: list[np.ndarray] = []
    all_labels: list[float] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for scenario_index, scenario in enumerate(scenarios):
            walls = _scenario_walls(scenario)
            for episode, (start, goal) in enumerate(_episode_cases(episodes, seed, scenario)):
                torch.manual_seed(seed * 100_000 + scenario_index * 10_000 + episode)
                env = PocketWorldEnv(walls=walls, agent_start=start, goal=goal)
                observation, _ = env.reset()
                sequences = _sample_action_sequences(rng, start, goal, candidates, horizon)
                # Add one map-aware route candidate so the binary target contains
                # successful as well as failed routes when the learned model is
                # systematically overconfident near the wall.
                wall_plan = random_shooting(
                    model,
                    observation,
                    goal,
                    horizon=horizon,
                    candidates=max(32, candidates),
                    collision_aware=True,
                    learned_collision=True,
                    route_objective=True,
                    route_execution_horizon=horizon,
                    wall_aware_route=True,
                    observation_history=[observation],
                )
                if len(wall_plan.actions):
                    padded = np.pad(
                        wall_plan.actions.astype(np.int64),
                        (0, max(0, horizon - len(wall_plan.actions))),
                        mode="constant",
                    )[:horizon]
                    sequences = np.concatenate((sequences, padded[None]), axis=0)
                # The existing RGB-only hybrid controller supplies a successful
                # route teacher when available. Its executed action prefix is
                # still relabeled by a fresh simulator rollout below, so the
                # predictor never trains on the teacher's private success flag.
                teacher_env = PocketWorldEnv(walls=walls, agent_start=start, goal=goal)
                teacher_observation, _ = teacher_env.reset()
                teacher = receding_horizon_plan(
                    model,
                    teacher_observation,
                    goal,
                    teacher_env.step,
                    max_steps=horizon,
                    rollout_horizon=min(16, horizon),
                    candidates=max(32, candidates),
                    collision_aware=True,
                    preserve_route=True,
                    route_tolerance=6.0,
                    learned_collision=True,
                    hybrid_collision=True,
                    use_history_velocity=True,
                    use_learned_velocity=True,
                    route_objective=True,
                    route_execution_horizon=min(12, horizon),
                    alignment_fallback_threshold=4.0,
                    wall_aware_route=True,
                )
                if len(teacher.actions):
                    padded = np.pad(
                        teacher.actions.astype(np.int64),
                        (0, max(0, horizon - len(teacher.actions))),
                        mode="constant",
                    )[:horizon]
                    sequences = np.concatenate((sequences, padded[None]), axis=0)
                features, _ = _imagined_route_features(
                    model, observation, goal, sequences, map_aware=map_aware
                )
                labels = np.asarray(
                    [_execute_route(start, goal, sequence, walls=walls) for sequence in sequences],
                    dtype=np.float32,
                )
                all_features.append(features)
                all_labels.append(labels)
    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


def _route_validation_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    predictions = probabilities >= 0.5
    positives = labels == 1.0
    negatives = labels == 0.0
    true_positive = float(np.sum(predictions & positives))
    false_positive = float(np.sum(predictions & negatives))
    true_negative = float(np.sum(~predictions & negatives))
    false_negative = float(np.sum(~predictions & positives))
    auc = float("nan")
    if positives.any() and negatives.any():
        order = np.argsort(probabilities, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
        auc = float(
            (ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2.0)
            / (positives.sum() * negatives.sum())
        )
    return {
        "samples": int(labels.size),
        "positive_rate": float(labels.mean()) if labels.size else float("nan"),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "accuracy": float(np.mean(predictions == positives)),
        "positive_precision": true_positive / max(1.0, true_positive + false_positive),
        "positive_recall": true_positive / max(1.0, true_positive + false_negative),
        "negative_recall": true_negative / max(1.0, true_negative + false_positive),
        "auroc": auc,
    }


def collect_selected_route_examples(
    model: PocketWorldModel,
    predictor: RouteCompletionPredictor,
    seeds: tuple[int, ...],
    episodes: int,
    candidates: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mine the routes actually selected by the current route planner."""
    features: list[np.ndarray] = []
    labels: list[float] = []
    for seed in seeds:
        for episode, (start, goal) in enumerate(_episode_cases(episodes, seed, "single_barrier")):
            torch.manual_seed(seed * 100_000 + episode)
            env = PocketWorldEnv(walls=SINGLE_BARRIER_WALLS, agent_start=start, goal=goal)
            observation, _ = env.reset()
            result = random_shooting(
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
                route_completion_model=predictor,
                route_completion_weight=ROUTE_COMPLETION_WEIGHT,
                collision_risk_budget=COLLISION_RISK_BUDGET,
                observation_history=[observation],
            )
            route_positions = result.imagined_positions[None]
            route_prefix = np.zeros((1, route_positions.shape[1]), dtype=np.float32)
            route_prefix[0, -1] = float(result.imagined_collision_risk)
            features.append(extract_route_features(route_positions, goal, route_prefix))
            labels.append(_execute_route(start, goal, result.actions))
    return np.concatenate(features, axis=0), np.asarray(labels, dtype=np.float32)


def _paired_success_delta(
    rows: dict[str, dict[str, list[dict[str, float | None]]]],
    baseline: str,
    challenger: str,
) -> dict[str, float | int | list[float]]:
    deltas: list[float] = []
    for seed_rows in rows.values():
        base = seed_rows[baseline]
        candidate = seed_rows[challenger]
        deltas.extend(
            float(candidate_row["real_success"]) - float(base_row["real_success"])
            for base_row, candidate_row in zip(base, candidate)
        )
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "baseline": baseline,
        "challenger": challenger,
        "episodes": int(values.size),
        "mean_delta_pp": float(values.mean() * 100.0),
        "std_delta_pp": float(values.std() * 100.0),
        "wins": int(np.sum(values > 0)),
        "losses": int(np.sum(values < 0)),
        "ties": int(np.sum(values == 0)),
        "per_seed_mean_delta_pp": [
            float(
                np.mean(
                    [
                        float(candidate_row["real_success"]) - float(base_row["real_success"])
                        for base_row, candidate_row in zip(
                            seed_rows[baseline], seed_rows[challenger]
                        )
                    ]
                )
                * 100.0
            )
            for seed_rows in rows.values()
        ],
    }


def evaluate_route_completion(
    checkpoint: str | Path,
    predictor_output: str | Path = "artifacts/route-completion-v1.pt",
    train_seeds: tuple[int, ...] = (101, 103),
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    train_episodes: int = 12,
    evaluation_episodes: int = 20,
    candidates: int = 256,
    horizon: int = 48,
    calibration_candidates: int = 32,
    epochs: int = 160,
    hard_negative_rounds: int = 0,
) -> dict[str, object]:
    model = _load_model(checkpoint)
    train_features, train_labels = collect_route_examples(
        model,
        train_seeds,
        episodes=train_episodes,
        candidates=calibration_candidates,
        horizon=horizon,
    )
    predictor = RouteCompletionPredictor()
    training = predictor.fit(train_features, train_labels, epochs=epochs)
    hard_negative_metrics: list[dict[str, float | int]] = []
    for round_index in range(max(0, int(hard_negative_rounds))):
        mined_features, mined_labels = collect_selected_route_examples(
            model,
            predictor,
            train_seeds,
            episodes=train_episodes,
            candidates=candidates,
            horizon=horizon,
        )
        train_features = np.concatenate((train_features, mined_features), axis=0)
        train_labels = np.concatenate((train_labels, mined_labels), axis=0)
        predictor = RouteCompletionPredictor()
        training = predictor.fit(train_features, train_labels, epochs=epochs, seed=7 + round_index)
        hard_negative_metrics.append(
            {
                "round": round_index + 1,
                "mined_samples": int(mined_features.shape[0]),
                "mined_positive_rate": float(mined_labels.mean()),
                "total_samples": int(train_features.shape[0]),
                "total_positive_rate": float(train_labels.mean()),
            }
        )
    validation_features, validation_labels = collect_route_examples(
        model,
        evaluation_seeds,
        episodes=evaluation_episodes,
        candidates=calibration_candidates,
        horizon=horizon,
    )
    validation_probabilities = predictor.predict_proba(validation_features)
    predictor.save(
        predictor_output,
        metadata={
            "checkpoint": str(checkpoint),
            "train_seeds": list(train_seeds),
            "train_episodes": train_episodes,
            "calibration_candidates": calibration_candidates,
            "horizon": horizon,
            "feature_names": list(ROUTE_FEATURE_NAMES),
        },
    )
    report = evaluate_planner_tournament(
        model,
        seeds=evaluation_seeds,
        episodes=evaluation_episodes,
        horizon=horizon,
        candidates=candidates,
        scenario="single_barrier",
        planners=(
            "learned_collision",
            "route_completion",
            "route_completion_safe_gate",
            "route_completion_rgb_only",
            "route_completion_soft",
            "route_completion_mpc",
            "route_aware_hybrid",
        ),
        route_models={
            "route_completion": predictor,
            "route_completion_safe_gate": predictor,
            "route_completion_rgb_only": predictor,
            "route_completion_soft": predictor,
            "route_completion_mpc": predictor,
        },
        risk_model_metadata={
            "route_predictor": str(predictor_output),
            "route_features": list(ROUTE_FEATURE_NAMES),
            "training_samples": int(train_features.shape[0]),
            "training_positive_rate": float(train_labels.mean()),
            "training_metrics": training,
            "validation_metrics": _route_validation_metrics(validation_probabilities, validation_labels),
            "hard_negative_rounds": hard_negative_metrics,
        },
        include_rows=True,
    )
    report["route_training"] = {
        "features": list(ROUTE_FEATURE_NAMES),
        "samples": int(train_features.shape[0]),
        "positive_rate": float(train_labels.mean()),
        "metrics": training,
        "predictor_output": str(predictor_output),
        "hard_negative_rounds": hard_negative_metrics,
    }
    report["route_validation"] = {
        "samples": int(validation_features.shape[0]),
        "metrics": _route_validation_metrics(validation_probabilities, validation_labels),
    }
    report["paired_route_comparison"] = _paired_success_delta(
        report["rows"], "learned_collision", "route_completion"
    )
    report["paired_mpc_comparison"] = _paired_success_delta(
        report["rows"], "learned_collision", "route_completion_mpc"
    )
    report["paired_safe_gate_comparison"] = _paired_success_delta(
        report["rows"], "learned_collision", "route_completion_safe_gate"
    )
    report["paired_rgb_only_comparison"] = _paired_success_delta(
        report["rows"], "learned_collision", "route_completion_rgb_only"
    )
    report["paired_soft_comparison"] = _paired_success_delta(
        report["rows"], "learned_collision", "route_completion_soft"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate route-level completion prediction")
    parser.add_argument("checkpoint", nargs="?", default="artifacts/pocketworld-map-suite-v3-final.pt")
    parser.add_argument("--predictor-output", default="artifacts/route-completion-v1.pt")
    parser.add_argument("--train-seeds", default="101,103")
    parser.add_argument("--evaluation-seeds", default="11,23,41")
    parser.add_argument("--train-episodes", type=int, default=12)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--calibration-candidates", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--hard-negative-rounds", type=int, default=0)
    parser.add_argument("--output", default="artifacts/evaluation-route-completion-barrier-v1.json")
    args = parser.parse_args()
    report = evaluate_route_completion(
        checkpoint=args.checkpoint,
        predictor_output=args.predictor_output,
        train_seeds=tuple(int(item) for item in args.train_seeds.split(",") if item.strip()),
        evaluation_seeds=tuple(int(item) for item in args.evaluation_seeds.split(",") if item.strip()),
        train_episodes=args.train_episodes,
        evaluation_episodes=args.evaluation_episodes,
        candidates=args.candidates,
        horizon=args.horizon,
        calibration_candidates=args.calibration_candidates,
        epochs=args.epochs,
        hard_negative_rounds=args.hard_negative_rounds,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
