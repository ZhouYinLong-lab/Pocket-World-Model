from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import _variant_walls, collect_random_rollouts
from .env import PocketWorldEnv, Rect
from .model import PocketWorldModel, observable_velocity_from_frames
from .planner import (
    estimate_agent_velocity,
    extract_agent_position,
    predictive_shift_score,
    random_shooting,
    receding_horizon_plan,
    estimate_speed_response,
)


def evaluate_prediction(model: PocketWorldModel, episodes: int = 20, seed: int = 11, ood: bool = False) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    horizons = (1, 5, 10, 20)
    image_errors = {str(horizon): [] for horizon in horizons}
    composited_image_errors = {str(horizon): [] for horizon in horizons}
    position_errors = {str(horizon): [] for horizon in horizons}
    composited_position_errors = {str(horizon): [] for horizon in horizons}
    latent_position_errors = {str(horizon): [] for horizon in horizons}
    model.eval()
    for _ in range(episodes):
        walls = _variant_walls(rng) if ood else None
        env = PocketWorldEnv(
            walls=walls,
            agent_start=(float(rng.integers(6, 15)), float(rng.integers(6, 15))),
            goal=(float(rng.integers(49, 58)), float(rng.integers(49, 58))),
            agent_speed_scale=float(rng.choice((0.8, 1.0, 1.2))) if ood else 1.0,
        )
        observation, _ = env.reset()
        actions = rng.integers(0, 4, size=max(horizons), dtype=np.int64)
        actual = [observation]
        for action in actions:
            next_observation, _, terminated, truncated, _ = env.step(int(action))
            actual.append(next_observation)
            if terminated or truncated:
                actual.extend([next_observation] * (len(actions) - len(actual) + 1))
                break
        start = torch.from_numpy(observation[None]).float() / 255.0
        action_tensor = torch.from_numpy(actions[None])
        imagined = model.imagine(start, action_tensor, compose_agent=False)[0].cpu()
        composited = model.imagine(start, action_tensor, compose_agent=True)[0].cpu()
        imagined_positions = model.imagine_positions(start, action_tensor)[0].cpu().numpy() * 64.0
        for horizon in horizons:
            target = torch.from_numpy(actual[horizon]).float() / 255.0
            prediction = imagined[horizon]
            composited_prediction = composited[horizon]
            image_errors[str(horizon)].append(float(torch.abs(prediction - target).mean()))
            composited_image_errors[str(horizon)].append(float(torch.abs(composited_prediction - target).mean()))
            predicted_position = extract_agent_position(prediction.numpy())
            composited_position = extract_agent_position(composited_prediction.numpy())
            actual_position = extract_agent_position(actual[horizon])
            if np.all(np.isfinite(predicted_position)) and np.all(np.isfinite(actual_position)):
                position_errors[str(horizon)].append(float(np.linalg.norm(predicted_position - actual_position)))
            if np.all(np.isfinite(composited_position)) and np.all(np.isfinite(actual_position)):
                composited_position_errors[str(horizon)].append(float(np.linalg.norm(composited_position - actual_position)))
            if np.all(np.isfinite(imagined_positions[horizon - 1])) and np.all(np.isfinite(actual_position)):
                latent_position_errors[str(horizon)].append(float(np.linalg.norm(imagined_positions[horizon - 1] - actual_position)))
    return {
        "image_mae": {horizon: _mean(values) for horizon, values in image_errors.items()},
        "composited_image_mae": {horizon: _mean(values) for horizon, values in composited_image_errors.items()},
        "position_error_px": {horizon: _mean(values) for horizon, values in position_errors.items()},
        "composited_position_error_px": {horizon: _mean(values) for horizon, values in composited_position_errors.items()},
        "latent_position_error_px": {horizon: _mean(values) for horizon, values in latent_position_errors.items()},
        "position_coverage": {horizon: len(position_errors[horizon]) / episodes for horizon in position_errors},
        "composited_position_coverage": {horizon: len(composited_position_errors[horizon]) / episodes for horizon in composited_position_errors},
    }


def _agent_circle_mask(position: np.ndarray, radius: float = 3.0) -> np.ndarray:
    yy, xx = np.mgrid[:64, :64]
    return ((xx - position[0]) ** 2 + (yy - position[1]) ** 2 <= radius**2)


def evaluate_agent_rendering(model: PocketWorldModel, episodes: int = 20, seed: int = 31, ood: bool = False) -> dict[str, dict[str, float]]:
    """Measure the learned mask head separately from the RGB decoder."""
    rng = np.random.default_rng(seed)
    horizons = (1, 5, 10, 20)
    position_errors = {str(horizon): [] for horizon in horizons}
    mask_ious = {str(horizon): [] for horizon in horizons}
    coverage = {str(horizon): 0 for horizon in horizons}
    model.eval()
    for _ in range(episodes):
        walls = _variant_walls(rng) if ood else None
        env = PocketWorldEnv(
            walls=walls,
            agent_start=(float(rng.integers(6, 15)), float(rng.integers(6, 15))),
            goal=(float(rng.integers(49, 58)), float(rng.integers(49, 58))),
            agent_speed_scale=float(rng.choice((0.8, 1.0, 1.2))) if ood else 1.0,
        )
        observation, info = env.reset()
        actions = rng.integers(0, 4, size=max(horizons), dtype=np.int64)
        actual_positions = []
        for action in actions:
            _, _, terminated, truncated, info = env.step(int(action))
            actual_positions.append(np.asarray(info["position"], dtype=np.float32))
            if terminated or truncated:
                actual_positions.extend([actual_positions[-1]] * (len(actions) - len(actual_positions)))
                break
        start = torch.from_numpy(observation[None]).float() / 255.0
        action_tensor = torch.from_numpy(actions[None])
        predicted_masks = model.imagine_agent_masks(start, action_tensor)[0, :, 0].cpu().numpy()
        for horizon in horizons:
            target_position = actual_positions[horizon - 1]
            predicted_mask = predicted_masks[horizon - 1] >= 0.5
            target_mask = _agent_circle_mask(target_position)
            intersection = np.logical_and(predicted_mask, target_mask).sum()
            union = np.logical_or(predicted_mask, target_mask).sum()
            mask_ious[str(horizon)].append(float(intersection / max(1, union)))
            ys, xs = np.where(predicted_mask)
            if len(xs):
                coverage[str(horizon)] += 1
                predicted_position = np.asarray((xs.mean(), ys.mean()), dtype=np.float32)
                position_errors[str(horizon)].append(float(np.linalg.norm(predicted_position - target_position)))
    return {
        "mask_iou": {horizon: _mean(values) for horizon, values in mask_ious.items()},
        "mask_position_error_px": {horizon: _mean(values) for horizon, values in position_errors.items()},
        "mask_position_coverage": {horizon: coverage[horizon] / episodes for horizon in coverage},
    }


def evaluate_planning(model: PocketWorldModel, episodes: int = 20, horizon: int = 16, candidates: int = 256, seed: int = 17) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    imagined_successes = 0
    real_successes = 0
    imagined_distances = []
    real_distances = []
    for episode in range(episodes):
        # Candidate generation is stochastic; seed each episode so a report
        # can be regenerated exactly from its declared seed.
        torch.manual_seed(seed * 100_000 + episode)
        env = PocketWorldEnv(
            agent_start=(float(rng.integers(7, 11)), float(rng.integers(7, 11))),
            goal=(float(rng.integers(16, 21)), float(rng.integers(28, 33))),
        )
        observation, info = env.reset()
        result = random_shooting(model, observation, tuple(info["goal"]), horizon=horizon, candidates=candidates)
        imagined_distances.append(result.imagined_distance)
        imagined_successes += int(result.imagined_distance <= env.goal_radius)
        for action in result.actions:
            _, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                break
        real_distances.append(float(info["distance_to_goal"]))
        real_successes += int(info["distance_to_goal"] <= env.goal_radius)
    return {
        "imagined_success_rate": imagined_successes / episodes,
        "real_success_rate": real_successes / episodes,
        "planning_gap": (imagined_successes - real_successes) / episodes,
        "mean_imagined_final_distance_px": _mean(imagined_distances),
        "mean_real_final_distance_px": _mean(real_distances),
    }


def evaluate_planning_sweep(model: PocketWorldModel, episodes: int = 20, horizons: tuple[int, ...] = (8, 16, 24, 32), candidates: int = 256, seed: int = 17) -> dict[str, dict[str, float]]:
    """Measure the imagined/real planning gap as the planning horizon grows."""
    return {str(horizon): evaluate_planning(model, episodes=episodes, horizon=horizon, candidates=candidates, seed=seed + index) for index, horizon in enumerate(horizons)}


def evaluate_obstacle_planning(
    model: PocketWorldModel,
    episodes: int = 20,
    horizon: int = 40,
    candidates: int = 512,
    seed: int = 71,
    learned_collision: bool = False,
    hybrid_collision: bool = False,
) -> dict[str, dict[str, float]]:
    """Compare unconstrained and wall-aware planning on a single barrier task."""
    rng = np.random.default_rng(seed)
    walls = (Rect(29, 10, 5, 44),)
    reports = {"unconstrained": [], "collision_aware": [], "collision_aware_receding": [], "collision_aware_chunked": [], "collision_aware_route": []}
    if learned_collision:
        reports["collision_aware_learned"] = []
        reports["collision_aware_learned_route"] = []
    if hybrid_collision:
        reports["collision_aware_hybrid"] = []
    for _ in range(episodes):
        start = (float(rng.integers(7, 13)), float(rng.integers(25, 39)))
        goal = (float(rng.integers(51, 57)), float(rng.integers(25, 39)))
        planners = (("unconstrained", False, False), ("collision_aware", True, False))
        if learned_collision:
            planners += (("collision_aware_learned", True, True),)
        if hybrid_collision:
            planners += (("collision_aware_hybrid", True, False),)
        for label, collision_aware, learned_collision_mode in planners:
            env = PocketWorldEnv(walls=walls, agent_start=start, goal=goal)
            observation, info = env.reset()
            result = random_shooting(
                model,
                observation,
                tuple(info["goal"]),
                horizon=horizon,
                candidates=candidates,
                collision_aware=collision_aware,
                learned_collision=learned_collision_mode,
                hybrid_collision=label == "collision_aware_hybrid",
            )
            imagined_success = result.imagined_distance <= env.goal_radius
            for action in result.actions:
                _, _, terminated, truncated, info = env.step(int(action))
                if terminated or truncated:
                    break
            reports[label].append((imagined_success, info["distance_to_goal"] <= env.goal_radius, info["distance_to_goal"]))
        env = PocketWorldEnv(walls=walls, agent_start=start, goal=goal)
        observation, info = env.reset()
        for label, commit_steps, preserve_route in (
            ("collision_aware_receding", 1, False),
            ("collision_aware_chunked", 4, False),
            ("collision_aware_route", 1, True),
        ):
            env = PocketWorldEnv(walls=walls, agent_start=start, goal=goal)
            observation, info = env.reset()
            result = receding_horizon_plan(
                model,
                observation,
                tuple(info["goal"]),
                env.step,
                max_steps=horizon,
                rollout_horizon=min(16, horizon),
                candidates=candidates,
                collision_aware=True,
                commit_steps=commit_steps,
                preserve_route=preserve_route,
            )
            reports[label].append((result.first_plan_distance <= env.goal_radius, result.final_info.get("distance_to_goal", float("inf")) <= env.goal_radius, result.final_info.get("distance_to_goal", float("inf"))))
        if learned_collision:
            env = PocketWorldEnv(walls=walls, agent_start=start, goal=goal)
            observation, info = env.reset()
            result = receding_horizon_plan(
                model,
                observation,
                tuple(info["goal"]),
                env.step,
                max_steps=horizon,
                rollout_horizon=horizon,
                candidates=candidates,
                collision_aware=True,
                preserve_route=True,
                route_tolerance=6.0,
                learned_collision=True,
            )
            reports["collision_aware_learned_route"].append((result.first_plan_distance <= env.goal_radius, result.final_info.get("distance_to_goal", float("inf")) <= env.goal_radius, result.final_info.get("distance_to_goal", float("inf"))))
    return {
        label: {
            "imagined_success_rate": float(np.mean([row[0] for row in rows])),
            "real_success_rate": float(np.mean([row[1] for row in rows])),
            "mean_real_final_distance_px": float(np.mean([row[2] for row in rows])),
        }
        for label, rows in reports.items()
    }


def evaluate_collision_prediction(model: PocketWorldModel, episodes: int = 20, horizon: int = 16, seed: int = 83, ood: bool = False) -> dict[str, float]:
    """Measure learned collision-event probabilities against simulator events."""
    rng = np.random.default_rng(seed)
    probabilities = []
    targets = []
    for _ in range(episodes):
        walls = _variant_walls(rng) if ood else None
        env = PocketWorldEnv(
            walls=walls,
            agent_start=(float(rng.integers(6, 58)), float(rng.integers(6, 58))),
            goal=(float(rng.integers(6, 58)), float(rng.integers(6, 58))),
        )
        observation, _ = env.reset()
        start_observation = observation.copy()
        actions = rng.integers(0, 4, size=horizon, dtype=np.int64)
        actual_collisions = []
        for action in actions:
            next_observation, _, terminated, truncated, info = env.step(int(action))
            actual_collisions.append(float(info["collision"]))
            observation = next_observation
            if terminated or truncated:
                break
        start = torch.from_numpy(start_observation[None]).float() / 255.0
        action_tensor = torch.from_numpy(actions[None])
        predicted = model.imagine_collision_probabilities(start, action_tensor)[0].cpu().numpy()
        probabilities.extend(predicted[:len(actual_collisions)].tolist())
        targets.extend(actual_collisions)
    predicted_labels = np.asarray(probabilities) >= 0.5
    target_array = np.asarray(targets, dtype=bool)
    true_positive = np.logical_and(predicted_labels, target_array).sum()
    return {
        "accuracy": float(np.mean(predicted_labels == target_array)),
        "positive_rate": float(np.mean(target_array)),
        "predicted_positive_rate": float(np.mean(predicted_labels)),
        "precision": float(true_positive / max(1, predicted_labels.sum())),
        "recall": float(true_positive / max(1, target_array.sum())),
    }


def evaluate_temporal_velocity(model: PocketWorldModel, episodes: int = 20, horizon: int = 8, seed: int = 97) -> dict[str, float]:
    """Compare learned temporal velocity representation with finite differences."""
    batch = collect_random_rollouts(
        episodes=episodes,
        horizon=horizon,
        seed=seed,
        sticky_probability=0.75,
        full_state_range=True,
    )
    observations = torch.from_numpy(batch.observations).float() / 255.0
    target = torch.from_numpy(batch.velocities).float()
    learned_errors = []
    finite_difference_errors = []
    model.eval()
    with torch.no_grad():
        for step in range(horizon + 1):
            history = observations[:, max(0, step + 1 - 4):step + 1]
            predicted, _ = model.temporal_velocity_stats(history)
            learned_error = (predicted * 3.0 - target[:, step]).norm(dim=-1)
            learned_errors.extend(learned_error.tolist())
            for episode in range(episodes):
                estimated = estimate_agent_velocity(batch.observations[episode, :step + 1])
                finite_difference_errors.append(float(np.linalg.norm(estimated - target[episode, step].numpy())))
    learned_array = np.asarray(learned_errors, dtype=np.float32)
    finite_array = np.asarray(finite_difference_errors, dtype=np.float32)
    return {
        "learned_velocity_mae_px": float(learned_array.mean()),
        "learned_velocity_rmse_px": float(np.sqrt(np.mean(learned_array ** 2))),
        "finite_difference_velocity_mae_px": float(finite_array.mean()),
        "improvement_over_finite_difference": float(1.0 - learned_array.mean() / max(1e-6, finite_array.mean())),
    }


def evaluate_uncertainty_calibration(
    model: PocketWorldModel,
    episodes: int = 20,
    horizon: int = 8,
    seed: int = 101,
    map_variant: bool = False,
    speed_scale: float = 1.0,
) -> dict[str, object]:
    """Measure empirical coverage of calibrated diagonal Gaussian transitions."""
    batch = collect_random_rollouts(
        episodes=episodes,
        horizon=horizon,
        seed=seed,
        sticky_probability=0.75,
        full_state_range=True,
        map_variant=map_variant,
        agent_speed_scale=speed_scale,
    )
    observations = torch.from_numpy(batch.observations).float() / 255.0
    actions = torch.from_numpy(batch.actions)
    positions = torch.from_numpy(batch.positions / 64.0).float()
    velocities = torch.from_numpy(batch.velocities / 3.0).float().clamp(-1.0, 1.0)
    normalized_states = torch.cat((positions, velocities), dim=-1)
    residuals = []
    scales = []
    model.eval()
    with torch.no_grad():
        for step in range(horizon):
            latent = model.encode(observations[:, step])
            history = observations[:, : step + 1]
            state = model.state_from_history(history)
            observed_velocity = torch.as_tensor(
                np.stack([
                    observable_velocity_from_frames(history[index].cpu().numpy())
                    for index in range(history.shape[0])
                ]),
                dtype=state.dtype,
                device=state.device,
            ) / 3.0
            state = torch.cat(
                (
                    state[..., :2],
                    (0.50 * state[..., 2:] + 0.50 * observed_velocity).clamp(-1.0, 1.0),
                ),
                dim=-1,
            )
            target = normalized_states[:, step + 1]
            mean, std = model.transition_state_stats(latent, state, actions[:, step])
            residuals.append((target - mean).abs())
            scales.append(std)
    residual = torch.cat(residuals, dim=0)
    scale = torch.cat(scales, dim=0).clamp_min(1e-6)
    z_values = {"0.50": 0.6745, "0.80": 1.2816, "0.90": 1.6449, "0.95": 1.9600}
    position_coverage = {}
    velocity_coverage = {}
    for level, z_value in z_values.items():
        covered = residual <= z_value * scale
        position_coverage[level] = float(covered[:, :2].float().mean())
        velocity_coverage[level] = float(covered[:, 2:].float().mean())
    nll = 0.5 * ((residual / scale).square() + 2.0 * scale.log()).mean()
    return {
        "position_coverage": position_coverage,
        "velocity_coverage": velocity_coverage,
        "position_coverage_error_90": abs(position_coverage["0.90"] - 0.90),
        "velocity_coverage_error_90": abs(velocity_coverage["0.90"] - 0.90),
        "mean_interval_width_px_90": float((2.0 * z_values["0.90"] * scale[:, :2] * 64.0).mean()),
        "state_gaussian_nll": float(nll),
        "calibration_scale": model.uncertainty_scale.detach().cpu().tolist(),
    }


def evaluate_uncertainty_calibration_matrix(
    model: PocketWorldModel,
    episodes: int = 20,
    horizon: int = 8,
    seed: int = 101,
) -> dict[str, dict[str, object]]:
    """Evaluate calibration across speed, map, and joint distribution shifts."""
    conditions = {
        "in_distribution": (False, 1.0),
        "ood_speed_slow": (False, 0.8),
        "ood_speed_fast": (False, 1.2),
        "ood_map": (True, 1.0),
        "ood_map_fast": (True, 1.2),
    }
    return {
        label: evaluate_uncertainty_calibration(
            model,
            episodes=episodes,
            horizon=horizon,
            seed=seed + index * 1000,
            map_variant=map_variant,
            speed_scale=speed_scale,
        )
        for index, (label, (map_variant, speed_scale)) in enumerate(conditions.items())
    }


def _transition_shift_scores(model: PocketWorldModel, batch) -> np.ndarray:
    """Collect one mature-window score per observable RGB rollout.

    Speed changes are not identifiable from a single sticky-action frame. The
    detector therefore reports the score after the final response window of
    each episode, matching the minimum history needed by the online monitor.
    """
    scores = []
    for episode in range(batch.observations.shape[0]):
        episode_scores = []
        for step in range(batch.actions.shape[1]):
            history_start = max(0, step + 1 - 16)
            history = batch.observations[episode, history_start:step + 1]
            action_history = batch.actions[episode, history_start:step + 1]
            episode_scores.append(
                predictive_shift_score(
                    model,
                    batch.observations[episode, step],
                    int(batch.actions[episode, step]),
                    batch.observations[episode, step + 1],
                    history,
                    action_history,
                )
            )
        if episode_scores:
            scores.append(episode_scores[-1])
    return np.asarray(scores, dtype=np.float32)


def fit_shift_threshold(
    model: PocketWorldModel,
    episodes: int = 20,
    horizon: int = 8,
    seed: int = 151,
    false_positive_rate: float = 0.05,
) -> dict[str, float]:
    """Fit an ID predictive-innovation threshold for online shift alarms."""
    if not 0.0 < false_positive_rate < 0.5:
        raise ValueError("false_positive_rate must be between 0 and 0.5")
    batch = collect_random_rollouts(
        episodes=episodes,
        horizon=horizon,
        seed=seed,
        sticky_probability=0.75,
        full_state_range=True,
    )
    response_values = []
    friction = float((0.50 + 0.49 * torch.sigmoid(model.friction_logit)).item())
    for episode in range(batch.observations.shape[0]):
        response = estimate_speed_response(
            batch.observations[episode],
            batch.actions[episode],
            friction=friction,
        )
        if np.isfinite(response):
            response_values.append(response)
    if response_values:
        response_array = np.asarray(response_values, dtype=np.float32)
        center = float(np.median(response_array))
        mad = float(np.median(np.abs(response_array - center)))
        model.speed_response_center.fill_(center)
        model.speed_response_scale.fill_(max(0.05, 1.4826 * mad))
    scores = _transition_shift_scores(model, batch)
    quantile = 1.0 - false_positive_rate
    return {
        "threshold": float(np.quantile(scores, quantile)),
        "false_positive_rate": false_positive_rate,
        "quantile": quantile,
        "calibration_samples": int(scores.size),
        "calibration_mean_score": float(scores.mean()),
        "calibration_p95_score": float(np.quantile(scores, 0.95)),
        "speed_response_center": float(model.speed_response_center.item()),
        "speed_response_scale": float(model.speed_response_scale.item()),
        "speed_response_samples": len(response_values),
    }


def _shift_detection_metrics(scores: np.ndarray, threshold: float, shifted: bool) -> dict[str, float | None]:
    triggered = scores >= threshold
    trigger_rate = float(triggered.mean()) if len(triggered) else 0.0
    return {
        "samples": int(scores.size),
        "mean_score": float(scores.mean()) if len(scores) else 0.0,
        "p95_score": float(np.quantile(scores, 0.95)) if len(scores) else 0.0,
        "threshold": float(threshold),
        "trigger_rate": trigger_rate,
        "false_positive_rate": trigger_rate if not shifted else None,
        "detection_rate": trigger_rate if shifted else None,
    }


def _roc_auc(negative: np.ndarray, positive: np.ndarray) -> float:
    """Compute AUROC without adding a scikit-learn dependency."""
    if len(negative) == 0 or len(positive) == 0:
        return 0.5
    comparisons = positive[:, None] - negative[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def evaluate_shift_detection_matrix(
    model: PocketWorldModel,
    episodes: int = 20,
    horizon: int = 8,
    seed: int = 151,
) -> dict[str, object]:
    """Evaluate online shift alarms on speed, map, and joint distribution shifts."""
    threshold_report = fit_shift_threshold(model, episodes, horizon, seed)
    threshold = threshold_report["threshold"]
    conditions = {
        "in_distribution": (False, 1.0, False),
        "ood_speed_slow": (False, 0.8, True),
        "ood_speed_fast": (False, 1.2, True),
        "ood_map": (True, 1.0, True),
        "ood_map_fast": (True, 1.2, True),
    }
    reports: dict[str, dict[str, float | None]] = {}
    id_batch = collect_random_rollouts(
        episodes=episodes,
        horizon=horizon,
        seed=seed + 5000,
        sticky_probability=0.75,
        full_state_range=True,
    )
    id_scores = _transition_shift_scores(model, id_batch)
    reports["in_distribution"] = _shift_detection_metrics(id_scores, threshold, shifted=False)
    for index, (label, (map_variant, speed_scale, shifted)) in enumerate(conditions.items()):
        if label == "in_distribution":
            continue
        batch = collect_random_rollouts(
            episodes=episodes,
            horizon=horizon,
            seed=seed + 6000 + index * 1000,
            sticky_probability=0.75,
            full_state_range=True,
            map_variant=map_variant,
            agent_speed_scale=speed_scale,
        )
        scores = _transition_shift_scores(model, batch)
        report = _shift_detection_metrics(scores, threshold, shifted=shifted)
        report["auroc_vs_in_distribution"] = _roc_auc(id_scores, scores)
        reports[label] = report
    return {"threshold": threshold_report, "conditions": reports}


def evaluate_action_effects(model: PocketWorldModel, repeat: int = 8) -> dict[str, dict[str, list[float] | float]]:
    """Compare predicted versus real displacement for each repeated action."""
    env = PocketWorldEnv(walls=(), agent_start=(32.0, 16.0), goal=(55.0, 55.0))
    observation, info = env.reset()
    start = torch.from_numpy(observation[None]).float() / 255.0
    result = {}
    for action, label in enumerate(("up", "down", "left", "right")):
        actions = torch.full((1, repeat), action, dtype=torch.long)
        predicted = model.imagine_positions(start, actions)[0, -1].cpu().numpy() * 64.0
        actual_env = PocketWorldEnv(walls=(), agent_start=(32.0, 16.0), goal=(55.0, 55.0))
        actual_env.reset()
        for _ in range(repeat):
            actual_env.step(action)
        actual = actual_env.position.copy()
        result[label] = {
            "predicted_position": predicted.tolist(),
            "actual_position": actual.tolist(),
            "position_error_px": float(np.linalg.norm(predicted - actual)),
        }
    return result


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _summarize(reports: list[dict]) -> dict:
    """Recursively summarize numeric report leaves across random seeds."""
    summary = {}
    for key in reports[0]:
        values = [report[key] for report in reports]
        if isinstance(values[0], dict):
            summary[key] = _summarize(values)
        elif isinstance(values[0], list):
            array = np.asarray(values, dtype=float)
            summary[key] = {"mean": array.mean(axis=0).tolist(), "std": array.std(axis=0).tolist()}
        elif values[0] is None:
            summary[key] = None
        else:
            summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-step prediction, OOD generalization, and imagined planning")
    parser.add_argument("checkpoint", nargs="?", default="artifacts/pocketworld.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--seeds", default="11,23,41", help="comma-separated evaluation seeds")
    parser.add_argument("--output", default="artifacts/evaluation.json")
    args = parser.parse_args()
    model = PocketWorldModel()
    payload = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if missing:
        print(f"warning: checkpoint is missing {len(missing)} structured-dynamics keys; use a freshly trained checkpoint for planning")
    if unexpected:
        print(f"warning: checkpoint has {len(unexpected)} legacy keys")
    collision_supervision = bool(payload.get("collision_supervision", False))
    agent_rendering = bool(payload.get("agent_rendering", False))
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    runs = []
    for seed in seeds:
        runs.append({
            "seed": seed,
            "in_distribution": evaluate_prediction(model, episodes=args.episodes, seed=seed),
            "out_of_distribution": evaluate_prediction(model, episodes=args.episodes, seed=seed + 1000, ood=True),
            "planning": evaluate_planning(model, episodes=args.episodes, candidates=args.candidates, seed=seed + 2000),
            "planning_sweep": evaluate_planning_sweep(model, episodes=args.episodes, candidates=args.candidates, seed=seed + 3000),
            "obstacle_planning": evaluate_obstacle_planning(
                model,
                episodes=args.episodes,
                candidates=args.candidates,
                seed=seed + 4000,
                learned_collision=collision_supervision,
                hybrid_collision=collision_supervision,
            ),
            "collision_prediction": {
                "in_distribution": evaluate_collision_prediction(model, episodes=args.episodes, seed=seed + 5000),
                "out_of_distribution": evaluate_collision_prediction(model, episodes=args.episodes, seed=seed + 6000, ood=True),
            } if collision_supervision else None,
            "temporal_velocity": evaluate_temporal_velocity(model, episodes=args.episodes, seed=seed + 9000),
            "uncertainty_calibration": evaluate_uncertainty_calibration_matrix(model, episodes=args.episodes, seed=seed + 10000),
            "shift_detection": evaluate_shift_detection_matrix(model, episodes=args.episodes, seed=seed + 11000),
            "agent_rendering": {
                "in_distribution": evaluate_agent_rendering(model, episodes=args.episodes, seed=seed + 7000),
                "out_of_distribution": evaluate_agent_rendering(model, episodes=args.episodes, seed=seed + 8000, ood=True),
            } if agent_rendering else None,
            "action_effects": evaluate_action_effects(model),
        })
    numeric_runs = [{key: value for key, value in run.items() if key != "seed"} for run in runs]
    report = {"config": {"episodes": args.episodes, "candidates": args.candidates, "seeds": seeds}, "runs": runs, "summary": _summarize(numeric_runs)}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
