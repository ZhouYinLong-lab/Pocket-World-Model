"""Compare ensemble and conformal collision-risk methods on paired routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import PocketWorldEnv
from .evaluate_planners import SINGLE_BARRIER_WALLS, _episode_cases, evaluate_planner_tournament
from .model import PocketWorldModel
from .planner import random_shooting
from .uncertainty import (
    ConformalCollisionRisk,
    PocketWorldEnsemble,
    fit_conformal_upper_bound,
)


def _load_model(checkpoint: str | Path) -> PocketWorldModel:
    payload = torch.load(checkpoint, map_location="cpu")
    model = PocketWorldModel()
    model.load_state_dict(payload["model"], strict=False)
    return model


def _collect_calibration(
    model: PocketWorldModel,
    risk_model: PocketWorldEnsemble,
    seeds: tuple[int, ...],
    episodes: int,
    horizon: int,
    candidates: int,
) -> tuple[np.ndarray, np.ndarray]:
    risks: list[float] = []
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
                collision_model=risk_model,
                observation_history=[observation],
            )
            collision_count = 0
            for action in result.actions:
                _, _, terminated, truncated, info = env.step(int(action))
                collision_count += int(info.get("collision", False))
                if terminated or truncated:
                    break
            risks.append(float(result.imagined_collision_risk))
            labels.append(float(collision_count > 0))
    return np.asarray(risks, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def _coverage(rows: dict[str, dict[str, list[dict[str, float | None]]]], planner: str) -> dict[str, float]:
    selected = [row for seed_rows in rows.values() for row in seed_rows[planner]]
    upper = np.asarray([float(row["imagined_collision_risk"]) for row in selected], dtype=np.float64)
    labels = np.asarray([float(row["collision_count"] > 0) for row in selected], dtype=np.float64)
    return {
        "empirical_upper_coverage": float(np.mean(upper >= labels)),
        "mean_upper_risk": float(upper.mean()),
        "real_collision_rate": float(labels.mean()),
        "routes": float(len(selected)),
    }


def evaluate_uncertainty_methods(
    checkpoint: str | Path,
    ensemble_checkpoints: list[str | Path],
    calibration_seeds: tuple[int, ...] = (101, 103),
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    calibration_episodes: int = 12,
    evaluation_episodes: int = 20,
    horizon: int = 48,
    candidates: int = 256,
    alpha: float = 0.10,
    disagreement_weight: float = 1.0,
) -> dict[str, object]:
    model = _load_model(checkpoint)
    ensemble = PocketWorldEnsemble.from_checkpoints(
        [str(path) for path in ensemble_checkpoints],
        disagreement_weight=disagreement_weight,
    )
    predicted, labels = _collect_calibration(
        model,
        ensemble,
        calibration_seeds,
        calibration_episodes,
        horizon,
        candidates,
    )
    calibration = fit_conformal_upper_bound(predicted, labels, alpha=alpha)
    conformal = ConformalCollisionRisk(ensemble, calibration)
    report = evaluate_planner_tournament(
        model,
        seeds=evaluation_seeds,
        episodes=evaluation_episodes,
        horizon=horizon,
        candidates=candidates,
        scenario="single_barrier",
        planners=("learned_collision", "ensemble_collision", "conformal_collision"),
        collision_models={
            "ensemble_collision": ensemble,
            "conformal_collision": conformal,
        },
        risk_model_metadata={
            "ensemble_checkpoints": [str(path) for path in ensemble_checkpoints],
            "ensemble_members": len(ensemble.members),
            "aggregation": "mean + disagreement_weight * member_std",
            "disagreement_weight": disagreement_weight,
            "conformal_alpha": calibration.alpha,
            "conformal_quantile": calibration.quantile,
            "conformal_samples": calibration.samples,
            "calibration_collision_rate": calibration.collision_rate,
            "calibration_seeds": list(calibration_seeds),
        },
        include_rows=True,
    )
    rows = report["rows"]
    report["risk_calibration"] = {
        "predicted_risk_mean": float(predicted.mean()),
        "collision_rate": float(labels.mean()),
        "conformal": {
            "alpha": calibration.alpha,
            "quantile": calibration.quantile,
            "samples": calibration.samples,
            "coverage_target": 1.0 - calibration.alpha,
        },
        "held_out_coverage": _coverage(rows, "conformal_collision"),
        "held_out_ensemble": _coverage(rows, "ensemble_collision"),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ensemble and conformal collision-risk estimators")
    parser.add_argument("checkpoint", nargs="?", default="artifacts/pocketworld-map-suite-v3-final.pt")
    parser.add_argument(
        "--ensemble-checkpoints",
        default="artifacts/pocketworld-map-suite-v3.pt,artifacts/pocketworld-map-suite-v3-calibrated.pt,artifacts/pocketworld-map-suite-v3-kinematics.pt",
    )
    parser.add_argument("--calibration-seeds", default="101,103")
    parser.add_argument("--evaluation-seeds", default="11,23,41")
    parser.add_argument("--calibration-episodes", type=int, default=12)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--disagreement-weight", type=float, default=1.0)
    parser.add_argument("--output", default="artifacts/evaluation-uncertainty-barrier-v1.json")
    args = parser.parse_args()
    report = evaluate_uncertainty_methods(
        checkpoint=args.checkpoint,
        ensemble_checkpoints=[item.strip() for item in args.ensemble_checkpoints.split(",") if item.strip()],
        calibration_seeds=tuple(int(item) for item in args.calibration_seeds.split(",") if item.strip()),
        evaluation_seeds=tuple(int(item) for item in args.evaluation_seeds.split(",") if item.strip()),
        calibration_episodes=args.calibration_episodes,
        evaluation_episodes=args.evaluation_episodes,
        horizon=args.horizon,
        candidates=args.candidates,
        alpha=args.alpha,
        disagreement_weight=args.disagreement_weight,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

