"""Reproducible maturity-gate evaluation for PocketWorld."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import evaluate_shift_detection_matrix, evaluate_uncertainty_calibration_matrix
from .model import PocketWorldModel


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def evaluate_maturity(
    model: PocketWorldModel,
    seeds: tuple[int, ...] = (11, 23, 41),
    episodes: int = 50,
    horizon: int = 8,
) -> dict[str, object]:
    """Measure calibration and OOD shift detection over independent seeds."""

    runs = []
    for seed in seeds:
        torch.manual_seed(seed)
        runs.append(
            {
                "seed": seed,
                "uncertainty_calibration": evaluate_uncertainty_calibration_matrix(
                    model, episodes=episodes, horizon=horizon, seed=seed + 10_000
                ),
                "shift_detection": evaluate_shift_detection_matrix(
                    model, episodes=episodes, horizon=horizon, seed=seed + 20_000
                ),
            }
        )

    calibration_summary = {}
    for condition in runs[0]["uncertainty_calibration"]:
        calibration_summary[condition] = {
            "position_coverage_90": _mean_std([
                run["uncertainty_calibration"][condition]["position_coverage"]["0.90"]
                for run in runs
            ]),
            "velocity_coverage_90": _mean_std([
                run["uncertainty_calibration"][condition]["velocity_coverage"]["0.90"]
                for run in runs
            ]),
        }
    shift_summary = {}
    for condition, report in runs[0]["shift_detection"]["conditions"].items():
        shift_summary[condition] = {
            "auroc_vs_in_distribution": _mean_std([
                run["shift_detection"]["conditions"][condition].get("auroc_vs_in_distribution", 0.5)
                for run in runs
            ]),
            "trigger_rate": _mean_std([
                run["shift_detection"]["conditions"][condition]["trigger_rate"]
                for run in runs
            ]),
        }
    return {
        "config": {"seeds": list(seeds), "episodes": episodes, "horizon": horizon},
        "runs": runs,
        "summary": {
            "uncertainty_calibration": calibration_summary,
            "shift_detection": shift_summary,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PocketWorld maturity gates")
    parser.add_argument("checkpoint", nargs="?", default="artifacts/pocketworld.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seeds", default="11,23,41")
    parser.add_argument("--output", default="artifacts/evaluation-maturity.json")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = PocketWorldModel()
    model.load_state_dict(payload["model"], strict=False)
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    report = evaluate_maturity(model, seeds=seeds, episodes=args.episodes, horizon=args.horizon)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
