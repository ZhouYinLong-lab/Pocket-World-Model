"""Run a fixed-checkpoint ablation of the RGB inertial route executor.

The route-field checkpoint and holdout tasks are held fixed.  Only the local
MPC horizon, beam width, velocity representation, and robustness envelope are
changed, so this script isolates execution-layer effects from representation
training variance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .evaluate_general_routes import evaluate_general_policy
from .route_field import RouteFieldPolicy


DEFAULT_CONFIGS = (
    {"name": "baseline_rgb_projection", "method": "distance_field_beam_rgb_projection"},
    {"name": "rgb_h2_b8", "horizon": 2, "beam_width": 8, "velocity_source": "rgb", "robust": False},
    {"name": "rgb_h4_b8", "horizon": 4, "beam_width": 8, "velocity_source": "rgb", "robust": False},
    {"name": "rgb_h4_b12", "horizon": 4, "beam_width": 12, "velocity_source": "rgb", "robust": False},
    {"name": "rgb_h6_b12", "horizon": 6, "beam_width": 12, "velocity_source": "rgb", "robust": False},
    {"name": "rgb_h4_b24", "horizon": 4, "beam_width": 24, "velocity_source": "rgb", "robust": False},
    {"name": "fused_h4_b12", "horizon": 4, "beam_width": 12, "velocity_source": "action_fused", "robust": False},
    {"name": "robust_rgb_h4_b12", "horizon": 4, "beam_width": 12, "velocity_source": "rgb", "robust": True},
)


def run_mpc_ablation(
    checkpoint: str | Path,
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    evaluation_episodes: int = 20,
    max_steps: int = 160,
    points: int = 13,
    configs: tuple[dict[str, object], ...] = DEFAULT_CONFIGS,
) -> dict[str, object]:
    policy = RouteFieldPolicy.load(checkpoint)
    results: dict[str, object] = {}
    for config in configs:
        if "method" in config:
            method = str(config["method"])
        else:
            method = (
                "distance_field_beam_robust_mpc"
                if bool(config.get("robust", False))
                else "distance_field_beam_mpc"
            )
        result = evaluate_general_policy(
            policy,
            evaluation_seeds,
            evaluation_episodes,
            max_steps,
            points,
            method,
            mpc_horizon=int(config.get("horizon", 6)),
            mpc_beam_width=int(config.get("beam_width", 4)),
            mpc_velocity_source=str(config.get("velocity_source", "rgb")),
        )
        results[str(config["name"])] = {
            "config": config,
            "evaluation": result,
        }
        for metric, values in result["summary"].items():
            if metric not in {"real_success", "collision_count"}:
                continue
            sample_count = len(result["rows"])
            mean = float(values["mean"])
            if metric == "real_success":
                half_width = 1.96 * math.sqrt(max(mean * (1.0 - mean), 0.0) / sample_count)
            else:
                sample = [float(row[metric]) for row in result["rows"]]
                variance = sum((item - mean) ** 2 for item in sample) / max(sample_count - 1, 1)
                half_width = 1.96 * math.sqrt(variance / sample_count)
            results[str(config["name"])] ["evaluation"]["summary"][metric]["ci95_half_width"] = half_width
    return {
        "protocol": {
            "checkpoint": str(checkpoint),
            "evaluation_seeds": list(evaluation_seeds),
            "evaluation_episodes_per_seed": evaluation_episodes,
            "max_steps": max_steps,
            "route_points": points,
            "student_evaluation_uses_astar": False,
            "task_split": "holdout",
            "configs": [item["name"] for item in configs],
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed-checkpoint RGB MPC configurations")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/general-route-sketch-v19-mpc-distance-field.pt",
    )
    parser.add_argument("--evaluation-seeds", default="11,23,41")
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--points", type=int, default=13)
    parser.add_argument("--output", default="artifacts/evaluation-mpc-ablation-v20.json")
    args = parser.parse_args()
    report = run_mpc_ablation(
        args.checkpoint,
        tuple(int(value) for value in args.evaluation_seeds.split(",") if value.strip()),
        args.evaluation_episodes,
        args.max_steps,
        args.points,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
