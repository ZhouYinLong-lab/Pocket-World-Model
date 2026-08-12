"""Compare route-level safety mechanisms with a fixed route predictor.

This entry point deliberately separates predictor training from planner
selection. It is useful for safety ablations because the dynamics checkpoint,
route predictor, paired tasks, and candidate budget can remain fixed while
only the safety mechanism changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .evaluate_planners import evaluate_planner_tournament
from .evaluate_route_completion import _load_model
from .route_completion import ROUTE_FEATURE_NAMES, RouteCompletionPredictor


SAFETY_METHOD_PLANNERS = (
    "learned_collision",
    "route_completion",
    "route_completion_safe_gate",
    "route_completion_rgb_only",
    "route_completion_soft",
    "route_completion_mpc",
    "route_aware_hybrid",
)


def evaluate_safety_methods(
    checkpoint: str | Path,
    route_predictor: str | Path,
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    evaluation_episodes: int = 20,
    candidates: int = 256,
    horizon: int = 48,
    scenario: str = "single_barrier",
    agent_speed_scale: float = 1.0,
    soft_rgb_penalty: float = 64.0,
    planners: tuple[str, ...] = SAFETY_METHOD_PLANNERS,
) -> dict[str, object]:
    """Run a fixed-predictor safety-method tournament."""
    model = _load_model(checkpoint)
    predictor = RouteCompletionPredictor.load(route_predictor)
    report = evaluate_planner_tournament(
        model,
        seeds=evaluation_seeds,
        episodes=evaluation_episodes,
        horizon=horizon,
        candidates=candidates,
        scenario=scenario,
        agent_speed_scale=agent_speed_scale,
        soft_rgb_penalty=soft_rgb_penalty,
        planners=planners,
        route_models={
            "route_completion": predictor,
            "route_completion_safe_gate": predictor,
            "route_completion_rgb_only": predictor,
            "route_completion_soft": predictor,
            "route_completion_mpc": predictor,
        },
        risk_model_metadata={
            "protocol": "fixed route predictor safety-method comparison",
            "route_predictor": str(route_predictor),
            "route_features": list(ROUTE_FEATURE_NAMES),
            "scenario": scenario,
            "agent_speed_scale": agent_speed_scale,
            "soft_rgb_penalty": soft_rgb_penalty,
        },
        include_rows=True,
    )
    rows = report["rows"]
    baseline = "learned_collision"
    for planner, key in (
        ("route_completion", "paired_route_comparison"),
        ("route_completion_safe_gate", "paired_safe_gate_comparison"),
        ("route_completion_rgb_only", "paired_rgb_only_comparison"),
        ("route_completion_soft", "paired_soft_comparison"),
        ("route_completion_mpc", "paired_mpc_comparison"),
    ):
        if planner not in planners:
            continue
        report[key] = _paired_success_delta(rows, baseline, planner)
    return report


def _paired_success_delta(
    rows: dict[str, dict[str, list[dict[str, float | None]]]],
    baseline: str,
    challenger: str,
) -> dict[str, object]:
    seed_deltas: list[float] = []
    wins = losses = ties = 0
    for seed_rows in rows.values():
        baseline_rows = seed_rows[baseline]
        challenger_rows = seed_rows[challenger]
        deltas = [
            float(challenger_row["real_success"]) - float(baseline_row["real_success"])
            for baseline_row, challenger_row in zip(baseline_rows, challenger_rows)
        ]
        seed_deltas.append(float(sum(deltas) / max(1, len(deltas)) * 100.0))
        wins += sum(delta > 0 for delta in deltas)
        losses += sum(delta < 0 for delta in deltas)
        ties += sum(delta == 0 for delta in deltas)
    all_deltas = [value for value in seed_deltas]
    return {
        "baseline": baseline,
        "challenger": challenger,
        "episodes": sum(len(seed_rows[baseline]) for seed_rows in rows.values()),
        "mean_delta_pp": float(sum(all_deltas) / max(1, len(all_deltas))),
        "per_seed_mean_delta_pp": seed_deltas,
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(ties),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed-predictor route safety mechanisms")
    parser.add_argument("checkpoint", nargs="?", default="artifacts/pocketworld-map-suite-v3-final.pt")
    parser.add_argument("--route-predictor", default="artifacts/route-completion-v3-safe-gate.pt")
    parser.add_argument("--evaluation-seeds", default="11,23,41")
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument(
        "--scenario",
        choices=("single_barrier", "barrier_shifted", "barrier_narrow_gap", "barrier_wide_gap"),
        default="single_barrier",
    )
    parser.add_argument("--agent-speed-scale", type=float, default=1.0)
    parser.add_argument("--soft-rgb-penalty", type=float, default=64.0)
    parser.add_argument("--output", default="artifacts/evaluation-safety-methods.json")
    args = parser.parse_args()
    report = evaluate_safety_methods(
        checkpoint=args.checkpoint,
        route_predictor=args.route_predictor,
        evaluation_seeds=tuple(int(item) for item in args.evaluation_seeds.split(",") if item.strip()),
        evaluation_episodes=args.evaluation_episodes,
        candidates=args.candidates,
        horizon=args.horizon,
        scenario=args.scenario,
        agent_speed_scale=args.agent_speed_scale,
        soft_rgb_penalty=args.soft_rgb_penalty,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
