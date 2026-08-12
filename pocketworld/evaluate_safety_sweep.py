"""Run reproducible soft-penalty and OOD safety-method sweeps.

The sweep keeps the dynamics checkpoint and route predictor fixed. Each row
is one independently reported environment condition, so weight selection on
the single-barrier protocol cannot be mistaken for OOD validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate_safety_methods import evaluate_safety_methods


DEFAULT_SCENARIOS = (
    "single_barrier",
    "barrier_shifted",
    "barrier_narrow_gap",
    "barrier_wide_gap",
)


def run_safety_sweep(
    checkpoint: str | Path,
    route_predictor: str | Path,
    penalties: tuple[float, ...] = (0.0, 8.0, 16.0, 32.0, 64.0, 128.0),
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS,
    speed_scales: tuple[float, ...] = (1.0,),
    evaluation_seeds: tuple[int, ...] = (11, 23, 41),
    evaluation_episodes: int = 20,
    candidates: int = 256,
    horizon: int = 48,
    resume_from: str | Path | None = None,
    max_conditions: int | None = None,
) -> dict[str, object]:
    """Evaluate a Cartesian product of penalty, map, and speed conditions."""
    rows: list[dict[str, object]] = []
    condition_index = 0
    if resume_from is not None and Path(resume_from).exists():
        previous = json.loads(Path(resume_from).read_text(encoding="utf-8"))
        rows.extend(previous.get("rows", []))
    completed_keys = {
        (row["scenario"], float(row["agent_speed_scale"]), float(row["soft_rgb_penalty"]))
        for row in rows
    }
    for scenario in scenarios:
        for speed_scale in speed_scales:
            for penalty in penalties:
                key = (scenario, float(speed_scale), float(penalty))
                if key in completed_keys:
                    continue
                if max_conditions is not None and condition_index >= max_conditions:
                    break
                condition_index += 1
                report = evaluate_safety_methods(
                    checkpoint=checkpoint,
                    route_predictor=route_predictor,
                    evaluation_seeds=evaluation_seeds,
                    evaluation_episodes=evaluation_episodes,
                    candidates=candidates,
                    horizon=horizon,
                    scenario=scenario,
                    agent_speed_scale=speed_scale,
                    soft_rgb_penalty=penalty,
                    planners=("learned_collision", "route_completion_soft"),
                )
                summary = report["summary"]
                soft = summary["route_completion_soft"]
                rows.append(
                    {
                        "scenario": scenario,
                        "agent_speed_scale": speed_scale,
                        "soft_rgb_penalty": penalty,
                        "real_success": soft["real_success"],
                        "collision_count": soft["collision_count"],
                        "rgb_route_collision": soft["rgb_route_collision"],
                        "estimated_model_queries": soft["estimated_model_queries"],
                        "paired_soft_comparison": report["paired_soft_comparison"],
                        "baseline_real_success": summary["learned_collision"]["real_success"],
                        "baseline_collision_count": summary["learned_collision"]["collision_count"],
                    }
                )
                if resume_from is not None:
                    destination = Path(resume_from)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(
                        json.dumps(
                            {
                                "config": {
                                    "checkpoint": str(checkpoint),
                                    "route_predictor": str(route_predictor),
                                    "penalties": list(penalties),
                                    "scenarios": list(scenarios),
                                    "speed_scales": list(speed_scales),
                                    "evaluation_seeds": list(evaluation_seeds),
                                    "evaluation_episodes": evaluation_episodes,
                                    "candidates": candidates,
                                    "horizon": horizon,
                                },
                                "rows": rows,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
            if max_conditions is not None and condition_index >= max_conditions:
                break
        if max_conditions is not None and condition_index >= max_conditions:
            break
    return {
        "config": {
            "checkpoint": str(checkpoint),
            "route_predictor": str(route_predictor),
            "penalties": list(penalties),
            "scenarios": list(scenarios),
            "speed_scales": list(speed_scales),
            "evaluation_seeds": list(evaluation_seeds),
            "evaluation_episodes": evaluation_episodes,
            "candidates": candidates,
            "horizon": horizon,
            "selection_rule": "soft penalty is selected only from single_barrier rows; OOD rows remain held out",
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep route safety penalties and OOD conditions")
    parser.add_argument("checkpoint", nargs="?", default="artifacts/pocketworld-map-suite-v3-final.pt")
    parser.add_argument("--route-predictor", default="artifacts/route-completion-v4-safety-methods.pt")
    parser.add_argument("--penalties", default="0,8,16,32,64,128")
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--speed-scales", default="1.0")
    parser.add_argument("--evaluation-seeds", default="11,23,41")
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-conditions", type=int, default=None)
    parser.add_argument("--output", default="artifacts/evaluation-safety-sweep.json")
    args = parser.parse_args()
    report = run_safety_sweep(
        checkpoint=args.checkpoint,
        route_predictor=args.route_predictor,
        penalties=tuple(float(item) for item in args.penalties.split(",") if item.strip()),
        scenarios=tuple(item.strip() for item in args.scenarios.split(",") if item.strip()),
        speed_scales=tuple(float(item) for item in args.speed_scales.split(",") if item.strip()),
        evaluation_seeds=tuple(int(item) for item in args.evaluation_seeds.split(",") if item.strip()),
        evaluation_episodes=args.evaluation_episodes,
        candidates=args.candidates,
        horizon=args.horizon,
        resume_from=args.resume_from,
        max_conditions=args.max_conditions,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
