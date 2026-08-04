from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .env import PocketWorldEnv, Rect
from .evaluate import _summarize
from .model import PocketWorldModel
from .planner import random_shooting, receding_horizon_plan


WALLS = (Rect(29, 10, 5, 44),)


def _episode_cases(episodes: int, seed: int) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    return [
        (
            (float(rng.integers(7, 13)), float(rng.integers(25, 39))),
            (float(rng.integers(51, 57)), float(rng.integers(25, 39))),
        )
        for _ in range(episodes)
    ]


def _open_loop_episode(
    model: PocketWorldModel,
    start: tuple[float, float],
    goal: tuple[float, float],
    horizon: int,
    candidates: int,
    uncertainty_radius_px: float = 0.0,
    uncertainty_growth_px: float = 0.0,
) -> dict[str, float]:
    env = PocketWorldEnv(walls=WALLS, agent_start=start, goal=goal)
    observation, info = env.reset()
    result = random_shooting(
        model,
        observation,
        goal,
        horizon=horizon,
        candidates=candidates,
        collision_aware=True,
        learned_collision=True,
        uncertainty_radius_px=uncertainty_radius_px,
        uncertainty_growth_px=uncertainty_growth_px,
    )
    collision_count = 0
    for action in result.actions:
        _, _, terminated, truncated, info = env.step(int(action))
        collision_count += int(info["collision"])
        if terminated or truncated:
            break
    return {
        "imagined_success": float(result.imagined_distance <= env.goal_radius),
        "real_success": float(info["distance_to_goal"] <= env.goal_radius),
        "real_final_distance_px": float(info["distance_to_goal"]),
        "collision_count": float(collision_count),
        "executed_actions": float(min(len(result.actions), env.steps)),
        "imagined_collision_risk": result.imagined_collision_risk,
    }


def _closed_loop_episode(
    model: PocketWorldModel,
    start: tuple[float, float],
    goal: tuple[float, float],
    horizon: int,
    candidates: int,
    uncertainty_radius_px: float = 0.0,
    uncertainty_growth_px: float = 0.0,
    use_learned_velocity: bool = False,
    probabilistic_uncertainty: bool = False,
) -> dict[str, float]:
    env = PocketWorldEnv(walls=WALLS, agent_start=start, goal=goal)
    observation, _ = env.reset()
    result = receding_horizon_plan(
        model,
        observation,
        goal,
        env.step,
        max_steps=horizon,
        rollout_horizon=horizon,
        candidates=candidates,
        collision_aware=True,
        preserve_route=True,
        route_tolerance=6.0,
        learned_collision=True,
        use_history_velocity=True,
        use_learned_velocity=use_learned_velocity,
        uncertainty_radius_px=uncertainty_radius_px,
        uncertainty_growth_px=uncertainty_growth_px,
        probabilistic_uncertainty=probabilistic_uncertainty,
        uncertainty_samples=16,
    )
    distance = float(result.final_info.get("distance_to_goal", float("inf")))
    return {
        "imagined_success": float(result.first_plan_distance <= env.goal_radius),
        "real_success": float(distance <= env.goal_radius),
        "real_final_distance_px": distance,
        "collision_count": float(result.collision_count),
        "executed_actions": float(len(result.actions)),
        "replans": float(result.replans),
    }


def evaluate_seed(
    model: PocketWorldModel,
    episodes: int,
    horizon: int,
    candidates: int,
    seed: int,
    only: str | None = None,
    include_new: bool = True,
) -> dict[str, dict[str, float]]:
    if only not in (None, "learned_velocity_probabilistic_closed"):
        raise ValueError(f"unsupported planner variant: {only}")
    if only == "learned_velocity_probabilistic_closed" and not include_new:
        raise ValueError("the checkpoint does not contain the learned temporal/probabilistic planner")
    labels = (only,) if only is not None else (
        "point_open",
        "uncertainty_open",
        "history_closed",
        "history_uncertainty_closed",
    ) + (("learned_velocity_probabilistic_closed",) if include_new else ())
    rows: dict[str, list[dict[str, float]]] = {label: [] for label in labels}
    for episode, (start, goal) in enumerate(_episode_cases(episodes, seed)):
        episode_seed = seed * 100_000 + episode
        if only == "learned_velocity_probabilistic_closed":
            torch.manual_seed(episode_seed)
            rows[only].append(
                _closed_loop_episode(
                    model,
                    start,
                    goal,
                    horizon,
                    candidates,
                    use_learned_velocity=True,
                    probabilistic_uncertainty=True,
                )
            )
            continue
        torch.manual_seed(episode_seed)
        rows["point_open"].append(_open_loop_episode(model, start, goal, horizon, candidates))
        torch.manual_seed(episode_seed)
        rows["uncertainty_open"].append(
            _open_loop_episode(model, start, goal, horizon, candidates, 0.5, 0.05)
        )
        torch.manual_seed(episode_seed)
        rows["history_closed"].append(_closed_loop_episode(model, start, goal, horizon, candidates))
        torch.manual_seed(episode_seed)
        rows["history_uncertainty_closed"].append(
            _closed_loop_episode(model, start, goal, horizon, candidates, 0.25, 0.025)
        )
        if include_new:
            torch.manual_seed(episode_seed)
            rows["learned_velocity_probabilistic_closed"].append(
                _closed_loop_episode(
                    model,
                    start,
                    goal,
                    horizon,
                    candidates,
                    use_learned_velocity=True,
                    probabilistic_uncertainty=True,
                )
            )
    return {
        label: {
            metric: float(np.mean([row[metric] for row in variant_rows]))
            for metric in variant_rows[0]
        }
        for label, variant_rows in rows.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate learned collision planning with history and uncertainty")
    parser.add_argument("checkpoint", nargs="?", default="artifacts/pocketworld-collision-v5.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--candidates", type=int, default=1024)
    parser.add_argument("--seeds", default="71,83,97")
    parser.add_argument("--only", choices=("learned_velocity_probabilistic_closed",), default=None, help="run one planner variant without the other comparison baselines")
    parser.add_argument("--output", default="artifacts/evaluation-collision-v3.json")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    model = PocketWorldModel()
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    has_temporal_probability = not missing and not unexpected
    if not has_temporal_probability:
        print(
            "warning: checkpoint predates the temporal/probabilistic heads; "
            "running the four legacy collision variants only"
        )
    if args.only == "learned_velocity_probabilistic_closed" and not has_temporal_probability:
        raise RuntimeError("--only learned_velocity_probabilistic_closed requires a temporal/probabilistic checkpoint")
    model.eval()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    runs = []
    for seed in seeds:
        report = evaluate_seed(
            model,
            args.episodes,
            args.horizon,
            args.candidates,
            seed,
            only=args.only,
            include_new=has_temporal_probability,
        )
        runs.append({"seed": seed, **report})
        print(json.dumps(runs[-1], indent=2))
    summary = _summarize([{key: value for key, value in run.items() if key != "seed"} for run in runs])
    report = {
        "config": {
            "checkpoint": args.checkpoint,
            "episodes": args.episodes,
            "horizon": args.horizon,
            "candidates": args.candidates,
            "seeds": seeds,
            "only": args.only,
        },
        "runs": runs,
        "summary": summary,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(destination), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
