"""Evaluate observable RGB/A* route-controller variants separately from learned planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .env import PocketWorldEnv
from .evaluate_planners import _episode_cases, _scenario_walls
from .evaluate_route_completion import _load_model
from .planner import route_controller_parameters, route_following_action, route_geometry_allow_diagonal


CONTROLLER_VARIANTS = {
    "cardinal_safe": {
        "clearance_radius": 4,
        "damping": 1.0,
        "lookahead_distance": 3.0,
        "allow_diagonal": False,
    },
    "cardinal_lookahead": {
        "clearance_radius": 4,
        "damping": 0.75,
        "lookahead_distance": 5.0,
        "allow_diagonal": False,
    },
    "tight_lookahead": {
        "clearance_radius": 3,
        "damping": 0.75,
        "lookahead_distance": 5.0,
        "allow_diagonal": False,
    },
    "diagonal_lookahead": {
        "clearance_radius": 4,
        "damping": 0.75,
        "lookahead_distance": 5.0,
        "allow_diagonal": True,
    },
    "adaptive_geometry": {
        "clearance_radius": 4,
        "damping": 0.75,
        "lookahead_distance": 5.0,
        "allow_diagonal": False,
        "adaptive_diagonal": True,
        "diagonal_detour_ratio_threshold": 1.7,
    },
    "adaptive_locked_geometry": {
        "clearance_radius": 4,
        "damping": 0.75,
        "lookahead_distance": 5.0,
        "allow_diagonal": False,
        "adaptive_diagonal": False,
        "diagonal_detour_ratio_threshold": 1.7,
        "lock_geometry": True,
    },
    "adaptive_locked_policy": {
        "clearance_radius": 4,
        "detour_ratio_threshold": 1.7,
        "lock_geometry": True,
        "adaptive_policy": True,
    },
}


def evaluate_route_controller(
    scenarios: tuple[str, ...] = ("single_barrier", "barrier_shifted", "barrier_narrow_gap", "barrier_wide_gap"),
    seeds: tuple[int, ...] = (11, 23, 41),
    episodes: int = 20,
    max_steps: int = 64,
    variants: tuple[str, ...] = tuple(CONTROLLER_VARIANTS),
    agent_speed_scale: float = 1.0,
) -> dict[str, object]:
    report: dict[str, object] = {
        "protocol": {
            "scenarios": list(scenarios),
            "seeds": list(seeds),
            "episodes": episodes,
            "max_steps": max_steps,
            "agent_speed_scale": agent_speed_scale,
            "controller_only": True,
            "geometry_source": "current RGB wall mask + A*",
        },
        "variants": {},
    }
    for variant in variants:
        if variant not in CONTROLLER_VARIANTS:
            raise ValueError(f"unknown controller variant: {variant}")
        rows: list[dict[str, object]] = []
        config = CONTROLLER_VARIANTS[variant]
        for scenario in scenarios:
            walls = _scenario_walls(scenario)
            for seed in seeds:
                for episode, (start, goal) in enumerate(_episode_cases(episodes, seed, scenario)):
                    env = PocketWorldEnv(
                        walls=walls,
                        agent_start=start,
                        goal=goal,
                        agent_speed_scale=agent_speed_scale,
                    )
                    observation, info = env.reset()
                    history = [observation]
                    locked_allow_diagonal = None
                    locked_policy = None
                    if config.get("lock_geometry", False):
                        threshold = float(config.get("detour_ratio_threshold", config.get("diagonal_detour_ratio_threshold", 1.7)))
                        if config.get("adaptive_policy", False):
                            locked_policy = route_controller_parameters(
                                observation,
                                goal,
                                clearance_radius=int(config["clearance_radius"]),
                                detour_ratio_threshold=threshold,
                            )
                            locked_allow_diagonal = bool(locked_policy["allow_diagonal"])
                        else:
                            locked_allow_diagonal = route_geometry_allow_diagonal(
                                observation,
                                goal,
                                clearance_radius=int(config["clearance_radius"]),
                                detour_ratio_threshold=threshold,
                            )
                    actions: list[int] = []
                    collisions = 0
                    remaining: list[float] = []
                    for _ in range(max_steps):
                        controller_config = dict(config)
                        controller_config.pop("lock_geometry", None)
                        controller_config.pop("diagonal_detour_ratio_threshold", None)
                        controller_config.pop("detour_ratio_threshold", None)
                        controller_config.pop("adaptive_policy", None)
                        if locked_allow_diagonal is not None:
                            controller_config["allow_diagonal"] = locked_allow_diagonal
                        if locked_policy is not None:
                            controller_config.update(locked_policy)
                        action, route_remaining = route_following_action(
                            observation,
                            goal,
                            observation_history=history,
                            **controller_config,
                        )
                        remaining.append(route_remaining)
                        observation, _, terminated, truncated, info = env.step(action)
                        history.append(observation)
                        history = history[-16:]
                        actions.append(int(action))
                        collisions += int(info.get("collision", False))
                        if terminated or truncated:
                            break
                    rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "episode": episode,
                            "real_success": float(info["distance_to_goal"] <= env.goal_radius),
                            "final_distance_px": float(info["distance_to_goal"]),
                            "collision_count": collisions,
                            "executed_actions": len(actions),
                            "initial_route_distance_px": remaining[0] if remaining else float("nan"),
                            "final_route_distance_px": remaining[-1] if remaining else float("nan"),
                            "route_regression_px": max(0.0, (remaining[-1] - min(remaining)) if remaining else 0.0),
                        }
                    )
        values = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in (
            "real_success", "final_distance_px", "collision_count", "executed_actions", "route_regression_px"
        )}
        report["variants"][variant] = {
            "config": config,
            "rows": rows,
            "summary": {
                key: {"mean": float(value.mean()), "std": float(value.std())}
                for key, value in values.items()
            },
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RGB/A* route-controller variants")
    parser.add_argument("--scenarios", default="single_barrier,barrier_shifted,barrier_narrow_gap,barrier_wide_gap")
    parser.add_argument("--seeds", default="11,23,41")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--agent-speed-scale", type=float, default=1.0)
    parser.add_argument("--variants", default=",".join(CONTROLLER_VARIANTS))
    parser.add_argument("--output", default="artifacts/evaluation-route-controller-v13.json")
    args = parser.parse_args()
    parse_strings = lambda value: tuple(item.strip() for item in value.split(",") if item.strip())
    report = evaluate_route_controller(
        scenarios=parse_strings(args.scenarios),
        seeds=tuple(int(item) for item in args.seeds.split(",") if item.strip()),
        episodes=args.episodes,
        max_steps=args.max_steps,
        variants=parse_strings(args.variants),
        agent_speed_scale=args.agent_speed_scale,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
