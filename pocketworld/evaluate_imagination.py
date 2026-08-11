"""Formal imagined-versus-real planning-gap evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import evaluate_planning_sweep
from .model import PocketWorldModel


def evaluate_imagination_gap(
    model: PocketWorldModel,
    seeds: tuple[int, ...] = (11, 23, 41),
    episodes: int = 50,
    horizons: tuple[int, ...] = (16, 24, 32),
    candidates: int = 256,
) -> dict[str, object]:
    """Compare model-selected imagined success with real execution.

    The same open-space task generator, candidate count, and action sequence
    are used for both sides. A negative gap means the model was conservative;
    a positive gap is the safety-relevant imagination bias.
    """
    runs = []
    for seed in seeds:
        torch.manual_seed(seed)
        runs.append({
            "seed": seed,
            "planning_sweep": evaluate_planning_sweep(
                model,
                episodes=episodes,
                horizons=horizons,
                candidates=candidates,
                seed=seed + 2000,
            ),
        })
    summary = {}
    for horizon in horizons:
        key = str(horizon)
        summary[key] = {}
        for metric in runs[0]["planning_sweep"][key]:
            values = np.asarray([run["planning_sweep"][key][metric] for run in runs], dtype=np.float32)
            summary[key][metric] = {"mean": float(values.mean()), "std": float(values.std())}
    primary_horizon = min((horizon for horizon in horizons if horizon >= 24), default=horizons[-1])
    primary_gap = summary[str(primary_horizon)]["planning_gap"]
    return {
        "config": {
            "seeds": list(seeds),
            "episodes": episodes,
            "horizons": list(horizons),
            "candidates": candidates,
            "primary_horizon": primary_horizon,
            "environment": "open-space planning reference",
        },
        "runs": runs,
        "summary": summary,
        "primary_absolute_gap": {
            "mean": abs(primary_gap["mean"]),
            "std": primary_gap["std"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate imagined-versus-real planning gaps")
    parser.add_argument("checkpoint", nargs="?", default="artifacts/pocketworld.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--horizons", default="16,24,32")
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--seeds", default="11,23,41")
    parser.add_argument("--output", default="artifacts/evaluation-imagination-gap.json")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    model = PocketWorldModel()
    model.load_state_dict(payload["model"], strict=False)
    horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    report = evaluate_imagination_gap(
        model,
        seeds=seeds,
        episodes=args.episodes,
        horizons=horizons,
        candidates=args.candidates,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
