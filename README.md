# PocketWorld

> A tiny and observable world model that learns 2D dynamics, imagines future trajectories, and plans through its learned environment.

PocketWorld is a deliberately small world-model laboratory. A deterministic 64×64 simulator produces RGB transitions, a CNN encoder/dynamics/decoder learns one-step prediction, and a random-shooting planner searches in the model's imagined future.

## The experiment

The central question is: **how far can a tiny model imagine before error breaks planning?**

The repository keeps the research loop inspectable:

- `pocketworld/env.py` — Gymnasium environment with inertia, collision, walls, and goal.
- `pocketworld/model.py` — CNN encoder + GRU image dynamics + CNN decoder, plus a supervised position/velocity state path with learned structured kinematics for planning.
- `pocketworld/data.py` — reproducible random-policy transition collection.
- `pocketworld/planner.py` — model rollout and random-shooting planning utilities.
- `pocketworld/train.py` — minimal one-step image-prediction training loop.
- `src/` — browser demo with side-by-side real/model rollouts and planning controls.

## Run the research core

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,export]"
pytest
python -m pocketworld.train --epochs 5 --episodes 100 --unroll-horizon 8
python -m pocketworld.evaluate artifacts/pocketworld.pt --episodes 20 --seeds 11,23,41
```

The checkpoint is written to `artifacts/pocketworld.pt`.
The evaluation command writes `artifacts/evaluation.json` with image error, decoded-position error, latent-position error, OOD generalization, and imagined/real planning success across 8/16/24/32-step planning horizons.

For a larger run, use 1,000 training trajectories and three evaluation seeds:

```bash
python -m pocketworld.train --epochs 12 --episodes 1000 --validation-episodes 200 --batch-size 32 --unroll-horizon 16 --output artifacts/pocketworld-large.pt
python -m pocketworld.evaluate artifacts/pocketworld-large.pt --episodes 50 --candidates 1024 --seeds 11,23,41 --output artifacts/evaluation-large.json
```

For the expanded state-coverage run used by the planning diagnostics:

```bash
python -m pocketworld.train --epochs 10 --episodes 1000 --validation-episodes 200 --batch-size 32 --unroll-horizon 16 --sticky-probability 0.75 --full-state-range --output artifacts/pocketworld-structured-wide.pt
python -m pocketworld.evaluate artifacts/pocketworld-structured-wide.pt --episodes 50 --candidates 1024 --seeds 11,23,41 --output artifacts/evaluation-structured-wide.json
```

The report contains both per-seed runs and recursive mean/std summaries so planning gaps are not based on one lucky rollout.

The latest large-scale results and the remaining gap are recorded in [the evaluation report](docs/evaluation-2026-08.md). The main result is 98% imagined / 96% real success at 16 planning steps, with the real-vs-imagined gap reaching 7 percentage points at 24–32 steps.

The compact planning state uses a transparent kinematic prior: action directions are fixed by the four-action environment, while acceleration, friction, and speed limit are learned from rollout state supervision. This prevents the planner state from collapsing to an action-insensitive average and makes the action-effect diagnostic interpretable.

To create a browser-loadable one-step model after training:

```bash
python -m pocketworld.export_onnx --output public/pocketworld.onnx
```

## Run the interactive demo

```bash
npm install
npm run dev
```

The frontend uses a pinned, stable Vite 5 toolchain so the demo does not depend on Vite 8's platform-specific Rolldown bindings.

The browser demo is intentionally self-contained so it can be deployed as a static site. It mirrors the simulator dynamics in the browser and makes model drift visible immediately; the Python checkpoint and ONNX export can be wired into the runtime for a fully learned browser rollout.

## Planned evaluation matrix

1. Compare 1/5/10/20-step image and position error.
2. Hold out wall, start, goal, and speed variants to measure distribution shift.
3. Compare imagined versus real planning success across planning horizons.

The first version intentionally avoids VAE, Transformer, diffusion, multi-agent worlds, and complex physics so the relationship between prediction error and planning failure stays visible.
