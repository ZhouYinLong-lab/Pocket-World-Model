# PocketWorld

> A tiny and observable world model that learns 2D dynamics, imagines future trajectories, and plans through its learned environment.

PocketWorld is a deliberately small world-model laboratory. A deterministic 64×64 simulator produces RGB transitions, a CNN encoder/dynamics/decoder learns one-step prediction, and a random-shooting planner searches in the model's imagined future.

## The experiment

The central question is: **how far can a tiny model imagine before error breaks planning?**

The repository keeps the research loop inspectable:

- `pocketworld/env.py` — Gymnasium environment with inertia, collision, walls, and goal.
- `pocketworld/model.py` — CNN encoder + action embedding + GRU dynamics + CNN decoder.
- `pocketworld/data.py` — reproducible random-policy transition collection.
- `pocketworld/planner.py` — model rollout and random-shooting planning utilities.
- `pocketworld/train.py` — minimal one-step image-prediction training loop.
- `src/` — browser demo with side-by-side real/model rollouts and planning controls.

## Run the research core

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
python -m pocketworld.train --epochs 5 --episodes 100
python -m pocketworld.evaluate artifacts/pocketworld.pt --episodes 20
```

The checkpoint is written to `artifacts/pocketworld.pt`.
The evaluation command writes `artifacts/evaluation.json` with in-distribution and out-of-distribution multi-step error plus imagined/real planning success.

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
