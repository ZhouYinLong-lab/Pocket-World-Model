# PocketWorld

<p align="center">
  <img src="docs/assets/pocketworld-hero.svg" alt="PocketWorld — a tiny observable world model laboratory" width="100%" />
</p>

<p align="center">
  <a href="https://github.com/ZhouYinLong-lab/Pocket-World-Model/actions/workflows/ci.yml"><img alt="PocketWorld CI" src="https://github.com/ZhouYinLong-lab/Pocket-World-Model/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://www.python.org/"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="https://gymnasium.farama.org/"><img alt="Gymnasium" src="https://img.shields.io/badge/Gymnasium-custom%20environment-0081A5"></a>
  <a href="LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-F7BE45.svg"></a>
  <img alt="Tests" src="https://img.shields.io/badge/tests-35%20passed-419400">
  <img alt="Coverage" src="https://img.shields.io/badge/coverage-76%25-69A94E">
</p>

## Why PocketWorld?

PocketWorld was inspired by the central idea behind [World Models](https://arxiv.org/abs/1803.10122): an agent can learn an internal model of its environment and use that model to reason about the future. It also follows the model-based reinforcement learning direction explored by [PlaNet](https://arxiv.org/abs/1811.04551) and [DreamerV3](https://arxiv.org/abs/2301.04104).

The project asks a deliberately concrete question:

> **How far can a tiny learned world model imagine before prediction error breaks planning?**

The motivation is that a one-step prediction can look convincing while a long imagined trajectory is already unsafe. PocketWorld therefore keeps both sides visible: the real simulator and the model's imagined future run side by side, and every plan is executed again in the real environment so the imagination gap can be measured rather than hidden behind a single reward number.

## What it does

PocketWorld is a small, reproducible world-model laboratory. A deterministic 64×64 RGB simulator contains an agent, inertia, walls, collisions, and a goal. From automatically collected interaction data, the model learns to predict the next observation and structured kinematics. A planner then rolls out many candidate action sequences inside the learned model, scores goal distance and collision risk, and executes the selected plan in the real simulator.

```text
real simulator → RGB/history + action → learned dynamics → imagined rollouts
       ↑                                                        ↓
       └──────────── execute, re-plan, and compare with reality ┘
```

## What problem it solves

This repository turns the vague claim “the model understands the environment” into testable failure modes:

- **Prediction versus control:** does a model that predicts the next frame also support useful multi-step planning?
- **Imagination bias:** when do imagined success and real success diverge?
- **Partial observability:** can recent RGB history recover velocity that is not visible in one frame?
- **Safety under model error:** can learned collision risk and a horizon-aware uncertainty boundary reduce fragile plans?
- **Reproducibility:** do the conclusions survive held-out maps, changed speeds, negative ablations, and three evaluation seeds?

The result is not a claim of general intelligence or state-of-the-art control. It is an observable testbed for understanding how world-model errors propagate from pixels to state estimates, collision decisions, action sequences, and real failures.

## What is implemented

- CNN encoder + GRU image dynamics + decoder, with supervised structured position and velocity prediction.
- State-conditioned RGB agent rendering so the learned trajectory remains visually inspectable even when a small decoded agent becomes blurry.
- Random-shooting and map-agnostic waypoint planning inside the learned model.
- Learned wall-relative collision-event prediction, history-based velocity initialization, closed-loop replanning, and a horizon-aware uncertainty risk boundary.
- Side-by-side real/imagination playback, ONNX export, browser inference, OOD evaluation, and machine-readable multi-seed reports.

## How it differs from related projects

| Project | Primary focus | PocketWorld's difference |
|---|---|---|
| [World Models](https://arxiv.org/abs/1803.10122) | Learn a compact world representation and train a controller in its hallucinated environment. | Keeps the same core intuition, but focuses on exposing prediction-to-planning failure in a tiny, inspectable setting. |
| [PlaNet](https://arxiv.org/abs/1811.04551) | Stochastic latent dynamics, reward prediction, and online latent planning from pixels. | Uses a simpler deterministic model and discrete actions, prioritizing transparent RGB/state/collision diagnostics over general control capacity. |
| [DreamerV3](https://arxiv.org/abs/2301.04104) | Train actor-critic agents from imagined trajectories across many tasks. | Does not aim to reproduce full RL training; it isolates the learned-model and planner interface in one controlled environment. |
| [TD-MPC2](https://arxiv.org/abs/2310.16828) | Large-scale decoder-free latent control for continuous domains. | Deliberately retains RGB decoding because seeing what the model imagines is part of the experiment, not just an implementation detail. |

In short, PocketWorld is best viewed as a **microscope for world-model reliability**: small enough to reproduce, visual enough to understand, and strict enough to compare imagined plans with real execution.

<p align="center">
  <img src="docs/assets/pocketworld-demo.gif" alt="PocketWorld real simulator and model imagination running side by side" width="900" />
</p>

<p align="center"><sub>The same action sequence unfolds in the real simulator and the model's imagination while drift accumulates.</sub></p>

## The experiment

The central question is: **how far can a tiny model imagine before error breaks planning?**

The repository keeps the research loop inspectable:

- `pocketworld/env.py` — Gymnasium environment with inertia, collision, walls, and goal.
- `pocketworld/model.py` — CNN encoder + GRU image dynamics + CNN decoder, plus supervised structured kinematics, collision heads, and state-conditioned RGB agent composition.
- `pocketworld/data.py` — reproducible random-policy transition collection.
- `pocketworld/planner.py` — model rollout and random-shooting planning utilities.
- `pocketworld/train.py` — minimal one-step image-prediction training loop.
- `src/` — browser demo with side-by-side real/model rollouts and planning controls.

<p align="center">
  <img src="docs/assets/pocketworld-architecture.svg" alt="PocketWorld encoder, dynamics, structured state, decoder, and planner architecture" width="100%" />
</p>

## Results at a glance

<p align="center">
  <img src="docs/assets/pocketworld-results.svg" alt="PocketWorld planning success, imagination gap, collision failure, and agent position error" width="100%" />
</p>

The main large-scale checkpoint reaches **98% imagined / 96% real success** at 16 planning steps. At 24–32 steps, imagined success stays at 100% while real success falls to 93%—a visible, measurable imagination gap. On the 150-episode, three-seed barrier validation, the point-estimate learned planner reaches 64% real success; history-aware uncertainty planning raises this to **76.7% ± 2.5pp** and reduces mean collisions by about 70%.

The full methodology, per-seed statistics, OOD results, and negative ablations are recorded in [the evaluation report](docs/evaluation-2026-08.md).

## Download a trained model

The `v0.1.0` release publishes the final renderer checkpoint and browser-ready ONNX graph outside Git history:

- [`pocketworld-renderer-v5.pt`](https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/download/v0.1.0/pocketworld-renderer-v5.pt) — PyTorch training/evaluation checkpoint.
- [`pocketworld-renderer-v5.onnx`](https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/download/v0.1.0/pocketworld-renderer-v5.onnx) — one-step RGB and structured-position inference.

The `v0.2.0` learned-collision milestone adds the focused barrier checkpoint and its compatible ONNX graph:

- [`pocketworld-collision-v5.pt`](https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/download/v0.2.0/pocketworld-collision-v5.pt) — collision curriculum, learned risk head, and calibrated kinematics.
- [`pocketworld-collision-v5.onnx`](https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/download/v0.2.0/pocketworld-collision-v5.onnx) — browser-compatible RGB and position outputs from the same checkpoint.

Only load checkpoints from the official release; PyTorch checkpoint files must be treated as trusted executable artifacts.

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

To train the collision-event ablation with single-barrier maps mixed into the data:

```bash
python -m pocketworld.train --epochs 10 --episodes 1000 --validation-episodes 200 --batch-size 32 --unroll-horizon 16 --sticky-probability 0.75 --full-state-range --barrier-probability 0.25 --output artifacts/pocketworld-collision-barrier.pt
```

For the focused learned-collision result, fine-tune the wall-relative risk head and then identify the three compact kinematic parameters from free transitions:

```bash
python -m pocketworld.train --resume artifacts/pocketworld-renderer-v5.pt --collision-only --epochs 10 --episodes 1500 --validation-episodes 300 --batch-size 64 --unroll-horizon 16 --sticky-probability 0.75 --full-state-range --barrier-probability 1 --collision-seek-probability 0.8 --output artifacts/pocketworld-collision-v4.pt
python -m pocketworld.train --resume artifacts/pocketworld-collision-v4.pt --kinematics-only --epochs 40 --episodes 500 --validation-episodes 100 --batch-size 64 --unroll-horizon 16 --sticky-probability 0.8 --full-state-range --output artifacts/pocketworld-collision-v5.pt
```

The report contains both per-seed runs and recursive mean/std summaries so planning gaps are not based on one lucky rollout.

The latest large-scale result is 98% imagined / 96% real success at 16 planning steps, with the real-vs-imagined gap reaching 7 percentage points at 24–32 steps.

The planner also includes an explicit `collision_aware=True` baseline, structured top/bottom detour proposals, and route-preserving replanning for barrier experiments. They are kept separate from the learned model so obstacle results remain honest: detour proposals improve the barrier distance, but reliable collision-state modeling is still needed for closed-loop success.

Collision-event and wall-relative-head variants are tracked as ablations in the evaluation report; the main checkpoint remains the structured-kinematics model without barrier-mix training.

Learned imagined collisions now freeze the compact state and zero velocity after an event. Peak collision risk is kept separate from geometric goal distance, so imagined success is no longer distorted by the planner's risk penalty.

The learned planner proposes map-agnostic two-bend waypoint routes and lets the wall-relative collision head rank them. It does not inspect wall boxes when running in pure learned mode; the explicit pixel planner remains a separately reported baseline.

Recent RGB frames can initialize velocity when temporal evidence exists, resolving information that is absent from a single frame. The uncertainty-aware planner re-scores a 64-candidate shortlist at the predicted position and four nearby states; its risk boundary grows with rollout depth.

To reproduce the frozen three-seed barrier comparison:

```bash
python -m pocketworld.evaluate_collision artifacts/pocketworld-collision-v5.pt --episodes 50 --horizon 48 --candidates 1024 --seeds 71,83,97 --output artifacts/evaluation-collision-v3.json
```

The committed [three-seed result](docs/results/evaluation-collision-v3.json) reports all four variants, per-seed metrics, and recursive mean/std summaries.

The planner also exposes `hybrid_collision=True`: it combines the learned collision event with a local wall-patch guard and is reported separately from the pure learned planner. This makes the engineering improvement useful without overstating what the learned collision head has solved.

The RGB path now reports raw decoder output separately from a state-conditioned composited frame. The composited frame restores a visible agent with 100% position coverage and about 1.7/2.3/3.1/3.9px position error at 1/5/10/20 steps on the renderer checkpoint; the learned mask head remains a diagnostic ablation because its shape IoU is still low.

The compact planning state uses a transparent kinematic prior: action directions are fixed by the four-action environment, while acceleration, friction, and speed limit are learned from rollout state supervision. This prevents the planner state from collapsing to an action-insensitive average and makes the action-effect diagnostic interpretable.

To create a browser-loadable one-step model after training:

```bash
python -m pocketworld.export_onnx --output public/pocketworld.onnx
```

For focused agent-renderer fine-tuning from an existing checkpoint:

```bash
python -m pocketworld.train --resume artifacts/pocketworld-renderer-v2.pt --agent-only --epochs 5 --episodes 1000 --validation-episodes 200 --unroll-horizon 16 --sticky-probability 0.75 --full-state-range --output artifacts/pocketworld-renderer-v5.pt
```

## Run the interactive demo

```bash
npm install
npm run dev
```

The frontend uses a pinned, stable Vite 5 toolchain so the demo does not depend on Vite 8's platform-specific Rolldown bindings.

The browser demo is intentionally self-contained so it can be deployed as a static site. It mirrors the simulator dynamics in the browser and makes model drift visible immediately. The ONNX export includes both the composited RGB frame and the supervised structured position, so the browser uses the stable position channel for the model marker while keeping RGB available for future visualization ablations.

## Regenerate the README visuals

The GIF and SVG files are generated locally with Pillow and contain no external assets:

```bash
python scripts/generate_readme_assets.py
```

## Planned evaluation matrix

1. Compare 1/5/10/20-step image and position error.
2. Hold out wall, start, goal, and speed variants to measure distribution shift.
3. Compare imagined versus real planning success across planning horizons.

The first version intentionally avoids VAE, Transformer, diffusion, multi-agent worlds, and complex physics so the relationship between prediction error and planning failure stays visible.

## Project governance

- [Contributing guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)
- [Citation metadata](CITATION.cff)
