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
  <img alt="Tests" src="https://img.shields.io/badge/tests-62%20passed-419400">
  <img alt="Coverage" src="https://img.shields.io/badge/coverage-79%25-69A94E">
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
- Learned wall-relative collision-event prediction, learned temporal velocity initialization, closed-loop replanning, and a horizon-aware uncertainty risk boundary.
- Side-by-side real/imagination playback, ONNX export, browser inference, OOD evaluation, and machine-readable multi-seed reports.

## How it differs from related projects

| Project | Primary focus | PocketWorld's difference |
|---|---|---|
| [World Models](https://arxiv.org/abs/1803.10122) | Learn a compact world representation and train a controller in its hallucinated environment. | Keeps the same core intuition, but focuses on exposing prediction-to-planning failure in a tiny, inspectable setting. |
| [PlaNet](https://arxiv.org/abs/1811.04551) | Stochastic latent dynamics, reward prediction, and online latent planning from pixels. | Uses a simpler deterministic model and discrete actions, prioritizing transparent RGB/state/collision diagnostics over general control capacity. |
| [DreamerV3](https://arxiv.org/abs/2301.04104) | Train actor-critic agents from imagined trajectories across many tasks. | Does not aim to reproduce full RL training; it isolates the learned-model and planner interface in one controlled environment. |
| [TD-MPC2](https://arxiv.org/abs/2310.16828) | Large-scale decoder-free latent control for continuous domains. | Deliberately retains RGB decoding because seeing what the model imagines is part of the experiment, not just an implementation detail. |

In short, PocketWorld is best viewed as a **microscope for world-model reliability**: small enough to reproduce, visual enough to understand, and strict enough to compare imagined plans with real execution.

## Current direction: learned temporal velocity and calibrated uncertainty

The latest research direction replaces two hand-designed shortcuts with measurable learned components:

- **Learned temporal velocity:** a lightweight RGB motion encoder, latent-frame differences, and a GRU learn velocity from the latest four observations. An auxiliary position head makes the motion features explicitly locate the small agent. Planning and calibration blend this learned estimate with an observable RGB finite-difference estimate, while `--temporal-only` supports frozen-world-model fine-tuning.
- **Calibrated probabilistic uncertainty:** a transition head predicts diagonal state standard deviations for position and velocity. A held-out rollout split calibrates one scale per coordinate with residual quantiles, then the planner samples landing states from that calibrated distribution when estimating collision probability.
- **Cost-controlled planning:** ordinary candidates are ranked with the point model first; only a shortlist receives probabilistic Monte Carlo rescoring. This keeps the uncertainty experiment inspectable and computationally bounded.
- **Route-level alignment:** candidates can be scored by endpoint distance, average along-route distance, regressions in progress, and accumulated collision risk. The executor reports imagined-vs-real route alignment error after every action and uses short route prefixes before replanning.
- **Wall-aware global fallback:** when alignment breaks, the RGB wall mask is footprint-inflated and searched with two 8-connected A* routes (top/bottom). The executor tracks only route bends and endpoints with the learned kinematics, while an explicit geometry gate rejects wall-intersecting rollouts. This follows the global-planner/local-controller split used by mature navigation stacks such as [Nav2 Planner Server](https://docs.nav2.org/configuration/packages/configuring-planner-server.html) and its predictive [MPPI controller](https://docs.nav2.org/configuration/packages/configuring-mppic.html), while keeping PocketWorld small and observable.
- **Route commitment and budget:** once the hybrid fallback chooses a side, subsequent replans stay on that side unless its observed path becomes infeasible. A* remaining distance is compared with the remaining action budget and route regressions are penalized; the executor reports route progress, remaining distance, and emergency side switches.
- **Online shift detection:** after each observed transition, a standardized position/velocity innovation score can trigger a fresh replan. Its threshold is fit from an in-distribution rollout quantile and is explicitly evaluated separately from the OOD label.

This is a marginal, split-calibrated Gaussian approximation—not a Bayesian posterior or an ensemble. The evaluation report now includes learned velocity error, finite-difference baseline error, 50/80/90/95% empirical coverage, interval width, and state Gaussian NLL.

The current calibration matrix evaluates fresh rollouts at nominal speed, 0.8x speed, 1.2x speed, a changed map, and the joint changed-map/1.2x-speed condition. The v3 kinematics checkpoint uses a learned temporal velocity representation blended with observable RGB velocity and a conservative post-fit uncertainty shrink. Across three seeds, 90% coverage is **94.0%/94.3% position/velocity in-distribution**, **89.3%/93.7% on joint map+fast OOD**, and speed/map shift AUROC is **0.885–0.955**. The detector uses an eight-frame mature RGB/action window, so the result is a calibrated online gate rather than a per-frame oracle. See [the v3 maturity report](docs/results/evaluation-maturity-v3-final-3seed.json).

The formal imagination-gap evaluator fixes candidate sampling per seed and reports the same selected plan in the learned model and the real simulator. On the released v3-final checkpoint, 24-step imagined/real success is **100.0% / 95.3%** (4.7pp gap) and 32-step success is **100.0% / 93.3%** (6.7pp gap). This is the central reliability result: short-horizon planning transfers closely, while longer imagined rollouts expose a measurable but bounded bias. See [the imagination-gap result](docs/results/evaluation-imagination-gap-v3-final.json).

The implementation details and current three-seed diagnostic slice are tracked in the [temporal/probabilistic experiment plan](docs/plans/learnable-temporal-probability.md), [v3 maturity matrix](docs/results/evaluation-maturity-v3-final-3seed.json), and [formal imagination-gap result](docs/results/evaluation-imagination-gap-v3-final.json).

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

The released v3-final checkpoint reaches **97.3% imagined / 97.3% real success** at 16 planning steps. At 24–32 steps, imagined success stays at 100% while real success is **95.3% / 93.3%**, giving a formal **4.7pp / 6.7pp** gap. On named train/holdout obstacle maps, the hybrid route controller reaches **98.9% / 93.3%** two-waypoint task success and **98.9% / 95.0%** three-waypoint task success across three seeds, with **0.40 / 0.45** and **0.41 / 0.57** collisions per leg. The strict single-barrier fallback reaches **50.67% ± 2.49pp** real success, **6.88 ± 0.14px** final distance, and **1.14 ± 0.23** collisions per episode. This is a mature, reproducible world-model/planning laboratory; it is not yet a claim that pure learned obstacle navigation is solved.

### Maturity gate

PocketWorld is considered mature as a research prototype when it passes all of these gates: deterministic three-seed reports; 20-step position error below 5px on train and holdout maps; at least 90% real waypoint success on unseen layouts; fewer than one collision per leg; a 24-step imagination gap below 10pp; 90% uncertainty coverage approximately within 85–95%; OOD shift AUROC above 0.85; and a published checkpoint plus machine-readable results. v3-final passes these gates on the stated protocols. The remaining red flag is scope-specific: the pure learned barrier planner still needs the explicit RGB/A* fallback for reliable obstacle crossing.

The full methodology, per-seed statistics, OOD results, and negative ablations are recorded in [the evaluation report](docs/evaluation-2026-08.md).

## Download a trained model

The `v0.1.0` release publishes the final renderer checkpoint and browser-ready ONNX graph outside Git history:

- [`pocketworld-renderer-v5.pt`](https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/download/v0.1.0/pocketworld-renderer-v5.pt) — PyTorch training/evaluation checkpoint.
- [`pocketworld-renderer-v5.onnx`](https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/download/v0.1.0/pocketworld-renderer-v5.onnx) — one-step RGB and structured-position inference.

The `v0.2.0` learned-collision milestone adds the focused barrier checkpoint and its compatible ONNX graph:

- [`pocketworld-collision-v5.pt`](https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/download/v0.2.0/pocketworld-collision-v5.pt) — collision curriculum, learned risk head, and calibrated kinematics.
- [`pocketworld-collision-v5.onnx`](https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/download/v0.2.0/pocketworld-collision-v5.onnx) — browser-compatible RGB and position outputs from the same checkpoint.

The `v0.3.0` route-aware probabilistic milestone publishes the v8 checkpoint and its complete three-seed result:

- [`pocketworld-temporal-probability-v8.pt`](https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/download/v0.3.0/pocketworld-temporal-probability-v8.pt) — temporal velocity, calibrated uncertainty, and route-aware planning checkpoint.
- [`evaluation-route-aware-temporal-probability-v8-full.json`](https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/download/v0.3.0/evaluation-route-aware-temporal-probability-v8-full.json) — exact protocol and per-seed metrics.

The `v0.4.0` maturity milestone publishes the calibrated v3-final checkpoint and formal imagination-gap evidence:

- [`pocketworld-map-suite-v3-final.pt`](https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/download/v0.4.0/pocketworld-map-suite-v3-final.pt) — learned kinematics, temporal velocity, calibrated uncertainty, and map-suite training metadata.
- [`pocketworld-map-suite-v3-final.onnx`](https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/download/v0.4.0/pocketworld-map-suite-v3-final.onnx) — browser-compatible one-step RGB/position graph.
- [`evaluation-imagination-gap-v3-final.json`](https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/download/v0.4.0/evaluation-imagination-gap-v3-final.json) — deterministic 16/24/32-step imagined-versus-real sweep.

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

To train and evaluate the learned temporal/probabilistic direction:

```bash
python -m pocketworld.train --epochs 8 --episodes 500 --validation-episodes 100 --batch-size 32 --unroll-horizon 8 --sticky-probability 0.75 --full-state-range --output artifacts/pocketworld-temporal-probability.pt
python -m pocketworld.train --resume artifacts/pocketworld-temporal-probability.pt --temporal-only --epochs 20 --episodes 500 --validation-episodes 100 --batch-size 32 --unroll-horizon 8 --sticky-probability 0.75 --full-state-range --output artifacts/pocketworld-temporal-probability-tuned.pt
python -m pocketworld.evaluate artifacts/pocketworld-temporal-probability-tuned.pt --episodes 50 --candidates 1024 --seeds 11,23,41 --output artifacts/evaluation-temporal-probability.json
```

The collision evaluator adds `learned_velocity_probabilistic_closed` alongside the existing point, history, and robust-radius baselines. Its probabilistic sampler is applied only to the planner shortlist so a larger candidate count remains practical.

To reproduce the completed v8 probabilistic barrier revalidation:

```bash
python -m pocketworld.evaluate_collision artifacts/pocketworld-temporal-probability-v8.pt --episodes 50 --horizon 48 --candidates 1024 --seeds 71,83,97 --only learned_velocity_probabilistic_closed --output artifacts/evaluation-temporal-probability-probabilistic-v8-full.json
```

The committed [v8 probabilistic result](docs/results/evaluation-temporal-probability-probabilistic-v8-full.json) contains per-seed metrics and mean/std summaries. The committed [OOD calibration matrix](docs/results/evaluation-temporal-probability-ood-v8.json) records the speed/map stability boundary.

## Map and task generalization

The original staggered-wall map remains the compatibility baseline, but the environment now exposes named layouts: `default`, `single_barrier`, `double_barrier`, `cross`, `zigzag`, and `open`. The `train` suite contains the first three; `cross` and `zigzag` are held out as unseen obstacle layouts. Rollout collection can train on the named suite instead of silently perturbing one map:

```bash
python -m pocketworld.train --resume artifacts/pocketworld-temporal-probability-v8.pt --epochs 12 --episodes 1000 --validation-episodes 200 --batch-size 32 --unroll-horizon 16 --sticky-probability 0.75 --full-state-range --map-suite train --output artifacts/pocketworld-map-suite-v1.pt
```

The generalization evaluator measures per-map multi-step prediction error and sequential two-waypoint tasks:

```bash
python -m pocketworld.evaluate_generalization artifacts/pocketworld-map-suite-v3-final.pt --episodes 20 --horizon 64 --candidates 32 --seeds 11,23,41 --train-suite train --holdout-suite holdout --waypoints 2 --output artifacts/evaluation-map-suite-v3-final-3seed.json
```

The original v1 diagnostic is preserved in [evaluation-map-suite-v1-full.json](docs/results/evaluation-map-suite-v1-full.json). The v3-final kinematics checkpoint preserves the four-action RGB-observed route follower, continuous-coordinate landing safety guard, and physically traversable zig-zag holdout while reducing 20-step position error to **3.02 ± 0.27px on train / 3.72 ± 0.32px on unseen maps**. The formal [three-seed two-waypoint result](docs/results/evaluation-map-suite-v3-kinematics-3seed.json) uses 20 episodes per map, 64-step execution budgets, and 32 candidates: task success is **98.9% ± 1.6pp / 93.3% ± 2.4pp**, waypoint completion is **98.9% / 94.2%**, and mean collisions are **0.40 / 0.45 per leg**. These are explicitly hybrid benchmarks: learned dynamics and collision-aware prediction plus RGB-only visible-route control.

The three-waypoint stress test uses the same three seeds and a 96-step budget. It reaches **98.9% train / 95.0% unseen task success**, **98.9% / 97.2% waypoint completion**, and **0.41 / 0.57 collisions per leg**; see [evaluation-map-suite-v3-kinematics-waypoint3-3seed.json](docs/results/evaluation-map-suite-v3-kinematics-waypoint3-3seed.json).

For the maturity-gate calibration and OOD protocol:

```bash
python -m pocketworld.evaluate_maturity artifacts/pocketworld-map-suite-v3-final.pt --episodes 50 --horizon 8 --seeds 11,23,41 --output artifacts/evaluation-maturity-v3-final-3seed.json
```

The [three-seed maturity report](docs/results/evaluation-maturity-v3-final-3seed.json) gives 90% position/velocity coverage of **94.0%/94.3% in-distribution**, **89.3%/93.7% on joint map+fast OOD**, and speed/map AUROC of **0.885–0.955**. The shift score waits for an eight-frame mature RGB/action window, keeping the detector observable and reproducible. The matrix is inside the practical 85–95% coverage band up to ordinary seed variation; it is not a guarantee of calibration under arbitrary physics changes.

To reproduce the formal imagination-gap sweep:

```bash
python -m pocketworld.evaluate_imagination artifacts/pocketworld-map-suite-v3-final.pt --episodes 50 --horizons 16,24,32 --candidates 256 --seeds 11,23,41 --output artifacts/evaluation-imagination-gap-v3-final.json
```

The [formal result](docs/results/evaluation-imagination-gap-v3-final.json) reports a **4.7pp** gap at 24 steps and **6.7pp** at 32 steps. The evaluator fixes candidate sampling per seed, so the imagined and real rates refer to the same selected action sequences.

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

Recent RGB frames can initialize velocity either with the legacy finite-difference estimator or with the learned temporal encoder. The calibrated probabilistic planner re-scores a shortlist by sampling future landing states; the older 64-point neighborhood and horizon-growing radius remain available as explicit robust baselines.

The route-aware planner adds `route_objective=True` and reports `route_alignment_error_px` / `max_route_alignment_error_px`. If alignment exceeds a configured threshold, it switches from learned route proposals to an explicit wall-aware hybrid fallback. To enable both alarms with the v8 thresholds:

```bash
python -m pocketworld.evaluate_collision artifacts/pocketworld-temporal-probability-v8.pt --episodes 50 --horizon 48 --candidates 1024 --seeds 71,83,97 --only learned_velocity_probabilistic_closed --alignment-fallback-threshold 4.0 --output artifacts/evaluation-route-aware-alignment4-v8-full.json
```

The completed [route-aware three-seed result](docs/results/evaluation-route-aware-temporal-probability-v8-full.json) is the no-fallback baseline. The [wall-aware A* fallback result](docs/results/evaluation-route-aware-astar-fallback-v8-full.json) reached **30.67% ± 2.49pp**. Adding route-side commitment and remaining-budget scoring raised this to **34.67% ± 6.80pp**. Triggering the fallback earlier at 4px alignment produces **50.67% ± 2.49pp** real success, **6.88 ± 0.14px** final distance, **1.14 ± 0.23** collisions, **60.68px** mean route progress, and **7.37px** remaining route distance, with zero emergency side switches. See the [alignment-4px result](docs/results/evaluation-route-aware-alignment4-v8-full.json). This is a meaningful transfer improvement, not a claim that the learned world model has solved general navigation.

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
2. Train on the named `train` map suite and hold out `cross`/`zigzag` layouts.
3. Compare single-goal and sequential-waypoint imagined versus real planning.
4. Improve route-level world-model alignment so low collision risk also produces real barrier crossings.

The first version intentionally avoids VAE, Transformer, diffusion, multi-agent worlds, and complex physics so the relationship between prediction error and planning failure stays visible.

## Project governance

- [Contributing guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Changelog](CHANGELOG.md)
- [Citation metadata](CITATION.cff)
