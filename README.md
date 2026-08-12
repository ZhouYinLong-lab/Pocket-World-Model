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
  <img alt="Tests" src="https://img.shields.io/badge/tests-121%20passed-419400">
  <img alt="Coverage" src="https://img.shields.io/badge/coverage-84%25-69A94E">
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

### Learned route-level planning comparison

The obstacle-navigation study now compares three distinct levels of abstraction:

| Method | Evaluation A* | Student signal | 240-task real success | Collisions / episode |
| --- | ---: | --- | ---: | ---: |
| Action-level RGB policy + DAgger | No | next discrete action | 59.2% | 17.63 |
| Route-mode RGB policy + waypoint controller | No | direct / top / bottom / gap | **100.0%** | **0.042** |
| Geometry-locked RGB/A* hybrid | Yes | visible global route | 99.2% | 0.242 |

The action-level student has reasonable held-out action accuracy (81.3%) but suffers from covariate shift: its real success drops to 0% on the shifted-barrier split. The route-level student predicts one persistent route mode from the initial RGB frame and goal, then uses an RGB-only local waypoint controller; it reaches 100% on all four barrier layouts across seeds `11,23,41` (20 tasks per layout and seed). This is a method comparison, not a claim that the learned policy has discovered general navigation: the route vocabulary and waypoint controller are deliberately specialized to this four-layout benchmark.

Machine-readable reports:

- [action-level student](docs/results/evaluation-learned-route-policy-v14.json)
- [route-level student](docs/results/evaluation-route-mode-policy-v15.json)
- [geometry-locked hybrid reference](docs/results/evaluation-route-aware-adaptive-v13-full.json)

Reproduce the route-level study with:

```bash
python -m pocketworld.evaluate_learned_route_policy --route-level \
  --train-seeds 101,103 --train-scenarios single_barrier,barrier_narrow_gap \
  --evaluation-seeds 11,23,41 --evaluation-episodes 20 --max-steps 64 \
  --epochs 300 --output artifacts/evaluation-route-mode-policy-v15.json
```

### Procedural multi-obstacle route study

The next comparison expands the fixed four-layout benchmark into deterministic
procedural maps with three or four vertical barriers and held-out endpoint
bands. It compares a continuous route-sketch student (v16) with a structured
per-barrier gap-center student (v17):

| Method | Learned output | Evaluation-time geometry | Real success | Collisions / episode |
| --- | --- | --- | ---: | ---: |
| Route sketch v16 | 5–9 continuous route points | RGB waypoint executor | 25.0% (9-point scale) | negative result |
| Gap route v17, raw | one gap center per barrier | none | **83.3%** | 1.417 |
| Gap route v17 + visible projection | one gap center per barrier | clamp to RGB-visible feasible gap | **100.0%** | **0.367** |

The v17 result uses 600 training tasks and 60 held-out tasks across three
evaluation seeds (`11,23,41`, 20 each). The student does **not** receive the
target gap center: its input is the initial RGB frame plus the goal, while the
gap center is only a teacher label. A* is used only as a data-quality filter to
discard disconnected procedural samples; labels, student inference, and
evaluation make no A* calls. The
projection is a separate RGB feasibility layer that clamps each prediction to
the visible gap interval; therefore the 100% number is a hybrid safety result,
not pure end-to-end learned obstacle navigation. The raw-vs-projected paired
ablation measures exactly how much that safety layer contributes.

The remaining limitation is scope: these procedural maps are left-to-right
vertical barriers with visible gaps. This is a controlled study of route
representation and model-error containment, not a claim of general 2D
navigation. See the [v17 machine-readable report](docs/results/evaluation-gap-route-policy-v17-honest-full.json).

Reproduce it with:

```bash
python -m pocketworld.evaluate_gap_route_policy \
  --train-seeds 101,103,107 --evaluation-seeds 11,23,41 \
  --train-episodes 200 --evaluation-episodes 20 --max-steps 140 \
  --epochs 360 --output artifacts/evaluation-gap-route-policy-v17.json
```

### v18 general obstacle representation study

v18 broadens the benchmark beyond vertical barriers to staggered blocks,
multi-channel walls, staircases, and offset L-shaped obstacles. Training uses
only `staggered_blocks` and `multi_channel`; the 60-task holdout uses all four
families across seeds `11,23,41`. The study compares route representations and
execution layers under one fixed protocol:

| Method | A* during evaluation | Real success | Collisions / episode |
| --- | ---: | ---: | ---: |
| Continuous route sketch | No | 6.7% | 15.35 |
| Route sketch + RGB projection | No | 1.7% | 8.15 |
| Coarse distance field | No | 50.0% | 6.20 |
| Distance field + RGB guard | No | 75.0% | 5.50 |
| Distance field + beam + RGB guard | No | **88.3%** | **2.20** |
| Learned route + A* fallback | Fallback only | 100.0% | 0.217 |
| RGB/A* reference | Yes | 100.0% | 0.067 |

The distance field is a learned 16×16 estimate of normalized route cost. A
fixed beam of four candidate grid paths is then followed using RGB edge checks;
the beam does not call A*. This improves generalization substantially over
fixed coordinate regression, but multi-channel maps remain the hard slice
(78.6% success, 5.93 collisions/episode). A conservative one-step action
shield was also tested and failed (63.3%, 40.62 collisions/episode), so it is
kept as a negative ablation rather than hidden from the comparison.

The reliable 100% methods still use observable RGB geometry plus repeated A*
planning. Thus v18 improves the learned route representation and quantifies
the remaining gap; it does not claim that pure learned obstacle navigation is
solved. See the [v18 machine-readable report](docs/results/evaluation-general-routes-v18-current-final.json).

### v23 coverage, density, and planner-method study

The latest study separates two questions that were previously confounded:
whether the model has seen enough obstacle families, and whether each family
has enough training density. Training is balanced by family and evaluated on
the same 60 held-out tasks for every condition:

| Training condition | Total samples | Samples / family / seed | Learned MPC success | Collisions / episode |
| --- | ---: | ---: | ---: | ---: |
| 2 families | 600 | 100 | 90.0% | 0.333 |
| 4 families | 600 | 50 | **95.0%** | **0.367** |
| 4 families | 1200 | 100 | 93.3% | 0.433 |

The result is deliberately non-monotonic: adding families at fixed total
budget helps this holdout, while doubling the four-family budget does not.
Coverage and per-family density therefore need to be reported as separate
variables rather than summarized as “more data is better.”

On the shared four-family/600 holdout, robust MPC matches ordinary MPC's
95.0% success and lowers collisions from 0.367 to 0.150, but costs 2.80 s of
planning per episode versus 255 ms. The guarded-MPC negative ablation reaches
only 66.7% success. The clearance-field variant is identical to the base
field on this task set, so its current loss is not yet demonstrated to change
the policy. RGB/A* reaches 100% with 0.067 collisions and remains a geometric
reference, not a pure learned result.

These are paired, fixed-protocol comparisons; no holdout result is used for
method selection. See the [coverage report](docs/results/evaluation-coverage-study-v23.json)
and [planner comparison](docs/results/evaluation-planner-comparison-v23.json).

The paired OOD matrix makes the robustness trade-off explicit:

| Condition | RGB projection | MPC | Robust MPC |
| --- | ---: | ---: | ---: |
| Nominal, speed 1.0 | 91.7% / 1.767 | **95.0% / 0.367** | **95.0% / 0.150** |
| Nominal, speed 1.25 | 76.7% / 15.083 | **95.0% / 0.533** | 93.3% / **0.433** |
| Walls +1, speed 1.0 | 73.7% / 2.456 | **78.9% / 0.474** | 77.2% / **0.298** |
| Walls +1, speed 1.25 | 64.9% / 12.070 | **78.9% / 0.877** | 73.7% / 1.053 |

Each cell is `real success / collisions per episode`. Robust MPC lowers risk
in three of four conditions, but loses success on the joint shift and costs
roughly 2.5–3.4 seconds of planning time per episode. This motivates the next
research step: a calibrated risk budget that can choose ordinary versus robust
MPC based on predicted uncertainty, instead of always enabling the expensive
envelope. See the [paired OOD report](docs/results/evaluation-coverage-study-v23-ood.json).

### v25 adaptive risk-gate study

The next experiment asked whether robust MPC should run on every step. An
adaptive gate computes an RGB/history-only risk score, enters robust MPC at a
calibrated threshold, and exits with hysteresis. Threshold selection uses
disjoint calibration seeds (`53,67`) and a fixed success-floor/collision/time
rule; the final holdout remains `11,23,41`.

| Method | Final success | Collisions / episode | Planning time | Robust calls / episode |
| --- | ---: | ---: | ---: | ---: |
| Ordinary MPC | **95.0%** | **0.367** | **178 ms** | 0 |
| Fixed robust MPC | 95.0% | **0.150** | 1.98 s | 77.7 |
| Calibrated adaptive MPC | 95.0% | 0.450 | 304 ms | 4.5 |

The calibrated gate reduces expensive robust calls by over 90% and is roughly
6.5× faster than fixed robust MPC, but it does not match ordinary MPC's
collision rate. Under the paired OOD matrix, adaptive MPC remains a mixed
result: it preserves 78.9% success on walls+1/speed1.25 with 1.035 collisions,
while ordinary MPC has 0.877 and fixed robust MPC has 1.053. The correct current
claim is therefore “compute-budget adaptation,” not calibrated safety.

This negative result is useful: the current risk score correlates with nominal
collisions but loses predictive value under joint map/speed shift. The next
research target is r…7714 tokens truncated… better search cannot repair an obstacle model that predicts the wrong world. See the [planner tournament protocol and analysis](docs/plans/planner-tournament.md), [open-space result](docs/results/evaluation-planner-tournament-open-v4.json), and [risk-budget barrier ablation](docs/results/evaluation-planner-tournament-barrier-v5-risk-budget.json).

Reproduce the comparison with:

```bash
python -m pocketworld.evaluate_planners artifacts/pocketworld-map-suite-v3-final.pt --episodes 20 --horizon 16 --candidates 256 --seeds 11,23,41 --scenario open --output artifacts/evaluation-planner-tournament-open-v3-final.json
python -m pocketworld.evaluate_planners artifacts/pocketworld-map-suite-v3-final.pt --episodes 20 --horizon 48 --candidates 256 --seeds 11,23,41 --scenario single_barrier --output artifacts/evaluation-planner-tournament-barrier-v3-final.json
```

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

### Risk-estimator comparison

To test whether the remaining barrier gap is caused by an underpowered risk estimator, the project now compares the single learned collision head with a three-checkpoint ensemble and a split-conformal upper-risk wrapper. The comparison keeps the v3-final dynamics model, 48-step horizon, 256-candidate budget, and paired three-seed barrier tasks fixed. The ensemble reaches **20.0% real success** and the conformal wrapper reaches **0.0%**, versus **61.7%** for the single learned collision planner. Conformal coverage is **100%**, but its calibration quantile is **0.9984** because every selected calibration route collided; it therefore stops almost immediately. This is an intentional negative result: stronger uncertainty guarantees can become unusable when the calibration unit is mismatched to route-level failure. See the [uncertainty-method comparison](docs/plans/uncertainty-methods.md) and [machine-readable report](docs/results/evaluation-uncertainty-barrier-v1.json).

The reliability bins sharpen the diagnosis: ensemble plans with predicted risk in the 0–0.20 range still collide on **100%** of held-out routes. The next optimization target is therefore route-conditioned collision supervision and route-completion prediction, not a larger search population.

### Route-conditioned completion study

The next experiment adds an explicit `RouteCompletionPredictor` trained on real simulator outcomes for complete candidate routes. It achieves **0.997 held-out AUROC**, but planner transfer is much harder: weight 10 reaches **63.3% real success** versus **61.7%** for learned collision, weight 64 collapses to **28.3%**, the risk-gated/hard-negative variants reach **51.7% / 53.3%**, and online route-completion MPC falls to **5.0%** despite replanning. The paired comparison is therefore a controlled negative result: candidate-level route classification is strong, but its score does not yet survive model-missed collisions during planning. See the [route-completion study](docs/plans/route-completion.md) and its [MPC report](docs/results/evaluation-route-completion-barrier-v6-mpc.json).

Reproduce the latest route-conditioned experiment with:

```bash
python -m pocketworld.evaluate_route_completion artifacts/pocketworld-map-suite-v3-final.pt \
  --train-seeds 101,103 --evaluation-seeds 11,23,41 --train-episodes 12 --evaluation-episodes 20 \
  --calibration-candidates 32 --candidates 256 --horizon 48 --hard-negative-rounds 1 \
  --predictor-output artifacts/route-completion-v2-hard-negative.pt \
  --output artifacts/evaluation-route-completion-barrier-v5-hard-negative.json
```

The follow-up adds an isolated `route_completion_safe_gate` ablation: before
the route score is trusted, the planner rejects imagined trajectories that
intersect the footprint-inflated wall mask extracted from the current RGB
observation. On the same 60 paired barrier tasks, this raises real success to
**65.0%** and lowers collisions to **2.20/episode**, versus **61.7%** and
**2.47/episode** for learned collision. The paired gain is only **+3.33pp**
(`0, 0, +10pp` across the three seeds), so the result is a bounded safety
improvement, not a claim that the learned model now solves obstacle
traversal. The high-query route-aware hybrid remains at **100.0%** real
success. See the [RGB safety-gate ablation](docs/plans/route-completion.md)
and [machine-readable v7 result](docs/results/evaluation-route-completion-barrier-v7-safe-gate.json).

The next ablation compares three ways to use the same RGB wall evidence. A
joint gate requires both learned-risk and RGB feasibility; an RGB-only gate
uses only the visible wall mask; a soft method adds a wall-intersection cost
without hard rejecting every risky route. On the same 60 paired barrier tasks:

| Method | Real success | Collisions/episode | Imagined RGB wall intersection | Paired delta |
| --- | ---: | ---: | ---: | ---: |
| Learned collision | 61.7% | 2.47 | 41.7% | — |
| Joint RGB gate | 65.0% | 2.20 | 11.7% | +3.33pp |
| RGB-only gate | 65.0% | 2.02 | 6.7% | +3.33pp |
| Soft RGB penalty | **71.7%** | **2.02** | 21.7% | **+10.00pp** |

The soft penalty is currently the strongest one-shot method, but its result is
still protocol-specific: three seeds, one barrier layout, and the same tasks
used for method comparison. The fixed-predictor runner makes this comparison
reproducible without retraining the route predictor:

```bash
python -m pocketworld.evaluate_safety_methods \
  artifacts/pocketworld-map-suite-v3-final.pt \
  --route-predictor artifacts/route-completion-v4-safety-methods.pt \
  --evaluation-seeds 11,23,41 --evaluation-episodes 20 \
  --candidates 256 --horizon 48 \
  --output artifacts/evaluation-safety-methods-barrier-v9.json
```

See the [safety-mechanism protocol](docs/plans/route-completion.md) and
[machine-readable v9 result](docs/results/evaluation-safety-methods-barrier-v9.json).

### Penalty selection and OOD generalization

The soft penalty was not chosen from the OOD results. A six-value scan on the
original single-barrier task (`0, 8, 16, 32, 64, 128`) selected **16** by a
predeclared rule: highest mean success, then smallest weight on ties. It gives
**71.7%** in-distribution success and **+10.0pp** paired improvement.

When frozen and moved to held-out shifted barriers, narrow/wide gaps, and
speed scales 0.8/1.2, the result reverses: soft planning is never better than
the learned-collision baseline, with paired deltas from **0.0pp to −25.0pp**.
The gap tasks were corrected to place start and goal outside the wall footprint
on opposite sides, so they require an actual detour. The current evidence is
therefore precise: soft RGB penalties improve this barrier distribution but do
not yet generalize across map geometry and dynamics shifts.

Reproduce the selection and held-out protocols with the resumable sweep runner:

```bash
python -m pocketworld.evaluate_safety_sweep \
  artifacts/pocketworld-map-suite-v3-final.pt \
  --route-predictor artifacts/route-completion-v4-safety-methods.pt \
  --penalties 0,8,16,32,64,128 --scenarios single_barrier \
  --speed-scales 1.0 --evaluation-seeds 11,23,41 --evaluation-episodes 20 \
  --candidates 256 --horizon 48 --resume-from artifacts/safety-sweep-selection-v11-progress.json \
  --output artifacts/evaluation-safety-sweep-selection-v11.json

python -m pocketworld.evaluate_safety_sweep \
  artifacts/pocketworld-map-suite-v3-final.pt \
  --route-predictor artifacts/route-completion-v4-safety-methods.pt \
  --penalties 16 --scenarios barrier_shifted,barrier_narrow_gap,barrier_wide_gap \
  --speed-scales 0.8,1.2 --evaluation-seeds 11,23,41 --evaluation-episodes 20 \
  --candidates 256 --horizon 48 --resume-from artifacts/safety-sweep-ood-v12-progress.json \
  --output artifacts/evaluation-safety-sweep-ood-v12.json
```

See the [selection report](docs/results/evaluation-safety-sweep-selection-v11.json)
and [OOD holdout report](docs/results/evaluation-safety-sweep-ood-v12.json).

### Executable hybrid route control

The OOD result exposed a second distinction: route-risk prediction and route
execution are separate problems. PocketWorld now includes an explicitly
reported geometry-locked controller. It computes a cardinal A* detour from the
current RGB wall mask once per task; if the visible detour is short enough, it
locks a diagonal/lookahead policy for gap crossing, otherwise it keeps
conservative cardinal braking for full barriers. The policy is not selected by
scenario name.

On a fresh three-seed matrix with 20 episodes per map and a 64-step execution
budget, the hybrid controller reaches:

| Map | Real success | Collisions/episode |
| --- | ---: | ---: |
| Single barrier | 100.0% | 0.650 |
| Shifted barrier | 100.0% | 0.233 |
| Narrow gap | 98.3% | 0.067 |
| Wide gap | 98.3% | 0.017 |

These runs use **zero learned model queries** in the route override; geometry
planning calls are reported separately. This makes the result a strong,
observable hybrid-navigation baseline and a useful upper bound for future
learned detour work, not evidence that the learned world model alone solves
obstacle traversal. See the [controller comparison](docs/results/evaluation-route-controller-v13.json)
and [adaptive hybrid report](docs/results/evaluation-route-aware-adaptive-v13-full.json).

### Map-aware route features: better risk signals, not yet better traversal

To broaden the method comparison, v0.13 trains two otherwise identical route
predictors. The 9D baseline sees only imagined trajectory statistics; the 17D
variant additionally reads geometry extracted from the current RGB wall mask
(direct blockage, clearances, and top/bottom detour lengths). Both use the same
training maps, seeds, planner random streams, and 48-step/64-candidate budget.

Across three seeds, four map layouts, and 0.8x/1.2x speed shifts, map-aware
features raise mean imagined route success from **63.3% to 70.8%** and reduce
collisions from **4.23 to 3.88 per episode**, but mean real route success is
unchanged at **15.0%**. This is a useful negative result: observable geometry
improves route-risk recognition, while the learned dynamics still fails to
generate reliably executable gap-crossing detours. The full protocol and
machine-readable result are in the [route-feature comparison](docs/plans/route-completion.md#map-aware-route-feature-comparison).

Reproduce the complete route-method comparison with:

```bash
python -m pocketworld.evaluate_route_completion artifacts/pocketworld-map-suite-v3-final.pt \
  --train-seeds 101,103 --evaluation-seeds 11,23,41 --train-episodes 12 --evaluation-episodes 20 \
  --calibration-candidates 32 --candidates 256 --horizon 48 --hard-negative-rounds 1 \
  --predictor-output artifacts/route-completion-v3-safe-gate.pt \
  --output artifacts/evaluation-route-completion-barrier-v7-safe-gate.json
```

Reproduce the risk-method study with:

```bash
python -m pocketworld.evaluate_uncertainty artifacts/pocketworld-map-suite-v3-final.pt \
  --ensemble-checkpoints artifacts/pocketworld-map-suite-v3.pt,artifacts/pocketworld-map-suite-v3-calibrated.pt,artifacts/pocketworld-map-suite-v3-kinematics.pt \
  --calibration-seeds 101,103 --evaluation-seeds 11,23,41 --calibration-episodes 12 --evaluation-episodes 20 \
  --horizon 48 --candidates 256 --output artifacts/evaluation-uncertainty-barrier-v1.json
```

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

