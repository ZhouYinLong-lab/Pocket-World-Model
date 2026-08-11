# Fixed-budget planner tournament

## Question

When the learned world model and the model-query budget are fixed, which planner transfers imagined progress into real execution most reliably?

This comparison is intentionally separate from model training. Every method receives the same checkpoint, paired start/goal cases, horizon, and nominal candidate budget per planning call. Closed-loop methods can call the planner more than once; their estimated total model queries are reported rather than hidden.

## Methods

- `random_shooting`: existing guided random-shooting baseline.
- `cem`: categorical Cross-Entropy Method. The total `candidates` budget is divided evenly across four iterations, so CEM does not receive more model rollouts than the baseline.
- `learned_collision`: existing learned collision-probability planner with probabilistic shortlist rescoring.
- `route_aware_hybrid`: existing closed-loop route-aware planner with RGB wall geometry, learned dynamics, and the 4px alignment-triggered fallback. It is a real-execution controller, not an open-loop imagination baseline.

## Protocol

- checkpoint: `pocketworld-map-suite-v3-final.pt`;
- seeds: `11, 23, 41`;
- open-space: 20 paired episodes, horizon 16, 256 candidate budget per call;
- single barrier: 20 paired episodes, horizon 48, 256 candidate budget per call;
- metrics: imagined success where defined, real success, final distance, collisions, executed actions, and planning score.

The route-aware hybrid intentionally reports `null` for imagined success and imagined final distance. Its action sequence is replanned against new RGB observations, so treating the first route score as a complete imagined rollout would create a misleading gap. It also consumes more total model queries than one-shot planners; the JSON reports `planning_calls` and `estimated_model_queries`.

## Results

### Open space

| Planner | Imagined success | Real success | Real final distance |
| --- | ---: | ---: | ---: |
| Random Shooting | 96.7% | 96.7% | 3.07px |
| CEM | 100.0% | 100.0% | 3.05px |
| Learned collision | 98.3% | 100.0% | 3.13px |
| Route-aware hybrid | — | 98.3% | 3.18px |

CEM improves the model-side endpoint distance from 1.30px to 0.37px under the same 256-query budget, but the real success improvement is small because the open-space baseline is already near saturation.

### Single barrier

| Planner | Imagined success | Real success | Real final distance | Collisions |
| --- | ---: | ---: | ---: | ---: |
| Random Shooting | 100.0% | 0.0% | 29.58px | 8.40 |
| CEM | 100.0% | 0.0% | 29.12px | 8.73 |
| Learned collision | 95.0% | 63.3% | 8.94px | 2.33 |
| Route-aware hybrid | — | 100.0% | 2.95px | 0.65 |

The main finding is negative and useful: better action-sequence search does not fix a misspecified obstacle model. CEM makes the imagined endpoint slightly better while preserving the 100pp imagined–real failure gap. Learned collision modeling reduces the gap but still fails on a substantial fraction of routes. Closed-loop route-aware control transfers best, at the cost of using explicit visible geometry.

Machine-readable reports:

- [open-space tournament](../results/evaluation-planner-tournament-open-v3-final.json)
- [single-barrier tournament](../results/evaluation-planner-tournament-barrier-v3-final.json)

The next comparison should add a collision-aware CEM variant and a discrete beam-search baseline, while keeping this fixed-budget protocol unchanged.
