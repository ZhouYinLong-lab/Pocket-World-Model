# Fixed-budget planner tournament

## Question

When the learned world model and the model-query budget are fixed, which planner transfers imagined progress into real execution most reliably?

This comparison is intentionally separate from model training. Every method receives the same checkpoint, paired start/goal cases, horizon, and nominal candidate budget per planning call. Closed-loop methods can call the planner more than once; their estimated total model queries are reported rather than hidden.

## Methods

- `random_shooting`: existing guided random-shooting baseline.
- `cem`: categorical Cross-Entropy Method. The total `candidates` budget is divided evenly across four iterations, so CEM does not receive more model rollouts than the baseline.
- `beam_search`: discrete prefix beam search. Its width is selected from `floor(candidates / (4 * horizon))`; at horizon 48 and budget 256 this is a width-one search with 192 expansions.
- `learned_collision`: existing learned collision-probability planner with probabilistic shortlist rescoring.
- `cem_collision`: categorical CEM scored with the learned collision head, without explicit wall geometry.
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
| Beam Search | 100.0% | 100.0% | 3.00px |
| Learned collision | 98.3% | 100.0% | 3.22px |
| Collision-aware CEM | 100.0% | 100.0% | 3.15px |
| Route-aware hybrid | — | 98.3% | 3.18px |

CEM improves the model-side endpoint distance from 1.30px to 0.37px under the same 256-query budget, but the real success improvement is small because the open-space baseline is already near saturation.

### Single barrier

| Planner | Imagined success | Real success | Real final distance | Collisions |
| --- | ---: | ---: | ---: | ---: |
| Random Shooting | 100.0% | 0.0% | 29.58px | 8.40 |
| CEM | 100.0% | 0.0% | 29.12px | 8.73 |
| Beam Search | 100.0% | 0.0% | 28.77px | 11.72 |
| Learned collision | 100.0% | 60.0% | 9.75px | 2.52 |
| Collision-aware CEM | 0.0% | 0.0% | 27.99px | 3.18 |
| Route-aware hybrid | — | 100.0% | 2.95px | 0.65 |

The main finding is negative and useful: better action-sequence search does not fix a misspecified obstacle model. CEM and Beam Search both preserve a 100pp imagined–real failure gap on the barrier. Adding the learned collision score to CEM makes the planner conservative, but it does not produce a successful route: collision-aware CEM reaches 0% imagined and 0% real success. Learned collision modeling reduces the gap to 40pp but still fails on a substantial fraction of routes. Closed-loop route-aware control transfers best, at the cost of using explicit visible geometry.

Machine-readable reports:

- [open-space tournament](../results/evaluation-planner-tournament-open-v4.json)
- [single-barrier tournament](../results/evaluation-planner-tournament-barrier-v4.json)

The next comparison should replace the learned collision head with an ensemble or conformal risk model and test whether better calibration improves CEM without relying on explicit RGB wall geometry.
