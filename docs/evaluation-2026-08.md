# PocketWorld evaluation — 2026-08-03

The current main checkpoint is trained with 1,000 trajectories, 200 validation trajectories, 16-step unrolls, sticky actions (`p=0.75`), and full-map start/goal sampling. The report was generated with 50 episodes per seed, 3 seeds (`11, 23, 41`), and 1,024 random-shooting candidates.

## Current result

| Metric | In-distribution | OOD |
| --- | ---: | ---: |
| 1-step latent position error | 1.22 px | 0.99 px |
| 5-step latent position error | 1.86 px | 1.59 px |
| 10-step latent position error | 2.50 px | 2.49 px |
| 20-step latent position error | 3.94 px | 4.19 px |
| 20-step image MAE | 0.0468 | 0.0704 |

Planning now uses a 16-step default horizon, with a sweep to expose imagination drift:

| Horizon | Imagined success | Real success | Gap |
| ---: | ---: | ---: | ---: |
| 8 | 0% | 0% | 0pp |
| 16 | 98% | 96% | 2pp |
| 24 | 100% | 93% | 7pp |
| 32 | 100% | 93% | 7pp |

The repeated-action diagnostic averages 4.04 px position error across up/down/left/right. This is the clearest evidence that the compact model is now action-sensitive; the previous sticky-only model had errors of roughly 9–35 px and could not produce meaningful imagined planning.

## What remains

- RGB decoding does not reliably preserve the small green agent: decoded-position coverage is 0% after five or more steps, even when latent position error is low.
- The structured planning state does not yet model wall collision explicitly, which explains the growing 10–12 percentage-point real-vs-imagined gap at 24–32 steps.
- A local-pixel decoder ablation reached 100% decoded-position coverage, but damaged the shared latent state and reduced real planning success to 0%; it is intentionally not the mainline result.

## Next experiments

1. Add a separate agent-rendering head or decoder skip path so pixel supervision cannot corrupt the planning state.
2. Add collision events and wall geometry to the compact state transition, then repeat the 24/32-step sweep.
3. Export the best checkpoint and show the latent planned trajectory alongside the decoded RGB rollout in the browser demo.

The next evaluation adds a single-barrier challenge. It compares the existing unconstrained planner with a collision-aware planner that extracts wall pixels from the current observation and penalizes trajectories after the first wall intersection. This is an explicit planning baseline; it is not counted as learned wall dynamics until the compact state model predicts collision events itself.

The first barrier result was intentionally a negative result: the unconstrained planner reached 100% imagined success but 0% real success. Structured top/bottom detour proposals now raise collision-aware open-loop success to 10% on the same 10-episode, 40-step check and reduce mean final distance from 28.18px to 21.34px. At horizon 64, the mean distance falls further to 17.54px, but success remains only 10%.

The implementation now includes `receding_horizon_plan`, which replans from the latest real observation after every action. Its barrier result is tracked separately from the open-loop baselines so closed-loop correction can be measured without changing the headline planning numbers.

The quick closed-loop check still reached 0% real success: 1-step replanning averaged 28.19px, 4-step commitment averaged 27.91px, and route-preserving replanning averaged 27.72px. All remain worse than executing the best structured detour open-loop (21.27px in the same check), confirming that the missing piece is collision-state modeling rather than route-cache lifetime. The route-preserving mode caches the imagined path and only replans on collision or >6px route deviation; its result is reported separately as `collision_aware_route`.
