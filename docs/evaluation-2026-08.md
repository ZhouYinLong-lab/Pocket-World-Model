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
