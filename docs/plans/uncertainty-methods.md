# Collision-risk methods: ensemble versus conformal

## Research question

The v0.6 planner tournament showed that categorical search and a calibrated
collision-risk budget do not fix the barrier failure. This tranche widens the
comparison at the risk-estimator level while keeping the primary dynamics
model, task generator, horizon, and candidate budget fixed:

- `learned_collision`: the existing single-checkpoint probabilistic collision head;
- `ensemble_collision`: three existing v3 checkpoints, aggregated as
  `mean + 1.0 * member_std`;
- `conformal_collision`: the same ensemble risk plus a split-conformal upper
  margin fitted on held-out barrier routes at `alpha=0.10`.

The ensemble members are `v3`, `v3-calibrated`, and `v3-kinematics`. They are
not claimed to be an independently bootstrapped research ensemble; they are a
transparent checkpoint-diversity ablation. This keeps the result reproducible
and makes the limitation explicit.

## Protocol

- single vertical barrier;
- evaluation seeds `11, 23, 41`;
- 20 paired episodes per seed, 60 routes total;
- 48-step horizon;
- 256 candidate action sequences per one-shot planning call;
- calibration seeds `101, 103`, 12 episodes each;
- same v3-final position/dynamics model for all three methods.

Reproduce it with:

```bash
python -m pocketworld.evaluate_uncertainty \
  artifacts/pocketworld-map-suite-v3-final.pt \
  --ensemble-checkpoints artifacts/pocketworld-map-suite-v3.pt,artifacts/pocketworld-map-suite-v3-calibrated.pt,artifacts/pocketworld-map-suite-v3-kinematics.pt \
  --calibration-seeds 101,103 --evaluation-seeds 11,23,41 \
  --calibration-episodes 12 --evaluation-episodes 20 \
  --horizon 48 --candidates 256 \
  --output artifacts/evaluation-uncertainty-barrier-v1.json
```

## Result

| Risk method | Imagined success | Real success | Real final distance | Collisions/episode | Held-out upper coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single learned collision | 96.7% | 61.7% | 9.07 px | 2.47 | — |
| Ensemble mean + disagreement | 26.7% | 20.0% | 22.99 px | 2.75 | 0.0% |
| Conformal upper risk | 0.0% | 0.0% | 44.77 px | 0.00 | 100.0% |

The conformal calibration quantile is `0.9984`. The calibration routes have a
100% observed collision rate, so the finite-sample upper bound becomes almost
one and the planner stops after one action. This is a useful negative result:
coverage is not the same as a usable route planner.

The ensemble also does not solve the obstacle problem. It changes the
selection distribution and lowers imagined completion, but selected routes
still collide. The ensemble's lower mean reported risk than the single model
must not be read as better calibration: it is partly caused by selecting
shorter, less-complete plans.

## Interpretation and next comparison

The bottleneck has moved from “which search method?” to “what is the unit of
uncertainty calibration?” A route-level label of “any collision in 48 steps”
is too coarse for a model whose risk head is trained on local transitions. The
next principled extension is a transition-level and route-conditioned
calibration matrix, followed by a Pareto sweep over risk coverage versus
completion. It should use diverse successful and failed routes, not only the
selected routes produced by one planner.

The machine-readable report is
[`evaluation-uncertainty-barrier-v1.json`](../results/evaluation-uncertainty-barrier-v1.json).

