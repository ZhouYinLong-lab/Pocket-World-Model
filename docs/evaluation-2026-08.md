# PocketWorld evaluation — 2026-08-03

The current main checkpoint is trained with 1,000 trajectories, 200 validation trajectories, 16-step unrolls, sticky actions (`p=0.75`), and full-map start/goal sampling. The report was generated with 50 episodes per seed, 3 seeds (`11, 23, 41`), and 1,024 random-shooting candidates.

The collision-supervised barrier-mix checkpoint is an ablation, trained with the same scale plus `--barrier-probability 0.25`. Its quick report uses 10 episodes per seed and 512 candidates, so it is not used to replace the larger main-checkpoint headline numbers.

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

## Renderer and collision follow-up

The renderer checkpoint adds a separate agent-mask head and a state-conditioned RGB composition path. Raw RGB decoding still loses the small agent, but the composited output has 100% position coverage and 1.69/2.32/3.15/3.94px position error at 1/5/10/20 steps in a 20-episode check. The raw decoder remains available as the honest pixel baseline; the composited output is the deployable visualization path.

The learned mask head is useful as a diagnostic but is not yet a good shape decoder: its 20-episode mask IoU is about 0.03–0.04 ID/OOD, despite 100% thresholded coverage after focused fine-tuning. This is why the final RGB path uses the structured position plus the known circular agent geometry rather than claiming that the mask head has solved pixel rendering.

The pure learned collision planner still reaches roughly 93% imagined / 0% real success on the 20-episode barrier check. A new `hybrid` planner adds a local wall-patch guard and post-collision freeze; it reaches about 5% real success and reduces mean final distance to about 25px, while the explicit pixel-wall baseline remains the stronger obstacle reference. The gap is now localized to event localization and detour selection, not missing API plumbing.

Next work should focus on wall-relative state transitions or a closed-loop planner that learns from collision-free waypoint progress. The current RGB deployment path, ONNX position contract, and negative collision result are already reproducible.

The barrier evaluation compares the unconstrained planner with a collision-aware planner that extracts wall pixels from the current observation and penalizes trajectories after the first wall intersection. This is an explicit planning baseline; it is not counted as learned wall dynamics until the compact state model predicts collision events itself.

The first barrier result was intentionally a negative result: the unconstrained planner reached 100% imagined success but 0% real success. Structured top/bottom detour proposals now raise collision-aware open-loop success to 10% on the same 10-episode, 40-step check and reduce mean final distance from 28.18px to 21.34px. At horizon 64, the mean distance falls further to 17.54px, but success remains only 10%.

The implementation now includes `receding_horizon_plan`, which replans from the latest real observation after every action. Its barrier result is tracked separately from the open-loop baselines so closed-loop correction can be measured without changing the headline planning numbers.

The quick closed-loop check still reached 0% real success: 1-step replanning averaged 28.19px, 4-step commitment averaged 27.91px, and route-preserving replanning averaged 27.72px. All remain worse than executing the best structured detour open-loop (21.27px in the same check), confirming that the missing piece is collision-state modeling rather than route-cache lifetime. The route-preserving mode caches the imagined path and only replans on collision or >6px route deviation; its result is reported separately as `collision_aware_route`.

The collision-supervised barrier-mix ablation reached 81.9% ID collision accuracy / 69.4% recall and 76.9% OOD accuracy / 56.1% recall. Its learned-collision planner still achieved 100% imagined but 0% real success on the barrier challenge, while the pixel wall baseline reached about 5% real success. This is a useful negative result: event labels alone do not make the compact state wall-relative.

A follow-up added a 7x7 wall-relative patch around the predicted landing point to the collision head. It improved the quick OOD collision accuracy to 82.9% / 65.9% recall, but learned-collision planning remained 100% imagined / 0% real. The subsequent post-collision response therefore changed the imagined state, rather than only classifying the event.

The post-collision response is now implemented: predicted events freeze position and clear velocity in learned imagined rollouts. The pure learned barrier result remains approximately 93–95% imagined / 0% real, so the missing issue is event localization on detour candidates, not the response rule itself. The hybrid visual guard is reported as a separate engineering baseline.

## Engineering completion checks

- Repository metadata: MIT license committed; GitHub topics set for world models, model-based RL, PyTorch, Gymnasium, and machine learning.
- Python verification: `20 passed`.
- ONNX export: verified against `pocketworld-renderer-v5.pt`; the compatible composited RGB model is written successfully.
- ONNX contract: the exported graph exposes `next_observation[batch,3,64,64]` and `next_position[batch,2]`; the browser prefers the supervised position channel for its model marker.
- Web verification: `npm install --no-audit --no-fund` followed by `npm run build` succeeds. Vite reports only a bundle-size warning for the ONNX Runtime WASM asset.

The remaining research gap is not build or packaging: the browser can load the one-step ONNX composited RGB model, while pure learned collision planning still fails the single-barrier real-execution test. The remaining product-level visualization path is usable; the remaining scientific question is whether collision-aware planning can learn wall-relative dynamics rather than rely on a visual guard.
